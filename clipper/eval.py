"""Hand-labeled regression scorecard.

Joins ``human_labels.json`` against the pipeline outputs (``candidates.json``,
``ranked.json`` if present, ``shaped.json`` if present) and emits a
``LabelScorecard`` with three numbers per video:

- ``scorer_recall``    — fraction of non-``rejected`` labels matched by a
                         scorer candidate at IoU ≥ ``iou_threshold``.
- ``ranker_inclusion`` — fraction of ``must_ship`` labels whose matched
                         candidate landed in the ranker's top-K. ``1.0``
                         when no ``must_ship`` labels exist or the ranker
                         did not run.
- ``shape_alignment``  — asymmetric per the hand-label doc:
                         * ``must_ship``/``nice`` reaching shape → ``keep`` is correct.
                         * ``rejected`` reaching shape → ``kill`` is correct.
                         * ``marginal`` is excluded (genuine fence-sitting).

Pure local computation — no Claude call.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clipper.schemas import (
    HumanLabels,
    LabeledClip,
    LabelMatch,
    LabelScorecard,
    ScoreResult,
    ShapeResult,
)


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0.0 else 0.0


def _candidate_id(c, positional: int) -> int:
    """Stable identity for matching against ranker/shape outputs.

    ``score_index`` is the contractual join key but may be ``None`` for
    artifacts written before that field existed. Falls back to position,
    matching shape's behavior.
    """
    return c.score_index if c.score_index is not None else positional


def _best_match(
    label: LabeledClip, scored: ScoreResult, iou_threshold: float
) -> tuple[int | None, float | None]:
    best_id: int | None = None
    best_iou: float | None = None
    for pos, c in enumerate(scored.candidates):
        iou = _iou(label.start, label.end, c.start_seconds, c.end_seconds)
        if iou < iou_threshold:
            continue
        if best_iou is None or iou > best_iou:
            best_iou = iou
            best_id = _candidate_id(c, pos)
    return best_id, best_iou


def _shape_verdict(
    matched_id: int | None,
    ranker_topk: set[int] | None,
    shaped: ShapeResult | None,
) -> str | None:
    if shaped is None or matched_id is None:
        return None
    by_index = {s.candidate_index: s.verdict for s in shaped.shaped}
    if matched_id in by_index:
        return by_index[matched_id]
    if ranker_topk is not None and matched_id not in ranker_topk:
        return "missing"
    return "failed"


def run_eval(
    video_stem: str,
    *,
    output_dir: Path,
    iou_threshold: float = 0.5,
    labels_path: Path | None = None,
    candidates_path: Path | None = None,
    ranked_path: Path | None = None,
    shaped_path: Path | None = None,
) -> LabelScorecard:
    """Compute the regression scorecard for one video.

    ``output_dir`` is the per-video root (``outputs/<stem>/``). Path
    overrides let callers point at non-default layouts.
    """
    labels_path = labels_path or output_dir / "human_labels.json"
    candidates_path = candidates_path or output_dir / "candidates.json"
    ranked_path = ranked_path or output_dir / "ranked.json"
    shaped_path = shaped_path or output_dir / "shaped.json"

    labels = HumanLabels.model_validate_json(labels_path.read_text())
    scored = ScoreResult.model_validate_json(candidates_path.read_text())
    # ranked.json is the filtered ScoreResult (survivors in rank order),
    # not the full RankedCandidates (which lives in ranking_decisions.json).
    # All eval needs is "which score_indexes survived" — that's right here.
    ranker_topk: set[int] | None = None
    if ranked_path.is_file():
        ranked = ScoreResult.model_validate_json(ranked_path.read_text())
        ranker_topk = {
            _candidate_id(c, pos) for pos, c in enumerate(ranked.candidates)
        }
    shaped = (
        ShapeResult.model_validate_json(shaped_path.read_text())
        if shaped_path.is_file()
        else None
    )

    matches: list[LabelMatch] = []
    for i, label in enumerate(labels.labels):
        matched_id, iou = _best_match(label, scored, iou_threshold)
        in_topk: bool | None
        if ranker_topk is None or matched_id is None:
            in_topk = None if ranker_topk is None else False
        else:
            in_topk = matched_id in ranker_topk
        matches.append(
            LabelMatch(
                label_index=i,
                tier=label.tier,
                reason=label.reason,
                matched_score_index=matched_id,
                iou=iou,
                in_ranker_topk=in_topk,
                shape_verdict=_shape_verdict(matched_id, ranker_topk, shaped),
            )
        )

    misses = [m.label_index for m in matches if m.matched_score_index is None]
    false_positives = [
        m.label_index
        for m in matches
        if m.tier == "rejected" and m.shape_verdict == "keep"
    ]

    non_rejected = [m for m in matches if m.tier != "rejected"]
    scorer_recall = (
        sum(1 for m in non_rejected if m.matched_score_index is not None)
        / len(non_rejected)
        if non_rejected
        else 1.0
    )

    must_ship = [m for m in matches if m.tier == "must_ship"]
    if not must_ship or ranker_topk is None:
        ranker_inclusion = 1.0
    else:
        ranker_inclusion = sum(1 for m in must_ship if m.in_ranker_topk) / len(must_ship)

    # Shape alignment: only labels that actually reached shape contribute.
    # `missing` (ranker-filtered) and `None` (no match / shape didn't run)
    # do not get counted against the shaper. `marginal` is excluded by
    # design — fence-sitting is real signal, not a regression.
    shape_relevant = [
        m for m in matches
        if m.tier != "marginal" and m.shape_verdict in {"keep", "kill", "failed"}
    ]
    if not shape_relevant:
        shape_alignment = 1.0
    else:
        correct = sum(
            1 for m in shape_relevant
            if (m.tier in {"must_ship", "nice"} and m.shape_verdict == "keep")
            or (m.tier == "rejected" and m.shape_verdict == "kill")
        )
        shape_alignment = correct / len(shape_relevant)

    return LabelScorecard(
        video_stem=video_stem,
        iou_threshold=iou_threshold,
        scorer_recall=scorer_recall,
        ranker_inclusion=ranker_inclusion,
        shape_alignment=shape_alignment,
        matches=matches,
        misses=misses,
        false_positives=false_positives,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
