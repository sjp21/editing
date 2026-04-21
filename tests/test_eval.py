from __future__ import annotations

from pathlib import Path

import pytest

from clipper.eval import _iou, run_eval
from clipper.schemas import (
    Candidate,
    HookTreatment,
    HumanLabels,
    LabeledClip,
    ScoreResult,
    Scores,
    ShapedClip,
    ShapedTrim,
    ShapeResult,
)


# ---------------------------------------------------------------------------
# Builders — small helpers that fill in the boring required fields.
# ---------------------------------------------------------------------------


def _scores() -> Scores:
    return Scores(
        hook_strength=5,
        emotional_peak=5,
        quotability=5,
        self_contained=5,
        visual_interest=5,
        reaction_worthy=5,
    )


def _candidate(*, start: float, end: float, score_index: int) -> Candidate:
    return Candidate(
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        scores=_scores(),
        overall_score=5.0,
        confidence=0.5,
        rationale="x",
        suggested_hook="y",
        score_index=score_index,
    )


def _score_result(candidates: list[Candidate]) -> ScoreResult:
    return ScoreResult(
        candidates=candidates,
        model="claude-opus-4-7",
        prompt_version="v2",
    )


def _ranked_survivors(
    scored: ScoreResult, survivor_score_indexes: list[int]
) -> ScoreResult:
    """Mimic ``apply_ranking`` — filter to survivors, preserving identity.

    ``ranked.json`` on disk is a ScoreResult (the filtered survivors in
    rank order), not a RankedCandidates. eval only needs the set of
    surviving ``score_index`` values, so that's what we write here.
    """
    by_id = {c.score_index: c for c in scored.candidates}
    return ScoreResult(
        candidates=[by_id[i] for i in survivor_score_indexes],
        model=scored.model,
        prompt_version=scored.prompt_version,
    )


def _shape_result(
    shaped: list[tuple[int, str]], source_ranker_prompt_version: str | None = "v1"
) -> ShapeResult:
    """``shaped`` is a list of ``(candidate_index, verdict)`` pairs."""
    clips: list[ShapedClip] = []
    for cid, verdict in shaped:
        if verdict == "keep":
            clips.append(
                ShapedClip(
                    candidate_index=cid,
                    source_start=0.0,
                    source_end=18.0,
                    verdict="keep",
                    trim=ShapedTrim(
                        start_seconds=0.0,
                        end_seconds=15.0,
                        start_word="Most",
                        end_word="end.",
                        duration_seconds=15.0,
                    ),
                    hook=HookTreatment(kind="overlay", text="hook", rationale="r"),
                    reasoning="r",
                    model="claude-opus-4-7",
                    prompt_version="v3",
                )
            )
        else:
            clips.append(
                ShapedClip(
                    candidate_index=cid,
                    source_start=0.0,
                    source_end=18.0,
                    verdict="kill",
                    kill_reason="not self-contained",
                    reasoning="r",
                    model="claude-opus-4-7",
                    prompt_version="v3",
                )
            )
    return ShapeResult(
        shaped=clips,
        model="claude-opus-4-7",
        prompt_version="v3",
        source_score_prompt_version="v2",
        source_ranker_prompt_version=source_ranker_prompt_version,
    )


def _write_inputs(
    root: Path,
    *,
    labels: HumanLabels,
    scored: ScoreResult,
    ranked: ScoreResult | None = None,
    shaped: ShapeResult | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "human_labels.json").write_text(labels.model_dump_json(indent=2))
    (root / "candidates.json").write_text(scored.model_dump_json(indent=2))
    if ranked is not None:
        (root / "ranked.json").write_text(ranked.model_dump_json(indent=2))
    if shaped is not None:
        (root / "shaped.json").write_text(shaped.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# IoU math
# ---------------------------------------------------------------------------


def test_iou_worked_example_from_doc():
    # The hand-label doc's worked example: label [60,80], candidate [58,74]
    # -> intersection 14, union 22, IoU ≈ 0.636.
    iou = _iou(60.0, 80.0, 58.0, 74.0)
    assert iou == pytest.approx(14.0 / 22.0)
    assert iou >= 0.5  # crosses default threshold -> match


def test_iou_disjoint_is_zero():
    assert _iou(60.0, 80.0, 100.0, 120.0) == 0.0


def test_iou_identical_is_one():
    assert _iou(60.0, 80.0, 60.0, 80.0) == 1.0


def test_iou_touching_is_zero():
    # Adjacent but non-overlapping: intersection length is 0.
    assert _iou(60.0, 80.0, 80.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# Tier asymmetry — `rejected` + shape kept it = false positive
# ---------------------------------------------------------------------------


def test_rejected_label_kept_by_shape_is_false_positive(tmp_path):
    labels = HumanLabels(
        video_stem="v",
        labeler="t",
        labeled_at="2026-04-21T00:00:00Z",
        labels=[
            LabeledClip(
                start=60.0, end=80.0, tier="rejected",
                reason="ranker loved it but lands on a hedge",
            ),
        ],
    )
    scored = _score_result([_candidate(start=58.0, end=74.0, score_index=0)])
    ranked = _ranked_survivors(scored, [0])
    shaped = _shape_result([(0, "keep")])
    _write_inputs(tmp_path, labels=labels, scored=scored, ranked=ranked, shaped=shaped)

    card = run_eval("v", output_dir=tmp_path)

    assert card.false_positives == [0]
    # `rejected` + shape=keep is the worst-case shape miss for this label.
    assert card.shape_alignment == 0.0


def test_rejected_label_killed_by_shape_is_correct(tmp_path):
    labels = HumanLabels(
        video_stem="v",
        labeler="t",
        labeled_at="2026-04-21T00:00:00Z",
        labels=[
            LabeledClip(
                start=60.0, end=80.0, tier="rejected", reason="hedge ending",
            ),
        ],
    )
    scored = _score_result([_candidate(start=58.0, end=74.0, score_index=0)])
    ranked = _ranked_survivors(scored, [0])
    shaped = _shape_result([(0, "kill")])
    _write_inputs(tmp_path, labels=labels, scored=scored, ranked=ranked, shaped=shaped)

    card = run_eval("v", output_dir=tmp_path)

    assert card.false_positives == []
    assert card.shape_alignment == 1.0


# ---------------------------------------------------------------------------
# Three scorecard numbers on a small fixture
# ---------------------------------------------------------------------------


def _fixture_labels() -> HumanLabels:
    return HumanLabels(
        video_stem="v",
        labeler="t",
        labeled_at="2026-04-21T00:00:00Z",
        labels=[
            # 0: must_ship — matches candidate 0, in ranker top-K, shape keeps
            LabeledClip(start=60.0, end=80.0, tier="must_ship", reason="A"),
            # 1: must_ship — no candidate matches at IoU >= 0.5 -> miss
            LabeledClip(start=400.0, end=420.0, tier="must_ship", reason="B"),
            # 2: nice — matches candidate 1, ranker dropped it -> shape "missing"
            LabeledClip(start=200.0, end=220.0, tier="nice", reason="C"),
            # 3: rejected — matches candidate 2, ranker kept it, shape kept it -> FP
            LabeledClip(start=300.0, end=320.0, tier="rejected", reason="D"),
            # 4: marginal — excluded from shape_alignment regardless
            LabeledClip(start=500.0, end=520.0, tier="marginal", reason="E"),
        ],
    )


def _fixture_scored() -> ScoreResult:
    return _score_result(
        [
            _candidate(start=58.0, end=74.0, score_index=0),    # IoU ≈ 0.64 with label 0
            _candidate(start=198.0, end=215.0, score_index=1),  # IoU ≈ 0.68 with label 2
            _candidate(start=298.0, end=315.0, score_index=2),  # IoU ≈ 0.68 with label 3
            _candidate(start=498.0, end=515.0, score_index=3),  # matches label 4 (marginal)
        ]
    )


def test_scorecard_numbers_on_fixture(tmp_path):
    labels = _fixture_labels()
    scored = _fixture_scored()
    # Ranker keeps candidates 0, 2, 3; drops 1.
    ranked = _ranked_survivors(scored, [0, 2, 3])
    # Shape keeps the must_ship hit and the rejected one (the bug we're trying
    # to detect); kills the marginal-matched candidate. Cand 1 was filtered by
    # the ranker so it never reached shape.
    shaped = _shape_result([(0, "keep"), (2, "keep"), (3, "kill")])
    _write_inputs(tmp_path, labels=labels, scored=scored, ranked=ranked, shaped=shaped)

    card = run_eval("v", output_dir=tmp_path)

    # Scorer recall: 4 non-rejected labels (0, 1, 2, 4); 3 of them matched.
    assert card.scorer_recall == pytest.approx(3 / 4)
    # Ranker inclusion: 2 must_ship labels; only label 0 has a matched
    # candidate that's in the ranker top-K. Label 1 has no match -> not in top-K.
    assert card.ranker_inclusion == pytest.approx(1 / 2)
    # Shape alignment: relevant labels are 0 (ms→keep ✓) and 3 (rej→keep ✗).
    # Label 2 maps to "missing" (ranker filtered) — excluded.
    # Label 4 is marginal — excluded.
    assert card.shape_alignment == pytest.approx(1 / 2)

    assert card.misses == [1]
    assert card.false_positives == [3]


def test_per_label_match_fields(tmp_path):
    labels = _fixture_labels()
    scored = _fixture_scored()
    ranked = _ranked_survivors(scored, [0, 2, 3])
    shaped = _shape_result([(0, "keep"), (2, "keep"), (3, "kill")])
    _write_inputs(tmp_path, labels=labels, scored=scored, ranked=ranked, shaped=shaped)

    card = run_eval("v", output_dir=tmp_path)
    by_idx = {m.label_index: m for m in card.matches}

    # Label 0 — must_ship matched, in top-K, shape kept.
    assert by_idx[0].matched_score_index == 0
    assert by_idx[0].in_ranker_topk is True
    assert by_idx[0].shape_verdict == "keep"

    # Label 1 — must_ship miss; ranker known to have run so in_ranker_topk=False.
    assert by_idx[1].matched_score_index is None
    assert by_idx[1].iou is None
    assert by_idx[1].in_ranker_topk is False
    assert by_idx[1].shape_verdict is None

    # Label 2 — nice matched candidate the ranker dropped -> "missing" at shape.
    assert by_idx[2].matched_score_index == 1
    assert by_idx[2].in_ranker_topk is False
    assert by_idx[2].shape_verdict == "missing"

    # Label 3 — rejected, ranker kept, shape kept (the false-positive case).
    assert by_idx[3].shape_verdict == "keep"


def test_no_ranker_means_inclusion_is_vacuously_one(tmp_path):
    labels = HumanLabels(
        video_stem="v", labeler="t", labeled_at="2026-04-21T00:00:00Z",
        labels=[LabeledClip(start=60.0, end=80.0, tier="must_ship", reason="r")],
    )
    scored = _score_result([_candidate(start=58.0, end=74.0, score_index=0)])
    _write_inputs(tmp_path, labels=labels, scored=scored)

    card = run_eval("v", output_dir=tmp_path)

    assert card.ranker_inclusion == 1.0
    assert card.matches[0].in_ranker_topk is None
    # Shape didn't run either; no relevant labels -> 1.0.
    assert card.shape_alignment == 1.0
    assert card.matches[0].shape_verdict is None


def test_no_shape_run_means_alignment_is_vacuously_one(tmp_path):
    labels = HumanLabels(
        video_stem="v", labeler="t", labeled_at="2026-04-21T00:00:00Z",
        labels=[
            LabeledClip(start=60.0, end=80.0, tier="must_ship", reason="r"),
            LabeledClip(start=300.0, end=320.0, tier="rejected", reason="r"),
        ],
    )
    scored = _score_result(
        [
            _candidate(start=58.0, end=74.0, score_index=0),
            _candidate(start=298.0, end=315.0, score_index=1),
        ]
    )
    ranked = _ranked_survivors(scored, [0, 1])
    _write_inputs(tmp_path, labels=labels, scored=scored, ranked=ranked)

    card = run_eval("v", output_dir=tmp_path)

    assert card.shape_alignment == 1.0
    assert all(m.shape_verdict is None for m in card.matches)


def test_score_index_falls_back_to_position_when_missing(tmp_path):
    # Old candidates.json files predate score_index. The fallback to
    # positional index keeps eval joinable.
    cand = _candidate(start=58.0, end=74.0, score_index=0)
    cand.score_index = None
    scored = _score_result([cand])
    labels = HumanLabels(
        video_stem="v", labeler="t", labeled_at="2026-04-21T00:00:00Z",
        labels=[LabeledClip(start=60.0, end=80.0, tier="must_ship", reason="r")],
    )
    _write_inputs(tmp_path, labels=labels, scored=scored)

    card = run_eval("v", output_dir=tmp_path)

    assert card.matches[0].matched_score_index == 0  # positional fallback


def test_iou_threshold_override_changes_match(tmp_path):
    labels = HumanLabels(
        video_stem="v", labeler="t", labeled_at="2026-04-21T00:00:00Z",
        labels=[LabeledClip(start=60.0, end=80.0, tier="nice", reason="r")],
    )
    # IoU ~ 0.636 — passes 0.5 default but fails a tightened 0.7.
    scored = _score_result([_candidate(start=58.0, end=74.0, score_index=0)])
    _write_inputs(tmp_path, labels=labels, scored=scored)

    loose = run_eval("v", output_dir=tmp_path, iou_threshold=0.5)
    strict = run_eval("v", output_dir=tmp_path, iou_threshold=0.7)

    assert loose.matches[0].matched_score_index == 0
    assert strict.matches[0].matched_score_index is None
    assert strict.misses == [0]
