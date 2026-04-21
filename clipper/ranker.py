"""Global ranker — comparative top-K selection across a full video.

The scorer dedupes by IoU within overlapping windows; it cannot catch
semantic duplicates (the same story told twice) and cannot rank across
the whole video. The ranker makes one Claude call that sees every
candidate at once and returns:

    - `ranked`:  up to K clips, best first, each with the three mandatory
                 justifications (hook_quote, payoff_in_window, comparative_reason).
    - `dropped`: every other input candidate, with a real reason.
    - `truncation_note`: honest when the model returns fewer than K.

It sits between score and shape. Shape remains ignorant of the ranker;
it keeps shaping whatever `ScoreResult` it is handed. Stable identity
across the seam rides on `Candidate.score_index`.

One Claude call per ranker run. No transcript excerpts — comparison is
on the scorer's distilled output.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import anthropic

from clipper._profiles import inject_profile, load_profile
from clipper.schemas import (
    DroppedClip,
    RankedCandidates,
    RankedClip,
    ScoreResult,
)

log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"
# Opus-tier required. The ranker is the one call in the pipeline that
# sees every candidate at once and has to make honest comparative taste
# calls (including the courage to truncate). Smaller models flatten the
# ranking into averaged scores.
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_K = 5
# Guardrail: above this many candidates the comparative call degrades
# into a pseudo-ranking regardless. Warn and cap rather than silently
# truncate.
MAX_CANDIDATES = 150


@dataclass
class RankerConfig:
    model: str = DEFAULT_MODEL
    prompt_path: Path | None = None
    max_tokens: int = 8192
    max_retries: int = 2
    effort: str = "high"
    profile: str | None = None
    k: int = DEFAULT_K
    log_path: Path | None = None
    profiles_dir: Path | None = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_rubric(prompt_path: Path | None) -> str:
    if prompt_path is not None:
        return Path(prompt_path).read_text()
    return (
        resources.files("clipper")
        .joinpath(f"prompts/{PROMPT_VERSION}/ranking.md")
        .read_text(encoding="utf-8")
    )


def _load_soul() -> str:
    return resources.files("clipper").joinpath("SOUL.md").read_text(encoding="utf-8")


def _build_system_prompt(
    soul: str,
    rubric: str,
    profile_text: str,
) -> str:
    body = inject_profile(rubric, profile_text)
    return f"{soul}\n\n---\n\n{body}"


# ---------------------------------------------------------------------------
# User-message serialization
# ---------------------------------------------------------------------------


def _build_user_message(scored: ScoreResult, k: int) -> str:
    lines: list[str] = []
    lines.append(f"REQUESTED_K: {k}")
    lines.append(f"VIDEO_PROMPT_VERSION: {scored.prompt_version}")
    lines.append("")
    lines.append(f"CANDIDATES (count={len(scored.candidates)}):")
    lines.append("")

    for cand in scored.candidates:
        idx = cand.score_index
        if idx is None:
            raise ValueError(
                "ScoreResult has candidates without score_index. This file "
                "predates the ranker; re-run score to populate score_index."
            )
        s = cand.scores
        lines.append(f"--- CANDIDATE score_index={idx} ---")
        lines.append(
            f"time: [{cand.start_seconds:.2f} - {cand.end_seconds:.2f}] "
            f"({cand.duration_seconds:.1f}s)"
        )
        lines.append(
            f"scores: hook={s.hook_strength:.1f} peak={s.emotional_peak:.1f} "
            f"quote={s.quotability:.1f} self_contained={s.self_contained:.1f} "
            f"visual={s.visual_interest:.1f} reaction={s.reaction_worthy:.1f}"
        )
        lines.append(
            f"overall_score: {cand.overall_score:.2f}  "
            f"confidence: {cand.confidence:.2f}"
        )
        lines.append(f"suggested_hook: {cand.suggested_hook!r}")
        lines.append(f"score_rationale: {cand.rationale}")
        lines.append("")

    lines.append(
        "Return ONLY the JSON object specified by the rubric. "
        "Every candidate above must appear in either `ranked` or `dropped`."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_ranker(
    scored: ScoreResult,
    cfg: RankerConfig | None = None,
    client: anthropic.Anthropic | None = None,
) -> RankedCandidates:
    cfg = cfg or RankerConfig()
    client = client or anthropic.Anthropic()

    if len(scored.candidates) > MAX_CANDIDATES:
        log.warning(
            "scorer returned %d candidates; capping ranker input at %d. "
            "Extra candidates are dropped *before* the ranker sees them.",
            len(scored.candidates),
            MAX_CANDIDATES,
        )
        scored = scored.model_copy(
            update={"candidates": scored.candidates[:MAX_CANDIDATES]}
        )

    soul = _load_soul()
    rubric = _load_rubric(cfg.prompt_path)
    profile_text = load_profile(cfg.profile, cfg.profiles_dir, PROMPT_VERSION)
    system_prompt = _build_system_prompt(soul, rubric, profile_text)
    user_text = _build_user_message(scored, cfg.k)

    call_id = uuid.uuid4().hex

    last_err: Exception | None = None
    hint: str | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            messages: list[dict] = [{"role": "user", "content": user_text}]
            if hint is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": hint,
                    }
                )
            response = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": cfg.effort},
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            )
            text = _extract_text(response)
            ranked = _parse_and_validate(text, scored, cfg, call_id)
            usage = _usage_dict(response.usage)
            ranked.total_input_tokens = usage.get("input_tokens", 0)
            ranked.total_output_tokens = usage.get("output_tokens", 0)
            ranked.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            ranked.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

            if cfg.log_path is not None:
                _append_log(cfg.log_path, ranked, usage, scored.prompt_version)
            return ranked
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            hint = (
                "Previous response was rejected: "
                f"{e}. Return ONLY the JSON object described in the rubric, "
                "with every candidate in either `ranked` or `dropped`."
            )
            log.warning("ranker attempt %d parse failure: %s", attempt, e)
            time.sleep(0.5 * (attempt + 1))
        except anthropic.APIError as e:
            last_err = e
            log.warning("ranker attempt %d api error: %s", attempt, e)
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"ranker exhausted retries: {last_err}")


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _extract_text(response) -> str:
    parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "\n".join(parts).strip()


def _usage_dict(usage) -> dict:
    fields = [
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    return {f: (getattr(usage, f, 0) or 0) for f in fields}


def _parse_and_validate(
    text: str,
    scored: ScoreResult,
    cfg: RankerConfig,
    call_id: str,
) -> RankedCandidates:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    if not t.startswith("{"):
        m = _JSON_BLOCK.search(t)
        if not m:
            raise ValueError(f"no JSON object found in model output: {text[:200]}")
        t = m.group(0)

    payload = json.loads(t)

    ranked = [RankedClip(**r) for r in payload.get("ranked", [])]
    dropped = [DroppedClip(**d) for d in payload.get("dropped", [])]
    requested_k = int(payload.get("requested_k", cfg.k))
    truncation_note = payload.get("truncation_note")

    known_indices = {c.score_index for c in scored.candidates}
    seen: set[int] = set()
    for r in ranked:
        if r.candidate_index not in known_indices:
            raise ValueError(
                f"ranked candidate_index={r.candidate_index} not in score indices"
            )
        seen.add(r.candidate_index)
    for d in dropped:
        if d.candidate_index not in known_indices:
            raise ValueError(
                f"dropped candidate_index={d.candidate_index} not in score indices"
            )
        seen.add(d.candidate_index)

    missing = known_indices - seen
    if missing:
        raise ValueError(
            f"ranker omitted candidates from its decision: {sorted(missing)}. "
            "Every score candidate must appear in either `ranked` or `dropped`."
        )
    if len(seen) != len(ranked) + len(dropped):
        # This is duplicate-across-lists, caught more explicitly by the
        # RankedCandidates validator below; included for defense in depth.
        raise ValueError("candidate appears in both ranked and dropped")

    # Profile provenance — ranker sets these, not the model.
    profile = cfg.profile
    profile_version = None

    return RankedCandidates(
        ranked=ranked,
        dropped=dropped,
        requested_k=requested_k,
        truncation_note=truncation_note,
        profile=profile,
        profile_version=profile_version,
        model=cfg.model,
        prompt_version=PROMPT_VERSION,
        source_score_prompt_version=scored.prompt_version,
        call_id=call_id,
    )


# ---------------------------------------------------------------------------
# Post-processing: filter ScoreResult to the ranker's survivors
# ---------------------------------------------------------------------------


def apply_ranking(scored: ScoreResult, ranked: RankedCandidates) -> ScoreResult:
    """Return a ScoreResult containing only the ranker's survivors, in rank order.

    Survivors keep their original `score_index`. Shape reads that field
    unchanged, so `ShapedClip.candidate_index` survives the filter. All
    other ScoreResult fields are passed through.
    """
    by_index = {c.score_index: c for c in scored.candidates}
    survivors = []
    for r in sorted(ranked.ranked, key=lambda c: c.rank):
        cand = by_index.get(r.candidate_index)
        if cand is None:
            # Validator upstream already asserts this, but belt-and-braces.
            raise ValueError(
                f"ranked candidate_index={r.candidate_index} not in scored candidates"
            )
        survivors.append(cand)

    return scored.model_copy(update={"candidates": survivors})


# ---------------------------------------------------------------------------
# JSONL log — one row per decision plus one summary row
# ---------------------------------------------------------------------------


def _append_log(
    path: Path,
    ranked: RankedCandidates,
    usage: dict,
    source_score_prompt_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    base = {
        "ts": ts,
        "stage": "ranker",
        "prompt_version": ranked.prompt_version,
        "model": ranked.model,
        "source_score_prompt_version": source_score_prompt_version,
        "profile": ranked.profile,
        "profile_version": ranked.profile_version,
        "call_id": ranked.call_id,
    }

    rows: list[dict] = []
    for r in ranked.ranked:
        rows.append(
            {
                **base,
                "candidate_index": r.candidate_index,
                "verdict": "ranked",
                "rank": r.rank,
                "hook_quote": r.hook_quote,
                "payoff_in_window": r.payoff_in_window,
                "payoff_reason": r.payoff_reason,
                "comparative_reason": r.comparative_reason,
                "reason_kind": None,
                "duplicate_of_candidate_index": None,
                "profile_conflict_note": r.profile_conflict_note,
            }
        )
    for d in ranked.dropped:
        rows.append(
            {
                **base,
                "candidate_index": d.candidate_index,
                "verdict": "dropped",
                "rank": None,
                "hook_quote": d.hook_quote,
                "payoff_in_window": None,
                "payoff_reason": None,
                "comparative_reason": d.comparative_reason,
                "reason_kind": d.reason_kind,
                "duplicate_of_candidate_index": d.duplicate_of_candidate_index,
                "profile_conflict_note": None,
            }
        )

    summary = {
        **base,
        "kind": "ranker_summary",
        "requested_k": ranked.requested_k,
        "returned_k": len(ranked.ranked),
        "truncation_note": ranked.truncation_note,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
    }

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")
