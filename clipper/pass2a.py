"""Pass 2a — Shaping.

For each candidate produced by Pass 1, send the candidate metadata, ±30s
of transcript context, and a word-level timestamp table to Claude. The
model returns a keep/kill verdict plus (on keep) an exact word-aligned
trim and a hook treatment. Output is a ``Pass2aResult``.

Reasoning only — no frames. Frames are Pass 2b's concern.

Observability:
- System prompt (rubric) is cached via ``cache_control: ephemeral``.
- Token usage aggregates into ``Pass2aResult``.
- Each shaped candidate appends one JSONL line to the pass2a log so the
  eventual feedback loop has the data regardless of outcome.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import anthropic

from clipper.schemas import (
    Candidate,
    HookTreatment,
    Pass1Result,
    Pass2aResult,
    ShapedClip,
    ShapedTrim,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_CONTEXT_PAD = 30.0
DEFAULT_WORD_TABLE_PAD = 5.0
DEFAULT_LOG_PATH = Path(".cache/logs/pass2a.jsonl")
TRIM_SLACK = 2.0
WORD_TIMESTAMP_TOLERANCE = 0.05


@dataclass
class Pass2aConfig:
    model: str = DEFAULT_MODEL
    prompt_path: Path | None = None
    max_tokens: int = 2048
    max_retries: int = 2
    effort: str = "high"
    context_pad: float = DEFAULT_CONTEXT_PAD
    word_table_pad: float = DEFAULT_WORD_TABLE_PAD
    log_path: Path | None = DEFAULT_LOG_PATH


def _load_rubric(prompt_path: Path | None) -> str:
    if prompt_path is not None:
        return Path(prompt_path).read_text()
    return (
        resources.files("clipper")
        .joinpath(f"prompts/{PROMPT_VERSION}/shaping.md")
        .read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Context & word-table extraction
# ---------------------------------------------------------------------------


def _extract_context(
    transcript: Transcript,
    start: float,
    end: float,
    pad: float = DEFAULT_CONTEXT_PAD,
) -> tuple[list[TranscriptSegment], list[TranscriptWord]]:
    """Return (segments overlapping [start-pad, end+pad], words in same range)."""
    lo = start - pad
    hi = end + pad

    segs: list[TranscriptSegment] = []
    for s in transcript.segments:
        if s.end < lo or s.start > hi:
            continue
        segs.append(s)

    words: list[TranscriptWord] = []
    for s in segs:
        for w in s.words:
            mid = (w.start + w.end) / 2.0
            if lo <= mid <= hi:
                words.append(w)
    words.sort(key=lambda w: w.start)
    return segs, words


def _segments_in_range(
    segments: list[TranscriptSegment], start: float, end: float
) -> list[TranscriptSegment]:
    return [s for s in segments if s.end >= start and s.start <= end]


def _format_transcript_block(segs: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{s.start:.2f} - {s.end:.2f}] {s.text}" for s in segs)


def _format_words_table(words: list[TranscriptWord]) -> str:
    return "\n".join(f't={w.start:.2f} "{w.word}"' for w in words)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_pass2a(
    pass1: Pass1Result,
    transcript: Transcript,
    cfg: Pass2aConfig | None = None,
    client: anthropic.Anthropic | None = None,
) -> Pass2aResult:
    cfg = cfg or Pass2aConfig()
    client = client or anthropic.Anthropic()
    rubric = _load_rubric(cfg.prompt_path)

    result = Pass2aResult(
        shaped=[],
        model=cfg.model,
        prompt_version=PROMPT_VERSION,
        source_pass1_prompt_version=pass1.prompt_version,
    )

    for idx, cand in enumerate(pass1.candidates):
        ctx_segs, ctx_words = _extract_context(
            transcript, cand.start_seconds, cand.end_seconds, pad=cfg.context_pad
        )
        try:
            shaped, usage = _shape_candidate(
                client, cand, idx, ctx_segs, ctx_words, rubric, cfg
            )
        except Exception as e:  # noqa: BLE001
            log.warning("candidate %d failed: %s", idx, e)
            result.candidates_failed += 1
            continue

        result.shaped.append(shaped)
        if shaped.verdict == "keep":
            result.candidates_shaped += 1
        else:
            result.candidates_killed += 1

        result.total_input_tokens += usage.get("input_tokens", 0)
        result.total_output_tokens += usage.get("output_tokens", 0)
        result.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        result.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        log.info(
            "cand %d done: verdict=%s in=%d out=%d cache_read=%d",
            idx,
            shaped.verdict,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
        )

        if cfg.log_path is not None:
            _append_log(cfg.log_path, shaped, usage, pass1.prompt_version)

    return result


def _shape_candidate(
    client: anthropic.Anthropic,
    cand: Candidate,
    idx: int,
    ctx_segs: list[TranscriptSegment],
    ctx_words: list[TranscriptWord],
    rubric: str,
    cfg: Pass2aConfig,
) -> tuple[ShapedClip, dict]:
    user_text = _build_user_message(cand, idx, ctx_segs, ctx_words, cfg)

    last_err: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": cfg.effort},
                system=[
                    {
                        "type": "text",
                        "text": rubric,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
            )
            text = _extract_text(response)
            shaped = _parse_and_validate(text, cand, idx, ctx_words, cfg)
            usage = _usage_dict(response.usage)
            return shaped, usage
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            log.warning("cand %d attempt %d parse failure: %s", idx, attempt, e)
            time.sleep(0.5 * (attempt + 1))
        except anthropic.APIError as e:
            last_err = e
            log.warning("cand %d attempt %d api error: %s", idx, attempt, e)
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"candidate {idx} exhausted retries: {last_err}")


def _build_user_message(
    cand: Candidate,
    idx: int,
    ctx_segs: list[TranscriptSegment],
    ctx_words: list[TranscriptWord],
    cfg: Pass2aConfig,
) -> str:
    before = [s for s in ctx_segs if s.end <= cand.start_seconds]
    inside = _segments_in_range(ctx_segs, cand.start_seconds, cand.end_seconds)
    after = [s for s in ctx_segs if s.start >= cand.end_seconds]

    table_lo = cand.start_seconds - cfg.word_table_pad
    table_hi = cand.end_seconds + cfg.word_table_pad
    table_words = [
        w for w in ctx_words if (w.start + w.end) / 2.0 >= table_lo
        and (w.start + w.end) / 2.0 <= table_hi
    ]

    header = (
        f"CANDIDATE #{idx}\n"
        f"source_start: {cand.start_seconds:.2f}\n"
        f"source_end: {cand.end_seconds:.2f}\n"
        f"overall_score: {cand.overall_score:.2f}\n"
        f"suggested_hook (from Pass 1): {cand.suggested_hook!r}\n"
        f"pass1_rationale: {cand.rationale}"
    )

    body_parts = [
        header,
        "",
        "[CONTEXT_BEFORE]",
        _format_transcript_block(before) or "(none)",
        "",
        "[CANDIDATE]",
        _format_transcript_block(inside) or "(none)",
        "",
        "[CONTEXT_AFTER]",
        _format_transcript_block(after) or "(none)",
        "",
        "WORD TABLE (candidate ± 5s; one word per line):",
        _format_words_table(table_words) or "(no words)",
        "",
        "Return ONLY the JSON object specified by the rubric.",
    ]
    return "\n".join(body_parts)


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
    cand: Candidate,
    idx: int,
    ctx_words: list[TranscriptWord],
    cfg: Pass2aConfig,
) -> ShapedClip:
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

    trim_raw = payload.get("trim")
    hook_raw = payload.get("hook")

    trim = ShapedTrim(**trim_raw) if trim_raw else None
    hook = HookTreatment(**hook_raw) if hook_raw else None

    shaped = ShapedClip(
        candidate_index=idx,
        source_start=cand.start_seconds,
        source_end=cand.end_seconds,
        verdict=payload["verdict"],
        kill_reason=payload.get("kill_reason"),
        trim=trim,
        hook=hook,
        reasoning=payload.get("reasoning", ""),
        model=cfg.model,
        prompt_version=PROMPT_VERSION,
    )

    if shaped.trim is not None:
        _enforce_trim_bounds(shaped.trim, cand)
        _enforce_word_alignment(shaped.trim, ctx_words)

    return shaped


def _enforce_trim_bounds(trim: ShapedTrim, cand: Candidate) -> None:
    lo = cand.start_seconds - TRIM_SLACK
    hi = cand.end_seconds + TRIM_SLACK
    if trim.start_seconds < lo - 1e-6 or trim.end_seconds > hi + 1e-6:
        raise ValueError(
            f"trim [{trim.start_seconds:.2f}, {trim.end_seconds:.2f}] outside "
            f"candidate ±{TRIM_SLACK}s bounds [{lo:.2f}, {hi:.2f}]"
        )


def _enforce_word_alignment(trim: ShapedTrim, words: list[TranscriptWord]) -> None:
    if not words:
        raise ValueError("cannot validate word alignment: empty word table")
    _match_word(trim.start_word, trim.start_seconds, words, edge="start")
    _match_word(trim.end_word, trim.end_seconds, words, edge="end")


def _match_word(
    quoted: str,
    timestamp: float,
    words: list[TranscriptWord],
    edge: str,
) -> None:
    """A word with matching timestamp (within tolerance) and same text must exist."""
    quoted_norm = _normalize_word(quoted)
    best_delta = float("inf")
    for w in words:
        ref = w.start if edge == "start" else w.end
        delta = abs(ref - timestamp)
        if delta > WORD_TIMESTAMP_TOLERANCE:
            continue
        if _normalize_word(w.word) == quoted_norm:
            return
        best_delta = min(best_delta, delta)
    raise ValueError(
        f"{edge}_word={quoted!r} at t={timestamp:.2f} does not match any "
        f"transcript word within {WORD_TIMESTAMP_TOLERANCE}s "
        f"(closest timestamp delta: {best_delta:.3f}s)"
    )


def _normalize_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w).strip().lower()


# ---------------------------------------------------------------------------
# JSONL log
# ---------------------------------------------------------------------------


def _append_log(
    path: Path,
    shaped: ShapedClip,
    usage: dict,
    source_pass1_prompt_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "pass2a",
        "prompt_version": shaped.prompt_version,
        "model": shaped.model,
        "source_pass1_prompt_version": source_pass1_prompt_version,
        "candidate_index": shaped.candidate_index,
        "source_start": shaped.source_start,
        "source_end": shaped.source_end,
        "verdict": shaped.verdict,
        "trim_start": shaped.trim.start_seconds if shaped.trim else None,
        "trim_end": shaped.trim.end_seconds if shaped.trim else None,
        "hook_kind": shaped.hook.kind if shaped.hook else None,
        "kill_reason": shaped.kill_reason,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
