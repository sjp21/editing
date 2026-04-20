"""Pass 1 — Candidate Identification.

For each window, send the transcript + signals summary + frames to Claude
with the two-stage filter-then-score prompt. Parse, validate, dedupe.

Observability / cost control:
- System prompt (the rubric) is cached via prompt caching.
- Usage counters are aggregated into Pass1Result for reproducibility.
- Every candidate carries model + prompt_version + window_index so the
  feedback loop envisaged in the plan can join labels to runs later.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import anthropic

from clipper.schemas import (
    AudioFeature,
    Candidate,
    Pass1Result,
    Scores,
    Signals,
    TextFeature,
    TranscriptSegment,
    Window,
)

log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-opus-4-7"
MAX_FRAMES_PER_WINDOW = 6


@dataclass
class Pass1Config:
    model: str = DEFAULT_MODEL
    # If None, the bundled prompt for PROMPT_VERSION is loaded via
    # importlib.resources so the installed CLI works from any cwd.
    prompt_path: Path | None = None
    max_tokens: int = 4096
    max_retries: int = 2
    frames_per_window: int = MAX_FRAMES_PER_WINDOW
    effort: str = "high"  # low | medium | high | max (opus-tier)


def _load_rubric(prompt_path: Path | None) -> str:
    if prompt_path is not None:
        return Path(prompt_path).read_text()
    return (
        resources.files("clipper")
        .joinpath(f"prompts/{PROMPT_VERSION}/candidate_identification.md")
        .read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Serialization helpers — turn a Window into prompt-ready pieces.
# ---------------------------------------------------------------------------


def _format_transcript(segs: list[TranscriptSegment]) -> str:
    lines = []
    for s in segs:
        lines.append(f"[{s.start:.2f} - {s.end:.2f}] {s.text}")
    return "\n".join(lines)


def _format_signals(signals: Signals) -> str:
    """Compact human-readable signal summary for the prompt."""
    lines: list[str] = []

    if signals.audio:
        lines.append("AUDIO SIGNALS:")
        for a in _top_audio(signals.audio):
            lines.append(f"  - t={a.time:.2f}s  {a.kind}  ({a.note or f'value={a.value:.2f}'})")
    else:
        lines.append("AUDIO SIGNALS: (none detected)")

    lines.append("")

    if signals.text:
        lines.append("TEXT SIGNALS:")
        for t in _top_text(signals.text):
            lines.append(f"  - t={t.time:.2f}s  {t.kind}  ({t.note or f'value={t.value:.2f}'})")
    else:
        lines.append("TEXT SIGNALS: (none detected)")

    return "\n".join(lines)


def _top_audio(features: list[AudioFeature], limit: int = 12) -> list[AudioFeature]:
    ranked = sorted(features, key=lambda a: (a.kind != "volume_spike", -a.value))
    return ranked[:limit]


def _top_text(features: list[TextFeature], limit: int = 12) -> list[TextFeature]:
    priority = {"exclamation": 0, "length_spike": 1, "interruption": 2, "repeated_word": 3}
    ranked = sorted(
        [t for t in features if t.kind != "sentence_boundary"],
        key=lambda t: (priority.get(t.kind, 99), -abs(t.value)),
    )
    return ranked[:limit]


def _encode_frame(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": data,
        },
    }


def _select_frames(window: Window, limit: int) -> list:
    """Prefer event-driven frames, fall back to base frames."""
    ordered = sorted(
        window.frames,
        key=lambda f: (f.trigger == "base", f.time),  # non-base first
    )
    return ordered[:limit]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_pass1(
    windows: list[Window],
    cfg: Pass1Config | None = None,
    client: anthropic.Anthropic | None = None,
) -> Pass1Result:
    cfg = cfg or Pass1Config()
    client = client or anthropic.Anthropic()

    rubric = _load_rubric(cfg.prompt_path)

    result = Pass1Result(
        candidates=[],
        model=cfg.model,
        prompt_version=PROMPT_VERSION,
    )

    for w in windows:
        try:
            candidates, usage = _score_window(client, w, rubric, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("window %d failed: %s", w.index, e)
            result.windows_rejected += 1
            continue

        if not candidates:
            result.windows_rejected += 1
        for c in candidates:
            c.window_index = w.index
            c.prompt_version = PROMPT_VERSION
            c.model = cfg.model
            result.candidates.append(c)

        result.windows_processed += 1
        result.total_input_tokens += usage.get("input_tokens", 0)
        result.total_output_tokens += usage.get("output_tokens", 0)
        result.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        result.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        log.info(
            "window %d done: %d candidate(s), in=%d out=%d cache_read=%d",
            w.index,
            len(candidates),
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
        )

    result.candidates = _dedupe_overlaps(result.candidates)
    result.candidates.sort(key=lambda c: c.overall_score, reverse=True)
    return result


def _score_window(
    client: anthropic.Anthropic,
    window: Window,
    rubric: str,
    cfg: Pass1Config,
) -> tuple[list[Candidate], dict]:
    """Call Claude for a single window. Returns (candidates, usage)."""
    user_content: list = []

    header = (
        f"WINDOW #{window.index}\n"
        f"Window start: {window.start:.2f}s\n"
        f"Window end: {window.end:.2f}s\n"
        f"Window duration: {window.duration:.2f}s\n\n"
        f"TRANSCRIPT:\n{_format_transcript(window.segments)}\n\n"
        f"{_format_signals(window.signals)}\n\n"
        f"Frames below are sampled from this window, in chronological order."
    )
    user_content.append({"type": "text", "text": header})

    for f in _select_frames(window, cfg.frames_per_window):
        user_content.append(
            {"type": "text", "text": f"Frame at t={f.time:.2f}s (trigger={f.trigger}):"}
        )
        try:
            user_content.append(_encode_frame(f.path))
        except FileNotFoundError:
            log.warning("frame missing: %s", f.path)

    user_content.append(
        {
            "type": "text",
            "text": (
                "Follow the two-stage process. "
                "Return ONLY the JSON object specified by the rubric."
            ),
        }
    )

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
                messages=[{"role": "user", "content": user_content}],
            )
            text = _extract_text(response)
            candidates = _parse_and_validate(text, window)
            usage = _usage_dict(response.usage)
            return candidates, usage
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            log.warning(
                "window %d attempt %d parse failure: %s", window.index, attempt, e
            )
            time.sleep(0.5 * (attempt + 1))
        except anthropic.APIError as e:
            last_err = e
            log.warning(
                "window %d attempt %d api error: %s", window.index, attempt, e
            )
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"window {window.index} exhausted retries: {last_err}")


def _extract_text(response) -> str:
    parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "\n".join(parts).strip()


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_and_validate(text: str, window: Window) -> list[Candidate]:
    """Strip fences, parse JSON, validate against pydantic schema, enforce bounds."""
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
    raw = payload.get("candidates", [])
    candidates: list[Candidate] = []

    for entry in raw:
        # Coerce scores if nested.
        if "scores" in entry and isinstance(entry["scores"], dict):
            entry["scores"] = Scores(**entry["scores"])

        cand = Candidate(**entry)

        # Clamp to window bounds; skip if out of range.
        if cand.start_seconds < window.start - 0.5 or cand.end_seconds > window.end + 0.5:
            log.warning(
                "candidate outside window [%.2f, %.2f]: %.2f-%.2f (skipped)",
                window.start,
                window.end,
                cand.start_seconds,
                cand.end_seconds,
            )
            continue

        candidates.append(cand)

    return candidates


def _usage_dict(usage) -> dict:
    fields = [
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    out = {}
    for f in fields:
        out[f] = getattr(usage, f, 0) or 0
    return out


# ---------------------------------------------------------------------------
# Deduplication across overlapping windows.
# ---------------------------------------------------------------------------


def _dedupe_overlaps(candidates: list[Candidate], iou_threshold: float = 0.6) -> list[Candidate]:
    """Greedy: keep highest-scoring candidate when two overlap heavily."""
    ordered = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
    kept: list[Candidate] = []

    for c in ordered:
        clash = False
        for k in kept:
            if _iou(c, k) >= iou_threshold:
                clash = True
                break
        if not clash:
            kept.append(c)

    return kept


def _iou(a: Candidate, b: Candidate) -> float:
    lo = max(a.start_seconds, b.start_seconds)
    hi = min(a.end_seconds, b.end_seconds)
    inter = max(0.0, hi - lo)
    union = (a.end_seconds - a.start_seconds) + (b.end_seconds - b.start_seconds) - inter
    return inter / union if union > 0 else 0.0
