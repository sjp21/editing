"""Per-window diagnostics: filter_reason capture, signal counts, call failures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

from clipper import pass1
from clipper.pass1 import Pass1Config, run_pass1
from clipper.schemas import (
    AudioFeature,
    Frame,
    Signals,
    TextFeature,
    TranscriptSegment,
    Window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _segment(start=100.0, end=118.0, text="A self-contained punchline lands clean.") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, words=[])


def _window(
    index: int = 0,
    start: float = 100.0,
    end: float = 125.0,
    n_text: int = 3,
    n_audio: int = 2,
    n_frames: int = 4,
    n_event_frames: int = 1,
) -> Window:
    """Build a Window with controlled signal / frame counts.

    Frame paths don't need to exist — `_score_window` only reads them when
    it actually posts to Claude, and we stub that path in these tests.
    """
    text_feats = [
        TextFeature(time=start + i, kind="exclamation", value=1.0)
        for i in range(n_text)
    ]
    audio_feats = [
        AudioFeature(time=start + i, kind="volume_spike", value=5.0)
        for i in range(n_audio)
    ]
    frames: list[Frame] = []
    # n_event_frames non-base + (n_frames - n_event_frames) base.
    for i in range(n_event_frames):
        frames.append(Frame(time=start + i, path=Path(f"/tmp/frame_event_{i}.jpg"), trigger="audio_spike"))
    for i in range(n_frames - n_event_frames):
        frames.append(Frame(time=start + 5 + i, path=Path(f"/tmp/frame_base_{i}.jpg"), trigger="base"))

    return Window(
        index=index,
        start=start,
        end=end,
        transcript="",
        segments=[_segment(start=start, end=end - 2.0)],
        frames=frames,
        signals=Signals(text=text_feats, audio=audio_feats),
    )


def _candidate_json(start=100.0, end=117.0) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": round(end - start, 2),
                    "scores": {
                        "hook_strength": 7.5,
                        "emotional_peak": 7.0,
                        "quotability": 8.0,
                        "self_contained": 7.0,
                        "visual_interest": 6.0,
                        "reaction_worthy": 7.5,
                    },
                    "overall_score": 7.3,
                    "confidence": 0.7,
                    "rationale": "Punchline lands at the exclamation.",
                    "suggested_hook": "The moment it clicks",
                }
            ]
        }
    )


def _reject_json(reason: str) -> str:
    return json.dumps({"candidates": [], "filter_reason": reason})


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=40, cache_read=0, cache_create=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_create


def _response(text: str, usage: _FakeUsage | None = None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage or _FakeUsage(),
    )


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        parent = self

        class _Messages:
            def create(self_inner, **kwargs):
                parent.calls.append(kwargs)
                if not parent._responses:
                    raise AssertionError("FakeClient out of responses")
                r = parent._responses.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

        self.messages = _Messages()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rejected_window_captures_filter_reason(monkeypatch):
    """Stage 1 rejection → WindowDiagnostic records the reason and zero candidates."""
    # Skip frame encoding (paths don't exist).
    monkeypatch.setattr(pass1, "_select_frames", lambda w, limit: [])
    monkeypatch.setattr(pass1.time, "sleep", lambda *_: None)

    w = _window(index=0, n_text=5, n_audio=3, n_frames=4, n_event_frames=2)
    client = _FakeClient([_response(_reject_json("pure logistics — setup banter"))])

    result = run_pass1([w], cfg=Pass1Config(), client=client)

    assert len(result.window_diagnostics) == 1
    diag = result.window_diagnostics[0]
    assert diag.index == 0
    assert diag.start == 100.0 and diag.end == 125.0
    assert diag.num_text_features == 5
    assert diag.num_audio_features == 3
    assert diag.num_frames == 4
    assert diag.num_event_frames == 2
    assert diag.candidates_returned == 0
    assert diag.filter_reason == "pure logistics — setup banter"
    assert diag.call_succeeded is True
    # Legacy counters still move.
    assert result.windows_processed == 1
    assert result.windows_rejected == 1


def test_accepted_window_has_no_filter_reason(monkeypatch):
    """Non-empty candidates → filter_reason stays None."""
    monkeypatch.setattr(pass1, "_select_frames", lambda w, limit: [])
    monkeypatch.setattr(pass1.time, "sleep", lambda *_: None)

    w = _window(index=2)
    client = _FakeClient([_response(_candidate_json(start=100.0, end=117.0))])

    result = run_pass1([w], cfg=Pass1Config(), client=client)

    assert len(result.candidates) == 1
    diag = result.window_diagnostics[0]
    assert diag.candidates_returned == 1
    assert diag.filter_reason is None
    assert diag.call_succeeded is True
    assert result.windows_processed == 1
    assert result.windows_rejected == 0


def test_call_failure_records_unsucceeded_diagnostic(monkeypatch):
    """Exhausted retries → WindowDiagnostic with call_succeeded=False."""
    monkeypatch.setattr(pass1, "_select_frames", lambda w, limit: [])
    monkeypatch.setattr(pass1.time, "sleep", lambda *_: None)

    w = _window(index=7)
    # Malformed on every attempt (max_retries=2 → 3 calls total).
    client = _FakeClient([_response("not json") for _ in range(5)])

    result = run_pass1([w], cfg=Pass1Config(), client=client)

    assert result.candidates == []
    assert len(result.window_diagnostics) == 1
    diag = result.window_diagnostics[0]
    assert diag.index == 7
    assert diag.call_succeeded is False
    assert diag.candidates_returned == 0
    assert diag.filter_reason and diag.filter_reason.startswith("call_failed:")
    # A throwing window bumps windows_rejected but not windows_processed.
    assert result.windows_processed == 0
    assert result.windows_rejected == 1


def test_rejection_without_filter_reason_stays_none(monkeypatch):
    """Older / non-conforming responses that omit filter_reason are tolerated."""
    monkeypatch.setattr(pass1, "_select_frames", lambda w, limit: [])
    monkeypatch.setattr(pass1.time, "sleep", lambda *_: None)

    w = _window(index=1)
    client = _FakeClient([_response(json.dumps({"candidates": []}))])

    result = run_pass1([w], cfg=Pass1Config(), client=client)

    diag = result.window_diagnostics[0]
    assert diag.candidates_returned == 0
    assert diag.filter_reason is None
    assert diag.call_succeeded is True


def test_prompt_version_stamped_on_candidate(monkeypatch):
    """run_pass1 stamps PROMPT_VERSION (v2) onto every candidate."""
    monkeypatch.setattr(pass1, "_select_frames", lambda w, limit: [])
    monkeypatch.setattr(pass1.time, "sleep", lambda *_: None)

    w = _window(index=0)
    client = _FakeClient([_response(_candidate_json())])
    result = run_pass1([w], cfg=Pass1Config(), client=client)

    assert result.prompt_version == "v2"
    assert result.candidates[0].prompt_version == "v2"
