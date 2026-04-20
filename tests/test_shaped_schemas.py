from __future__ import annotations

import pytest
from pydantic import ValidationError

from clipper.schemas import HookTreatment, ShapedClip, ShapedTrim


def _trim(**overrides) -> ShapedTrim:
    kwargs = dict(
        start_seconds=100.0,
        end_seconds=115.0,
        start_word="Most",
        end_word="three.",
        duration_seconds=15.0,
    )
    kwargs.update(overrides)
    return ShapedTrim(**kwargs)


def _hook(**overrides) -> HookTreatment:
    kwargs = dict(kind="rewrite", text="Most founders undercharge.", rationale="hook")
    kwargs.update(overrides)
    return HookTreatment(**kwargs)


def _shaped(**overrides) -> ShapedClip:
    kwargs = dict(
        candidate_index=0,
        source_start=100.0,
        source_end=115.0,
        verdict="keep",
        trim=_trim(),
        hook=_hook(),
        reasoning="landed on specific claim",
        model="claude-opus-4-7",
        prompt_version="v1",
    )
    kwargs.update(overrides)
    return ShapedClip(**kwargs)


# ---- ShapedTrim ------------------------------------------------------------


def test_trim_consistent_ok():
    t = _trim()
    assert t.duration_seconds == pytest.approx(t.end_seconds - t.start_seconds)


def test_trim_duration_too_short():
    with pytest.raises(ValidationError):
        _trim(start_seconds=100.0, end_seconds=111.0, duration_seconds=11.0)


def test_trim_duration_too_long():
    with pytest.raises(ValidationError):
        _trim(start_seconds=100.0, end_seconds=123.0, duration_seconds=23.0)


def test_trim_tolerates_at_lower_bound():
    t = _trim(start_seconds=100.0, end_seconds=112.0, duration_seconds=12.0)
    assert t.duration_seconds == 12.0


def test_trim_tolerates_at_upper_bound():
    t = _trim(start_seconds=100.0, end_seconds=122.0, duration_seconds=22.0)
    assert t.duration_seconds == 22.0


def test_trim_mismatched_duration_rejected():
    with pytest.raises(ValidationError):
        _trim(start_seconds=100.0, end_seconds=120.0, duration_seconds=15.0)


def test_trim_inverted_timestamps_rejected():
    with pytest.raises(ValidationError):
        _trim(start_seconds=120.0, end_seconds=100.0, duration_seconds=15.0)


# ---- HookTreatment --------------------------------------------------------


def test_hook_rewrite_requires_text():
    with pytest.raises(ValidationError):
        HookTreatment(kind="rewrite", text=None, rationale="x")


def test_hook_overlay_requires_text():
    with pytest.raises(ValidationError):
        HookTreatment(kind="overlay", text="   ", rationale="x")


def test_hook_none_allows_missing_text():
    h = HookTreatment(kind="none", rationale="native open is already strong")
    assert h.text is None


# ---- ShapedClip ------------------------------------------------------------


def test_keep_requires_trim():
    with pytest.raises(ValidationError):
        _shaped(trim=None)


def test_keep_requires_hook():
    with pytest.raises(ValidationError):
        _shaped(hook=None)


def test_kill_requires_reason():
    with pytest.raises(ValidationError):
        _shaped(verdict="kill", trim=None, hook=None, kill_reason=None)


def test_kill_ok_with_reason():
    c = _shaped(
        verdict="kill",
        trim=None,
        hook=None,
        kill_reason="payoff lands outside the window",
    )
    assert c.verdict == "kill"
