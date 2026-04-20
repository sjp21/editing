from __future__ import annotations

import pytest
from pydantic import ValidationError

from clipper.schemas import Candidate, Scores


def _scores() -> Scores:
    return Scores(
        hook_strength=5,
        emotional_peak=5,
        quotability=5,
        self_contained=5,
        visual_interest=5,
        reaction_worthy=5,
    )


def _candidate(**overrides) -> Candidate:
    kwargs = dict(
        start_seconds=10.0,
        end_seconds=28.0,
        duration_seconds=18.0,
        scores=_scores(),
        overall_score=5.0,
        confidence=0.5,
        rationale="x",
        suggested_hook="y",
    )
    kwargs.update(overrides)
    return Candidate(**kwargs)


def test_consistent_candidate_ok():
    c = _candidate()
    assert c.end_seconds - c.start_seconds == pytest.approx(c.duration_seconds)


def test_within_tolerance_ok():
    c = _candidate(start_seconds=10.0, end_seconds=28.05, duration_seconds=18.0)
    assert c.duration_seconds == 18.0


def test_mismatched_duration_rejected():
    with pytest.raises(ValidationError):
        _candidate(start_seconds=10.0, end_seconds=30.0, duration_seconds=18.0)


def test_inverted_timestamps_rejected():
    with pytest.raises(ValidationError):
        _candidate(start_seconds=30.0, end_seconds=10.0, duration_seconds=18.0)


def test_equal_timestamps_rejected():
    with pytest.raises(ValidationError):
        _candidate(start_seconds=10.0, end_seconds=10.0, duration_seconds=18.0)


def test_duration_out_of_range_still_rejected():
    with pytest.raises(ValidationError):
        _candidate(start_seconds=10.0, end_seconds=22.0, duration_seconds=12.0)
