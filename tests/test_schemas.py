from __future__ import annotations

import pytest
from pydantic import ValidationError

from clipper.schemas import (
    Candidate,
    DroppedClip,
    RankedCandidates,
    RankedClip,
    Scores,
)


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


def test_candidate_score_index_default_none():
    """Back-compat: old candidates.json files have no score_index."""
    c = _candidate()
    assert c.score_index is None


def test_candidate_score_index_roundtrip():
    c = _candidate(score_index=3)
    assert c.score_index == 3


# ---------------------------------------------------------------------------
# Ranker schema validators
# ---------------------------------------------------------------------------


def _rc(idx: int, rank: int) -> RankedClip:
    return RankedClip(
        candidate_index=idx,
        rank=rank,
        hook_quote="q",
        payoff_in_window=True,
        payoff_reason="lands inside",
        comparative_reason="more specific than the rest",
    )


def _dc(idx: int, dup_of: int | None = None) -> DroppedClip:
    return DroppedClip(
        candidate_index=idx,
        reason_kind="semantic_duplicate" if dup_of is not None else "generic",
        reason_detail="detail",
        comparative_reason="lost on specificity",
        duplicate_of_candidate_index=dup_of,
    )


def _ranked(**overrides) -> RankedCandidates:
    kwargs = dict(
        ranked=[_rc(0, 1)],
        dropped=[_dc(1), _dc(2)],
        requested_k=5,
        truncation_note="one clear winner",
        model="m",
        prompt_version="v1",
        source_score_prompt_version="v2",
        call_id="c",
    )
    kwargs.update(overrides)
    return RankedCandidates(**kwargs)


def test_ranked_contiguous_ok():
    rc = _ranked(
        ranked=[_rc(0, 1), _rc(1, 2)],
        dropped=[_dc(2)],
        truncation_note="padded",
    )
    assert len(rc.ranked) == 2


def test_ranked_non_contiguous_rejected():
    with pytest.raises(ValidationError):
        _ranked(
            ranked=[_rc(0, 1), _rc(1, 3)],
            dropped=[_dc(2)],
            truncation_note="t",
        )


def test_duplicate_rank_rejected():
    with pytest.raises(ValidationError):
        _ranked(
            ranked=[_rc(0, 1), _rc(1, 1)],
            dropped=[_dc(2)],
            truncation_note="t",
        )


def test_candidate_in_both_lists_rejected():
    with pytest.raises(ValidationError):
        _ranked(
            ranked=[_rc(0, 1)],
            dropped=[_dc(0), _dc(1)],
            truncation_note="t",
        )


def test_truncation_note_required_when_short():
    with pytest.raises(ValidationError, match="truncation_note required"):
        _ranked(ranked=[_rc(0, 1)], dropped=[_dc(1)], truncation_note=None)


def test_truncation_note_forbidden_when_full():
    with pytest.raises(ValidationError, match="must be absent"):
        _ranked(
            ranked=[_rc(0, 1), _rc(1, 2), _rc(2, 3), _rc(3, 4), _rc(4, 5)],
            dropped=[],
            requested_k=5,
            truncation_note="should not have one",
        )


def test_ranked_exceeds_requested_k_rejected():
    with pytest.raises(ValidationError, match="exceeds"):
        _ranked(
            ranked=[_rc(0, 1), _rc(1, 2)],
            dropped=[_dc(2)],
            requested_k=1,
        )


def test_duplicate_of_must_be_ranked():
    # #1 claims it's a duplicate of #2, but #2 is dropped, not ranked.
    with pytest.raises(ValidationError, match="duplicate_of_candidate_index"):
        _ranked(
            ranked=[_rc(0, 1)],
            dropped=[_dc(1, dup_of=2), _dc(2)],
            truncation_note="t",
        )
