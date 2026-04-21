"""Ranker parsing, validation, and mocked end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper import ranker
from clipper.ranker import (
    PROMPT_VERSION,
    RankerConfig,
    _parse_and_validate,
    apply_ranking,
    run_ranker,
)
from clipper.schemas import (
    Candidate,
    RankedCandidates,
    ScoreResult,
    Scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scores(v: float = 7.0) -> Scores:
    return Scores(
        hook_strength=v,
        emotional_peak=v,
        quotability=v,
        self_contained=v,
        visual_interest=v,
        reaction_worthy=v,
    )


def _cand(start: float = 10.0, end: float = 27.0, idx: int | None = 0) -> Candidate:
    return Candidate(
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        scores=_scores(),
        overall_score=7.5,
        confidence=0.8,
        rationale="specific claim lands",
        suggested_hook="The pricing line",
        prompt_version="v2",
        model="claude-opus-4-7",
        score_index=idx,
    )


def _scored(n: int = 3) -> ScoreResult:
    cands = [
        _cand(start=10.0 + 30.0 * i, end=27.0 + 30.0 * i, idx=i) for i in range(n)
    ]
    return ScoreResult(candidates=cands, model="m", prompt_version="v2")


def _ranked_json(
    ranked: list[dict],
    dropped: list[dict],
    requested_k: int = 5,
    truncation_note: str | None = None,
) -> str:
    payload: dict = {
        "ranked": ranked,
        "dropped": dropped,
        "requested_k": requested_k,
    }
    if truncation_note is not None:
        payload["truncation_note"] = truncation_note
    return json.dumps(payload)


def _rank_row(idx: int, rank: int, profile_conflict: str | None = None) -> dict:
    return {
        "candidate_index": idx,
        "rank": rank,
        "hook_quote": "Most founders undercharge by a factor of three.",
        "payoff_in_window": True,
        "payoff_reason": "Specific claim lands at t=12.3 inside the window.",
        "comparative_reason": f"More specific than the other candidates at rank {rank}.",
        "profile_conflict_note": profile_conflict,
    }


def _drop_row(
    idx: int,
    reason_kind: str = "semantic_duplicate",
    dup_of: int | None = None,
) -> dict:
    return {
        "candidate_index": idx,
        "reason_kind": reason_kind,
        "reason_detail": "Same point as the ranked survivor but abstract.",
        "comparative_reason": "Lost on specificity.",
        "duplicate_of_candidate_index": dup_of,
        "hook_quote": "Some quoted line.",
    }


# ---------------------------------------------------------------------------
# Fake client (mirrors tests/test_shape.py)
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, input_tokens=200, output_tokens=80, cache_read=0, cache_create=0):
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
    def __init__(self, responses: list):
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
# _parse_and_validate
# ---------------------------------------------------------------------------


def _parse(text: str, scored: ScoreResult, k: int = 5) -> RankedCandidates:
    return _parse_and_validate(text, scored, RankerConfig(k=k), call_id="abc123")


def test_parse_valid_ranking():
    scored = _scored(3)
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[
            _drop_row(1, "semantic_duplicate", dup_of=0),
            _drop_row(2, "payoff_absent"),
        ],
        requested_k=5,
        truncation_note="Only one clip clears the bar.",
    )
    r = _parse(text, scored)
    assert len(r.ranked) == 1
    assert r.ranked[0].rank == 1
    assert len(r.dropped) == 2
    assert r.call_id == "abc123"
    assert r.prompt_version == PROMPT_VERSION
    assert r.source_score_prompt_version == "v2"


def test_parse_rejects_unknown_index():
    scored = _scored(2)
    text = _ranked_json(
        ranked=[_rank_row(99, 1)],
        dropped=[_drop_row(0), _drop_row(1)],
        requested_k=5,
        truncation_note="only one picked",
    )
    with pytest.raises(ValueError, match="not in score indices"):
        _parse(text, scored)


def test_parse_rejects_incomplete_coverage():
    scored = _scored(3)
    # Candidate 2 missing from both lists.
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="only one picked",
    )
    with pytest.raises(ValueError, match="omitted"):
        _parse(text, scored)


def test_parse_rejects_duplicate_rank():
    scored = _scored(3)
    text = _ranked_json(
        ranked=[_rank_row(0, 1), _rank_row(1, 1)],
        dropped=[_drop_row(2)],
        requested_k=5,
        truncation_note="padded",
    )
    with pytest.raises(ValueError, match="contiguous"):
        _parse(text, scored)


def test_parse_fenced_json():
    scored = _scored(1)
    inner = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[],
        requested_k=5,
        truncation_note="only one candidate",
    )
    text = "```json\n" + inner + "\n```"
    r = _parse(text, scored)
    assert len(r.ranked) == 1


def test_parse_rejects_appearing_in_both_lists():
    scored = _scored(2)
    # Candidate 0 ranked AND dropped.
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(0), _drop_row(1)],
        requested_k=5,
        truncation_note="only one picked",
    )
    with pytest.raises(ValueError, match="both"):
        _parse(text, scored)


def test_parse_rejects_duplicate_of_not_in_ranked():
    scored = _scored(3)
    # Dropped #1 claims it is a duplicate of #2, but #2 is also dropped.
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1, dup_of=2), _drop_row(2)],
        requested_k=5,
        truncation_note="one picked",
    )
    with pytest.raises(ValueError, match="duplicate_of_candidate_index"):
        _parse(text, scored)


def test_parse_rejects_truncation_note_missing():
    scored = _scored(3)
    # Returns 1 but no truncation_note while requested_k=5.
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1), _drop_row(2)],
        requested_k=5,
    )
    with pytest.raises(ValueError, match="truncation_note required"):
        _parse(text, scored)


# ---------------------------------------------------------------------------
# run_ranker
# ---------------------------------------------------------------------------


def test_run_ranker_single_call(monkeypatch):
    scored = _scored(2)
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="one clear winner",
    )
    client = _FakeClient([_response(text)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    result = run_ranker(scored, cfg=RankerConfig(k=5), client=client)
    assert len(client.calls) == 1  # core invariant: one ranker call per run
    assert len(result.ranked) == 1
    assert result.ranked[0].candidate_index == 0


def test_run_ranker_retry_then_success(monkeypatch):
    scored = _scored(2)
    good = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="one winner",
    )
    client = _FakeClient([_response("not json"), _response(good)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    result = run_ranker(scored, cfg=RankerConfig(k=5), client=client)
    assert len(client.calls) == 2
    assert len(result.ranked) == 1


def test_run_ranker_caches_system_prompt(monkeypatch):
    scored = _scored(2)
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="one winner",
    )
    client = _FakeClient([_response(text)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    run_ranker(scored, cfg=RankerConfig(k=5), client=client)
    sent = client.calls[0]
    sys_block = sent["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}
    assert sent["model"] == RankerConfig().model


def test_run_ranker_logs_every_decision(monkeypatch, tmp_path):
    scored = _scored(3)
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[
            _drop_row(1, "semantic_duplicate", dup_of=0),
            _drop_row(2, "payoff_absent"),
        ],
        requested_k=5,
        truncation_note="one strong pick",
    )
    client = _FakeClient([_response(text, _FakeUsage(300, 120, 50, 10))])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    log_path = tmp_path / "ranker.jsonl"
    cfg = RankerConfig(k=5, log_path=log_path)
    run_ranker(scored, cfg=cfg, client=client)

    lines = log_path.read_text().splitlines()
    # 3 decision rows + 1 summary row
    assert len(lines) == 4
    rows = [json.loads(x) for x in lines]
    call_ids = {r["call_id"] for r in rows}
    assert len(call_ids) == 1

    summary = rows[-1]
    assert summary["kind"] == "ranker_summary"
    assert summary["requested_k"] == 5
    assert summary["returned_k"] == 1
    assert summary["truncation_note"] == "one strong pick"
    assert summary["input_tokens"] == 300
    assert summary["output_tokens"] == 120

    verdicts = [r["verdict"] for r in rows[:-1]]
    assert verdicts.count("ranked") == 1
    assert verdicts.count("dropped") == 2


def test_run_ranker_populates_token_counts(monkeypatch):
    scored = _scored(2)
    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="one winner",
    )
    client = _FakeClient([_response(text, _FakeUsage(400, 90, 200, 50))])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    r = run_ranker(scored, cfg=RankerConfig(k=5), client=client)
    assert r.total_input_tokens == 400
    assert r.total_output_tokens == 90
    assert r.cache_read_tokens == 200
    assert r.cache_creation_tokens == 50


def test_truncation_returns_fewer_than_k(monkeypatch):
    scored = _scored(5)
    # k=5, ranker returns only 3.
    text = _ranked_json(
        ranked=[_rank_row(0, 1), _rank_row(1, 2), _rank_row(2, 3)],
        dropped=[_drop_row(3), _drop_row(4)],
        requested_k=5,
        truncation_note="the remaining two are forced picks",
    )
    client = _FakeClient([_response(text)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    r = run_ranker(scored, cfg=RankerConfig(k=5), client=client)
    assert len(r.ranked) == 3
    assert r.truncation_note == "the remaining two are forced picks"
    assert len(r.dropped) == 2


def test_profile_missing_raises(monkeypatch, tmp_path):
    scored = _scored(1)
    cfg = RankerConfig(
        k=5,
        profile="tiktok",
        profiles_dir=tmp_path / "does_not_exist",
    )
    client = _FakeClient([])  # should not be called
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    with pytest.raises(FileNotFoundError, match="tiktok"):
        run_ranker(scored, cfg=cfg, client=client)
    assert len(client.calls) == 0


def test_profile_loaded_and_injected(monkeypatch, tmp_path):
    scored = _scored(2)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "tiktok.md").write_text("TIKTOK_SENTINEL_PROFILE_TEXT")

    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[_drop_row(1)],
        requested_k=5,
        truncation_note="one winner",
    )
    client = _FakeClient([_response(text)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    r = run_ranker(
        scored,
        cfg=RankerConfig(k=5, profile="tiktok", profiles_dir=profiles_dir),
        client=client,
    )
    assert r.profile == "tiktok"
    # Profile text must be injected into the system prompt.
    sys_text = client.calls[0]["system"][0]["text"]
    assert "TIKTOK_SENTINEL_PROFILE_TEXT" in sys_text


def test_profile_with_backslashes_is_literal(monkeypatch, tmp_path):
    """Profile text containing backslash sequences (Windows paths, regex
    backreferences, LaTeX) must round-trip verbatim — not be interpreted
    as a re.sub replacement template."""
    scored = _scored(1)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    hazard = "See C:\\Users\\foo; regex example: \\1 captures group 1; LaTeX: \\gamma"
    (profiles_dir / "tiktok.md").write_text(hazard)

    text = _ranked_json(
        ranked=[_rank_row(0, 1)],
        dropped=[],
        requested_k=5,
        truncation_note="only one candidate",
    )
    client = _FakeClient([_response(text)])
    monkeypatch.setattr(ranker.time, "sleep", lambda *_: None)

    r = run_ranker(
        scored,
        cfg=RankerConfig(k=5, profile="tiktok", profiles_dir=profiles_dir),
        client=client,
    )
    assert r.profile == "tiktok"
    sys_text = client.calls[0]["system"][0]["text"]
    assert hazard in sys_text


# ---------------------------------------------------------------------------
# apply_ranking
# ---------------------------------------------------------------------------


def test_apply_ranking_preserves_indices():
    scored = _scored(4)
    # Ranker puts score #3 at rank 1, #0 at rank 2.
    ranked = RankedCandidates(
        ranked=[
            _RankedClipRow(3, 1),
            _RankedClipRow(0, 2),
        ],
        dropped=[
            _DroppedClipRow(1),
            _DroppedClipRow(2),
        ],
        requested_k=5,
        truncation_note="two real winners",
        model="m",
        prompt_version=PROMPT_VERSION,
        source_score_prompt_version="v2",
        call_id="c",
    )
    out = apply_ranking(scored, ranked)
    assert len(out.candidates) == 2
    assert out.candidates[0].score_index == 3  # rank 1 first
    assert out.candidates[1].score_index == 0

    # Shape reads cand.score_index, so indices survive.
    # Rest of ScoreResult passes through unchanged.
    assert out.prompt_version == scored.prompt_version
    assert out.model == scored.model


def _RankedClipRow(candidate_index: int, rank: int):
    from clipper.schemas import RankedClip

    return RankedClip(
        candidate_index=candidate_index,
        rank=rank,
        hook_quote="quote",
        payoff_in_window=True,
        payoff_reason="lands inside",
        comparative_reason="more specific than the rest",
    )


def _DroppedClipRow(candidate_index: int):
    from clipper.schemas import DroppedClip

    return DroppedClip(
        candidate_index=candidate_index,
        reason_kind="generic",
        reason_detail="no named things",
        comparative_reason="lost on specificity",
    )


# ---------------------------------------------------------------------------
# apply_ranking → shape integration (shape sees filtered survivors)
# ---------------------------------------------------------------------------


def test_apply_ranking_then_shape(monkeypatch, tmp_path):
    from clipper import shape as shape_mod
    from clipper.shape import ShapeConfig, run_shape
    from clipper.schemas import Transcript, TranscriptSegment, TranscriptWord

    scored = _scored(3)
    ranked = RankedCandidates(
        ranked=[_RankedClipRow(2, 1)],  # only score #2 survives
        dropped=[_DroppedClipRow(0), _DroppedClipRow(1)],
        requested_k=5,
        truncation_note="one clear winner",
        model="m",
        prompt_version=PROMPT_VERSION,
        source_score_prompt_version="v2",
        call_id="c",
    )
    survivors = apply_ranking(scored, ranked)
    assert len(survivors.candidates) == 1

    # Build a transcript that covers the survivor. Words are 1s each so
    # "Most"→"day" spans 13s — enough to satisfy ShapedTrim's 12s minimum.
    surv = survivors.candidates[0]
    words = []
    t = surv.start_seconds - 1.0
    text = [
        "Setup",  # prefix
        "Most", "founders", "undercharge", "by", "a", "factor", "of",
        "three", "which", "means", "you", "are", "charging",
        "less", "every", "single", "day",
        "now",  # suffix
    ]
    for word in text:
        words.append(TranscriptWord(start=t, end=t + 1.0, word=word))
        t += 1.0
    seg = TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word for w in words),
        words=words,
    )
    tr = Transcript(language="en", duration=words[-1].end + 10, segments=[seg])

    start_word = next(w for w in words if w.word == "Most")
    end_word = next(w for w in words if w.word == "day")
    keep_text = json.dumps({
        "verdict": "keep",
        "trim": {
            "start_seconds": start_word.start,
            "end_seconds": end_word.end,
            "start_word": "Most",
            "end_word": "day",
            "duration_seconds": round(end_word.end - start_word.start, 2),
        },
        "hook": {
            "kind": "rewrite",
            "text": "Most founders undercharge by a factor of three.",
            "rationale": "specific lead",
        },
        "reasoning": "payoff lands inside trim",
    })

    client = _FakeClient([_response(keep_text)])
    monkeypatch.setattr(shape_mod.time, "sleep", lambda *_: None)

    result = run_shape(
        survivors,
        tr,
        cfg=ShapeConfig(log_path=tmp_path / "shape.jsonl"),
        client=client,
        source_ranker_prompt_version=PROMPT_VERSION,
    )

    # Shape was called exactly once — for the single survivor, not for all 3.
    assert len(client.calls) == 1
    assert len(result.shaped) == 1
    # ShapedClip.candidate_index must equal the original score_index (2), not 0.
    assert result.shaped[0].candidate_index == 2
    assert result.source_ranker_prompt_version == PROMPT_VERSION


# ---------------------------------------------------------------------------
# Prompt + package-data regression
# ---------------------------------------------------------------------------


def test_ranking_rubric_loads_from_package():
    rubric = ranker._load_rubric(None)
    assert "[PLATFORM_PROFILE]" in rubric
    assert "hook_quote" in rubric
    assert "comparative_reason" in rubric


def test_soul_loads_from_package():
    soul = ranker._load_soul()
    assert "editor" in soul.lower()
