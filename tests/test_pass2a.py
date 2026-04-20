"""Pass 2a parsing, validation, and mocked end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper import pass2a
from clipper.pass2a import (
    Pass2aConfig,
    _parse_and_validate,
    _extract_context,
    _format_words_table,
    run_pass2a,
)
from clipper.schemas import (
    Candidate,
    Pass1Result,
    Scores,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scores() -> Scores:
    return Scores(
        hook_strength=7,
        emotional_peak=7,
        quotability=8,
        self_contained=7,
        visual_interest=5,
        reaction_worthy=8,
    )


def _candidate(start=100.0, end=117.0) -> Candidate:
    return Candidate(
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        scores=_scores(),
        overall_score=7.5,
        confidence=0.8,
        rationale="specific claim lands",
        suggested_hook="The three things",
        prompt_version="v1",
        model="claude-opus-4-7",
    )


def _build_transcript(words_spec: list[tuple[float, float, str]]) -> Transcript:
    """words_spec: list of (start, end, text). Packs them all into a single segment."""
    words = [TranscriptWord(start=s, end=e, word=w) for s, e, w in words_spec]
    seg = TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word for w in words),
        words=words,
    )
    return Transcript(language="en", duration=words[-1].end + 5.0, segments=[seg])


def _default_words() -> list[tuple[float, float, str]]:
    """Words spanning [97, 120]. Candidate lives at [100, 117].

    Every word is exactly 1.0s. All words are unique so `next()` lookups
    are unambiguous. Key boundary words:
      - "Most"     starts at 100.0 (candidate start)
      - "charging" ends   at 115.0 (used as a clean trim exit)
      - "than"     ends   at 117.0 (candidate end)
    """
    texts = [
        # prefix
        "Setup", "context", "intro",
        # candidate payload
        "Most", "founders", "undercharge", "by", "a", "factor", "of",
        "three", "which", "means", "you", "are", "charging",
        "less", "every", "day", "than",
        # suffix
        "trailing", "noise", "here",
    ]
    words = []
    t = 97.0
    for text in texts:
        words.append((t, t + 1.0, text))
        t += 1.0
    return words


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50, cache_read=80, cache_create=0):
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

        # nested `messages.create` to mirror the real SDK shape.
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
# Helpers
# ---------------------------------------------------------------------------


def _keep_json(start=100.0, end=113.0, start_word="Most", end_word="charging"):
    return json.dumps({
        "verdict": "keep",
        "trim": {
            "start_seconds": start,
            "end_seconds": end,
            "start_word": start_word,
            "end_word": end_word,
            "duration_seconds": round(end - start, 2),
        },
        "hook": {
            "kind": "rewrite",
            "text": "Most founders undercharge by a factor of three.",
            "rationale": "specific claim leads",
        },
        "reasoning": "payoff inside the trim",
    })


def _kill_json():
    return json.dumps({
        "verdict": "kill",
        "kill_reason": "payoff lands after window and ending hedges",
        "reasoning": "no survivable reshape",
    })


def _find_word_at(words, timestamp, edge="start"):
    """Return the word whose edge-timestamp sits at `timestamp`."""
    for s, e, w in words:
        ref = s if edge == "start" else e
        if abs(ref - timestamp) < 1e-6:
            return w
    raise AssertionError(f"no word found at {edge}={timestamp}")


# ---------------------------------------------------------------------------
# _extract_context
# ---------------------------------------------------------------------------


def test_extract_context_respects_pad():
    tr = _build_transcript(_default_words())
    segs, words = _extract_context(tr, 100.0, 117.0, pad=5.0)
    assert len(segs) == 1
    # Words returned must all have midpoints inside [95, 122]
    for w in words:
        mid = (w.start + w.end) / 2
        assert 95.0 <= mid <= 122.0


def test_format_words_table_is_one_per_line():
    words = [TranscriptWord(start=12.34, end=12.5, word="Hi"),
             TranscriptWord(start=12.56, end=12.8, word="there.")]
    out = _format_words_table(words)
    assert out.splitlines() == ['t=12.34 "Hi"', 't=12.56 "there."']


# ---------------------------------------------------------------------------
# _parse_and_validate
# ---------------------------------------------------------------------------


def _parse(text, cand, words):
    return _parse_and_validate(text, cand, idx=0, ctx_words=words, cfg=Pass2aConfig())


def test_parse_keep_valid():
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand = _candidate(100.0, 117.0)
    start_word = _find_word_at(words_spec, 100.0, "start")
    # find the timestamp where "three" ends
    end_ts = next(e for s, e, w in words_spec if w == "charging")
    text = _keep_json(
        start=100.0, end=end_ts, start_word=start_word, end_word="charging"
    )
    _, ctx_words = _extract_context(tr, cand.start_seconds, cand.end_seconds, pad=30.0)
    shaped = _parse(text, cand, ctx_words)
    assert shaped.verdict == "keep"
    assert shaped.trim is not None
    assert shaped.trim.start_word == start_word


def test_parse_fenced_json():
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand = _candidate(100.0, 117.0)
    start_word = _find_word_at(words_spec, 100.0, "start")
    end_ts = next(e for s, e, w in words_spec if w == "charging")
    inner = _keep_json(start=100.0, end=end_ts, start_word=start_word, end_word="charging")
    text = "```json\n" + inner + "\n```"
    _, ctx_words = _extract_context(tr, cand.start_seconds, cand.end_seconds, pad=30.0)
    shaped = _parse(text, cand, ctx_words)
    assert shaped.verdict == "keep"


def test_parse_malformed_json_raises():
    cand = _candidate(100.0, 117.0)
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse("not json at all", cand, [])


def test_parse_kill_verdict():
    cand = _candidate(100.0, 117.0)
    shaped = _parse(_kill_json(), cand, [])
    assert shaped.verdict == "kill"
    assert shaped.trim is None
    assert shaped.kill_reason


def test_parse_rejects_trim_outside_slack():
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand = _candidate(100.0, 117.0)
    # Propose a trim starting at Setup (t=97.0) — outside source_start±2s=[98, 119].
    start_word = _find_word_at(words_spec, 97.0, "start")  # "Setup"
    end_word = _find_word_at(words_spec, 112.0, "end")      # "are"
    text = _keep_json(start=97.0, end=112.0, start_word=start_word, end_word=end_word)
    _, ctx_words = _extract_context(tr, cand.start_seconds, cand.end_seconds, pad=30.0)
    with pytest.raises(ValueError, match="outside"):
        _parse(text, cand, ctx_words)


def test_parse_rejects_word_timestamp_mismatch():
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand = _candidate(100.0, 117.0)
    # Claim start_word="Most" at t=101.0, but Most actually starts at t=100.0.
    end_word = _find_word_at(words_spec, 115.0, "end")  # "every"
    text = _keep_json(
        start=101.0, end=115.0, start_word="Most", end_word=end_word
    )
    _, ctx_words = _extract_context(tr, cand.start_seconds, cand.end_seconds, pad=30.0)
    with pytest.raises(ValueError, match="does not match"):
        _parse(text, cand, ctx_words)


def test_parse_retry_then_success(monkeypatch, tmp_path):
    # End-to-end via run_pass2a so the retry loop is exercised.
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand = _candidate(100.0, 117.0)
    pass1 = Pass1Result(candidates=[cand], model="m", prompt_version="v1")

    start_word = _find_word_at(words_spec, 100.0, "start")
    end_ts = next(e for s, e, w in words_spec if w == "charging")
    good = _keep_json(start=100.0, end=end_ts, start_word=start_word, end_word="charging")

    client = _FakeClient([_response("not json"), _response(good)])
    monkeypatch.setattr(pass2a.time, "sleep", lambda *_: None)
    cfg = Pass2aConfig(log_path=tmp_path / "log.jsonl")

    result = run_pass2a(pass1, tr, cfg=cfg, client=client)
    assert result.candidates_shaped == 1
    assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# run_pass2a — aggregation + log line
# ---------------------------------------------------------------------------


def test_run_pass2a_aggregates_and_logs(monkeypatch, tmp_path):
    words_spec = _default_words()
    tr = _build_transcript(words_spec)
    cand_a = _candidate(100.0, 117.0)
    cand_b = _candidate(100.0, 117.0)
    pass1 = Pass1Result(
        candidates=[cand_a, cand_b],
        model="m",
        prompt_version="v1",
    )

    start_word = _find_word_at(words_spec, 100.0, "start")
    end_ts = next(e for s, e, w in words_spec if w == "charging")
    keep_text = _keep_json(start=100.0, end=end_ts, start_word=start_word, end_word="charging")

    log_path = tmp_path / "pass2a.jsonl"
    monkeypatch.setattr(pass2a.time, "sleep", lambda *_: None)

    client = _FakeClient([
        _response(keep_text, _FakeUsage(100, 50, 0, 200)),
        _response(_kill_json(), _FakeUsage(110, 30, 200, 0)),
    ])

    cfg = Pass2aConfig(log_path=log_path)
    result = run_pass2a(pass1, tr, cfg=cfg, client=client)

    assert result.candidates_shaped == 1
    assert result.candidates_killed == 1
    assert result.candidates_failed == 0
    assert result.total_input_tokens == 210
    assert result.total_output_tokens == 80
    assert result.cache_read_tokens == 200
    assert result.cache_creation_tokens == 200
    assert result.source_pass1_prompt_version == "v1"
    assert result.prompt_version == "v1"

    # candidate_index is preserved positionally.
    indices = [s.candidate_index for s in result.shaped]
    assert indices == [0, 1]

    # One JSONL line per shaped candidate.
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    entry0 = json.loads(lines[0])
    entry1 = json.loads(lines[1])
    assert entry0["stage"] == "pass2a"
    assert entry0["source_pass1_prompt_version"] == "v1"
    assert entry0["verdict"] == "keep"
    assert entry0["trim_start"] == 100.0
    assert entry0["hook_kind"] == "rewrite"
    assert entry1["verdict"] == "kill"
    assert entry1["trim_start"] is None
    assert entry1["kill_reason"]


def test_run_pass2a_counts_failures(monkeypatch, tmp_path):
    tr = _build_transcript(_default_words())
    pass1 = Pass1Result(
        candidates=[_candidate(100.0, 117.0)],
        model="m",
        prompt_version="v1",
    )
    monkeypatch.setattr(pass2a.time, "sleep", lambda *_: None)

    # Always malformed — retries exhaust.
    responses = [_response("not json") for _ in range(5)]
    client = _FakeClient(responses)

    cfg = Pass2aConfig(log_path=tmp_path / "log.jsonl", max_retries=1)
    result = run_pass2a(pass1, tr, cfg=cfg, client=client)
    assert result.candidates_failed == 1
    assert result.candidates_shaped == 0
    assert not (tmp_path / "log.jsonl").exists()


def test_run_pass2a_caches_system_prompt(monkeypatch, tmp_path):
    tr = _build_transcript(_default_words())
    pass1 = Pass1Result(
        candidates=[_candidate(100.0, 117.0)],
        model="m",
        prompt_version="v1",
    )
    monkeypatch.setattr(pass2a.time, "sleep", lambda *_: None)

    client = _FakeClient([_response(_kill_json())])
    cfg = Pass2aConfig(log_path=tmp_path / "log.jsonl")
    run_pass2a(pass1, tr, cfg=cfg, client=client)

    sent = client.calls[0]
    sys_block = sent["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert sent["model"] == Pass2aConfig().model
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def test_shaping_rubric_loads_from_package():
    rubric = pass2a._load_rubric(None)
    assert "Shaping" in rubric or "shaping" in rubric
    assert "[PLATFORM_PROFILE]" in rubric
