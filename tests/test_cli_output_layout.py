"""CLI output-layout guardrail.

Asserts that `clipper shape` resolves its artifacts under
`outputs/<video-stem>/` by default and under the user-supplied path when
`--output-dir` is passed. The rest of the pipeline is monkeypatched out so
the test is fast and network-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from clipper import cli
from clipper.schemas import (
    Candidate,
    CaptionStyle,
    DroppedClip,
    PostedClip,
    RankedCandidates,
    RankedClip,
    RenderedClip,
    RenderResult,
    RenderSettings,
    ScoreResult,
    ShapeResult,
    Scores,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _make_transcript() -> Transcript:
    w = TranscriptWord(start=0.0, end=0.5, word="hello")
    seg = TranscriptSegment(start=0.0, end=0.5, text="hello", words=[w])
    return Transcript(language="en", duration=1.0, segments=[seg])


def _make_scored() -> ScoreResult:
    cand = Candidate(
        start_seconds=0.0,
        end_seconds=17.0,
        duration_seconds=17.0,
        scores=Scores(
            hook_strength=7,
            emotional_peak=7,
            quotability=7,
            self_contained=7,
            visual_interest=5,
            reaction_worthy=7,
        ),
        overall_score=7.0,
        confidence=0.8,
        rationale="ok",
        suggested_hook="hook",
        prompt_version="v1",
        model="m",
    )
    return ScoreResult(candidates=[cand], model="m", prompt_version="v1")


def _empty_shape() -> ShapeResult:
    return ShapeResult(
        shaped=[],
        model="m",
        prompt_version="v1",
        source_score_prompt_version="v1",
    )


@pytest.fixture
def _stub_pipeline(monkeypatch):
    transcript = _make_transcript()
    monkeypatch.setattr(cli, "extract_audio", lambda video, cache: Path("/dev/null"))
    monkeypatch.setattr(cli, "transcribe", lambda audio, cfg: transcript)
    monkeypatch.setattr(cli, "run_shape", lambda *a, **kw: _empty_shape())
    return transcript


def _write_candidates(path: Path) -> None:
    path.write_text(_make_scored().model_dump_json())


def test_shape_defaults_to_outputs_stem(tmp_path, monkeypatch, _stub_pipeline):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")
    cands = tmp_path / "candidates.json"
    _write_candidates(cands)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["shape", str(video), "--candidates", str(cands)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    root = tmp_path / "outputs" / "myvideo"
    assert (root / "shaped.json").exists()
    assert (root / "transcript.json").exists()
    # The old double-extension sidecar must not be created.
    assert not (root / "shaped.json.transcript.json").exists()


def _make_scored_multi(n: int = 3) -> ScoreResult:
    cands = []
    for i in range(n):
        cands.append(
            Candidate(
                start_seconds=0.0 + 30.0 * i,
                end_seconds=17.0 + 30.0 * i,
                duration_seconds=17.0,
                scores=Scores(
                    hook_strength=7, emotional_peak=7, quotability=7,
                    self_contained=7, visual_interest=5, reaction_worthy=7,
                ),
                overall_score=7.0,
                confidence=0.8,
                rationale="ok",
                suggested_hook="hook",
                prompt_version="v2",
                model="m",
                score_index=i,
            )
        )
    return ScoreResult(candidates=cands, model="m", prompt_version="v2")


def _make_ranked_for(scored: ScoreResult) -> RankedCandidates:
    """Pick the first candidate, drop the rest as semantic duplicates."""
    survivor = scored.candidates[0]
    ranked = [
        RankedClip(
            candidate_index=survivor.score_index,
            rank=1,
            hook_quote="quote",
            payoff_in_window=True,
            payoff_reason="lands inside",
            comparative_reason="more specific than the rest",
        )
    ]
    dropped = [
        DroppedClip(
            candidate_index=c.score_index,
            reason_kind="semantic_duplicate",
            reason_detail="same as rank 1",
            comparative_reason="lost to rank 1 on specificity",
            duplicate_of_candidate_index=survivor.score_index,
        )
        for c in scored.candidates[1:]
    ]
    return RankedCandidates(
        ranked=ranked,
        dropped=dropped,
        requested_k=5,
        truncation_note="only one clip clears the bar",
        model="m",
        prompt_version="v1",
        source_score_prompt_version=scored.prompt_version,
        call_id="callid",
    )


def test_rank_subcommand_writes_both_artifacts(tmp_path, monkeypatch):
    scored = _make_scored_multi(3)
    cands = tmp_path / "candidates.json"
    cands.write_text(scored.model_dump_json())

    ranked = _make_ranked_for(scored)
    monkeypatch.setattr(cli, "run_ranker", lambda sc, cfg=None, client=None: ranked)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "rank",
            "--candidates", str(cands),
            "--output-dir", str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    decisions = tmp_path / "ranking_decisions.json"
    survivors_path = tmp_path / "ranked.json"
    assert decisions.exists()
    assert survivors_path.exists()

    survivors = ScoreResult.model_validate_json(survivors_path.read_text())
    assert len(survivors.candidates) == 1
    assert survivors.candidates[0].score_index == 0


def test_rank_rejects_candidates_without_score_index(tmp_path):
    # Simulate an old candidates.json from before score_index was added.
    old_scored = _make_scored_multi(2).model_dump()
    for c in old_scored["candidates"]:
        c["score_index"] = None
    cands = tmp_path / "candidates.json"
    import json as _json
    cands.write_text(_json.dumps(old_scored))

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["rank", "--candidates", str(cands), "--output-dir", str(tmp_path)],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "score_index" in result.output


def test_process_rank_then_shape_filters_candidates(tmp_path, monkeypatch):
    """--rank --shape passes only the ranker's survivors to shape."""
    scored = _make_scored_multi(3)
    transcript = _make_transcript()
    ranked = _make_ranked_for(scored)

    shape_calls = []

    def _fake_run_shape(sc, tr, cfg=None, client=None, source_ranker_prompt_version=None):
        shape_calls.append(sc)
        return _empty_shape()

    from clipper.schemas import Signals
    # Stub out the whole score pipeline.
    monkeypatch.setattr(cli, "extract_audio", lambda video, cache: Path("/dev/null"))
    monkeypatch.setattr(cli, "transcribe", lambda audio, cfg: transcript)
    monkeypatch.setattr(cli, "compute_signals", lambda tr, audio: Signals())
    monkeypatch.setattr(cli, "extract_frames", lambda **kw: [])
    monkeypatch.setattr(cli, "build_windows", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "run_score", lambda windows, cfg=None: scored)
    monkeypatch.setattr(cli, "run_ranker", lambda sc, cfg=None, client=None: ranked)
    monkeypatch.setattr(cli, "run_shape", _fake_run_shape)

    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "process", str(video),
            "--output-dir", str(tmp_path / "out"),
            "--rank",
            "--shape",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assert len(shape_calls) == 1
    filtered_scored = shape_calls[0]
    assert len(filtered_scored.candidates) == 1
    assert filtered_scored.candidates[0].score_index == 0

    root = tmp_path / "out"
    assert (root / "candidates.json").exists()
    assert (root / "ranking_decisions.json").exists()
    assert (root / "ranked.json").exists()
    assert (root / "shaped.json").exists()


def test_shape_forwards_ranker_prompt_version_from_sibling(tmp_path, monkeypatch):
    """When ranking_decisions.json sits next to the input candidates,
    `shape` must pick up its prompt_version and forward it to run_shape
    so the shaped artifact + jsonl rows can rejoin the ranker call."""
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")

    scored = _make_scored_multi(2)
    cands = tmp_path / "ranked.json"
    cands.write_text(scored.model_dump_json())
    decisions = _make_ranked_for(scored)
    (tmp_path / "ranking_decisions.json").write_text(decisions.model_dump_json())

    captured: dict = {}

    def _fake_run_shape(sc, tr, cfg=None, client=None, source_ranker_prompt_version=None):
        captured["source_ranker_prompt_version"] = source_ranker_prompt_version
        return _empty_shape()

    monkeypatch.setattr(cli, "extract_audio", lambda video, cache: Path("/dev/null"))
    monkeypatch.setattr(cli, "transcribe", lambda audio, cfg: _make_transcript())
    monkeypatch.setattr(cli, "run_shape", _fake_run_shape)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["shape", str(video), "--candidates", str(cands)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert captured["source_ranker_prompt_version"] == decisions.prompt_version


def test_shape_without_sibling_decisions_forwards_none(tmp_path, monkeypatch):
    """No ranking_decisions.json → source_ranker_prompt_version stays
    None, matching the naive (pre-rank) shape workflow."""
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")
    cands = tmp_path / "candidates.json"
    _write_candidates(cands)

    captured: dict = {}

    def _fake_run_shape(sc, tr, cfg=None, client=None, source_ranker_prompt_version=None):
        captured["source_ranker_prompt_version"] = source_ranker_prompt_version
        return _empty_shape()

    monkeypatch.setattr(cli, "extract_audio", lambda video, cache: Path("/dev/null"))
    monkeypatch.setattr(cli, "transcribe", lambda audio, cfg: _make_transcript())
    monkeypatch.setattr(cli, "run_shape", _fake_run_shape)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["shape", str(video), "--candidates", str(cands)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert captured["source_ranker_prompt_version"] is None


def test_process_profile_threads_to_ranker_and_shape(tmp_path, monkeypatch):
    """--profile must land on both RankerConfig and ShapeConfig."""
    scored = _make_scored_multi(3)
    transcript = _make_transcript()
    ranked = _make_ranked_for(scored)

    ranker_cfgs = []
    shape_cfgs = []

    def _fake_run_ranker(sc, cfg=None, client=None):
        ranker_cfgs.append(cfg)
        return ranked

    def _fake_run_shape(sc, tr, cfg=None, client=None, source_ranker_prompt_version=None):
        shape_cfgs.append(cfg)
        return _empty_shape()

    from clipper.schemas import Signals
    monkeypatch.setattr(cli, "extract_audio", lambda video, cache: Path("/dev/null"))
    monkeypatch.setattr(cli, "transcribe", lambda audio, cfg: transcript)
    monkeypatch.setattr(cli, "compute_signals", lambda tr, audio: Signals())
    monkeypatch.setattr(cli, "extract_frames", lambda **kw: [])
    monkeypatch.setattr(cli, "build_windows", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "run_score", lambda windows, cfg=None: scored)
    monkeypatch.setattr(cli, "run_ranker", _fake_run_ranker)
    monkeypatch.setattr(cli, "run_shape", _fake_run_shape)

    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "process", str(video),
            "--output-dir", str(tmp_path / "out"),
            "--rank",
            "--shape",
            "--profile", "default",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert len(ranker_cfgs) == 1
    assert ranker_cfgs[0].profile == "default"
    assert len(shape_cfgs) == 1
    assert shape_cfgs[0].profile == "default"


def test_post_writes_posted_jsonl_under_stem_logs(tmp_path):
    """`clipper post` resolves the MP4 via the default rendered.json lookup
    and writes <stem>/logs/posted.jsonl with the matched shaped_candidate_index."""
    stem_root = tmp_path / "myvideo"
    clips_dir = stem_root / "clips"
    clips_dir.mkdir(parents=True)
    mp4 = clips_dir / "clip-0.mp4"
    mp4.touch()

    rendered_clip = RenderedClip(
        shaped_candidate_index=4,
        source_start=100.0,
        source_end=115.0,
        trim_start=100.0,
        trim_end=115.0,
        output_path=mp4,
        output_sha256="OUT",
        input_fingerprint="fp",
        output_duration_seconds=15.0,
        measured_integrated_lufs=-14.0,
        face_coverage=0.9,
        hook_overlay_text=None,
        settings=RenderSettings(),
        caption_style=CaptionStyle(),
        render_wall_seconds=1.0,
        render_version="r1",
        source_video_sha256="SRC",
        source_transcript_sha256="TR",
        font_file_sha256="f",
        source_shaped_prompt_version="v1",
        source_score_prompt_version="v2",
    )
    manifest = RenderResult(
        rendered=[rendered_clip],
        render_version="r1",
        source_video_sha256="SRC",
        source_transcript_sha256="TR",
        source_shaped_prompt_version="v1",
        source_score_prompt_version="v2",
        renders_completed=1,
    )
    (clips_dir / "rendered.json").write_text(manifest.model_dump_json())

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "post", str(mp4),
            "--platform", "shorts",
            "--post-id", "abc",
            "--post-url", "https://youtube.com/shorts/abc",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    log_path = stem_root / "logs" / "posted.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    row = PostedClip.model_validate_json(lines[0])
    assert row.shaped_candidate_index == 4
    assert row.video_stem == "myvideo"
    assert row.platform == "shorts"
    assert row.post_id == "abc"


def test_shape_honors_output_dir_override(tmp_path, monkeypatch, _stub_pipeline):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "myvideo.mp4"
    video.write_bytes(b"")
    cands = tmp_path / "candidates.json"
    _write_candidates(cands)

    custom = tmp_path / "foo"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "shape", str(video),
            "--candidates", str(cands),
            "--output-dir", str(custom),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assert (custom / "shaped.json").exists()
    assert (custom / "transcript.json").exists()
    assert not (tmp_path / "outputs").exists()
