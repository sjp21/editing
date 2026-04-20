"""Input fingerprint stability + stale-skip guardrail — no ffmpeg."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper import render as rmod
from clipper.render import (
    RENDER_VERSION,
    RenderConfig,
    _compute_input_fingerprint,
    _prior_manifest_for_skip,
    _should_skip,
    run_pass2b,
)
from clipper.schemas import (
    CaptionStyle,
    HookTreatment,
    Pass2aResult,
    Pass2bResult,
    RenderedClip,
    RenderSettings,
    ShapedClip,
    ShapedTrim,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _trim() -> ShapedTrim:
    return ShapedTrim(
        start_seconds=100.0,
        end_seconds=115.0,
        start_word="Most",
        end_word="three.",
        duration_seconds=15.0,
    )


def _shaped_keep(idx=0) -> ShapedClip:
    return ShapedClip(
        candidate_index=idx,
        source_start=100.0,
        source_end=115.0,
        verdict="keep",
        trim=_trim(),
        hook=HookTreatment(kind="overlay", text="Hook", rationale="r"),
        reasoning="r",
        model="m",
        prompt_version="v1",
    )


def _shaped_kill(idx=1) -> ShapedClip:
    return ShapedClip(
        candidate_index=idx,
        source_start=200.0,
        source_end=215.0,
        verdict="kill",
        kill_reason="no survivable reshape",
        reasoning="r",
        model="m",
        prompt_version="v1",
    )


def _transcript() -> Transcript:
    words = [
        TranscriptWord(start=100.0 + i * 0.5, end=100.5 + i * 0.5, word=f"w{i}")
        for i in range(30)
    ]
    seg = TranscriptSegment(start=100.0, end=115.0, text=" ".join(w.word for w in words), words=words)
    return Transcript(language="en", duration=120.0, segments=[seg])


# ---- _compute_input_fingerprint -----------------------------------------------


def test_fingerprint_stable_across_invocations():
    sc = _shaped_keep()
    fp1 = _compute_input_fingerprint(
        sc, "src", "txt", "font", RenderSettings(), CaptionStyle()
    )
    fp2 = _compute_input_fingerprint(
        sc, "src", "txt", "font", RenderSettings(), CaptionStyle()
    )
    assert fp1 == fp2


@pytest.mark.parametrize(
    "mutator",
    [
        "source_sha",
        "transcript_sha",
        "font_sha",
        "caption_color",
        "setting_crf",
        "shaped_trim",
    ],
)
def test_fingerprint_flips_on_each_input(mutator, monkeypatch):
    sc = _shaped_keep()
    base = _compute_input_fingerprint(
        sc, "src", "txt", "font", RenderSettings(), CaptionStyle()
    )
    if mutator == "source_sha":
        alt = _compute_input_fingerprint(
            sc, "src2", "txt", "font", RenderSettings(), CaptionStyle()
        )
    elif mutator == "transcript_sha":
        alt = _compute_input_fingerprint(
            sc, "src", "txt2", "font", RenderSettings(), CaptionStyle()
        )
    elif mutator == "font_sha":
        alt = _compute_input_fingerprint(
            sc, "src", "txt", "font2", RenderSettings(), CaptionStyle()
        )
    elif mutator == "caption_color":
        cs = CaptionStyle(active_color="#00FF00")
        alt = _compute_input_fingerprint(sc, "src", "txt", "font", RenderSettings(), cs)
    elif mutator == "setting_crf":
        rs = RenderSettings(video_crf=23)
        alt = _compute_input_fingerprint(sc, "src", "txt", "font", rs, CaptionStyle())
    elif mutator == "shaped_trim":
        sc2 = _shaped_keep()
        sc2.trim = ShapedTrim(
            start_seconds=100.0, end_seconds=116.0, start_word="Most",
            end_word="three.", duration_seconds=16.0,
        )
        alt = _compute_input_fingerprint(sc2, "src", "txt", "font", RenderSettings(), CaptionStyle())
    assert alt != base


def test_fingerprint_changes_on_render_version_bump(monkeypatch):
    sc = _shaped_keep()
    base = _compute_input_fingerprint(
        sc, "src", "txt", "font", RenderSettings(), CaptionStyle()
    )
    monkeypatch.setattr(rmod, "RENDER_VERSION", "v2")
    bumped = _compute_input_fingerprint(
        sc, "src", "txt", "font", RenderSettings(), CaptionStyle()
    )
    assert base != bumped


# ---- _should_skip guardrail ---------------------------------------------------


def _write_manifest(path: Path, rendered_path: Path, fp: str, out_sha: str) -> None:
    pr = Pass2bResult(
        rendered=[
            RenderedClip(
                shaped_candidate_index=0,
                source_start=100.0, source_end=115.0,
                trim_start=100.0, trim_end=115.0,
                output_path=rendered_path,
                output_sha256=out_sha,
                input_fingerprint=fp,
                output_duration_seconds=15.0,
                measured_integrated_lufs=-14.0,
                face_coverage=0.9,
                hook_overlay_text=None,
                settings=RenderSettings(),
                caption_style=CaptionStyle(),
                render_wall_seconds=1.0,
                render_version=RENDER_VERSION,
                source_video_sha256="s",
                source_transcript_sha256="t",
                font_file_sha256="f",
                source_shaped_prompt_version="v1",
                source_pass1_prompt_version="v2",
            )
        ],
        render_version=RENDER_VERSION,
        source_video_sha256="s",
        source_transcript_sha256="t",
        source_shaped_prompt_version="v1",
        source_pass1_prompt_version="v2",
    )
    path.write_text(pr.model_dump_json())


def test_skip_guardrail_triggers_on_full_match(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    mp4 = out / "clip-0.mp4"
    mp4.write_bytes(b"fake mp4")
    import hashlib
    sha = hashlib.sha256(b"fake mp4").hexdigest()
    manifest = out / "rendered.json"
    _write_manifest(manifest, mp4, fp="FP", out_sha=sha)

    prior = _prior_manifest_for_skip(manifest)
    assert 0 in prior
    assert _should_skip(0, "FP", prior) is not None
    assert _should_skip(0, "OTHER", prior) is None  # fingerprint mismatch


def test_skip_fails_when_output_missing(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    mp4 = out / "clip-0.mp4"
    mp4.write_bytes(b"fake mp4")
    import hashlib
    sha = hashlib.sha256(b"fake mp4").hexdigest()
    manifest = out / "rendered.json"
    _write_manifest(manifest, mp4, fp="FP", out_sha=sha)
    prior = _prior_manifest_for_skip(manifest)
    mp4.unlink()  # simulate missing output
    assert _should_skip(0, "FP", prior) is None


def test_skip_fails_when_output_sha_mismatch(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    mp4 = out / "clip-0.mp4"
    mp4.write_bytes(b"fake mp4")
    manifest = out / "rendered.json"
    _write_manifest(manifest, mp4, fp="FP", out_sha="wrong_sha")
    prior = _prior_manifest_for_skip(manifest)
    assert _should_skip(0, "FP", prior) is None


# ---- run_pass2b integration paths without ffmpeg ------------------------------


def test_run_pass2b_skips_kills_without_invoking_render(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video bytes")
    shaped = Pass2aResult(
        shaped=[_shaped_kill(0), _shaped_kill(1)],
        model="m",
        prompt_version="v1",
        source_pass1_prompt_version="v2",
    )

    # Force _render_one to blow up if called.
    def _boom(*args, **kwargs):
        raise AssertionError("_render_one must not run for kill verdicts")
    monkeypatch.setattr(rmod, "_render_one", _boom)

    cfg = RenderConfig(
        out_dir=tmp_path / "clips",
        log_path=tmp_path / "log.jsonl",
        skip_existing=False,
    )
    result = run_pass2b(shaped, _transcript(), "T_SHA", video, cfg=cfg)
    assert result.renders_skipped_kill == 2
    assert result.renders_completed == 0
    log_lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(log_lines) == 2
    for ln in log_lines:
        entry = json.loads(ln)
        assert entry["skipped_reason"] == "kill"


def test_run_pass2b_fingerprint_skip_bypasses_render(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video bytes")
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    mp4 = out_dir / "clip-0.mp4"
    mp4.write_bytes(b"fake mp4 output")

    shaped = Pass2aResult(
        shaped=[_shaped_keep(0)],
        model="m",
        prompt_version="v1",
        source_pass1_prompt_version="v2",
    )
    # Freeze font resolution so the fingerprint is predictable.
    monkeypatch.setattr(rmod, "_resolve_font", lambda style, cache_dir: (tmp_path / "font.ttf", "FONT_SHA"))
    (tmp_path / "font.ttf").write_bytes(b"font")
    monkeypatch.setattr(rmod, "_sha256_file", lambda p: {
        str(video): "SRC_SHA",
        str(mp4): "OUT_SHA",
    }.get(str(p), "OTHER"))

    fp = _compute_input_fingerprint(
        shaped.shaped[0], "SRC_SHA", "T_SHA", "FONT_SHA",
        RenderSettings(), CaptionStyle(),
    )

    # Seed a manifest matching that fingerprint.
    _write_manifest(out_dir / "rendered.json", mp4, fp=fp, out_sha="OUT_SHA")

    def _boom(*args, **kwargs):
        raise AssertionError("_render_one must not run on fingerprint skip")
    monkeypatch.setattr(rmod, "_render_one", _boom)

    cfg = RenderConfig(out_dir=out_dir, log_path=tmp_path / "log.jsonl", skip_existing=True)
    result = run_pass2b(shaped, _transcript(), "T_SHA", video, cfg=cfg)
    assert result.renders_skipped_fingerprint == 1
    assert result.renders_completed == 0
    entry = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert entry["skipped_reason"] == "fingerprint_match"


def test_run_pass2b_fingerprint_mismatch_forces_render(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video")
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    mp4 = out_dir / "clip-0.mp4"
    mp4.write_bytes(b"fake")

    shaped = Pass2aResult(
        shaped=[_shaped_keep(0)],
        model="m",
        prompt_version="v1",
        source_pass1_prompt_version="v2",
    )
    monkeypatch.setattr(rmod, "_resolve_font", lambda style, cache_dir: (tmp_path / "font.ttf", "FONT_SHA"))
    (tmp_path / "font.ttf").write_bytes(b"font")
    monkeypatch.setattr(rmod, "_sha256_file", lambda p: "ANY")

    # Seed a stale fingerprint so the skip guard sees a mismatch.
    _write_manifest(out_dir / "rendered.json", mp4, fp="STALE_FP", out_sha="ANY")

    render_called = {"n": 0}

    def _fake_render(sc, transcript, source, source_sha, transcript_sha, font_path, font_sha, cfg, fp, shaped_arg):
        render_called["n"] += 1
        return RenderedClip(
            shaped_candidate_index=sc.candidate_index,
            source_start=sc.source_start,
            source_end=sc.source_end,
            trim_start=sc.trim.start_seconds,
            trim_end=sc.trim.end_seconds,
            output_path=mp4,
            output_sha256="OUT_SHA",
            input_fingerprint=fp,
            output_duration_seconds=15.0,
            measured_integrated_lufs=-14.0,
            face_coverage=0.8,
            hook_overlay_text=None,
            settings=cfg.settings,
            caption_style=cfg.caption_style,
            render_wall_seconds=1.0,
            render_version=RENDER_VERSION,
            source_video_sha256=source_sha,
            source_transcript_sha256=transcript_sha,
            font_file_sha256=font_sha,
            source_shaped_prompt_version=shaped_arg.prompt_version,
            source_pass1_prompt_version=shaped_arg.source_pass1_prompt_version,
        )

    monkeypatch.setattr(rmod, "_render_one", _fake_render)

    cfg = RenderConfig(out_dir=out_dir, log_path=tmp_path / "log.jsonl", skip_existing=True)
    result = run_pass2b(shaped, _transcript(), "T_SHA", video, cfg=cfg)
    assert render_called["n"] == 1
    assert result.renders_completed == 1
    assert result.renders_skipped_fingerprint == 0
