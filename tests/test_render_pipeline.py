"""End-to-end rendering test — requires ffmpeg on PATH.

Synthesizes a tiny silent test video with a flat color + a single face-like
rectangle via ffmpeg's lavfi source, then runs the full render pipeline. Skips
cleanly when ffmpeg is missing or mediapipe/cv2 are unavailable (we still
exercise the pipeline but with the center-crop fallback).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper.render import RenderConfig, run_pass2b
from clipper.schemas import (
    CaptionStyle,
    HookTreatment,
    Pass2aResult,
    RenderSettings,
    ShapedClip,
    ShapedTrim,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_fixture_video(path: Path, seconds: float = 6.0) -> None:
    """Synthesize a 720p video with a recognizable flat color + silent audio."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-f", "lavfi", "-i", f"color=c=teal:s=1280x720:d={seconds}:r=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _make_transcript(seconds: float) -> Transcript:
    words: list[TranscriptWord] = []
    t = 0.0
    vocab = ["most", "founders", "undercharge", "by", "a", "factor", "of", "three"]
    i = 0
    step = 0.4
    while t + step <= seconds:
        words.append(TranscriptWord(start=t, end=t + step, word=vocab[i % len(vocab)]))
        t += step
        i += 1
    seg = TranscriptSegment(
        start=0.0, end=words[-1].end if words else seconds,
        text=" ".join(w.word for w in words), words=words,
    )
    return Transcript(language="en", duration=seconds, segments=[seg])


def _shaped(trim_start=0.5, trim_end=13.5) -> Pass2aResult:
    # 13.0s is outside ShapedTrim's [12, 22] range; use 13 exactly to stay valid.
    duration = round(trim_end - trim_start, 2)
    trim = ShapedTrim(
        start_seconds=trim_start, end_seconds=trim_end,
        start_word="most", end_word="three",
        duration_seconds=duration,
    )
    return Pass2aResult(
        shaped=[
            ShapedClip(
                candidate_index=0,
                source_start=0.0, source_end=15.0,
                verdict="keep",
                trim=trim,
                hook=HookTreatment(
                    kind="overlay",
                    text="Short hook line.",
                    rationale="r",
                ),
                reasoning="r",
                model="m",
                prompt_version="v1",
            ),
            ShapedClip(
                candidate_index=1,
                source_start=20.0, source_end=35.0,
                verdict="kill",
                kill_reason="no payoff",
                reasoning="r",
                model="m",
                prompt_version="v1",
            ),
        ],
        model="m",
        prompt_version="v1",
        source_pass1_prompt_version="v2",
    )


def test_end_to_end_render(tmp_path):
    video = tmp_path / "fixture.mp4"
    _make_fixture_video(video, seconds=16.0)

    transcript = _make_transcript(16.0)
    tr_path = tmp_path / "tr.json"
    tr_path.write_text(transcript.model_dump_json())
    tr_sha = _sha256_file(tr_path)

    pass2a = _shaped(trim_start=0.5, trim_end=13.5)

    cfg = RenderConfig(
        out_dir=tmp_path / "clips",
        log_path=tmp_path / "pass2b.jsonl",
        skip_existing=True,
    )

    result = run_pass2b(pass2a, transcript, tr_sha, video, cfg=cfg)
    assert result.renders_completed == 1
    assert result.renders_skipped_kill == 1
    assert result.renders_failed == 0

    out_mp4 = tmp_path / "clips" / "clip-0.mp4"
    assert out_mp4.exists()
    assert not out_mp4.with_suffix(".mp4.tmp").exists()

    # Dimensions check via ffprobe.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(out_mp4),
        ],
        check=True, capture_output=True, text=True,
    )
    w, h = probe.stdout.strip().split("x")
    assert (int(w), int(h)) == (1080, 1920)

    # Duration within tolerance of (trim_end - trim_start).
    expected = 13.0
    assert abs(result.rendered[0].output_duration_seconds - expected) < 0.2

    # Output sha is populated and matches the file on disk.
    assert result.rendered[0].output_sha256 == _sha256_file(out_mp4)

    # Manifest file written atomically (no .tmp leftover).
    assert (tmp_path / "clips" / "rendered.json").exists()
    assert not (tmp_path / "clips" / "rendered.json.tmp").exists()


def test_rerun_hits_fingerprint_skip(tmp_path):
    video = tmp_path / "fixture.mp4"
    _make_fixture_video(video, seconds=16.0)
    transcript = _make_transcript(16.0)
    tr_path = tmp_path / "tr.json"
    tr_path.write_text(transcript.model_dump_json())
    tr_sha = _sha256_file(tr_path)
    pass2a = _shaped(trim_start=0.5, trim_end=13.5)

    cfg = RenderConfig(out_dir=tmp_path / "clips", log_path=tmp_path / "log.jsonl")
    r1 = run_pass2b(pass2a, transcript, tr_sha, video, cfg=cfg)
    sha1 = r1.rendered[0].output_sha256

    r2 = run_pass2b(pass2a, transcript, tr_sha, video, cfg=cfg)
    assert r2.renders_skipped_fingerprint == 1
    assert r2.renders_completed == 0
    # Same sha — fingerprint skip returned the prior record.
    assert r2.rendered[0].output_sha256 == sha1


def test_failed_render_leaves_no_partial(tmp_path, monkeypatch):
    video = tmp_path / "fixture.mp4"
    _make_fixture_video(video, seconds=6.0)
    transcript = _make_transcript(6.0)
    tr_path = tmp_path / "tr.json"
    tr_path.write_text(transcript.model_dump_json())
    tr_sha = _sha256_file(tr_path)

    # Shape a clip whose trim is inside the fixture.
    pass2a = _shaped(trim_start=0.5, trim_end=13.0)  # end past fixture duration
    cfg = RenderConfig(out_dir=tmp_path / "clips", log_path=tmp_path / "log.jsonl")

    # Force _render_ffmpeg to fail mid-way by substituting the binary.
    from clipper import render as rmod
    real = rmod._render_ffmpeg

    def _fail(*args, **kwargs):
        # Let ffmpeg start, write some bytes, then raise.
        out_path = kwargs.get("out_path") or args[-1]
        Path(out_path).write_bytes(b"PARTIAL")
        raise RuntimeError("synthetic mid-render failure")
    monkeypatch.setattr(rmod, "_render_ffmpeg", _fail)

    result = run_pass2b(pass2a, transcript, tr_sha, video, cfg=cfg)
    assert result.renders_failed == 1

    mp4 = tmp_path / "clips" / "clip-0.mp4"
    assert not mp4.exists()
    tmp = tmp_path / "clips" / "clip-0.mp4.tmp"
    assert not tmp.exists()
