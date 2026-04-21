"""Unit tests for `clipper post` — identity resolution + append-only log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper._posted import append_posted, resolve_identity
from clipper.schemas import (
    CaptionStyle,
    PostedClip,
    RenderedClip,
    RenderResult,
    RenderSettings,
)


def _rendered_clip(mp4: Path, *, idx: int = 0) -> RenderedClip:
    return RenderedClip(
        shaped_candidate_index=idx,
        source_start=100.0,
        source_end=115.0,
        trim_start=100.0,
        trim_end=115.0,
        output_path=mp4,
        output_sha256="OUT_SHA",
        input_fingerprint="fp",
        output_duration_seconds=15.0,
        measured_integrated_lufs=-14.0,
        face_coverage=0.9,
        hook_overlay_text=None,
        settings=RenderSettings(),
        caption_style=CaptionStyle(),
        render_wall_seconds=1.0,
        render_version="r1",
        source_video_sha256="SRC_SHA",
        source_transcript_sha256="TR_SHA",
        font_file_sha256="f",
        source_shaped_prompt_version="v1",
        source_score_prompt_version="v2",
    )


def _write_manifest(stem_root: Path, clips: list[RenderedClip]) -> Path:
    out_dir = stem_root / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in clips:
        Path(c.output_path).touch()
    manifest_path = out_dir / "rendered.json"
    result = RenderResult(
        rendered=clips,
        render_version="r1",
        source_video_sha256="SRC_SHA",
        source_transcript_sha256="TR_SHA",
        source_shaped_prompt_version="v1",
        source_score_prompt_version="v2",
        renders_completed=len(clips),
    )
    manifest_path.write_text(result.model_dump_json())
    return manifest_path


def _posted_row(matched: RenderedClip, stem_root: Path, **overrides) -> PostedClip:
    kwargs = dict(
        ts="2026-04-20T12:00:00Z",
        shaped_candidate_index=matched.shaped_candidate_index,
        video_stem=stem_root.name,
        source_video_sha256=matched.source_video_sha256,
        output_path=str(Path(matched.output_path).resolve()),
        output_sha256=matched.output_sha256,
        render_version=matched.render_version,
        source_shaped_prompt_version=matched.source_shaped_prompt_version,
        source_score_prompt_version=matched.source_score_prompt_version,
        platform="shorts",
        post_id="ABC123",
        posted_at="2026-04-20T12:00:00Z",
    )
    kwargs.update(overrides)
    return PostedClip(**kwargs)


def test_post_resolves_identity_and_appends(tmp_path):
    stem_root = tmp_path / "myvideo"
    mp4 = stem_root / "clips" / "clip-0.mp4"
    clip = _rendered_clip(mp4, idx=7)
    manifest = _write_manifest(stem_root, [clip])

    matched, resolved_root = resolve_identity(mp4, manifest)
    assert matched.shaped_candidate_index == 7
    assert resolved_root == stem_root

    log_path = resolved_root / "logs" / "posted.jsonl"
    row = _posted_row(matched, resolved_root)
    append_posted(log_path, row)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    parsed = PostedClip.model_validate_json(lines[0])
    assert parsed.shaped_candidate_index == 7
    assert parsed.video_stem == "myvideo"
    assert parsed.source_video_sha256 == "SRC_SHA"
    assert parsed.output_sha256 == "OUT_SHA"
    assert parsed.platform == "shorts"


def test_post_missing_manifest_raises(tmp_path):
    stem_root = tmp_path / "myvideo"
    (stem_root / "clips").mkdir(parents=True)
    mp4 = stem_root / "clips" / "clip-0.mp4"
    mp4.touch()
    manifest = stem_root / "clips" / "rendered.json"  # doesn't exist

    with pytest.raises(FileNotFoundError) as exc:
        resolve_identity(mp4, manifest)
    assert str(manifest) in str(exc.value)


def test_post_clip_not_in_manifest_raises(tmp_path):
    stem_root = tmp_path / "myvideo"
    # Manifest contains clip-0 only.
    listed = stem_root / "clips" / "clip-0.mp4"
    clip = _rendered_clip(listed, idx=0)
    manifest = _write_manifest(stem_root, [clip])

    # User points at a different MP4 under the same directory.
    other = stem_root / "clips" / "clip-1.mp4"
    other.touch()
    with pytest.raises(ValueError) as exc:
        resolve_identity(other, manifest)
    msg = str(exc.value)
    assert str(other) in msg
    assert str(listed) in msg


def test_post_appends_two_rows_for_cross_post(tmp_path):
    stem_root = tmp_path / "myvideo"
    mp4 = stem_root / "clips" / "clip-0.mp4"
    clip = _rendered_clip(mp4, idx=3)
    manifest = _write_manifest(stem_root, [clip])

    matched, root = resolve_identity(mp4, manifest)
    log_path = root / "logs" / "posted.jsonl"

    append_posted(
        log_path, _posted_row(matched, root, platform="shorts", post_id="yt-1")
    )
    append_posted(
        log_path, _posted_row(matched, root, platform="tiktok", post_id="tt-1")
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [PostedClip.model_validate_json(line) for line in lines]
    assert [p.platform for p in parsed] == ["shorts", "tiktok"]
    assert [p.post_id for p in parsed] == ["yt-1", "tt-1"]
    assert all(p.shaped_candidate_index == 3 for p in parsed)


@pytest.mark.parametrize(
    "platform, post_id",
    [("", "abc"), ("   ", "abc"), ("shorts", ""), ("shorts", "   ")],
)
def test_post_empty_platform_or_post_id_raises(tmp_path, platform, post_id):
    stem_root = tmp_path / "myvideo"
    mp4 = stem_root / "clips" / "clip-0.mp4"
    clip = _rendered_clip(mp4, idx=0)
    manifest = _write_manifest(stem_root, [clip])
    matched, root = resolve_identity(mp4, manifest)

    with pytest.raises(Exception) as exc:
        _posted_row(matched, root, platform=platform, post_id=post_id)
    # Pydantic's ValidationError names the offending field.
    msg = str(exc.value)
    assert "platform" in msg or "post_id" in msg
