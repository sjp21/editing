"""Pass 2b schema round-trip and invariant tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.schemas import (
    CaptionStyle,
    Pass2bResult,
    RenderSettings,
    RenderedClip,
)


def _rendered(**overrides) -> RenderedClip:
    kwargs = dict(
        shaped_candidate_index=0,
        source_start=100.0,
        source_end=115.0,
        trim_start=100.0,
        trim_end=115.0,
        output_path=Path("clips/clip-0.mp4"),
        output_sha256="a" * 64,
        input_fingerprint="f" * 64,
        output_duration_seconds=15.0,
        measured_integrated_lufs=-14.1,
        face_coverage=0.92,
        hook_overlay_text="Most founders undercharge.",
        settings=RenderSettings(),
        caption_style=CaptionStyle(),
        render_wall_seconds=12.5,
        render_version="v1",
        source_video_sha256="9" * 64,
        source_transcript_sha256="7" * 64,
        font_file_sha256="4" * 64,
        model_file_sha256="3" * 64,
        source_shaped_prompt_version="v1",
        source_pass1_prompt_version="v2",
    )
    kwargs.update(overrides)
    return RenderedClip(**kwargs)


def test_rendered_roundtrip_preserves_provenance():
    r = _rendered()
    data = r.model_dump_json()
    back = RenderedClip.model_validate_json(data)
    assert back.input_fingerprint == r.input_fingerprint
    assert back.source_transcript_sha256 == r.source_transcript_sha256
    assert back.render_version == "v1"
    assert back.face_coverage == pytest.approx(0.92)
    assert back.model_file_sha256 == "3" * 64


def test_face_coverage_bounds():
    _rendered(face_coverage=0.0)
    _rendered(face_coverage=1.0)
    with pytest.raises(ValidationError):
        _rendered(face_coverage=-0.01)
    with pytest.raises(ValidationError):
        _rendered(face_coverage=1.01)


def test_pass2b_result_roundtrip():
    p = Pass2bResult(
        rendered=[_rendered()],
        render_version="v1",
        source_video_sha256="9" * 64,
        source_transcript_sha256="7" * 64,
        source_shaped_prompt_version="v1",
        source_pass1_prompt_version="v2",
        renders_completed=1,
        renders_skipped_kill=2,
        renders_skipped_fingerprint=3,
        renders_failed=4,
    )
    data = p.model_dump_json()
    back = Pass2bResult.model_validate_json(data)
    assert back.renders_completed == 1
    assert back.renders_skipped_kill == 2
    assert back.renders_skipped_fingerprint == 3
    assert back.renders_failed == 4
    assert back.rendered[0].input_fingerprint == "f" * 64


def test_settings_and_caption_defaults():
    s = RenderSettings()
    assert s.width == 1080
    assert s.height == 1920
    assert s.loudness_target_lufs == -14.0
    assert s.face_tracker == "mediapipe_blazeface_short_range"

    c = CaptionStyle()
    assert c.words_visible == 3
    assert c.position == "bottom_third"
    assert c.active_color.startswith("#")
