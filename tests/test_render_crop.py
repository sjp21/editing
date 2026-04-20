"""Crop path smoothing + fallback — no ffmpeg, no mediapipe."""

from __future__ import annotations

import pytest

from clipper.render import (
    CropPoint,
    _build_crop_filter,
    _face_coverage,
    _median_crop_center,
    _smooth_crop_path,
)


def _pts(values: list[tuple[float, float | None, float | None]]) -> list[CropPoint]:
    return [CropPoint(t=t, cx=cx, cy=cy) for t, cx, cy in values]


def test_smoothing_damps_single_spike():
    # Steady at 0.5 with one spike to 0.9 at index 3.
    base = [(float(i), 0.5, 0.5) for i in range(20)]
    base[10] = (10.0, 0.9, 0.9)
    sm = _smooth_crop_path(_pts(base), halflife_s=0.5)
    # Smoothed value at the spike must be between 0.5 and 0.9 — not the raw spike.
    spike_sm = sm[10]
    assert spike_sm.cx is not None
    assert 0.5 < spike_sm.cx < 0.9


def test_all_none_track_yields_zero_coverage_and_none_center():
    pts = _pts([(float(i), None, None) for i in range(10)])
    sm = _smooth_crop_path(pts, halflife_s=0.5)
    assert _face_coverage(sm) == 0.0
    cx, cy = _median_crop_center(sm)
    assert cx is None and cy is None


def test_raw_partial_coverage_fraction():
    raw = [
        (0.0, 0.5, 0.5),
        (1.0, None, None),
        (2.0, 0.6, 0.6),
        (3.0, None, None),
        (4.0, 0.55, 0.55),
    ]
    # Coverage is computed on the raw input (what the detector saw), not after
    # smoothing's fill-forward — otherwise any single detection would report
    # ~1.0 coverage for the rest of the clip.
    assert _face_coverage(_pts(raw)) == pytest.approx(3 / 5)


def test_build_crop_filter_centered_on_face():
    # 1920x1080 source, face at (0.75, 0.5) normalized.
    f = _build_crop_filter(1920, 1080, 0.75, 0.5, 1080, 1920)
    # Full-height, 9:16 crop width = 1080 * 9/16 = 607.5 -> 608
    assert "crop=608:1080:" in f
    # Centered near 0.75*1920 - 608/2 = 1440 - 304 = 1136
    parts = f.split(":")
    x = int(parts[2])
    assert 1100 <= x <= 1140


def test_build_crop_filter_fallback_center():
    f = _build_crop_filter(1920, 1080, None, None, 1080, 1920)
    # Center: x = (1920 - 608) / 2 = 656
    assert "crop=608:1080:656:0" == f


def test_build_crop_filter_clamps_out_of_frame_face():
    # Face at right edge — crop rect should clamp inside.
    f = _build_crop_filter(1920, 1080, 0.99, 0.5, 1080, 1920)
    # x must not exceed src_w - crop_w = 1920 - 608 = 1312
    parts = f.split(":")
    x = int(parts[2])
    assert x <= 1312
