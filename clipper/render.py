"""Pass 2b — Rendering.

Deterministic renderer that consumes a ``ShapedClip`` + pinned ``Transcript``
+ source video and produces a 9:16 MP4 with burned word-by-word captions,
EBU R128 loudness-normalized audio, and an optional hook overlay.

No Claude call. No STT call. This stage is dumb on purpose: any change to
visible output (typography, crop, codec, loudness, font) must bump
``RENDER_VERSION`` so old ``rendered.json`` files stay joinable to outcome
data. Mirrors the ``PROMPT_VERSION`` discipline in ``pass1.py`` / ``pass2a.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from clipper.schemas import (
    CaptionStyle,
    Pass2aResult,
    Pass2bResult,
    RenderedClip,
    RenderSettings,
    ShapedClip,
    Transcript,
    TranscriptWord,
)

log = logging.getLogger(__name__)


RENDER_VERSION = "v1"
DEFAULT_LOG_PATH = Path(".cache/logs/pass2b.jsonl")
BUNDLED_FONT_RESOURCE = "assets/fonts/DejaVuSans.ttf"
CROP_SAMPLE_FPS = 5.0


@dataclass
class RenderConfig:
    settings: RenderSettings = field(default_factory=RenderSettings)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    out_dir: Path = Path("clips")
    log_path: Path | None = DEFAULT_LOG_PATH
    skip_existing: bool = True
    manifest_path: Path | None = None  # rendered.json; defaults to out_dir/rendered.json
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    cache_dir: Path = Path(".cache")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def run_pass2b(
    shaped: Pass2aResult,
    transcript: Transcript,
    transcript_sha256: str,
    source_video: Path,
    cfg: RenderConfig | None = None,
) -> Pass2bResult:
    cfg = cfg or RenderConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    source_sha = _sha256_file(source_video)
    font_path, font_sha = _resolve_font(cfg.caption_style, cfg.cache_dir)

    prior = _prior_manifest_for_skip(_manifest_path(cfg)) if cfg.skip_existing else {}

    result = Pass2bResult(
        rendered=[],
        render_version=RENDER_VERSION,
        source_video_sha256=source_sha,
        source_transcript_sha256=transcript_sha256,
        source_shaped_prompt_version=shaped.prompt_version,
        source_pass1_prompt_version=shaped.source_pass1_prompt_version,
    )

    for sc in shaped.shaped:
        if sc.verdict == "kill":
            result.renders_skipped_kill += 1
            if cfg.log_path is not None:
                _append_log(
                    cfg.log_path,
                    _log_entry_kill(sc, shaped),
                )
            continue

        try:
            fp = _compute_input_fingerprint(
                shaped_clip=sc,
                source_sha=source_sha,
                transcript_sha=transcript_sha256,
                font_sha=font_sha,
                settings=cfg.settings,
                caption_style=cfg.caption_style,
            )
            skip = _should_skip(sc.candidate_index, fp, prior)
            if skip is not None:
                result.rendered.append(skip)
                result.renders_skipped_fingerprint += 1
                if cfg.log_path is not None:
                    _append_log(cfg.log_path, _log_entry_fp_skip(skip, shaped))
                continue

            rendered = _render_one(
                sc,
                transcript,
                source_video,
                source_sha,
                transcript_sha256,
                font_path,
                font_sha,
                cfg,
                fp,
                shaped,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("render %d failed: %s", sc.candidate_index, e)
            result.renders_failed += 1
            if cfg.log_path is not None:
                _append_log(cfg.log_path, _log_entry_failed(sc, shaped, str(e)))
            continue

        result.rendered.append(rendered)
        result.renders_completed += 1
        if cfg.log_path is not None:
            _append_log(cfg.log_path, _log_entry_rendered(rendered, shaped))

    # Write manifest atomically.
    _write_manifest(result, _manifest_path(cfg))
    return result


def _manifest_path(cfg: RenderConfig) -> Path:
    return cfg.manifest_path or (cfg.out_dir / "rendered.json")


# ---------------------------------------------------------------------------
# Fingerprinting & skip
# ---------------------------------------------------------------------------


def _compute_input_fingerprint(
    shaped_clip: ShapedClip,
    source_sha: str,
    transcript_sha: str,
    font_sha: str,
    settings: RenderSettings,
    caption_style: CaptionStyle,
) -> str:
    payload = {
        "render_version": RENDER_VERSION,
        "source_video_sha256": source_sha,
        "source_transcript_sha256": transcript_sha,
        "font_file_sha256": font_sha,
        "shaped_clip": shaped_clip.model_dump(mode="json"),
        "render_settings": settings.model_dump(mode="json"),
        "caption_style": caption_style.model_dump(mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prior_manifest_for_skip(manifest_path: Path) -> dict[int, RenderedClip]:
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text())
        prior = Pass2bResult.model_validate(data)
    except Exception as e:  # noqa: BLE001
        log.warning("could not load prior manifest %s: %s", manifest_path, e)
        return {}
    return {r.shaped_candidate_index: r for r in prior.rendered}


def _should_skip(
    idx: int, fingerprint: str, prior: dict[int, RenderedClip]
) -> RenderedClip | None:
    """Return the prior RenderedClip if all three guardrails match; else None."""
    p = prior.get(idx)
    if p is None:
        return None
    if p.input_fingerprint != fingerprint:
        return None
    if not p.output_path.exists():
        return None
    on_disk = _sha256_file(p.output_path)
    if on_disk != p.output_sha256:
        return None
    return p


# ---------------------------------------------------------------------------
# Font resolution (importlib.resources, same pattern as rubric loading)
# ---------------------------------------------------------------------------


def _resolve_font(style: CaptionStyle, cache_dir: Path) -> tuple[Path, str]:
    if style.font_file is not None:
        p = Path(style.font_file)
        return p, _sha256_file(p)
    # Read bytes via the Traversable so we don't depend on as_file's Path being
    # live later — under zipapp / PyInstaller the yielded path is a temp file
    # that as_file cleans up on __exit__. Materialize once into a stable cache
    # location so the later ffmpeg subprocess (and fontsdir lookup) can read it.
    ref = resources.files("clipper").joinpath(BUNDLED_FONT_RESOURCE)
    data = ref.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    out = cache_dir / "fonts" / f"DejaVuSans-{sha[:12]}.ttf"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, out)
    return out, sha


# ---------------------------------------------------------------------------
# Loudness measurement (two-pass EBU R128)
# ---------------------------------------------------------------------------


_LOUDNORM_JSON = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)


def _measure_loudness(
    source: Path, trim_start: float, trim_end: float, cfg: RenderConfig
) -> dict:
    cmd = [
        cfg.ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-ss", f"{trim_start:.3f}",
        "-to", f"{trim_end:.3f}",
        "-i", str(source),
        "-af",
        (
            f"loudnorm=I={cfg.settings.loudness_target_lufs}"
            f":TP={cfg.settings.loudness_true_peak_dbfs}"
            f":LRA={cfg.settings.loudness_range_lu}"
            ":print_format=json"
        ),
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # ffmpeg writes loudnorm's JSON to stderr.
    blocks = _LOUDNORM_JSON.findall(proc.stderr)
    if not blocks:
        raise RuntimeError("loudnorm produced no JSON block")
    return json.loads(blocks[-1])


# ---------------------------------------------------------------------------
# Captions — word-by-word ASS
# ---------------------------------------------------------------------------


_ASS_HEADER_TEMPLATE = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Active,{font},{size},{active},&H00000000,{outline},&H00000000,-1,0,0,0,100,100,0,0,1,{ow},0,{align},40,40,{mv},1
Style: Inactive,{font},{size},{inactive},&H00000000,{outline},&H00000000,0,0,0,0,100,100,0,0,1,{ow},0,{align},40,40,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _hex_to_ass(color: str) -> str:
    """'#RRGGBB' -> '&H00BBGGRR' (ASS uses BGR little-endian with AA prefix)."""
    s = color.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #RRGGBB, got {color!r}")
    r, g, b = s[0:2], s[2:4], s[4:6]
    return f"&H00{b}{g}{r}".upper()


def _fmt_ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    cs = int(round(s * 100))
    sec = cs // 100
    cs = cs % 100
    # Re-carry if rounding pushed us to 60s.
    if sec >= 60:
        m += sec // 60
        sec = sec % 60
    if m >= 60:
        h += m // 60
        m = m % 60
    return f"{h:d}:{m:02d}:{sec:02d}.{cs:02d}"


def _ass_alignment(position: str) -> tuple[int, int]:
    """Return (alignment code, margin_v). ASS numpad: 2=bottom-center, 5=middle-center, 8=top-center."""
    if position == "top_third":
        return 8, 200
    if position == "middle":
        return 5, 0
    return 2, 200  # bottom_third default


def _words_in_trim(
    words: list[TranscriptWord], trim_start: float, trim_end: float
) -> list[TranscriptWord]:
    out: list[TranscriptWord] = []
    for w in words:
        mid = (w.start + w.end) / 2.0
        if mid < trim_start or mid > trim_end:
            continue
        # Truncate to trim bounds.
        ws = max(w.start, trim_start)
        we = min(w.end, trim_end)
        if we <= ws:
            continue
        out.append(TranscriptWord(start=ws, end=we, word=w.word, probability=w.probability))
    out.sort(key=lambda w: w.start)
    return out


def _all_words(transcript: Transcript) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for seg in transcript.segments:
        words.extend(seg.words)
    words.sort(key=lambda w: w.start)
    return words


def _ass_escape(text: str) -> str:
    """ASS dialogue text needs braces and backslashes neutralized."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _build_ass(
    words: list[TranscriptWord],
    trim_start: float,
    trim_end: float,
    style: CaptionStyle,
    width: int,
    height: int,
) -> str:
    clipped = _words_in_trim(words, trim_start, trim_end)
    align, margin_v = _ass_alignment(style.position)

    header = _ASS_HEADER_TEMPLATE.format(
        w=width,
        h=height,
        font=style.font_family,
        size=style.font_size,
        active=_hex_to_ass(style.active_color),
        inactive=_hex_to_ass(style.inactive_color),
        outline=_hex_to_ass(style.outline_color),
        ow=style.outline_width,
        align=align,
        mv=margin_v,
    )

    lines: list[str] = []
    n_visible = max(1, style.words_visible)
    half = n_visible // 2

    for i, w in enumerate(clipped):
        start_local = w.start - trim_start
        end_local = w.end - trim_start
        if end_local <= start_local:
            continue

        # Build the visible window around this word.
        lo = max(0, i - half)
        hi = min(len(clipped), lo + n_visible)
        lo = max(0, hi - n_visible)  # right-align if near the end

        parts: list[str] = []
        for j in range(lo, hi):
            text = _ass_escape(clipped[j].word.strip())
            if j == i:
                parts.append("{\\rActive}" + text)
            else:
                parts.append("{\\rInactive}" + text)
        dialogue_text = "{\\rInactive}" + " ".join(parts)

        lines.append(
            f"Dialogue: 0,{_fmt_ass_time(start_local)},{_fmt_ass_time(end_local)},"
            f"Inactive,,0,0,0,,{dialogue_text}"
        )

    return header + "\n".join(lines) + ("\n" if lines else "")


def _write_ass_file(
    words: list[TranscriptWord],
    trim_start: float,
    trim_end: float,
    style: CaptionStyle,
    width: int,
    height: int,
    out_path: Path,
) -> None:
    out_path.write_text(_build_ass(words, trim_start, trim_end, style, width, height), encoding="utf-8")


# ---------------------------------------------------------------------------
# Crop path (MediaPipe face detection + EMA smoothing)
# ---------------------------------------------------------------------------


@dataclass
class CropPoint:
    t: float
    cx: float | None  # normalized [0, 1] in source frame
    cy: float | None


def _compute_crop_path(
    source: Path, trim_start: float, trim_end: float, cfg: RenderConfig
) -> list[CropPoint]:
    """Sample faces at CROP_SAMPLE_FPS across the trim.

    Uses OpenCV's bundled Haar cascade. No external model download; deterministic
    across machines with the same opencv build. See docs/technical-decisions.md
    for the rationale and the upgrade path to a deep-learning detector.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("opencv unavailable (%s); falling back to center crop", e)
        return [CropPoint(t=trim_start, cx=None, cy=None)]

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        log.warning("haar cascade missing at %s; falling back to center crop", cascade_path)
        return [CropPoint(t=trim_start, cx=None, cy=None)]
    detector = cv2.CascadeClassifier(str(cascade_path))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / CROP_SAMPLE_FPS)))
    start_frame = int(round(trim_start * src_fps))
    end_frame = int(round(trim_end * src_fps))

    points: list[CropPoint] = []
    try:
        for f_idx in range(start_frame, end_frame + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            t = f_idx / src_fps
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(max(30, w // 20), max(30, h // 20)),
            )
            if len(faces) > 0:
                # Largest detection wins (single-subject case).
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                cx = (x + fw / 2) / w
                cy = (y + fh / 2) / h
                points.append(CropPoint(t=t, cx=cx, cy=cy))
            else:
                points.append(CropPoint(t=t, cx=None, cy=None))
    finally:
        cap.release()

    if not points:
        return [CropPoint(t=trim_start, cx=None, cy=None)]
    return points


def _smooth_crop_path(
    pts: list[CropPoint], halflife_s: float
) -> list[CropPoint]:
    """EMA-smooth the (cx, cy) series. Gaps (None) are filled by the last valid center."""
    if not pts:
        return []
    if halflife_s <= 0:
        alpha = 1.0
    else:
        # Smoothing happens at the detector cadence (CROP_SAMPLE_FPS), not the
        # output video fps — the series we're smoothing is one sample per
        # CROP_SAMPLE_FPS step, regardless of what the final render rate is.
        dt = 1.0 / CROP_SAMPLE_FPS
        alpha = 1.0 - math.exp(-math.log(2) * dt / halflife_s)

    smoothed: list[CropPoint] = []
    last_cx: float | None = None
    last_cy: float | None = None
    state_cx: float | None = None
    state_cy: float | None = None

    for p in pts:
        if p.cx is not None and p.cy is not None:
            last_cx, last_cy = p.cx, p.cy
        cx, cy = last_cx, last_cy
        if cx is None or cy is None:
            smoothed.append(CropPoint(t=p.t, cx=None, cy=None))
            continue
        if state_cx is None or state_cy is None:
            state_cx, state_cy = cx, cy
        else:
            state_cx = alpha * cx + (1 - alpha) * state_cx
            state_cy = alpha * cy + (1 - alpha) * state_cy
        smoothed.append(CropPoint(t=p.t, cx=state_cx, cy=state_cy))

    return smoothed


def _face_coverage(pts: list[CropPoint]) -> float:
    if not pts:
        return 0.0
    n_face = sum(1 for p in pts if p.cx is not None and p.cy is not None)
    return n_face / len(pts)


def _median_crop_center(pts: list[CropPoint]) -> tuple[float | None, float | None]:
    xs = sorted(p.cx for p in pts if p.cx is not None)
    ys = sorted(p.cy for p in pts if p.cy is not None)
    if not xs or not ys:
        return None, None
    mid = len(xs) // 2
    cx = xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    cy = ys[mid] if len(ys) % 2 else 0.5 * (ys[mid - 1] + ys[mid])
    return cx, cy


# ---------------------------------------------------------------------------
# ffmpeg invocation (single render pass, measured loudnorm applied)
# ---------------------------------------------------------------------------


def _probe_video_dims(source: Path, ffprobe_bin: str) -> tuple[int, int]:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(source),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    w_str, h_str = out.split("x")
    return int(w_str), int(h_str)


def _build_crop_filter(
    src_w: int,
    src_h: int,
    cx_norm: float | None,
    cy_norm: float | None,
    target_w: int,
    target_h: int,
) -> str:
    """Return ffmpeg crop filter.

    Target aspect is 9:16. Compute the largest 9:16 rect that fits in the source
    and center it on (cx, cy). If no face info, center-crop.
    """
    # Crop rect size: height-constrained by default, width = height * 9/16.
    ar_target = target_w / target_h  # 9/16
    # First try full-height crop; if source is narrower than target aspect, full-width.
    crop_h = src_h
    crop_w = int(round(crop_h * ar_target))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w / ar_target))

    if cx_norm is None or cy_norm is None:
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2
    else:
        x = int(round(cx_norm * src_w - crop_w / 2))
        y = int(round(cy_norm * src_h - crop_h / 2))
        x = max(0, min(x, src_w - crop_w))
        y = max(0, min(y, src_h - crop_h))

    return f"crop={crop_w}:{crop_h}:{x}:{y}"


def _is_finite_number(s: str) -> bool:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _loudnorm_apply_filter(measured: dict, cfg: RenderConfig) -> str:
    """Build the loudnorm apply filter.

    If the measurement pass produced non-finite values (silent input, etc.),
    fall back to a single-pass loudnorm — the two-pass correctness guarantee
    doesn't apply to silent audio anyway.
    """
    keys = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
    vals = {k: str(measured.get(k, "")) for k in keys}
    if not all(_is_finite_number(v) for v in vals.values()):
        return (
            f"loudnorm=I={cfg.settings.loudness_target_lufs}"
            f":TP={cfg.settings.loudness_true_peak_dbfs}"
            f":LRA={cfg.settings.loudness_range_lu}"
            ":print_format=summary"
        )
    return (
        f"loudnorm=I={cfg.settings.loudness_target_lufs}"
        f":TP={cfg.settings.loudness_true_peak_dbfs}"
        f":LRA={cfg.settings.loudness_range_lu}"
        f":measured_I={vals['input_i']}"
        f":measured_LRA={vals['input_lra']}"
        f":measured_TP={vals['input_tp']}"
        f":measured_thresh={vals['input_thresh']}"
        f":offset={vals['target_offset']}"
        ":linear=true:print_format=summary"
    )


def _render_ffmpeg(
    source: Path,
    trim_start: float,
    trim_end: float,
    crop_filter: str,
    ass_path: Path,
    font_dir: Path,
    hook: str | None,
    measured: dict,
    cfg: RenderConfig,
    font_file: Path,
    out_path: Path,
) -> None:
    # libass reads styles from the ASS file; we force fontsdir so the host
    # fontconfig cache is bypassed (portability + determinism).
    subtitles_filter = (
        f"subtitles={_ffmpeg_escape_path(str(ass_path))}"
        f":fontsdir={_ffmpeg_escape_path(str(font_dir))}"
    )
    vf = f"{crop_filter},scale={cfg.settings.width}:{cfg.settings.height}:flags=lanczos,{subtitles_filter}"
    if hook:
        hook_duration = 0.8
        safe_hook = _drawtext_escape(hook)
        drawtext = (
            "drawtext=fontfile=" + _ffmpeg_escape_path(str(font_file))
            + f":text='{safe_hook}'"
            + ":fontcolor=white:fontsize=56:borderw=4:bordercolor=black"
            + ":x=(w-text_w)/2:y=(h*0.18)"
            + f":enable='between(t,0,{hook_duration})'"
        )
        vf = f"{vf},{drawtext}"

    af = _loudnorm_apply_filter(measured, cfg)

    cmd = [
        cfg.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-ss", f"{trim_start:.3f}",
        "-to", f"{trim_end:.3f}",
        "-i", str(source),
        "-vf", vf,
        "-af", af,
        "-c:v", cfg.settings.video_codec,
        "-preset", "medium",
        "-crf", str(cfg.settings.video_crf),
        "-pix_fmt", "yuv420p",
        "-r", str(cfg.settings.fps),
        "-video_track_timescale", str(cfg.settings.fps * 1000),
        "-c:a", cfg.settings.audio_codec,
        "-ar", str(cfg.settings.audio_sample_rate),
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _ffmpeg_escape_path(p: str) -> str:
    """Escape characters that break ffmpeg's filter-graph argument parsing.

    ``:`` separates options within a filter; ``,`` separates filters within a
    chain; ``[`` / ``]`` delimit stream labels; ``'`` and ``\\`` are the filter
    parser's quoting / escape characters. Paths under cfg.out_dir are
    caller-controlled, so any of these can realistically appear.
    """
    return (
        p.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def _drawtext_escape(text: str) -> str:
    # The text is wrapped in single quotes by the caller. ffmpeg's filter-graph
    # parser does not honor backslash escapes inside single-quoted regions — a
    # literal `'` always terminates the quoted value — so apostrophes must use
    # the close-escape-reopen idiom: `foo'\''bar`. Backslashes still need
    # doubling and `%` stays escaped because drawtext does %{...} expansion
    # regardless of quoting. Colons are safe inside single-quoted text.
    return (
        text.replace("\\", "\\\\")
        .replace("%", r"\%")
        .replace("'", r"'\''")
    )


def _probe_duration_and_lufs(path: Path, cfg: RenderConfig) -> tuple[float, float]:
    dur_cmd = [
        cfg.ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    dur = float(subprocess.run(dur_cmd, check=True, capture_output=True, text=True).stdout.strip())

    loud_cmd = [
        cfg.ffmpeg_bin,
        "-hide_banner", "-nostats",
        "-i", str(path),
        "-af",
        (
            f"loudnorm=I={cfg.settings.loudness_target_lufs}"
            f":TP={cfg.settings.loudness_true_peak_dbfs}"
            f":LRA={cfg.settings.loudness_range_lu}"
            ":print_format=json"
        ),
        "-f", "null", "-",
    ]
    proc = subprocess.run(loud_cmd, check=True, capture_output=True, text=True)
    blocks = _LOUDNORM_JSON.findall(proc.stderr)
    if not blocks:
        raise RuntimeError("post-render loudnorm produced no JSON block")
    measured = json.loads(blocks[-1])
    raw = measured.get("input_i", "-inf")
    try:
        integrated = float(raw)
    except (TypeError, ValueError):
        integrated = float("-inf")
    # Silent output reports -inf; clamp to a sentinel that still serializes.
    if integrated == float("-inf") or integrated != integrated:
        integrated = -70.0
    return dur, integrated


# ---------------------------------------------------------------------------
# Single-clip render driver
# ---------------------------------------------------------------------------


def _render_one(
    sc: ShapedClip,
    transcript: Transcript,
    source: Path,
    source_sha: str,
    transcript_sha: str,
    font_path: Path,
    font_sha: str,
    cfg: RenderConfig,
    fingerprint: str,
    shaped: Pass2aResult,
) -> RenderedClip:
    assert sc.trim is not None  # keep-path

    trim_start = sc.trim.start_seconds
    trim_end = sc.trim.end_seconds

    out_path = cfg.out_dir / f"clip-{sc.candidate_index}.mp4"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    ass_path = cfg.out_dir / f"clip-{sc.candidate_index}.ass"

    wall_start = time.time()

    # Loudness measurement
    measured = _measure_loudness(source, trim_start, trim_end, cfg)

    # Caption timeline
    _write_ass_file(
        words=_all_words(transcript),
        trim_start=trim_start,
        trim_end=trim_end,
        style=cfg.caption_style,
        width=cfg.settings.width,
        height=cfg.settings.height,
        out_path=ass_path,
    )

    # Crop path — static median for v1.
    src_w, src_h = _probe_video_dims(source, cfg.ffprobe_bin)
    crop_points = _compute_crop_path(source, trim_start, trim_end, cfg)
    smoothed = _smooth_crop_path(crop_points, cfg.settings.crop_smoothing_halflife_s)
    # face_coverage documents the detector hit rate, so it must be measured on
    # the raw series — _smooth_crop_path fills gaps forward from the last valid
    # center, which would inflate coverage to ~1.0 after any single detection.
    coverage = _face_coverage(crop_points)
    med_cx, med_cy = _median_crop_center(smoothed)
    crop_filter = _build_crop_filter(
        src_w, src_h, med_cx, med_cy, cfg.settings.width, cfg.settings.height
    )

    hook_overlay = None
    if sc.hook is not None and sc.hook.kind == "overlay" and sc.hook.text:
        hook_overlay = sc.hook.text

    # ffmpeg render → tmp → os.replace.
    try:
        _render_ffmpeg(
            source=source,
            trim_start=trim_start,
            trim_end=trim_end,
            crop_filter=crop_filter,
            ass_path=ass_path,
            font_dir=font_path.parent,
            hook=hook_overlay,
            measured=measured,
            cfg=cfg,
            font_file=font_path,
            out_path=tmp_path,
        )
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, out_path)

    out_duration, measured_lufs = _probe_duration_and_lufs(out_path, cfg)
    out_sha = _sha256_file(out_path)

    return RenderedClip(
        shaped_candidate_index=sc.candidate_index,
        source_start=sc.source_start,
        source_end=sc.source_end,
        trim_start=trim_start,
        trim_end=trim_end,
        output_path=out_path,
        output_sha256=out_sha,
        input_fingerprint=fingerprint,
        output_duration_seconds=out_duration,
        measured_integrated_lufs=measured_lufs,
        face_coverage=coverage,
        hook_overlay_text=hook_overlay,
        settings=cfg.settings,
        caption_style=cfg.caption_style,
        render_wall_seconds=time.time() - wall_start,
        render_version=RENDER_VERSION,
        source_video_sha256=source_sha,
        source_transcript_sha256=transcript_sha,
        font_file_sha256=font_sha,
        source_shaped_prompt_version=shaped.prompt_version,
        source_pass1_prompt_version=shaped.source_pass1_prompt_version,
    )


# ---------------------------------------------------------------------------
# Hashing + manifest + logs
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _write_manifest(result: Pass2bResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(result.model_dump_json(indent=2))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _append_log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_entry(shaped: Pass2aResult) -> dict:
    return {
        "ts": _now_iso(),
        "stage": "pass2b",
        "render_version": RENDER_VERSION,
        "source_pass1_prompt_version": shaped.source_pass1_prompt_version,
        "source_shaped_prompt_version": shaped.prompt_version,
    }


def _log_entry_rendered(r: RenderedClip, shaped: Pass2aResult) -> dict:
    e = _base_entry(shaped)
    e.update({
        "shaped_candidate_index": r.shaped_candidate_index,
        "trim_start": r.trim_start,
        "trim_end": r.trim_end,
        "output_path": str(r.output_path),
        "input_fingerprint": r.input_fingerprint,
        "output_sha256": r.output_sha256,
        "source_video_sha256": r.source_video_sha256,
        "source_transcript_sha256": r.source_transcript_sha256,
        "font_file_sha256": r.font_file_sha256,
        "output_duration": r.output_duration_seconds,
        "measured_lufs": r.measured_integrated_lufs,
        "hook_overlay_text": r.hook_overlay_text,
        "face_coverage": r.face_coverage,
        "render_wall_seconds": r.render_wall_seconds,
        "skipped_reason": None,
        "failed": False,
    })
    return e


def _log_entry_fp_skip(r: RenderedClip, shaped: Pass2aResult) -> dict:
    e = _log_entry_rendered(r, shaped)
    e["skipped_reason"] = "fingerprint_match"
    return e


def _log_entry_kill(sc: ShapedClip, shaped: Pass2aResult) -> dict:
    e = _base_entry(shaped)
    e.update({
        "shaped_candidate_index": sc.candidate_index,
        "trim_start": None,
        "trim_end": None,
        "output_path": None,
        "skipped_reason": "kill",
        "kill_reason": sc.kill_reason,
        "failed": False,
    })
    return e


def _log_entry_failed(sc: ShapedClip, shaped: Pass2aResult, err: str) -> dict:
    e = _base_entry(shaped)
    e.update({
        "shaped_candidate_index": sc.candidate_index,
        "trim_start": sc.trim.start_seconds if sc.trim else None,
        "trim_end": sc.trim.end_seconds if sc.trim else None,
        "output_path": None,
        "skipped_reason": None,
        "failed": True,
        "error": err,
    })
    return e
