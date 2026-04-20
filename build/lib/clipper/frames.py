"""Frame extraction via ffmpeg.

Base strategy: 1 frame every ~20s at 512px wide. Event-driven frames
get added at volume spikes, transcript peaks, and sentence boundaries
so the model has visual context for the charged moments.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from clipper.schemas import Frame, Signals

log = logging.getLogger(__name__)

_FRAME_WIDTH = 512
_MIN_EVENT_GAP = 2.0  # seconds — dedupe nearby events


def extract_frames(
    video: Path,
    duration: float,
    signals: Signals,
    cache_dir: Path,
    base_stride: float = 20.0,
) -> list[Frame]:
    """Return a deduplicated list of Frames covering the video."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_hash = _video_hash(video)

    targets: list[tuple[float, str]] = []
    t = 0.0
    while t < duration:
        targets.append((t, "base"))
        t += base_stride
    if duration - targets[-1][0] > base_stride / 2:
        targets.append((duration - 0.5, "base"))

    for a in signals.audio:
        if a.kind == "volume_spike":
            targets.append((a.time, "audio_spike"))

    for tx in signals.text:
        if tx.kind in ("exclamation", "length_spike", "interruption"):
            targets.append((tx.time, "text_peak"))
        elif tx.kind == "sentence_boundary":
            targets.append((tx.time, "sentence_boundary"))

    targets.sort(key=lambda x: x[0])
    deduped: list[tuple[float, str]] = []
    for time, trig in targets:
        if time < 0 or time > duration:
            continue
        if deduped and (time - deduped[-1][0]) < _MIN_EVENT_GAP:
            # Prefer non-base triggers when collapsing.
            if deduped[-1][1] == "base" and trig != "base":
                deduped[-1] = (time, trig)
            continue
        deduped.append((time, trig))

    frames: list[Frame] = []
    for time, trig in deduped:
        path = cache_dir / f"frame-{video_hash}-{int(time * 1000):08d}.jpg"
        if not path.exists():
            _extract_single_frame(video, time, path)
        if path.exists() and path.stat().st_size > 0:
            frames.append(Frame(time=time, path=path, trigger=trig))

    log.info("extracted %d frames (from %d targets)", len(frames), len(deduped))
    return frames


def _extract_single_frame(video: Path, time: float, out: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{time:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale={_FRAME_WIDTH}:-2",
        "-q:v", "3",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        log.warning("frame extract failed at t=%.2f: %s", time, e.stderr.decode()[:200])


def _video_hash(video: Path) -> str:
    h = hashlib.sha1()
    h.update(str(video.resolve()).encode())
    st = video.stat()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:10]
