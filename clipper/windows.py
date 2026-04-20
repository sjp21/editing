"""Build overlapping windows from transcript + signals + frames.

Each window is ~25s with a ~12s stride. Candidate clips must land inside
a window but final durations are constrained later (15-20s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clipper.schemas import (
    Frame,
    Signals,
    Transcript,
    Window,
)

log = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    size: float = 25.0
    stride: float = 12.0


def build_windows(
    transcript: Transcript,
    signals: Signals,
    frames: list[Frame],
    cfg: WindowConfig | None = None,
) -> list[Window]:
    cfg = cfg or WindowConfig()
    duration = transcript.duration
    windows: list[Window] = []
    start = 0.0
    idx = 0

    while start < duration:
        end = min(start + cfg.size, duration)
        if end - start < 15.0 and windows:
            # Too short to hold a 15-20s candidate; extend the last one.
            windows[-1] = _rebuild(
                windows[-1].index,
                windows[-1].start,
                end,
                transcript,
                signals,
                frames,
            )
            break

        windows.append(_rebuild(idx, start, end, transcript, signals, frames))
        idx += 1
        start += cfg.stride

    log.info("built %d windows (size=%.1fs stride=%.1fs)", len(windows), cfg.size, cfg.stride)
    return windows


def _rebuild(
    idx: int,
    start: float,
    end: float,
    transcript: Transcript,
    signals: Signals,
    frames: list[Frame],
) -> Window:
    segs = [s for s in transcript.segments if s.end >= start and s.start <= end]
    frames_in = [f for f in frames if start <= f.time <= end]

    sig = Signals(
        audio=[a for a in signals.audio if start <= a.time <= end],
        text=[t for t in signals.text if start <= t.time <= end],
    )

    transcript_text = " ".join(s.text for s in segs).strip()

    return Window(
        index=idx,
        start=start,
        end=end,
        transcript=transcript_text,
        segments=segs,
        frames=frames_in,
        signals=sig,
    )
