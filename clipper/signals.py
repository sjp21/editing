"""Signal enrichment: proxy cues for emotional peaks, pacing, punchlines.

Two sources:
- Transcript (cheap) — exclamations, interruptions, repeated words, length spikes.
- Audio (lightweight) — volume envelope + silence detection via ffmpeg RMS.

These get injected into the prompt as a compact per-window summary so
the model has hints about where the charged moments likely are.
"""

from __future__ import annotations

import logging
import re
import wave
from pathlib import Path

import numpy as np

from clipper.schemas import (
    AudioFeature,
    Signals,
    TextFeature,
    Transcript,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text signals
# ---------------------------------------------------------------------------


_EXCLAIM = re.compile(r"[!?]{1,}")
_REPEAT = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)
_INTERRUPT = re.compile(r"[-–—]\s*$|\.\.\.\s*$")


def extract_text_features(transcript: Transcript) -> list[TextFeature]:
    features: list[TextFeature] = []

    # Global length statistics for length-spike detection.
    lengths = [len(s.text.split()) for s in transcript.segments] or [0]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths)) or 1.0

    for seg in transcript.segments:
        mid = (seg.start + seg.end) / 2.0

        if _EXCLAIM.search(seg.text):
            features.append(
                TextFeature(
                    time=mid,
                    kind="exclamation",
                    value=1.0,
                    note=seg.text[:80],
                )
            )

        if _INTERRUPT.search(seg.text):
            features.append(
                TextFeature(
                    time=seg.end,
                    kind="interruption",
                    value=1.0,
                    note=seg.text[-40:],
                )
            )

        if _REPEAT.search(seg.text):
            features.append(
                TextFeature(
                    time=mid,
                    kind="repeated_word",
                    value=1.0,
                    note=seg.text[:80],
                )
            )

        word_count = len(seg.text.split())
        z = (word_count - mean_len) / std_len
        if abs(z) >= 1.5:
            features.append(
                TextFeature(
                    time=mid,
                    kind="length_spike",
                    value=float(z),
                    note=f"{word_count}w (z={z:.2f})",
                )
            )

        # Sentence boundary — useful for event-driven frame extraction.
        features.append(
            TextFeature(
                time=seg.end,
                kind="sentence_boundary",
                value=0.0,
            )
        )

    return features


# ---------------------------------------------------------------------------
# Audio signals
# ---------------------------------------------------------------------------


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        width = w.getsampwidth()
        raw = w.readframes(n_frames)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (2**31)

    return data, sample_rate


def extract_audio_features(audio: Path) -> list[AudioFeature]:
    """Compute volume envelope + silence gaps on a 16kHz mono WAV.

    Uses 100ms windows. Volume spikes fire on +1.5 std above local median;
    silence gaps fire when envelope stays below -30dB for >0.6s.
    """
    samples, sr = _read_wav_mono(audio)
    if len(samples) == 0:
        return []

    win = int(sr * 0.1)  # 100 ms
    n_win = len(samples) // win
    if n_win < 2:
        return []

    trimmed = samples[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)

    times = np.arange(n_win) * 0.1
    features: list[AudioFeature] = []

    # Volume spikes — peaks that rise sharply vs. a rolling median.
    rolling = _rolling_median(db, window=50)  # ~5s
    excess = db - rolling
    std = float(np.std(excess)) or 1.0
    spike_thresh = max(3.0, 1.5 * std)

    last_spike = -10.0
    for i, e in enumerate(excess):
        if e >= spike_thresh and (times[i] - last_spike) >= 0.5:
            features.append(
                AudioFeature(
                    time=float(times[i]),
                    kind="volume_spike",
                    value=float(e),
                    note=f"{db[i]:.1f}dB (+{e:.1f} over rolling median)",
                )
            )
            last_spike = float(times[i])

    # Silence gaps — sustained quiet regions.
    silent = db < -35.0
    run_start: int | None = None
    for i, s in enumerate(silent):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            run_len = (i - run_start) * 0.1
            if run_len >= 0.6:
                features.append(
                    AudioFeature(
                        time=float(times[run_start]),
                        kind="silence_gap",
                        value=float(run_len),
                        note=f"{run_len:.2f}s silence",
                    )
                )
            run_start = None

    # Speech rate changes — approximated from envelope zero-crossings per second.
    # Done cheaply by looking at rolling change in nonzero-db fraction.
    active = (db > -30.0).astype(np.float32)
    rate = _rolling_mean(active, window=50)
    rate_delta = np.abs(np.diff(rate, prepend=rate[0]))
    rd_std = float(np.std(rate_delta)) or 1.0
    rate_thresh = max(0.15, 2.0 * rd_std)
    last_rate_change = -10.0
    for i, d in enumerate(rate_delta):
        if d >= rate_thresh and (times[i] - last_rate_change) >= 1.0:
            features.append(
                AudioFeature(
                    time=float(times[i]),
                    kind="speech_rate_change",
                    value=float(d),
                )
            )
            last_rate_change = float(times[i])

    return features


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) <= window:
        return np.full_like(arr, np.median(arr))
    out = np.empty_like(arr)
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    for i in range(len(arr)):
        out[i] = np.median(padded[i : i + window])
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="same")


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def compute_signals(transcript: Transcript, audio: Path) -> Signals:
    return Signals(
        text=extract_text_features(transcript),
        audio=extract_audio_features(audio),
    )
