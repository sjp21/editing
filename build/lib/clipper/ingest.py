"""Audio extraction and transcription.

Audio is extracted once per video (cached on disk). Transcripts are
cached as JSON keyed by the audio file hash so re-runs during dev
don't re-invoke Whisper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipper.schemas import Transcript, TranscriptSegment, TranscriptWord

log = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    whisper_model: str = "small"
    whisper_device: str = "auto"          # cpu | cuda | auto
    whisper_compute_type: str = "default"  # int8 | float16 | default
    language: str | None = None            # autodetect if None
    cache_dir: Path = Path(".cache")


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()[:16]


def extract_audio(video: Path, cache_dir: Path) -> Path:
    """Extract mono 16kHz WAV suitable for Whisper. Cached by video hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_file(video)
    out = cache_dir / f"audio-{key}.wav"
    if out.exists():
        log.info("audio cache hit: %s", out)
        return out

    log.info("extracting audio with ffmpeg: %s -> %s", video, out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return out


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(audio: Path, cfg: IngestConfig) -> Transcript:
    """Run faster-whisper and return a post-processed transcript.

    Caches the raw transcript JSON on disk. Post-processing (sentence
    grouping, punctuation normalization) runs on every load so we can
    tweak it without re-transcribing.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_file(audio)
    raw_cache = cfg.cache_dir / f"whisper-{cfg.whisper_model}-{key}.json"

    if raw_cache.exists():
        log.info("whisper cache hit: %s", raw_cache)
        raw = json.loads(raw_cache.read_text())
    else:
        raw = _run_whisper(audio, cfg)
        tmp = raw_cache.with_suffix(raw_cache.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(raw))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, raw_cache)

    return _postprocess(raw)


def _run_whisper(audio: Path, cfg: IngestConfig) -> dict:
    # Imported lazily — faster-whisper is heavy and pulls CTranslate2.
    from faster_whisper import WhisperModel

    log.info("loading whisper model=%s device=%s", cfg.whisper_model, cfg.whisper_device)
    compute_type = cfg.whisper_compute_type
    if compute_type == "default":
        compute_type = "int8" if cfg.whisper_device == "cpu" else "float16"

    model = WhisperModel(
        cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio),
        language=cfg.language,
        word_timestamps=True,
        vad_filter=True,
    )

    raw_segments: list[dict] = []
    for seg in segments:
        raw_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            }
        )

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": raw_segments,
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


_SENT_END = re.compile(r"[.!?]+[\"')\]]*\s*$")


def _postprocess(raw: dict) -> Transcript:
    """Group Whisper segments into coherent sentence-aligned chunks."""
    segments: list[TranscriptSegment] = []
    buffer_words: list[TranscriptWord] = []
    buffer_texts: list[str] = []
    buffer_start: float | None = None

    def flush(end: float) -> None:
        nonlocal buffer_words, buffer_texts, buffer_start
        if buffer_start is None or not buffer_texts:
            buffer_words, buffer_texts, buffer_start = [], [], None
            return
        text = " ".join(t.strip() for t in buffer_texts if t.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=buffer_start,
                    end=end,
                    text=text,
                    words=buffer_words,
                )
            )
        buffer_words, buffer_texts, buffer_start = [], [], None

    for seg in raw["segments"]:
        if buffer_start is None:
            buffer_start = seg["start"]
        buffer_texts.append(seg["text"])
        buffer_words.extend(TranscriptWord(**w) for w in seg["words"])

        ends_sentence = bool(_SENT_END.search(seg["text"]))
        long_enough = (seg["end"] - buffer_start) >= 3.0

        if ends_sentence and long_enough:
            flush(seg["end"])

    if buffer_start is not None:
        flush(raw["segments"][-1]["end"])

    return Transcript(
        language=raw.get("language"),
        duration=raw["duration"],
        segments=segments,
    )
