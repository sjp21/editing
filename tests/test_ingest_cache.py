"""Atomicity tests for ingest cache writes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from clipper import ingest
from clipper.ingest import IngestConfig, extract_audio, transcribe


def _write_fake_ffmpeg(dir_: Path, body: str) -> Path:
    """Create a shell script named ``ffmpeg`` that executes ``body``.

    The script parses the output path from ``-i <in> ... <out>`` argv shape
    used by extract_audio and honors whatever ``body`` does with it.
    """
    script = dir_ / "ffmpeg"
    # Use bash explicitly so ${@: -1} works regardless of /bin/sh being dash.
    script.write_text(
        "#!/usr/bin/env bash\n"
        'out="${@: -1}"\n'
        f"{body}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _prepend_path(monkeypatch, d: Path) -> None:
    monkeypatch.setenv("PATH", f"{d}:{os.environ.get('PATH', '')}")


def test_extract_audio_no_partial_on_failure(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Fake ffmpeg writes a partial file then exits nonzero.
    _write_fake_ffmpeg(bin_dir, 'printf "RIFF" > "$out"\nexit 1')
    _prepend_path(monkeypatch, bin_dir)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"not really a video")

    cache = tmp_path / "cache"

    with pytest.raises(subprocess.CalledProcessError):
        extract_audio(video, cache)

    # Final path must not exist; only the tmp has been cleaned up.
    key = ingest._hash_file(video)
    final = cache / f"audio-{key}.wav"
    assert not final.exists(), "final cache file must not exist after failure"
    tmp = final.with_suffix(final.suffix + ".tmp")
    assert not tmp.exists(), "tmp file must be cleaned up after failure"


def test_extract_audio_success_atomic(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_ffmpeg(bin_dir, 'printf "RIFFDATA" > "$out"')
    _prepend_path(monkeypatch, bin_dir)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video bytes")
    cache = tmp_path / "cache"

    out = extract_audio(video, cache)
    assert out.exists()
    assert out.read_bytes() == b"RIFFDATA"
    # No stray tmp left behind.
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_extract_audio_cache_hit(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # If ffmpeg runs, test fails.
    _write_fake_ffmpeg(bin_dir, "echo should-not-run >&2\nexit 99")
    _prepend_path(monkeypatch, bin_dir)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video bytes")
    cache = tmp_path / "cache"
    cache.mkdir()
    key = ingest._hash_file(video)
    pre_existing = cache / f"audio-{key}.wav"
    pre_existing.write_bytes(b"cached")

    out = extract_audio(video, cache)
    assert out == pre_existing
    assert out.read_bytes() == b"cached"


def test_transcribe_atomic_success(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio bytes")
    cfg = IngestConfig(cache_dir=tmp_path / "cache")

    fake_raw = {
        "language": "en",
        "duration": 1.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hi.", "words": []},
        ],
    }
    monkeypatch.setattr(ingest, "_run_whisper", lambda a, c: fake_raw)

    transcript = transcribe(audio, cfg)
    assert transcript.language == "en"

    key = ingest._hash_file(audio)
    cache_file = cfg.cache_dir / f"whisper-{cfg.whisper_model}-{key}.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text()) == fake_raw
    assert not cache_file.with_suffix(cache_file.suffix + ".tmp").exists()


def test_transcribe_no_partial_on_whisper_failure(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio bytes")
    cfg = IngestConfig(cache_dir=tmp_path / "cache")

    def boom(a, c):
        raise RuntimeError("whisper blew up")

    monkeypatch.setattr(ingest, "_run_whisper", boom)

    with pytest.raises(RuntimeError):
        transcribe(audio, cfg)

    key = ingest._hash_file(audio)
    cache_file = cfg.cache_dir / f"whisper-{cfg.whisper_model}-{key}.json"
    assert not cache_file.exists()
    assert not cache_file.with_suffix(cache_file.suffix + ".tmp").exists()
