"""CLI entry point — `python -m clipper process video.mp4 --out candidates.json`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from clipper.frames import extract_frames
from clipper.ingest import IngestConfig, extract_audio, transcribe
from clipper.pass1 import DEFAULT_MODEL, Pass1Config, run_pass1
from clipper.pass2a import Pass2aConfig, run_pass2a
from clipper.schemas import Pass1Result
from clipper.signals import compute_signals
from clipper.windows import WindowConfig, build_windows

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def process(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    out: Path = typer.Option(Path("candidates.json"), "--out", "-o", help="Output JSON"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Claude model id"),
    whisper_model: str = typer.Option("small", "--whisper", help="faster-whisper model size"),
    whisper_device: str = typer.Option("auto", "--device", help="cpu | cuda | auto"),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache", help="Cache directory"),
    window_size: float = typer.Option(25.0, "--window", help="Window size (seconds)"),
    window_stride: float = typer.Option(12.0, "--stride", help="Window stride (seconds)"),
    frames_per_window: int = typer.Option(6, "--frames", help="Max frames/window sent to Claude"),
    effort: str = typer.Option("high", "--effort", help="low | medium | high | max"),
    max_windows: int | None = typer.Option(
        None, "--max-windows", help="Process only the first N windows (dev)"
    ),
    shape: bool = typer.Option(False, "--shape/--no-shape", help="Also run Pass 2a shaping"),
    shape_out: Path = typer.Option(
        Path("shaped.json"), "--shape-out", help="Pass 2a output JSON (when --shape)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process a video into ranked candidate clips via Pass 1."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    ingest_cfg = IngestConfig(
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        cache_dir=cache_dir,
    )

    log.info("1/5 extracting audio")
    audio = extract_audio(video, cache_dir)

    log.info("2/5 transcribing (%s)", whisper_model)
    transcript = transcribe(audio, ingest_cfg)
    log.info("transcript: %d segments, %.1fs, language=%s",
             len(transcript.segments), transcript.duration, transcript.language)

    log.info("3/5 computing signals")
    signals = compute_signals(transcript, audio)
    log.info("signals: %d audio, %d text", len(signals.audio), len(signals.text))

    log.info("4/5 extracting frames")
    frames = extract_frames(
        video=video,
        duration=transcript.duration,
        signals=signals,
        cache_dir=cache_dir / "frames",
    )

    windows = build_windows(
        transcript, signals, frames,
        cfg=WindowConfig(size=window_size, stride=window_stride),
    )
    if max_windows is not None:
        windows = windows[:max_windows]
        log.info("limited to first %d windows", len(windows))

    log.info("5/5 running Pass 1 on %d windows (model=%s)", len(windows), model)
    pass1_cfg = Pass1Config(
        model=model,
        frames_per_window=frames_per_window,
        effort=effort,
    )
    result = run_pass1(windows, cfg=pass1_cfg)

    out.write_text(result.model_dump_json(indent=2))
    log.info(
        "wrote %s — %d candidates (%d windows processed, %d rejected)",
        out,
        len(result.candidates),
        result.windows_processed,
        result.windows_rejected,
    )
    log.info(
        "tokens: in=%d out=%d cache_read=%d cache_write=%d",
        result.total_input_tokens,
        result.total_output_tokens,
        result.cache_read_tokens,
        result.cache_creation_tokens,
    )

    _print_summary(result, top_k=10)

    if shape:
        log.info("running Pass 2a on %d candidates", len(result.candidates))
        shaped = run_pass2a(result, transcript, cfg=Pass2aConfig(model=model, effort=effort))
        shape_out.write_text(shaped.model_dump_json(indent=2))
        log.info(
            "wrote %s — %d shaped, %d killed, %d failed",
            shape_out,
            shaped.candidates_shaped,
            shaped.candidates_killed,
            shaped.candidates_failed,
        )


@app.command()
def shape(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    candidates: Path = typer.Option(
        ..., "--candidates", "-c", exists=True, readable=True,
        help="Pass1Result JSON from a previous `process` run",
    ),
    out: Path = typer.Option(Path("shaped.json"), "--out", "-o", help="Output JSON"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Claude model id"),
    whisper_model: str = typer.Option("small", "--whisper", help="faster-whisper model size"),
    whisper_device: str = typer.Option("auto", "--device", help="cpu | cuda | auto"),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache", help="Cache directory"),
    effort: str = typer.Option("high", "--effort", help="low | medium | high | max"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Pass 2a shaping on a saved Pass 1 result."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    pass1 = Pass1Result.model_validate_json(candidates.read_text())
    log.info("loaded %d candidates from %s", len(pass1.candidates), candidates)

    ingest_cfg = IngestConfig(
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        cache_dir=cache_dir,
    )
    audio = extract_audio(video, cache_dir)
    transcript = transcribe(audio, ingest_cfg)

    shaped = run_pass2a(pass1, transcript, cfg=Pass2aConfig(model=model, effort=effort))
    out.write_text(shaped.model_dump_json(indent=2))
    log.info(
        "wrote %s — %d shaped, %d killed, %d failed",
        out,
        shaped.candidates_shaped,
        shaped.candidates_killed,
        shaped.candidates_failed,
    )
    log.info(
        "tokens: in=%d out=%d cache_read=%d cache_write=%d",
        shaped.total_input_tokens,
        shaped.total_output_tokens,
        shaped.cache_read_tokens,
        shaped.cache_creation_tokens,
    )


def _print_summary(result, top_k: int = 10) -> None:
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"TOP {min(top_k, len(result.candidates))} CANDIDATES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for i, c in enumerate(result.candidates[:top_k], 1):
        print(
            f"{i:>2}. [{c.start_seconds:7.2f} - {c.end_seconds:7.2f}] "
            f"score={c.overall_score:5.2f}  conf={c.confidence:.2f}  "
            f"({c.duration_seconds:.1f}s)",
            file=sys.stderr,
        )
        print(f"    hook: {c.suggested_hook}", file=sys.stderr)
        print(f"    why:  {c.rationale}", file=sys.stderr)


if __name__ == "__main__":
    app()
