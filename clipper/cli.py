"""CLI entry point — `python -m clipper process video.mp4 --out candidates.json`."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

from clipper._posted import append_posted, resolve_identity
from clipper.eval import run_eval
from clipper.frames import extract_frames
from clipper.ingest import IngestConfig, extract_audio, transcribe
from clipper.score import DEFAULT_MODEL, ScoreConfig, run_score
from clipper.shape import ShapeConfig, run_shape
from clipper.ranker import (
    DEFAULT_K as RANKER_DEFAULT_K,
    RankerConfig,
    apply_ranking,
    run_ranker,
)
from clipper.render import RenderConfig, run_render
from clipper.schemas import PostedClip, RankedCandidates, ScoreResult, ShapeResult, Transcript
from clipper.signals import compute_signals
from clipper.windows import WindowConfig, build_windows

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def process(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-O",
        help="Per-video output root (default: outputs/<video-stem>)",
    ),
    out: Path = typer.Option(
        None, "--out", "-o",
        help="Pass 1 output JSON (default: <output-dir>/candidates.json)",
    ),
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
    rank: bool = typer.Option(
        False, "--rank/--no-rank",
        help="Also run the global ranker between score and shape",
    ),
    k: int = typer.Option(
        RANKER_DEFAULT_K, "--k",
        help="Ranker top-K ceiling (may truncate lower when clips don't clear the bar)",
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Platform profile name for the ranker (optional)"
    ),
    shape: bool = typer.Option(False, "--shape/--no-shape", help="Also run shape stage"),
    shape_out: Path = typer.Option(
        None, "--shape-out",
        help="Shape output JSON (default: <output-dir>/shaped.json)",
    ),
    render: bool = typer.Option(
        False, "--render/--no-render", help="Also run render stage (requires --shape)"
    ),
    render_dir: Path = typer.Option(
        None, "--render-dir",
        help="Output directory for rendered MP4s (default: <output-dir>/clips)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process a video into ranked candidate clips via the scorer."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    # Fail fast before the pipeline — the scorer runs a paid Claude call per window.
    if render and not shape:
        raise typer.BadParameter("--render requires --shape")
    if render and not rank:
        log.warning(
            "rendering un-ranked candidates — you may burn GPU on clips the "
            "ranker would drop. Pass --rank to filter first."
        )

    root = output_dir or Path("outputs") / video.stem
    out = out or root / "candidates.json"
    shape_out = shape_out or root / "shaped.json"
    render_dir = render_dir or root / "clips"
    log_dir = root / "logs"

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

    log.info("5/5 running score on %d windows (model=%s)", len(windows), model)
    score_cfg = ScoreConfig(
        model=model,
        frames_per_window=frames_per_window,
        effort=effort,
    )
    result = run_score(windows, cfg=score_cfg)

    _atomic_write_json(out, result)
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

    shape_input = result
    ranker_prompt_version: str | None = None
    if rank:
        log.info(
            "running ranker on %d candidates (k=%d, profile=%s)",
            len(result.candidates),
            k,
            profile,
        )
        ranker_cfg = RankerConfig(
            model=model,
            effort=effort,
            k=k,
            profile=profile,
            log_path=log_dir / "ranker.jsonl",
        )
        ranked = run_ranker(result, cfg=ranker_cfg)
        _atomic_write_json(root / "ranking_decisions.json", ranked)
        shape_input = apply_ranking(result, ranked)
        _atomic_write_json(root / "ranked.json", shape_input)
        ranker_prompt_version = ranked.prompt_version
        log.info(
            "ranker kept %d of %d candidates%s",
            len(shape_input.candidates),
            len(result.candidates),
            f" (truncated: {ranked.truncation_note})" if ranked.truncation_note else "",
        )

    if shape:
        log.info("running shape on %d candidates", len(shape_input.candidates))
        shaped = run_shape(
            shape_input,
            transcript,
            cfg=ShapeConfig(
                model=model,
                effort=effort,
                profile=profile,
                log_path=log_dir / "shape.jsonl",
            ),
            source_ranker_prompt_version=ranker_prompt_version,
        )
        _atomic_write_json(shape_out, shaped)
        # Persist the transcript sidecar so a standalone `render` call consumes
        # bit-identical word timings. Downstream render hashes this file.
        transcript_sidecar = _write_transcript_sidecar(root, transcript)
        log.info(
            "wrote %s — %d shaped, %d killed, %d failed",
            shape_out,
            shaped.candidates_shaped,
            shaped.candidates_killed,
            shaped.candidates_failed,
        )

        if render:
            log.info("running render into %s", render_dir)
            transcript_sha = _sha256_file(transcript_sidecar)
            rendered = run_render(
                shaped, transcript, transcript_sha, video,
                cfg=RenderConfig(
                    out_dir=render_dir,
                    log_path=log_dir / "render.jsonl",
                ),
            )
            log.info(
                "rendered %d, skipped_kill=%d, skipped_fp=%d, failed=%d",
                rendered.renders_completed,
                rendered.renders_skipped_kill,
                rendered.renders_skipped_fingerprint,
                rendered.renders_failed,
            )


@app.command()
def shape(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    candidates: Path = typer.Option(
        ..., "--candidates", "-c", exists=True, readable=True,
        help="ScoreResult JSON from a previous `process` run",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-O",
        help="Per-video output root (default: outputs/<video-stem>)",
    ),
    out: Path = typer.Option(
        None, "--out", "-o",
        help="Output JSON (default: <output-dir>/shaped.json)",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Claude model id"),
    whisper_model: str = typer.Option("small", "--whisper", help="faster-whisper model size"),
    whisper_device: str = typer.Option("auto", "--device", help="cpu | cuda | auto"),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache", help="Cache directory"),
    effort: str = typer.Option("high", "--effort", help="low | medium | high | max"),
    profile: str | None = typer.Option(
        None, "--profile", help="Platform profile name for the shaper (optional)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run shape on a saved score result."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    root = output_dir or Path("outputs") / video.stem
    out = out or root / "shaped.json"
    log_dir = root / "logs"

    scored = ScoreResult.model_validate_json(candidates.read_text())
    log.info("loaded %d candidates from %s", len(scored.candidates), candidates)

    # If the user just ran `clipper rank` in the same directory, the sibling
    # ranking_decisions.json carries the prompt_version of the ranker call
    # that produced these survivors. Forward it so shape's shaped.json and
    # its jsonl rows can be joined back to that ranker call at retro time.
    ranker_prompt_version: str | None = None
    decisions_path = candidates.parent / "ranking_decisions.json"
    if decisions_path.is_file():
        try:
            decisions = RankedCandidates.model_validate_json(
                decisions_path.read_text()
            )
            ranker_prompt_version = decisions.prompt_version
            log.info(
                "joined ranker provenance from %s (prompt_version=%s)",
                decisions_path,
                ranker_prompt_version,
            )
        except Exception as e:
            log.warning("ignoring unreadable %s: %s", decisions_path, e)

    ingest_cfg = IngestConfig(
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        cache_dir=cache_dir,
    )
    audio = extract_audio(video, cache_dir)
    transcript = transcribe(audio, ingest_cfg)

    shaped = run_shape(
        scored,
        transcript,
        cfg=ShapeConfig(
            model=model,
            effort=effort,
            profile=profile,
            log_path=log_dir / "shape.jsonl",
        ),
        source_ranker_prompt_version=ranker_prompt_version,
    )
    _atomic_write_json(out, shaped)
    sidecar = _write_transcript_sidecar(root, transcript)
    log.info(
        "wrote %s (+ %s) — %d shaped, %d killed, %d failed",
        out,
        sidecar.name,
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


@app.command()
def rank(
    candidates: Path = typer.Option(
        ..., "--candidates", "-c", exists=True, readable=True,
        help="ScoreResult JSON from a previous `process` run",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-O",
        help="Per-video output root (default: <candidates-parent>)",
    ),
    k: int = typer.Option(
        RANKER_DEFAULT_K, "--k",
        help="Ranker top-K ceiling (may truncate lower when clips don't clear the bar)",
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Platform profile name (optional)"
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Claude model id"),
    effort: str = typer.Option("high", "--effort", help="low | medium | high | max"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the global ranker on a saved score result."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    scored = ScoreResult.model_validate_json(candidates.read_text())
    if any(c.score_index is None for c in scored.candidates):
        raise typer.BadParameter(
            f"{candidates} was written before score_index was populated. "
            "Re-run `clipper process` to regenerate candidates.json."
        )

    root = output_dir or candidates.parent
    log_dir = root / "logs"

    log.info(
        "ranking %d candidates (k=%d, profile=%s)",
        len(scored.candidates),
        k,
        profile,
    )
    ranker_cfg = RankerConfig(
        model=model,
        effort=effort,
        k=k,
        profile=profile,
        log_path=log_dir / "ranker.jsonl",
    )
    ranked = run_ranker(scored, cfg=ranker_cfg)

    survivors = apply_ranking(scored, ranked)
    _atomic_write_json(root / "ranking_decisions.json", ranked)
    _atomic_write_json(root / "ranked.json", survivors)
    log.info(
        "wrote %s and %s — kept %d of %d%s",
        root / "ranking_decisions.json",
        root / "ranked.json",
        len(survivors.candidates),
        len(scored.candidates),
        f" (truncated: {ranked.truncation_note})" if ranked.truncation_note else "",
    )
    log.info(
        "tokens: in=%d out=%d cache_read=%d cache_write=%d",
        ranked.total_input_tokens,
        ranked.total_output_tokens,
        ranked.cache_read_tokens,
        ranked.cache_creation_tokens,
    )


@app.command()
def render(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    shaped: Path = typer.Option(
        ..., "--shaped", "-s", exists=True, readable=True,
        help="ShapeResult JSON from a previous `shape` run",
    ),
    transcript: Path = typer.Option(
        None, "--transcript", "-t",
        help="Transcript JSON sidecar (default: <shaped-parent>/transcript.json)",
    ),
    out_dir: Path = typer.Option(
        None, "--out-dir", "-o",
        help="Output directory for rendered MP4s (default: <shaped-parent>/clips)",
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Manifest JSON path (defaults to <out-dir>/rendered.json)"
    ),
    no_skip: bool = typer.Option(
        False, "--no-skip", help="Disable fingerprint-keyed skip (force re-render)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Render clips from a saved shape result + transcript sidecar."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    root = shaped.parent
    transcript_path = transcript or root / "transcript.json"
    if not transcript_path.exists():
        raise typer.BadParameter(
            f"transcript sidecar not found at {transcript_path}. "
            f"Run `clipper shape` first — it writes transcript.json next to shaped.json."
        )
    out_dir = out_dir or root / "clips"

    shape_result = ShapeResult.model_validate_json(shaped.read_text())
    tr = Transcript.model_validate_json(transcript_path.read_text())
    tr_sha = _sha256_file(transcript_path)

    cfg = RenderConfig(
        out_dir=out_dir,
        manifest_path=manifest,
        skip_existing=not no_skip,
        log_path=root / "logs" / "render.jsonl",
    )

    result = run_render(shape_result, tr, tr_sha, video, cfg=cfg)
    log.info(
        "rendered %d, skipped_kill=%d, skipped_fp=%d, failed=%d",
        result.renders_completed,
        result.renders_skipped_kill,
        result.renders_skipped_fingerprint,
        result.renders_failed,
    )


@app.command()
def post(
    clip_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to rendered MP4"
    ),
    platform: str = typer.Option(
        ..., "--platform", help="Platform name, e.g. shorts, tiktok, linkedin"
    ),
    post_id: str = typer.Option(
        ..., "--post-id", help="Platform-assigned post identifier"
    ),
    post_url: str | None = typer.Option(None, "--post-url"),
    title: str | None = typer.Option(None, "--title"),
    posted_at: str | None = typer.Option(
        None, "--posted-at", help="ISO8601 UTC; defaults to now"
    ),
    notes: str | None = typer.Option(None, "--notes"),
    manifest: Path | None = typer.Option(
        None, "--manifest",
        help="rendered.json path (defaults to <clip-parent>/rendered.json)",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-O",
        help="Per-video output root (defaults to <rendered.json parent>/..)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record that a rendered clip was posted to a platform."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    manifest_path = manifest or clip_path.parent / "rendered.json"
    try:
        matched, stem_root = resolve_identity(clip_path, manifest_path)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    root = output_dir or stem_root
    log_path = root / "logs" / "posted.jsonl"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        row = PostedClip(
            ts=now,
            shaped_candidate_index=matched.shaped_candidate_index,
            video_stem=stem_root.name,
            source_video_sha256=matched.source_video_sha256,
            output_path=str(Path(matched.output_path).resolve()),
            output_sha256=matched.output_sha256,
            render_version=matched.render_version,
            source_shaped_prompt_version=matched.source_shaped_prompt_version,
            source_score_prompt_version=matched.source_score_prompt_version,
            platform=platform,
            post_id=post_id,
            post_url=post_url,
            title=title,
            posted_at=posted_at or now,
            notes=notes,
        )
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    append_posted(log_path, row)
    log.info(
        "wrote %s — shaped_candidate_index=%d platform=%s post_id=%s",
        log_path,
        row.shaped_candidate_index,
        row.platform,
        row.post_id,
    )


@app.command(name="eval")
def eval_cmd(
    video_stem: str = typer.Option(
        ..., "--video-stem", help="Per-video stem (resolves to outputs/<stem>/)"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", "-O",
        help="Per-video output root (default: outputs/<video-stem>)",
    ),
    labels: Path = typer.Option(
        None, "--labels", help="human_labels.json (default: <output-dir>/human_labels.json)"
    ),
    candidates: Path = typer.Option(
        None, "--candidates", help="candidates.json (default: <output-dir>/candidates.json)"
    ),
    ranked: Path = typer.Option(
        None, "--ranked", help="ranked.json (default: <output-dir>/ranked.json)"
    ),
    shaped: Path = typer.Option(
        None, "--shaped", help="shaped.json (default: <output-dir>/shaped.json)"
    ),
    iou_threshold: float = typer.Option(
        0.5, "--iou-threshold", help="Min IoU to count a candidate as matching a label"
    ),
    out: Path = typer.Option(
        None, "--out", "-o",
        help="Scorecard output JSON (default: <output-dir>/eval.json)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score the pipeline against hand-written labels."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("clipper.cli")

    root = output_dir or Path("outputs") / video_stem
    out = out or root / "eval.json"
    log_path = root / "logs" / "eval.jsonl"

    scorecard = run_eval(
        video_stem,
        output_dir=root,
        iou_threshold=iou_threshold,
        labels_path=labels,
        candidates_path=candidates,
        ranked_path=ranked,
        shaped_path=shaped,
    )

    _atomic_write_json(out, scorecard)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(scorecard.model_dump_json() + "\n")

    log.info(
        "wrote %s — recall=%.2f inclusion=%.2f shape=%.2f "
        "(misses=%d false_positives=%d)",
        out,
        scorecard.scorer_recall,
        scorecard.ranker_inclusion,
        scorecard.shape_alignment,
        len(scorecard.misses),
        len(scorecard.false_positives),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, model) -> Path:
    """Write a pydantic model to `path` atomically (.tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(model.model_dump_json(indent=2))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def _write_transcript_sidecar(root: Path, transcript: Transcript) -> Path:
    """Write `<root>/transcript.json` atomically."""
    return _atomic_write_json(root / "transcript.json", transcript)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


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
