# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```sh
# Install (editable) + dev deps
pip install -e '.[dev]'

# Run the CLI end-to-end
python -m clipper process path/to/video.mp4 --out candidates.json

# Iterate cheaply: first N windows only, verbose logs, cheaper model
python -m clipper process clip.mp4 --max-windows 3 --verbose --model claude-sonnet-4-6

# Force CPU whisper (no CUDA / small machine)
python -m clipper process clip.mp4 --device cpu --whisper tiny

# Tests
pytest                                    # full suite
pytest tests/test_schemas.py              # single file
pytest tests/test_schemas.py::test_consistent_candidate_ok   # single test
```

`ANTHROPIC_API_KEY` must be set for `pass1` to run. `ffmpeg` + `ffprobe` must be on `PATH`.

## Architecture

Single-command pipeline (`clipper.cli:process`) that runs five stages over one video. Each stage's output is a pydantic model in `schemas.py` — that schema is the contract between stages and is the seam the project is explicitly designed around ("all intermediate outputs MUST be structured to support" a future feedback loop).

```
video → ingest (audio + whisper) → signals → frames → windows → pass1 → Pass1Result JSON
```

Stage-by-stage responsibilities:

- **`ingest.py`** — ffmpeg extracts 16kHz mono WAV; `faster-whisper` transcribes with word timestamps; Whisper segments get re-grouped into sentence-aligned `TranscriptSegment`s (see `_postprocess`).
- **`signals.py`** — two independent feature streams. Text features (exclamations, interruptions, repeated words, length-spikes via z-score, sentence boundaries) come from the transcript. Audio features (volume spikes vs. rolling median, silence gaps, speech-rate changes) come from RMS of the WAV. These are proxies for emotional peaks / pacing that the LLM can't derive from a still frame.
- **`frames.py`** — base frame every ~20s plus *event-driven* frames at audio spikes / text peaks / sentence boundaries, deduped if within 2s. Event triggers survive on the `Frame.trigger` field so `pass1` can prioritize them.
- **`windows.py`** — 25s windows with 12s stride. Windows overlap on purpose so a clip that straddles a boundary gets scored twice; dedup happens post-scoring.
- **`pass1.py`** — per-window call to Claude with (a) system prompt = the rubric (cached via `cache_control: ephemeral`), (b) user content = transcript + compact signal summary + up to 6 frames (event-driven preferred over base). Output is parsed, validated against `Candidate`, clamped to window bounds, and finally deduped across overlapping windows by IoU ≥ 0.6 keeping the highest `overall_score`.

### Key invariants to preserve

- **`Candidate.duration_seconds` must be in `[15, 20]`** and must match `end - start` within 0.1s. This is enforced by pydantic validators in `schemas.py` and tested in `test_schemas.py` — if you widen the clip-length range, both move together.
- **Prompt versioning is load-bearing.** `pass1.PROMPT_VERSION` is pinned to the directory under `clipper/prompts/`, the string is copied into every `Candidate.prompt_version`, and the `.md` file ships as package data (`pyproject.toml` `package-data`). When editing the rubric in a way that changes output semantics, create `clipper/prompts/v2/…` and bump `PROMPT_VERSION` rather than editing v1 in place — old `candidates.json` files stay joinable.
- **Cache writes are atomic.** `extract_audio` and `transcribe` write to `.tmp` then `os.replace` — do not "simplify" this. `test_ingest_cache.py` asserts no partial files on failure.
- **Default model is `claude-opus-4-7`** (`pass1.DEFAULT_MODEL`). `messages.create` uses `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — opus-tier features; don't strip them when editing.
- **Rubric loading uses `importlib.resources`**, not relative paths, so the installed `clipper` console script works from any cwd (`test_prompt_loading.py` guards this).

### Caching layout

Everything lands under `.cache/` (gitignored):

- `audio-<sha1-of-video>.wav` — one per unique video bytes.
- `whisper-<model>-<sha1-of-audio>.json` — raw Whisper output; post-processing runs on every load so you can tweak `_postprocess` without re-transcribing.
- `frames/frame-<video-hash>-<ms>.jpg` — video hash is `(path, size, mtime)`, not content, so replacing a file at the same path invalidates its frames.
- Claude system-prompt cache is server-side via `cache_control: ephemeral`; check `cache_read_input_tokens` in `Pass1Result` to confirm hits.

## Scope note

Per `video_to_clips_mvp_updated.md`, only **Pass 1 (candidate identification)** is in scope. Pass 2 (clip crafting — cropping, captioning, reframing) is explicitly deferred. Don't build Pass 2 features into the Pass 1 path.
