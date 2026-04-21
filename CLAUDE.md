# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```sh
# Install (editable) + dev deps
pip install -e '.[dev]'

# Run the CLI end-to-end (artifacts land under outputs/<video-stem>/)
python -m clipper process path/to/video.mp4

# Full pipeline in one call (score + rank + shape + render)
python -m clipper process path/to/video.mp4 --rank --shape --render

# Standalone ranker on an existing candidates.json
python -m clipper rank --candidates outputs/myvideo/candidates.json --k 5

# Write the whole per-video tree somewhere else
python -m clipper process path/to/video.mp4 --output-dir /tmp/run1 --rank --shape --render

# Iterate cheaply: first N windows only, verbose logs, cheaper model
python -m clipper process clip.mp4 --max-windows 3 --verbose --model claude-sonnet-4-6

# Force CPU whisper (no CUDA / small machine)
python -m clipper process clip.mp4 --device cpu --whisper tiny

# Tests
pytest                                    # full suite
pytest tests/test_schemas.py              # single file
pytest tests/test_schemas.py::test_consistent_candidate_ok   # single test
```

`ANTHROPIC_API_KEY` must be set for `score` to run. `ffmpeg` + `ffprobe` must be on `PATH`.

## Architecture

Single-command pipeline (`clipper.cli:process`) that runs five stages over one video. Each stage's output is a pydantic model in `schemas.py` — that schema is the contract between stages and is the seam the project is explicitly designed around ("all intermediate outputs MUST be structured to support" a future feedback loop).

```
video → ingest (audio + whisper) → signals → frames → windows → score → [ranker] → shape → render
```

Stage-by-stage responsibilities:

- **`ingest.py`** — ffmpeg extracts 16kHz mono WAV; `faster-whisper` transcribes with word timestamps; Whisper segments get re-grouped into sentence-aligned `TranscriptSegment`s (see `_postprocess`).
- **`signals.py`** — two independent feature streams. Text features (exclamations, interruptions, repeated words, length-spikes via z-score, sentence boundaries) come from the transcript. Audio features (volume spikes vs. rolling median, silence gaps, speech-rate changes) come from RMS of the WAV. These are proxies for emotional peaks / pacing that the LLM can't derive from a still frame.
- **`frames.py`** — base frame every ~20s plus *event-driven* frames at audio spikes / text peaks / sentence boundaries, deduped if within 2s. Event triggers survive on the `Frame.trigger` field so `score` can prioritize them.
- **`windows.py`** — 25s windows with 12s stride. Windows overlap on purpose so a clip that straddles a boundary gets scored twice; dedup happens post-scoring.
- **`score.py`** — per-window call to Claude with (a) system prompt = the rubric (cached via `cache_control: ephemeral`), (b) user content = transcript + compact signal summary + up to 6 frames (event-driven preferred over base). Output is parsed, validated against `Candidate`, clamped to window bounds, and finally deduped across overlapping windows by IoU ≥ 0.6 keeping the highest `overall_score`. After dedupe+sort, each surviving candidate gets a stable `score_index` — the identity downstream stages (and JSONL rows) join on.
- **`ranker.py`** — single Claude call that sees every candidate at once and returns `RankedCandidates` (top-K with mandatory `hook_quote` + `payoff_in_window` + `comparative_reason`, plus a `dropped` list covering the rest). Catches the semantic duplicates the scorer's IoU dedup can't (same story told twice). `apply_ranking(scored, ranked)` filters `ScoreResult` to the survivors, preserving `score_index`; shape stays ignorant of the ranker and keeps shaping whatever `ScoreResult` it's handed.
- **`shape.py`** — shaping. Per-candidate Claude call with ±30s transcript context and a word-level timestamp table. Returns a keep/kill verdict, an exact word-aligned trim, and a hook treatment. `ShapedClip.candidate_index` is set from `cand.score_index` so it survives the ranker's filter/reorder.
- **`render.py`** — deterministic render: crop to 9:16 with face tracking, burn word-by-word captions, normalize loudness. Produces one MP4 per kept clip plus a `RenderResult` manifest.

### Key invariants to preserve

- **`Candidate.duration_seconds` must be in `[15, 20]`** and must match `end - start` within 0.1s. This is enforced by pydantic validators in `schemas.py` and tested in `test_schemas.py` — if you widen the clip-length range, both move together.
- **Prompt versioning is load-bearing.** `score.PROMPT_VERSION` and `ranker.PROMPT_VERSION` are pinned to directories under `clipper/prompts/`, the string is copied into every `Candidate.prompt_version` / `RankedCandidates.prompt_version`, and the `.md` files ship as package data (`pyproject.toml` `package-data`). When editing a rubric in a way that changes output semantics, create `clipper/prompts/vN/…` and bump `PROMPT_VERSION` rather than editing in place — old artifacts stay joinable. Pure provenance fields (`Candidate.score_index`) do NOT bump the prompt version.
- **`Candidate.score_index` is the stable identity across the ranker seam.** The scorer assigns it after dedupe+sort; the ranker reorders/filters candidates but never rewrites it; shape reads it unconditionally (with a fallback to the positional `idx` for old `candidates.json` files). This is what keeps `ShapedClip.candidate_index`, `RenderResult.shaped_candidate_index`, and every JSONL row joinable back to the original scorer identity.
- **One Claude call per ranker run.** The whole point of the ranker is the comparative view; per-candidate ranker calls defeat it. `test_ranker.py::test_run_ranker_single_call` guards this.
- **Cache writes are atomic.** `extract_audio` and `transcribe` write to `.tmp` then `os.replace` — do not "simplify" this. `test_ingest_cache.py` asserts no partial files on failure. The shared `cli._atomic_write_json(path, model)` helper uses the same pattern for `candidates.json`, `shaped.json`, `ranked.json`, `ranking_decisions.json`, and the transcript sidecar.
- **Default model is `claude-opus-4-7`** (`score.DEFAULT_MODEL` and `ranker.DEFAULT_MODEL`). `messages.create` uses `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — opus-tier features; don't strip them when editing. The ranker in particular needs Opus-tier comparative judgment — smaller models flatten the ranking into averaged scores.
- **Rubric loading uses `importlib.resources`**, not relative paths, so the installed `clipper` console script works from any cwd (`test_prompt_loading.py` guards this). `SOUL.md` lives at `clipper/SOUL.md` (shipped via `package-data`) and is loaded the same way — keep it inside the package, not at the repo root.

### Output layout

Every run of `process` (or `shape` / `render`) writes to a per-video tree:

```
outputs/<video-stem>/
├── candidates.json          # score
├── ranking_decisions.json   # ranker (if --rank): full RankedCandidates with drops + reasons
├── ranked.json              # ranker: filtered ScoreResult, survivors in rank order
├── shaped.json              # shape (runs on ranked.json when --rank is set, else candidates.json)
├── transcript.json          # pinned transcript sidecar (hashed by render)
├── clips/clip-<i>.mp4       # render output + ASS subs + rendered.json manifest
└── logs/
    ├── ranker.jsonl         # N per-candidate decisions + 1 summary row, all sharing a call_id
    ├── shape.jsonl          # append-only feedback-loop log for shape
    ├── render.jsonl         # append-only feedback-loop log for render
    └── posted.jsonl         # post: one row per (clip, platform) posting (append-only)
```

Pass `--output-dir PATH` to send the whole tree somewhere else. Override individual
artifacts with `--out`, `--shape-out`, `--render-dir` if needed. `outputs/` is gitignored.

### Caching layout

Content-addressed caches live under `.cache/` (gitignored) and are shared across runs:

- `audio-<sha1-of-video>.wav` — one per unique video bytes.
- `whisper-<model>-<sha1-of-audio>.json` — raw Whisper output; post-processing runs on every load so you can tweak `_postprocess` without re-transcribing.
- `frames/frame-<video-hash>-<ms>.jpg` — video hash is `(path, size, mtime)`, not content, so replacing a file at the same path invalidates its frames.
- `fonts/`, `models/` — bundled assets materialized on first use.
- Claude system-prompt cache is server-side via `cache_control: ephemeral`; check `cache_read_input_tokens` in `ScoreResult` to confirm hits.

## Skills to invoke

- **Before editing `score.py`, `shape.py`, or any new module that imports `anthropic`:** invoke the `claude-api` skill first. It codifies the caching / adaptive-thinking / effort / model-migration patterns this project depends on (see the "Key invariants" above).
- **Before any non-trivial refactor or new stage (ranker, platform profiles, feedback loop):** invoke `andrej-karpathy-skills:karpathy-guidelines` first. Guards against speculative abstractions and unrequested cleanup — matches the "don't build shape/render into score" and "don't simplify the atomic write" invariants.
- **Before touching >3 files you haven't read this session:** delegate discovery to an `Explore` subagent instead of grepping in the main thread. `score.py`, `shape.py`, and `render.py` are large enough that ad-hoc exploration fills the context window.

## Scope note

Score, the global ranker, shape, render, and platform profiles are all built (ranker + shaper both load the active profile via `clipper/_profiles.py`; one profile — `default.md` — ships today). The next stage on the roadmap is the **feedback loop** (the `logs/*.jsonl` rows are structured for retrospective joins; no retro job runs yet). Manual outcome entry (`clipper post`) is the first feedback-loop brick; outcome-metric capture and retro-rubric updates are the next bricks. See `VISION.md` for the full arc, build order, and non-goals.

The seam discipline from `VISION.md` still governs every change: each stage's output is typed JSON at the boundary, stages can be swapped or upgraded independently, and logic must not leak across seams (don't bake rendering into shaping, don't bake shaping into score, don't collapse stages to save a call, don't teach shape about the ranker beyond the one-line `score_index` provenance pass-through). If a change would erase or muddy a stage boundary, stop and reconsider.
