# clipper

MVP CLI that takes a 0–30 minute video and outputs ranked candidate 15–20 second clips suitable for short-form (TikTok, Reels, Shorts).

Pipeline (Pass 1 only — candidate identification):

```
video
  └─ ffmpeg ─ audio (.wav, 16kHz mono)
               └─ faster-whisper ─ transcript (sentence-grouped)
                    │                  └─ text signals (exclamations, length spikes, interruptions)
                    └─ RMS envelope ── audio signals (volume spikes, silence, speech-rate changes)
                                         │
  └─ ffmpeg ─ frames (base every 20s + event-driven at peaks)
                                         │
  windows (25s, stride 12s)  ←───────────┘
    └─ Claude (two-stage: filter → score) ── candidates.json
```

## Install

```sh
# System
brew install ffmpeg                 # or: apt-get install ffmpeg

# Python
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configure

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```sh
python -m clipper process path/to/video.mp4 --out candidates.json
```

Useful flags during dev:

```sh
# Limit to first 3 windows so you don't burn tokens while iterating
python -m clipper process clip.mp4 --max-windows 3 --verbose

# Force CPU whisper
python -m clipper process clip.mp4 --device cpu --whisper tiny

# Try a cheaper model for bulk runs
python -m clipper process clip.mp4 --model claude-sonnet-4-6
```

## Output

`candidates.json` is a `Pass1Result` — validated candidates plus provenance (model, prompt version, window index, token usage). Candidates are deduplicated across overlapping windows (IoU ≥ 0.6) and sorted by `overall_score`.

The schema is intentionally stable so human labels / downstream metrics can be joined later — that's the feedback loop called out in the plan.

## Layout

```
clipper/
  ingest.py     # ffmpeg audio + faster-whisper transcription (cached)
  signals.py    # text + audio feature extraction
  frames.py     # base + event-driven frame extraction
  windows.py    # 25s/12s sliding windows with attached signals+frames
  pass1.py      # Claude API call, JSON parsing, retry, dedup
  schemas.py    # pydantic models — contract for the whole pipeline
  cli.py        # typer CLI
  prompts/v1/
    candidate_identification.md   # versioned; PROMPT_VERSION pinned in pass1.py
```

## Costs

All caching is on by default:
- Audio files cached by content hash.
- Whisper transcripts cached by audio hash + model size.
- Frames cached by path + mtime + size.
- Claude system prompt cached via prompt caching (`cache_control: ephemeral`) — check `cache_read_input_tokens` in the result to confirm hits.
