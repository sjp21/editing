# Project: Video-to-Clips Pipeline (MVP)

*Historical artifact — the original MVP spec.* The stage named here as
"Pass 1" is now called **score** (module `clipper/score.py`). Filename
preserved because older PRs and conversations reference this document; body
rewritten against the current names. See [plan.md](plan.md) for the rename.

## Objective
Build a CLI tool that takes a 0–30 minute video and outputs ranked candidate
15–20 second clips optimized for short-form platforms (TikTok, Reels, Shorts).

The system uses:
- Whisper (via faster-whisper) for transcript generation
- ffmpeg for frame extraction
- Claude for editorial judgment (selection + scoring)

**Scope: score stage only (Candidate Identification).**

---

# Architecture

## High-Level Design
Two-pass system (only the score stage implemented in MVP):

- Score — Candidate Identification
- Pass 2 — Clip Crafting (deferred; later built out as shape + render)

---

## Critical Change: Chunked Processing

- Segment transcript into overlapping windows (~20–30s, stride ~10–15s)
- Score windows in batches
- Aggregate + deduplicate results

---

# Data Pipeline

## Step 1 — Audio + Transcript
- Extract audio via ffmpeg
- Transcribe with faster-whisper

### Post-processing (required)
- Sentence segmentation
- Punctuation normalization
- Group into coherent segments

---

## Step 2 — Signal Enrichment (REQUIRED)

To compensate for lack of full video understanding, inject proxy signals:

### 1. Transcript-derived energy signals
- sentence length variance
- interruptions / cutoffs
- exclamations
- repeated words

### 2. Audio-derived features (lightweight)
- volume spikes
- speech rate changes
- silence gaps

These approximate:
- emotional peaks
- pacing
- punchlines

### 3. Frame targeting (event-driven)
In addition to fixed 20s frames:
- extract frames at:
  - audio spikes
  - transcript peaks
  - sentence boundaries

---

## Step 3 — Frame Extraction
- Base: 1 frame every ~20s (512px)
- Event-driven frames (see above)

---

## Step 4 — Window Construction
Each window includes:
- structured transcript
- associated frames
- derived signals (energy, audio cues)

---

# Score — Candidate Identification

## Two-Stage Reasoning

### Stage 1 — Filtering
Reject if:
- not self-contained
- no clear moment
- mid-thought
- no emotional/narrative beat

### Stage 2 — Scoring

Dimensions:
- Hook strength
- Emotional peak
- Quotability
- Self-contained narrative
- Visual interest
- Reaction-worthiness

Model must consider:
- transcript
- frames
- injected signals (energy + audio)

---

## Output Schema

{
  "start_seconds": float,
  "end_seconds": float,
  "duration_seconds": float,
  "scores": {...},
  "overall_score": float,
  "confidence": float,
  "rationale": "string",
  "suggested_hook": "string"
}

---

## Validation Rules
- 15 <= duration_seconds <= 20
- Valid JSON
- Retry on failure

---

# Prompt Design

Must include:
- Two-stage filtering + scoring
- Clear rubric definitions
- Explicit instruction to use:
  - transcript
  - frames
  - energy/audio signals

---

# Tech Stack
- Python 3.11+
- anthropic SDK
- ffmpeg
- faster-whisper
- pydantic
- typer

---

# CLI

python -m clipper process path/to/video.mp4 --out candidates.json

---

# Repo Structure

clipper/
  ingest.py
  signals.py      # NEW: energy + audio feature extraction
  windows.py
  score.py
  schemas.py
  cli.py
  prompts/v1/
    candidate_identification.md

---

# Cost Control
- Use short clips during dev
- Cache transcripts, frames, signals
- Avoid full transcript sends

---

# Determinism & Logging
- Low temperature
- Log model, prompt version, token usage

---

# Execution Plan
1. Scaffold repo
2. Build ingest.py
3. Build signals.py (energy + audio features)
4. Implement windowing
5. Write prompt
6. Build score + CLI
7. Run end-to-end

---

# Success Criteria
- 20–40 candidates
- Valid clip durations
- Strong alignment with emotional/audio peaks
- Minimal low-signal clips

---
# Future Direction

The system is designed with a future feedback-driven improvement loop:

video
→ transcript + signals
→ LLM candidate generation
→ structured logging
→ feedback collection (human or downstream metrics)
→ pattern memory layer
→ re-weighted scoring priors

This loop is intentionally excluded from Phase 1 implementation.
However, all intermediate outputs **MUST** be structured to support it.
