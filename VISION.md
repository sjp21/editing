# Vision

This document describes the long-term direction for the project. It is read on demand by Claude (and human contributors) when implementing sub-tasks, so that local decisions serve the larger arc. For operational rules and current conventions, see `CLAUDE.md`. For setup and usage, see `README.md`. For current library choices and other revisable implementation decisions, see `docs/technical-decisions.md`.

`VISION.md` changes slowly.

## How to use this document

When implementing a sub-task:

- Identify which stage it belongs to. If it spans stages, a seam is probably being violated.
- Check that outputs are typed JSON at stage boundaries.
- Check that anything new the stage produces is written to the log.
- If the task seems to require changing the vision, raise it explicitly rather than drifting. Surface the conflict rather than silently resolving it.

## Core goal

Turn long-form video into short-form social content that actually performs. The system should:

1. Ingest a 0–30 minute video.
2. Identify and rank the best moments.
3. Produce finished, platform-ready clips from those moments.
4. Learn from real-world performance and improve over time.

Pillars 1 and 2 are partially built (Pass 1). Pillars 3 and 4 are the work ahead.

## Architectural principle: extend by seams

The project has a clean typed JSON seam at `Pass1Result`. Every future stage follows the same discipline: a typed JSON boundary between stages, so any stage can be swapped, upgraded, or replaced without touching the others. This is the single most important structural rule. Do not collapse stages for convenience; do not let Pass 2 logic leak into Pass 1; do not bypass the seams to save a call.

If a change would erase or muddy a stage boundary, that is a signal to stop and reconsider.

## Stage map

### Pass 1 — Candidate identification (built)
Finds 15–20s candidate segments. Output: `Pass1Result`. Do not extend this stage with cropping, captioning, ranking, or shaping logic.

### Global ranker (next conceptual stage)
Pass 1's deduplication is per-window and cannot catch semantic near-duplicates (e.g., the same story told twice) or rank across the whole video. A single Claude call that sees all candidates at once returns a ranked top-K with per-clip justifications and a diversity constraint. Output: `RankedCandidates`. This is the stage that makes "best moments" mean best, not merely "passed threshold."

### Pass 2a — Shaping (reasoning)
Per-candidate Claude call with transcript ±30s of context and word-level timestamps. Returns an exact trim (to the word), a hook rewrite or overlay suggestion, and a keep/kill verdict. Output: `ShapedClip`. Pure reasoning — cacheable and testable in isolation.

### Pass 2b — Rendering (deterministic)
A deterministic pipeline that takes a `ShapedClip` and produces a finished, platform-ready file. It must:

- **Reframe to 9:16** using a face/subject tracker with a smoothed crop window. Multi-subject scenes may need a more capable segmenter.
- **Burn word-by-word highlighted captions** sourced from the existing word-level timestamps. This is the single highest-leverage retention feature.
- **Normalize loudness** to a platform-appropriate target. Non-negotiable; skipping this is why AI-made clips sound "off."
- **Emphasize the hook** — optionally, a brief zoom or text overlay on the first ~0.8s, sourced from the shaping step.

Output: a rendered file plus a `RenderedClip` manifest. Specific library choices (face tracker, segmenter, loudness target, renderer) live in `docs/technical-decisions.md` because they age faster than this document should.

### Platform profiles
What "good" means is platform-specific. Externalize this as versioned prompt fragments (`profiles/tiktok.md`, `profiles/linkedin.md`, `profiles/shorts.md`) describing hook expectations, arousal level, length sweet spot, and caption density. The ranker and shaper both load the active profile. Benefits: one place to iterate on taste, natural A/B testing across profiles, and a clean target for the feedback loop to rewrite later.

### Feedback loop
Treated as its own first-class stage, not an afterthought. See the next section.

## The feedback loop

Two honest constraints shape the design:

- Outcome signal is noisy. A clip can be excellent and flop, mediocre and pop.
- There will not be enough data to train a model from scratch for a long time.

Given those constraints, the approach is:

**Log everything from day one.** Every candidate, every feature, every score, every decision, every rendered output — persisted to a local analytical store with a stable schema. Cheap now, irreplaceable later. Any new stage must write to the log.

**Two-tier feedback capture.** Implicit: which clips the user actually posts (a quiet but strong preference signal). Explicit: platform performance metrics pulled back via the platforms' public APIs, keyed on post IDs. A `published_clips` table joins renders to outcomes.

**Retrospective rubric updates, not model training.** Periodically (e.g., every ~20 posted clips), a retro job has Claude read top and bottom performers side by side with their features and produces an updated rubric appendix. That appendix becomes part of the cached system prompt for future runs. This is RAG on the project's own performance history, using Claude's reasoning in place of a model there is not yet data to train. It is also legible — a human can read why the system changed its mind.

**A lightweight learned ranker, only later.** Once hundreds of outcomes exist, a gradient-boosted ranker over candidate features (predicting retention%) can serve as a cheap pre-filter before the Claude call. Do not build this early; the retro-rubric approach dominates until the data is there.

## Build order

Each step is independently valuable and preserves the seams:

1. Pass 2b rendering (crop + captions + loudness) — unlocks shipping anything at all.
2. Global ranker — makes selection meaningful.
3. Pass 2a shaping — sharpens trims and hooks.
4. Platform profiles — externalizes taste.
5. Logging infrastructure and manual outcome entry — starts the flywheel.
6. Platform API integrations — automates outcome capture.
7. Retrospective rubric updates — closes the learning loop.
8. Learned ranker — only when data supports it.

## Non-goals (for now)

- Full auto-publishing to platforms. The user stays in the loop on what goes out.
- Training custom models from scratch.
- Multi-speaker diarization-driven reframing beyond what a tracker with smoothing handles.
- Generating B-roll, music, or synthetic voiceover.

These may become in-scope later, but sub-tasks should not quietly pull them in.
