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

Pillars 1 and 2 are partially built (score). Pillars 3 and 4 are the work ahead.

## Architectural principle: extend by seams

The project has a clean typed JSON seam at `ScoreResult`. Every future stage follows the same discipline: a typed JSON boundary between stages, so any stage can be swapped, upgraded, or replaced without touching the others. This is the single most important structural rule. Do not collapse stages for convenience; do not let shape/render logic leak into score; do not bypass the seams to save a call.

If a change would erase or muddy a stage boundary, that is a signal to stop and reconsider.

Taste lives in one more cross-stage artifact: `clipper/SOUL.md`, loaded by both the ranker and the shaper. The JSON seam governs *format*; SOUL governs *editorial priors*. Platform profiles override SOUL for platform-specific taste; the feedback loop overrides both. Keeping taste in one shared file — rather than letting it fragment into per-stage rubrics — is what makes retro-rubric updates (below) a single thing to rewrite instead of N things kept in sync.

## Stage map

### Score — Candidate identification (built)
Finds 15–20s candidate segments. Output: `ScoreResult`. Do not extend this stage with cropping, captioning, ranking, or shaping logic. The scorer's `hook_strength` dimension measures attention capture in the first ~2 seconds; this is load-bearing for the whole pipeline — a clip with a weak first beat cannot be salvaged downstream, so scorer thresholds on hook_strength dominate selection. The 15–20s duration constraint is a default inherited from the profile layer — see §Platform profiles.

### Global ranker (next conceptual stage)
The scorer's deduplication is per-window and cannot catch semantic near-duplicates (e.g., the same story told twice) or rank across the whole video. A single Claude call that sees all candidates at once returns a ranked top-K with per-clip justifications and a diversity constraint. Output: `RankedCandidates`. This is the stage that makes "best moments" mean best, not merely "passed threshold."

### Shape (reasoning)
Per-candidate Claude call with transcript ±30s of context and word-level timestamps. Returns an exact trim (to the word), a hook rewrite or overlay suggestion, and a keep/kill verdict. Output: `ShapedClip`. Pure reasoning — cacheable and testable in isolation. The shaper is expected to trim aggressively from the top — the first ~0.8 seconds decides whether the rest is watched, so over-trimming the opening is a recoverable error and under-trimming is not.

### Render (deterministic)
A deterministic pipeline that takes a `ShapedClip` and produces a finished, platform-ready file. It must:

- **Reframe to 9:16** using a face/subject tracker with a smoothed crop window. Multi-subject scenes may need a more capable segmenter.
- **Burn word-by-word highlighted captions** sourced from the existing word-level timestamps. This is the single highest-leverage retention feature.
- **Normalize loudness** to a platform-appropriate target. Non-negotiable; skipping this is why AI-made clips sound "off."
- **Reinforce the shaped hook** — optionally, a brief zoom or text overlay on the first ~0.8s, sourced from the shaping step. This is presentation, not selection; a clip that needs the overlay to have a hook should have been killed by shape.

Output: a rendered file plus a `RenderedClip` manifest. Specific library choices (face tracker, segmenter, loudness target, renderer) live in `docs/technical-decisions.md` because they age faster than this document should.

### Manual editing handoff (future)
Optional post-render export for users who want to take over the final edit themselves. When a shaped clip is composed of multiple sub-segments stitched from non-contiguous parts of the source (e.g., a 10–15s clip the shaper assembled from several takes), the system should be able to emit each sub-segment as its own file cut from the original video, plus a manifest of in/out timecodes keyed to the `ShapedClip`. The target use case is DaVinci Resolve and similar NLEs: the system does identification, selection, and assembly; the user does the polish. Seam-respecting shape: a separate stage reading `ShapedClip` + source video, writing loose source-cut sub-clips and a timecode manifest, leaving the main render output untouched.

### Platform profiles
What "good" means is platform-specific. Externalize this as versioned prompt fragments (`profiles/tiktok.md`, `profiles/linkedin.md`, `profiles/shorts.md`) describing hook expectations, arousal level, length sweet spot, and caption density. The ranker and shaper both load the active profile. Benefits: one place to iterate on taste, natural A/B testing across profiles, and a clean target for the feedback loop to rewrite later.

The current 15–20s hard cap on `Candidate.duration_seconds` is an unvalidated default, not a platform-universal truth — TikTok's FYP sweet spot runs longer, Shorts caps at 60s, LinkedIn tolerates 45–90s explanatory clips. Length targets are a per-profile parameter; the schema's hard cap will be relaxed to a per-profile range once the feedback loop (or the hand-labeled eval) shows a profile is length-starved. Until then, 15–20s is a working assumption that errs toward short-form-native pacing.

### Feedback loop
Treated as its own first-class stage, not an afterthought. See the next section.

## The feedback loop

Two honest constraints shape the design:

- Outcome signal is noisy. A clip can be excellent and flop, mediocre and pop.
- There will not be enough data to train a model from scratch for a long time.

Given those constraints, the approach is:

**Log everything from day one.** Every candidate, every feature, every score, every decision, every rendered output — persisted to a local analytical store with a stable schema. Cheap now, irreplaceable later. Any new stage must write to the log.

**Two-tier feedback capture.** Implicit: which clips the user actually posts (a quiet but strong preference signal). Explicit: platform performance metrics pulled back via the platforms' public APIs, keyed on post IDs. A `published_clips` table joins renders to outcomes.

**Precision backstop via hand labels.** Both tiers above require a clip to be posted. Clips the system would have shipped that the user vetoes in review never get posted, so they never generate outcome data — platform metrics alone measure recall of good picks but not precision against bad ones. A hand-labeled eval set (see `hand-labeled-regression-process.md`) serves as an interim precision signal until enough rejection-equivalent data accumulates automatically. Labels are per-video, tiered (`must_ship`, `nice`, `marginal`, `rejected`), and feed the retro job's input alongside platform metrics — the `rejected` tier being the only signal that ever sees false positives.

**Retrospective rubric updates, not model training.** Periodically (e.g., every ~20 posted clips), a retro job has Claude read top and bottom performers side by side with their features and produces an updated rubric appendix. That appendix becomes part of the cached system prompt for future runs. This is RAG on the project's own performance history, using Claude's reasoning in place of a model there is not yet data to train. It is also legible — a human can read why the system changed its mind.

**A lightweight learned ranker, only later.** Once hundreds of outcomes exist, a gradient-boosted ranker over candidate features (predicting retention%) can serve as a cheap pre-filter before the Claude call. Do not build this early; the retro-rubric approach dominates until the data is there.

## Build order

Each step is independently valuable and preserves the seams:

1. Render (crop + captions + loudness) — unlocks shipping anything at all.
2. Global ranker — makes selection meaningful.
3. Shape — sharpens trims and hooks.
4. Platform profiles — externalizes taste.
5. Logging infrastructure and manual outcome entry — starts the flywheel.
6. Hand-labeled eval scorecard — interim precision signal and regression gate for every rubric change until platform metrics exist.
7. Platform API integrations — automates outcome capture.
8. Retrospective rubric updates — closes the learning loop.
9. Learned ranker — only when data supports it.

## Non-goals (for now)

- Full auto-publishing to platforms. The user stays in the loop on what goes out.
- Training custom models from scratch.
- Multi-speaker diarization-driven reframing beyond what a tracker with smoothing handles.
- Generating B-roll, music, or synthetic voiceover.
- Music-synced editing (cutting long performance/pose footage to a chosen music track, for bands or fashion-style creators). Achievable with beat detection + visual motion energy, but a distinct product with different users and quality bars — a plausible v2 direction, deliberately out of scope until the speech pipeline has closed its feedback loop.

These may become in-scope later, but sub-tasks should not quietly pull them in.
