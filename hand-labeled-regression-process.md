# Hand-Labeled Regression Process

How to build a small human-labeled eval set for the clipper pipeline, so every future rubric change has a regression signal without needing real platform outcome data.

## Why this exists

The feedback loop (`VISION.md` build-order steps 6–7) needs real posted-clip outcome metrics to close. Until those exist, a hand-labeled eval set is the cheapest stand-in: **you are the outcome signal**. Label 3–5 videos once, then re-run after every rubric change to see whether the change helped or hurt on clips you already have opinions about.

The labels also outlive this interim phase — the `reason` field on each label is the highest-value signal for the eventual retro job, because it lets Claude compare your reasoning to its own and find systematic blind spots.

## Budget

Roughly 30 minutes per video once you have the rhythm. 5 videos ≈ 2.5 hours, one-time cost.

## Step-by-step (one video)

### 1. Prep

```sh
python -m clipper process path/to/video.mp4
```

This produces `outputs/<stem>/transcript.json` (used in step 4 for fast timestamp lookup) and `outputs/<stem>/candidates.json`. **Do not open `candidates.json` yet** — anchoring trap (see traps below).

### 2. Watch blind

Play the video at 1.5×. Keep a scratch pad (paper is fine). When something makes you think *"I'd actually post this,"* note the rough start time. Don't over-think, don't rewind every 30s. You're capturing instinctive reactions, not a careful analysis.

### 3. Filter your scratch list

Pick 3–7 real candidates. Assign a tier:

- **`must_ship`** (2–3 per video): "if only these shipped, I'd be happy"
- **`nice`** (1–3 per video): "would post, not my favorite"
- **`marginal`** (0–2 per video): "genuinely undecided — testing the system's judgment here"

### 4. Find exact timestamps

Open `outputs/<stem>/transcript.json` — it has word-level timestamps. Search the text for the line you remember hearing; pull the timestamp from there. Much faster than scrubbing a video player. A ~17-second clip doesn't need frame-accurate boundaries — whole-second resolution is fine because `shape` re-trims to word boundaries anyway.

### 5. Write reasons

One sentence per clip. Be specific.

- Bad: *"good clip"*
- Good: *"'undercharge by 3x' hook, $47K specific example lands at 12s"*

The reason is the highest-value field. The eventual retro job reads it alongside Claude's reasoning to find systematic blind spots. If you can't articulate a reason, mark `marginal` and write *"gut feel, no specific reason"* — honest. Forcing a fake rationale pollutes the retro.

### 6. Compare against Claude's picks

*Now* open `outputs/<stem>/candidates.json`.

- Clips Claude found that you missed → add them to your labels at the appropriate tier. You undershot, not Claude.
- Clips Claude picked that you *don't* want to ship → add them with tier `rejected` + one-sentence reason. **These are the most valuable labels** — they measure Claude's false positives, which no other signal catches.

### 7. Encode as JSON

Save to `outputs/<stem>/human_labels.json`:

```json
{
  "video_stem": "founder-interview-2026",
  "labeler": "scotton",
  "labeled_at": "2026-04-21T14:32:00Z",
  "labels": [
    {
      "start": 63.0,
      "end": 80.5,
      "tier": "must_ship",
      "reason": "'undercharge by 3x' hook + $47K example as in-window payoff at ~12s."
    },
    {
      "start": 412.0,
      "end": 430.0,
      "tier": "rejected",
      "reason": "Ranker loved it but the punchline lands at 19s and ends on a hedge — scroll bait."
    },
    {
      "start": 195.5,
      "end": 211.0,
      "tier": "marginal",
      "reason": "Genuinely unsure — nice line but requires respecting the speaker."
    }
  ]
}
```

## Schema

```python
class LabeledClip(BaseModel):
    start: float                  # seconds from video start
    end: float                    # seconds from video start
    tier: Literal["must_ship", "nice", "marginal", "rejected"]
    reason: str                   # one sentence, your reasoning

class HumanLabels(BaseModel):
    video_stem: str
    labeler: str
    labeled_at: str               # ISO8601 UTC
    labels: list[LabeledClip]
```

Tier meanings:

| Tier | Meaning | Signal to the retro |
|---|---|---|
| `must_ship` | Strong keep; if the ranker misses this, that's a regression | Strong positive |
| `nice` | Would post, not your favorite | Moderate positive |
| `marginal` | Genuinely undecided; testing the system's judgment | Neutral — disagreements are fine |
| `rejected` | System will probably pick this; you don't think it should ship | **Strong negative** — catches false positives |

## Design notes

Four methodology decisions. The last two directly affect how to write good labels; the first two affect how to read the eval results later.

### IoU threshold is 0.5

The comparison counts a candidate as matching a label when IoU (intersection-over-union of their time ranges) ≥ 0.5. Concrete math — label `[60, 80]` and candidate `[58, 74]`:

- intersection = `[60, 74]` = 14s
- union = `[58, 80]` = 22s
- IoU = 14/22 ≈ 0.64 → **match**

Why not tighter:

- **Windowing drift.** 25s windows on 12s stride means the scorer's output boundaries are jittered by several seconds relative to where the content actually lives.
- **Shape re-trimming.** `shape` snaps to word boundaries, shifting edges another 0.5–3s.
- **Your own labeling precision.** ±2s from scratch-pad estimates is normal.

Stacked, natural drift is 3–4s per edge on a 17s clip — IoU ~0.6 territory. Demanding 0.7+ would flag as misses matches a human would call identical. 0.3 is too loose — weak overlaps where only the setup or only the payoff aligns would count as matches.

**What this means for labeling**: don't stress about boundary precision. Whole-second resolution on a rough scratch-pad estimate is fine. The math accommodates drift on purpose.

### LLM noise is real

Score, rank, and shape each use adaptive-thinking Claude calls; outputs vary run-to-run. Typical variance on the same video:

- **Scorer**: 3–5 of ~20 candidates differ between runs; stable candidates' boundaries shift 1–3s.
- **Ranker**: top-3 stable; top-5+ noisier (marginal picks swap in and out).
- **Shape**: stable on obvious keep/kill; ~20% flip on borderline clips.

Mitigation ladder, cheapest first:

1. **Use 5 videos, not 1.** Cross-video patterns swamp single-video noise.
2. **Run eval twice after a rubric change.** Real regressions reproduce.
3. **Record and replay LLM outputs.** Deterministic re-runs. Only worth the infra for serious rubric experimentation.

**What this means for labeling**: your labels stay stable — you only write them once. But don't read a single-video, single-run delta as a regression. Wait for the pattern to reproduce across multiple videos or multiple runs.

### The `reason` field is the retro's richest input

Numerical labels capture *what* you picked. Reasons capture *the editorial instinct behind the pick* — which is exactly what the rubric is trying to encode. The retro job (when it exists) will read your reasons alongside the ranker's `comparative_reason` and shape's decision reasons, then pattern-match across disagreements.

Concrete retro mechanics (once implemented):

- **False positive pattern**: *"In 4 of 5 `rejected` labels, Claude cited 'strong hook' while you cited 'payoff outside window' or 'ends on a hedge'"* → rubric over-weights hook, under-weights payoff.
- **False negative pattern**: *"All 3 missed `must_ship` labels cite 'specific number' in your reason"* → rubric under-values specificity.

This mechanic only works if reasons are **honest and specific enough to compare**. Three principles:

- `"Gut feel, no specific rationale"` is valid data. Fabricating a rule to fill the field is not — the retro learns from the fabrication.
- Don't write elaborate reasons for picks you love and terse ones for picks you don't. The asymmetry poisons the data.
- `SOUL.md`'s "Contradictions You Hold" section is literal: real editors have inconsistent instincts. Forcing coherence where there is none fabricates signal.

**Practical rule**: one sentence per label, naming the specific thing (a quote, a number, a story beat) that drove the decision. If you can't name a specific thing, write `"gut feel"` and move on — still useful data.

### Tier asymmetry — disagreements aren't equal

The four tiers encode asymmetric importance for the scorecard:

| Disagreement | Signal | Meaning |
|---|---|---|
| `must_ship` missed by ranker | **Strong regression** | Top-K broken — ranker didn't surface your best clip |
| `rejected` kept by shape | **Strong regression** | False positive — system shipping clips you wouldn't ship |
| `nice` missed by ranker | Moderate | Ranker under-weighting this kind of clip |
| `marginal` disagreement | No signal | You were undecided; expecting Claude to match a coin flip is unfair |

**Practical guidance for labeling**:

- `must_ship` and `rejected` are **accountability tiers**. Use them only when you have a confident opinion — disagreements here trigger real regression alarms.
- `nice` and `marginal` are **context tiers**. Use for "would post but not my favorite" and "genuinely undecided." Disagreements here don't flag regressions.
- Don't collapse `marginal` into `nice` or `rejected` just to avoid fence-sitting. Forcing yourself off the fence in a given mood introduces fake signal; `marginal` preserves honest uncertainty as real data.
- `rejected` is the highest-value tier to label for — it's the only way to measure Claude's **false positives**, clips the system would ship that you wouldn't. These never show up in "did you post this" data, so without `rejected` labels they're invisible to the retro. Don't skip them because they feel negative.

## Traps to avoid

**Anchoring to Claude's picks.** Do not open `candidates.json` before step 6. The moment you see Claude's picks, your labels drift toward "agrees with Claude," contaminating the blind signal.

**Forcing reasons you don't have.** If a clip is gut-feel, say so. `"gut feel, no specific reason"` is better data than a fabricated rationale. (See *The `reason` field* in Design notes for why.)

**Labeling at 1× speed.** Ten-minute videos become half-hour labeling sessions. Stay at 1.5–2× — you're capturing reactions, not transcribing.

## What to run against your labels

Once labels exist, a simple comparison (either manual or via a future `clipper eval` subcommand) computes three numbers per video:

- **Scorer recall**: fraction of non-`rejected` labels covered by a candidate at IoU ≥ 0.5.
- **Ranker inclusion**: fraction of `must_ship` labels that landed in top-K.
- **Shape alignment**: for labels the ranker kept, did `shape` keep or kill them? For `rejected` labels that reached `shape`, did it kill them?

See *Design notes* above for IoU choice and how to interpret deltas under LLM noise.

## Relationship to the feedback loop

These labels complement (don't replace) real outcome metrics. When platform API metrics eventually arrive (`VISION.md` build-order step 6), the retro job will have two signals per clip:

1. **Implicit + explicit outcome data** (posted or not; views/retention if posted).
2. **Human-labeled intent** (would you have posted, and why).

The second signal is particularly useful for clips the system chose that you *wouldn't* have posted — they never make it to the outcome data at all, so without human labels, the system's false positives are invisible to the retro.

## When to formalize this

Zero-infra (this doc + hand-written JSON) is fine for the first 5 videos. If you find yourself labeling more than that, or if re-running the comparison manually becomes a chore, build a `clipper eval` subcommand that reads `human_labels.json` + the pipeline outputs and writes a scorecard to `outputs/<stem>/logs/eval.jsonl`. At that point it becomes a proper regression test that joins cleanly to the existing log discipline.
