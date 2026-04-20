# Pass 2a — Shaping: next implementation step

## Context

[VISION.md](VISION.md) describes a four-stage pipeline (Pass 1 → global ranker → Pass 2a shaping → Pass 2b rendering) plus platform profiles and a feedback loop. Audit of the current codebase against VISION:

| VISION stage | Status | Evidence |
|---|---|---|
| Pass 1 — Candidate identification | **Built** | [clipper/pass1.py](clipper/pass1.py), [schemas.py](clipper/schemas.py) (`Candidate`, `Pass1Result`), [prompts/v1/candidate_identification.md](clipper/prompts/v1/candidate_identification.md) |
| Global ranker | Not implemented | No `RankedCandidates` schema; no ranker module |
| Pass 2a — Shaping | Not implemented | No `ShapedClip` schema; no module |
| Pass 2b — Rendering | Not implemented | No reframer, captioner, loudness step, or rendering deps in [pyproject.toml](pyproject.toml) |
| Platform profiles | Not implemented | No `profiles/` directory |
| Feedback loop | Not implemented | Provenance fields exist on `Candidate` but nothing reads them |

VISION's stated build order starts with Pass 2b (rendering), but that stage bundles 4 sub-components (9:16 reframe, word-by-word captions, loudness normalization, hook emphasis) and pulls in heavy dependencies. **Pass 2a is a realistically-scoped next step**: pure reasoning, reuses the Pass 1 Claude-call pattern, consumes word-level data Whisper already produces, no new heavy deps. Its `verdict: keep|kill` also gives us a second-pass filter without building the ranker yet.

## Goal

Implement Pass 2a exactly as VISION describes: a per-candidate Claude call that reads transcript ±30s of context plus word-level timestamps, and returns an exact word-aligned trim, a hook treatment, and a keep/kill verdict. Output: a typed `Pass2aResult` JSON at the seam between Pass 1 and (future) Pass 2b.

## Scope

**In scope:** schemas, prompt, module, CLI wiring, tests for Pass 2a only. Minimal per-shape JSONL log line (see §6).
**Out of scope:** ranker, Pass 2b rendering, full logging/feedback infrastructure beyond the per-shape log line, platform profiles *content* (but the prompt reserves an injection point — see §3). These remain future steps and must not be silently pulled in.

## Design

### 1. Schemas — add to [clipper/schemas.py](clipper/schemas.py)

```python
class ShapedTrim(BaseModel):
    start_seconds: float
    end_seconds: float
    start_word: str                 # sanity check: model must quote a real word
    end_word: str
    duration_seconds: float         # validator: 12.0..22.0, matches end-start ±0.1

class HookTreatment(BaseModel):
    kind: Literal["rewrite", "overlay", "none"]
    text: str | None = None         # required when kind != "none"
    rationale: str

class ShapedClip(BaseModel):
    candidate_index: int            # position in source Pass1Result.candidates
    source_start: float             # audit trail of original Candidate
    source_end: float
    verdict: Literal["keep", "kill"]
    kill_reason: str | None = None
    trim: ShapedTrim | None = None
    hook: HookTreatment | None = None
    reasoning: str
    model: str
    prompt_version: str

    @model_validator(mode="after")
    def _check_verdict(self): ...    # keep→trim+hook required; kill→kill_reason required

class Pass2aResult(BaseModel):
    shaped: list[ShapedClip]
    model: str
    prompt_version: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    candidates_shaped: int = 0
    candidates_killed: int = 0
    candidates_failed: int = 0
    source_pass1_prompt_version: str   # joinability across prompt revisions
```

**Trim invariant:** shaped trim must satisfy `start >= candidate.start - 2.0` and `end <= candidate.end + 2.0` (snap-to-word slack only — expansion exists so word-boundary snapping can cross a word that straddles the candidate edge, *not* to let shaping re-select a different moment). `start_word`/`end_word` must match an actual transcript word within 0.05s. The ±30s of transcript is **context for reasoning, not a trim range** — allowing trim anywhere in ±30s would let shaping re-select and violate the Pass 1 seam.

**Duration range `[12, 22]` (vs `Candidate`'s `[15, 20]`):** this is a deliberate new range for a new schema, not a widening of the Pass 1 invariant. Justification: word-snapping at both edges plus aggressive top-of-clip tightening (per [SOUL.md](SOUL.md) — *"if the clip can survive a brutal 2-second trim from the top, take it"*) can legitimately move `Candidate`'s 15–20s into the 12–22s range. The `Candidate` invariant in [schemas.py:138-140](clipper/schemas.py#L138-L140) is untouched; [CLAUDE.md's](CLAUDE.md) "widen together" rule applies to `Candidate`, not to downstream schemas.

### 2. New module — [clipper/pass2a.py](clipper/pass2a.py)

Mirrors the [pass1.py](clipper/pass1.py) shape (PROMPT_VERSION = "v1", Pass2aConfig dataclass, `_load_rubric` via `importlib.resources`, `cache_control: ephemeral` on system, `thinking={"type": "adaptive"}`, `output_config={"effort": cfg.effort}`, retry loop with JSONDecodeError/APIError handling). Default model = `claude-opus-4-7` (matches `pass1.DEFAULT_MODEL`). Key signatures:

```python
def run_pass2a(
    pass1: Pass1Result,
    transcript: Transcript,
    cfg: Pass2aConfig | None = None,
    client: anthropic.Anthropic | None = None,
) -> Pass2aResult

def _extract_context(transcript, start, end, pad=30.0
    ) -> tuple[list[TranscriptSegment], list[TranscriptWord]]
def _format_words_table(words) -> str       # "t=12.34 \"word\"" lines
def _shape_candidate(client, cand, idx, ctx, rubric, cfg) -> tuple[ShapedClip, dict]
def _parse_and_validate(text, cand, idx, words) -> ShapedClip
```

`_parse_and_validate` enforces: pydantic validity, `start_word`/`end_word` match a transcript word within 0.05s of the claimed timestamp, trim inside `[cand.start-2.0, cand.end+2.0]`, provenance (`model`, `prompt_version`) set.

### 3. New prompt — [clipper/prompts/v1/shaping.md](clipper/prompts/v1/shaping.md)

**Derived from [SOUL.md](SOUL.md).** SOUL explicitly says *"You are loaded into the ranker and the shaper"* (SOUL.md#L3). The v1 shaping rubric must carry SOUL's craft values forward — not as a pointer, but as embedded rubric content. Specifically import: the "How You Shape (Pass 2a Stage)" section verbatim (trim to the word, first 0.8s decides retention, brutal 2-second top trim, honest kill verdict), the anti-patterns list (SOUL.md#L34-L43), and the worked examples (SOUL.md#L78-L95) as in-prompt exemplars. The rubric is the single source of truth at runtime; SOUL is where it originates and where changes get reviewed.

Prompt structure:

- **System (cached):** SOUL-derived rubric for boundary selection, hook craft, kill criteria, and the strict JSON schema specification. Static → cache hit after the first candidate. Include an explicit `[PLATFORM_PROFILE]` placeholder block (empty in v1) immediately after the rubric but before the schema spec — this reserves the profile-injection seam VISION.md#L55 requires without adding any content yet. When profiles land, the loader injects `profiles/<active>.md` at this marker; v1 → v2 is a content swap, not a structural rewrite.
- **User (per candidate):** candidate header (index, original bounds, `overall_score`, `suggested_hook`); `[CONTEXT_BEFORE]`/`[CANDIDATE]`/`[CONTEXT_AFTER]` transcript markers; a word-level table for candidate ± 5s, one word per line `t=12.34 "word"`; instruction to emit only the JSON.
- **No frames.** Pass 2a is reasoning over language/timing only — frames are Pass 2b's concern.

Word data is already available: `TranscriptSegment.words: list[TranscriptWord]` in [schemas.py:33](clipper/schemas.py#L33), preserved by `_postprocess` in [ingest.py:171](clipper/ingest.py#L171). No ingest changes.

### 4. CLI — edit [clipper/cli.py](clipper/cli.py)

Add one subcommand and one optional flag to `process`:

- `clipper shape <video> --candidates candidates.json --out shaped.json` — loads `Pass1Result` from disk, re-runs `extract_audio` + `transcribe` (both hit cache), calls `run_pass2a`, writes `Pass2aResult`. Lets the shaping prompt be iterated on without re-running Pass 1 — preserves the "cacheable and testable in isolation" property from [VISION.md:42](VISION.md#L42).
- `clipper process ... [--shape/--no-shape] [--shape-out shaped.json]` — default **off** initially; once stable, flip to on. When enabled, runs Pass 2a inline on the Pass 1 result and writes `shaped.json` next to `candidates.json`.

### 5. Package data — [pyproject.toml](pyproject.toml)

Line 28 already globs `prompts/v1/*.md`, so adding `shaping.md` needs no edit.

### 6. Minimal per-shape log — new `.cache/logs/pass2a.jsonl`

VISION.md#L69: *"Any new stage must write to the log."* No log infra exists yet for Pass 1, so this is the minimum viable compliance. One JSONL line appended per shaped candidate, written inside `run_pass2a` before returning:

```json
{"ts": "2026-04-19T12:34:56Z", "stage": "pass2a", "prompt_version": "v1",
 "model": "claude-opus-4-7", "source_pass1_prompt_version": "v1",
 "candidate_index": 0, "source_start": 120.3, "source_end": 137.5,
 "verdict": "keep", "trim_start": 121.1, "trim_end": 136.8,
 "hook_kind": "rewrite", "kill_reason": null,
 "input_tokens": 3421, "output_tokens": 214, "cache_read_tokens": 3200}
```

Atomic append (`open(..., "a")` is atomic for short lines on POSIX); file path configurable via `Pass2aConfig.log_path`, default `.cache/logs/pass2a.jsonl`. No rotation, no indexing — the real log infra gets built when the feedback loop does; this just ensures the data exists when that happens.

## Critical files to modify

- [clipper/schemas.py](clipper/schemas.py) — add `ShapedTrim`, `HookTreatment`, `ShapedClip`, `Pass2aResult`
- [clipper/pass2a.py](clipper/pass2a.py) — new module (mirror [pass1.py](clipper/pass1.py))
- [clipper/prompts/v1/shaping.md](clipper/prompts/v1/shaping.md) — new rubric (SOUL-derived, with `[PLATFORM_PROFILE]` placeholder)
- [clipper/cli.py](clipper/cli.py) — new `shape` subcommand + optional `--shape` flag on `process`
- [tests/test_shaped_schemas.py](tests/test_shaped_schemas.py) — new
- [tests/test_pass2a.py](tests/test_pass2a.py) — new (parsing + clamping + mocked end-to-end + JSONL log line written)

Reuse (do not reimplement):
- `_extract_text`, `_usage_dict`, fenced-JSON stripping pattern from [pass1.py:269-290](clipper/pass1.py#L269-L290) — factor into a small `_llm.py` helper shared by pass1 and pass2a, or copy if factoring feels like over-reach for one duplication.
- `extract_audio` + `transcribe` from [ingest.py](clipper/ingest.py) — cached by content hash, safe to call repeatedly.
- `Pass1Config`/`_load_rubric` pattern from [pass1.py:45-64](clipper/pass1.py#L45-L64).

## Verification

1. **Unit — schemas:** `pytest tests/test_shaped_schemas.py` — keep-requires-trim, kill-requires-reason, duration bounds, trim-outside-candidate-slack rejection, word-timestamp-mismatch rejection.
2. **Unit — pass2a parsing:** mocked `client.messages.create` returning canned JSON → valid `ShapedClip`; fenced JSON; malformed JSON triggers retry; kill-verdict path.
3. **Unit — end-to-end:** fake Anthropic client returning one response per candidate; assert `Pass2aResult` aggregates token usage, preserves `candidate_index`, writes `source_pass1_prompt_version`, and emits one JSONL line per shaped candidate to a tmp log path.
4. **Regression:** existing `pytest` suite (12 tests across [tests/](tests/)) remains green — Pass 1 must be untouched.
5. **Manual E2E** (requires `ANTHROPIC_API_KEY`): run `clipper process clip.mp4 --max-windows 3 --out cands.json`, then `clipper shape clip.mp4 --candidates cands.json --out shaped.json`. Inspect `shaped.json` by hand — trim boundaries should land on real word edges; at least one `verdict: kill` is plausible on noisy input. Inspect `.cache/logs/pass2a.jsonl` — one line per candidate.
6. **Cache sanity:** second run of `clipper shape` on the same video should show `cache_read_tokens > 0` in the output JSON (system-prompt cache hit after first candidate).

## Seam note for future stages

When the global ranker lands, its output `RankedCandidates` will replace `Pass1Result` as the input to `run_pass2a`. That change is contained to the `pass2a.run_pass2a` signature and the CLI glue — `ShapedClip` itself doesn't move. Keeping `candidate_index` + `source_start`/`source_end` in `ShapedClip` (rather than a richer ref) keeps that swap cheap.

When platform profiles land, the `[PLATFORM_PROFILE]` placeholder in [prompts/v1/shaping.md](clipper/prompts/v1/shaping.md) becomes the injection point — the loader reads `profiles/<active>.md` and substitutes. No prompt-structure change; the cached system prompt gets a new suffix, and SOUL.md#L96-L102's authority rule (feedback loop > profile > craft) is enforced by ordering rubric → profile → schema in the system message.
