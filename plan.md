# Feedback loop, brick 1: `clipper post` + `posted.jsonl`

## Context

The pipeline's first five stages are built and have the full provenance chain: `score → rank → shape → render`, with platform profiles wired into both ranker and shape. (Note: profile work is complete in the working tree but uncommitted — `clipper/_profiles.py`, `clipper/prompts/v1/profiles/shorts.md`, schema `ShapeResult.profile` field, CLI threading, four new tests. Committing that work is a prerequisite housekeeping step; it is not part of this plan.)

Per `VISION.md` build order step 5 ("Logging infrastructure and manual outcome entry — starts the flywheel") and the honest constraints from `VISION.md:62–64` (noisy outcome signal, not enough data to train), the next stage is the feedback loop. The order is forced: **capture implicit signal first** — *which clips the user actually posts* — because that is a quiet but strong preference signal, and it is the one upstream of everything the feedback loop will eventually do (retro-rubric updates, published_clips joins, learned ranker).

This plan ships the smallest brick that starts the flywheel: a `clipper post <clip.mp4> --platform <name> --post-id <id>` subcommand that resolves the MP4 back to its identity via the existing `rendered.json` manifest and appends a row to `outputs/<stem>/logs/posted.jsonl`. No platform APIs. No retro job. No schema migration. The join key (`shaped_candidate_index`) is the same one the three existing logs already share, so when metric-pull and retro jobs land later, they plug into this log without rework.

## Scope

1. Add a `PostedClip` pydantic model and an append helper.
2. Add a `post` subcommand to `clipper/cli.py`.
3. Resolve MP4 path → clip identity via the render manifest, with a clear error when the MP4 isn't in any manifest under the resolved stem root.
4. Tests: identity resolution, append-only (cross-posting), error paths.

**Non-goals (do not quietly pull in):** no platform API integration, no outcome metrics capture (views/retention/etc.), no retro-rubric job, no `published_clips` DB, no UI, no auto-discovery walking the whole `outputs/` tree, no enum/whitelist of platform names (keep it a non-empty string — VISION.md §Platform profiles only names three today but more will exist), no fingerprint-keyed skip (cross-posting the same clip to two platforms appends two rows, by design).

## Design

### Path → identity resolution

The render manifest already holds everything needed. Given `clip_path = outputs/<stem>/clips/clip-0.mp4`:

1. Look for `clip_path.parent / "rendered.json"` (this matches `render.py:150` — the default manifest path is `out_dir / "rendered.json"`, and `cli.py:443` sets `out_dir = shaped_parent / "clips"`).
2. Load it as `RenderResult.model_validate_json(...)` — the schema already exists at `clipper/schemas.py:344`.
3. Find the `RenderedClip` whose `output_path.resolve() == clip_path.resolve()`. Raise a clear `typer.BadParameter` if none matches, listing the manifest that was consulted and the output_paths it contains.
4. `<stem>_root = rendered.json.parent.parent`; posted log goes at `<stem>_root / "logs" / "posted.jsonl"`.
5. Allow `--manifest PATH` to override the lookup for non-default layouts (mirrors `render`'s `--manifest`). Allow `--output-dir PATH` to override the posted-log destination. Neither is needed for the golden path.

From the matched `RenderedClip`, carry forward: `shaped_candidate_index`, `source_video_sha256`, `output_sha256`, `render_version`, `source_shaped_prompt_version`, `source_score_prompt_version`. These are the provenance anchors a retro job will eventually join on.

### Row shape

`PostedClip` (new model in `clipper/schemas.py`, placed near `RenderResult` at `schemas.py:344`):

| Field | Type | Source |
|---|---|---|
| `ts` | `str` (ISO8601 UTC) | generated at append time |
| `stage` | `Literal["post"]` | constant |
| `shaped_candidate_index` | `int` | `RenderedClip.shaped_candidate_index` — **the join key** |
| `video_stem` | `str` | resolved from `<stem>_root.name` |
| `source_video_sha256` | `str` | `RenderedClip.source_video_sha256` |
| `output_path` | `str` | absolute path of the MP4 |
| `output_sha256` | `str` | `RenderedClip.output_sha256` |
| `render_version` | `str` | `RenderedClip.render_version` |
| `source_shaped_prompt_version` | `str` | `RenderedClip.source_shaped_prompt_version` |
| `source_score_prompt_version` | `str` | `RenderedClip.source_score_prompt_version` |
| `platform` | `str` | `--platform` |
| `post_id` | `str` | `--post-id` |
| `post_url` | `str \| None` | `--post-url` (optional) |
| `title` | `str \| None` | `--title` (optional) |
| `posted_at` | `str \| None` (ISO8601) | `--posted-at` (optional; defaults to `ts`) |
| `notes` | `str \| None` | `--notes` (optional) |

`platform` and `post_id` validators: non-empty after `.strip()`. No enum — creators add platforms faster than we add entries.

### Append

Mirror `shape.py:_append_log` / `render.py:_append_log`: `posted_log.parent.mkdir(parents=True, exist_ok=True)`, then `open(path, "a", encoding="utf-8")` and write one `row.model_dump_json()` + `"\n"`. Append-only; duplicates are intentional (cross-posting). No atomic-write helper needed — JSONL is line-oriented and every existing log does plain append.

## Edits

### 1. `clipper/schemas.py` (~line 356, after `RenderResult`)

Add `PostedClip` model with the fields above. Match the schema discipline used elsewhere in the file. Add non-empty validators on `platform` and `post_id`.

### 2. `clipper/cli.py` (new subcommand, after the existing `render` block at `cli.py:406`)

```python
@app.command()
def post(
    clip_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to rendered MP4"),
    platform: str = typer.Option(..., "--platform", help="Platform name, e.g. shorts, tiktok, linkedin"),
    post_id: str = typer.Option(..., "--post-id", help="Platform-assigned post identifier"),
    post_url: str | None = typer.Option(None, "--post-url"),
    title: str | None = typer.Option(None, "--title"),
    posted_at: str | None = typer.Option(None, "--posted-at", help="ISO8601 UTC; defaults to now"),
    notes: str | None = typer.Option(None, "--notes"),
    manifest: Path | None = typer.Option(None, "--manifest",
        help="rendered.json path (defaults to <clip-parent>/rendered.json)"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-O",
        help="Per-video output root (defaults to <rendered.json parent>/..)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record that a rendered clip was posted to a platform."""
```

Body: resolve manifest → match `output_path` → build `PostedClip` → append to `<stem>_root / "logs" / "posted.jsonl"`. Print the path written to so the user sees the row landed.

Error messages must be concrete:
- Manifest not found → `"no rendered.json at <path>; pass --manifest if your layout is non-default"`.
- Clip not in manifest → `"clip <path> not in <manifest>; manifest contains: <listing of output_paths>"`. (Likely the user is pointing at an MP4 from a different render run or stem.)

### 3. `clipper/_posted.py` (new, small)

Extract the two pure functions:
```python
def resolve_identity(clip_path: Path, manifest_path: Path) -> tuple[RenderedClip, Path]: ...
def append_posted(log_path: Path, row: PostedClip) -> None: ...
```
CLI calls both. Tests import both directly to bypass typer boilerplate. Keeps `cli.py` thin and matches the existing `_profiles.py` / `_atomic_write_json` pattern of leaving shared pure logic outside the command body.

### 4. `tests/test_post.py` (new)

Mirror the unit shape of `tests/test_ranker.py` / `tests/test_shape.py`: construct a tiny `RenderResult` in-memory, write it to `tmp_path/clips/rendered.json`, touch `tmp_path/clips/clip-0.mp4`, then drive `resolve_identity` + `append_posted` directly.

- `test_post_resolves_identity_and_appends` — happy path; assert the JSONL row parses back as `PostedClip`, keys match the matched `RenderedClip`, and `shaped_candidate_index` round-trips.
- `test_post_missing_manifest_raises` — no `rendered.json` under `<clip-parent>`; exception message names the looked-up path.
- `test_post_clip_not_in_manifest_raises` — manifest exists but `output_path` doesn't match; message lists the manifest's output_paths.
- `test_post_appends_two_rows_for_cross_post` — invoke twice with two different `(platform, post_id)` pairs; assert two lines in the file and both parse.
- `test_post_empty_platform_or_post_id_raises` — pydantic validator on `PostedClip` rejects empty/whitespace.

### 5. `tests/test_cli_output_layout.py` (extend)

One assertion: `clipper post <clip> --platform shorts --post-id abc` writes `<stem>/logs/posted.jsonl` (not somewhere else), and the row's `shaped_candidate_index` matches the `RenderedClip` in the stubbed manifest. Uses the same typer CliRunner pattern as the existing tests in that file.

### 6. `CLAUDE.md` — Output layout section

Add one line to the `outputs/<video-stem>/` tree diagram under `logs/`:
```
│   ├── posted.jsonl         # post: one row per (clip, platform) posting (append-only)
```
And one line to the Scope note: "Manual outcome entry (`clipper post`) is the first feedback-loop brick; outcome-metric capture and retro-rubric updates are the next bricks."

### 7. `VISION.md` — no edit

Lines 67–75 describe the two-tier feedback capture this brick implements. No new vision content.

## Critical files

| File | Role |
|---|---|
| `clipper/schemas.py` (insert ~line 356) | **New model.** `PostedClip` |
| `clipper/cli.py` (insert after line 463) | **New subcommand.** `post` |
| `clipper/_posted.py` | **New.** `resolve_identity` + `append_posted` pure helpers |
| `clipper/render.py:150` | Reference — confirms default manifest path for resolution |
| `clipper/schemas.py:319–354` | Reference — `RenderedClip` / `RenderResult` fields the resolver reads |
| `clipper/shape.py:409` | Reference — append-log pattern to mirror |
| `tests/test_post.py` | **New.** Five unit tests |
| `tests/test_cli_output_layout.py` | Extend with one CLI end-to-end assertion |
| `CLAUDE.md` (output layout + scope note) | Document the new log and brick status |

## Skills to invoke before editing

- `andrej-karpathy-skills:karpathy-guidelines` — before writing `_posted.py` and the `PostedClip` model, because the surface is small and there's real pull toward speculative additions (enum of platforms, outcome-metrics fields, fingerprint-skip on duplicates). The non-goals list above names the ones to resist.
- No `claude-api` skill needed — this brick touches zero Claude calls.

## Verification

```sh
# Targeted tests
pytest -x tests/test_post.py tests/test_cli_output_layout.py

# Full suite — guards that PostedClip schema addition doesn't leak
pytest -x

# End-to-end smoke — assumes profile work is committed and a prior run exists
python -m clipper process test.webm --rank --shape --render --profile shorts --max-windows 2
python -m clipper post outputs/test/clips/clip-0.mp4 --platform shorts --post-id ABC123 --post-url https://youtube.com/shorts/ABC123
#   → outputs/test/logs/posted.jsonl has one line
#   → jq on that line shows shaped_candidate_index matching rendered.json

# Cross-post (same clip, different platform) appends, doesn't dedup
python -m clipper post outputs/test/clips/clip-0.mp4 --platform tiktok --post-id tt-999
#   → outputs/test/logs/posted.jsonl now has two lines

# Error paths
python -m clipper post /tmp/not-a-real-clip.mp4 --platform shorts --post-id x
#   → clear error before any write; no posted.jsonl created
python -m clipper post outputs/test/clips/clip-0.mp4 --platform "" --post-id x
#   → pydantic validation error naming `platform`
```

## After this ships

The second feedback-loop brick (also out of scope here) follows naturally:

1. **Outcome metrics capture.** Add `clipper metrics <stem>` that reads `posted.jsonl`, pulls metrics from platform APIs keyed on `post_id`, and appends to `outputs/<stem>/logs/metrics.jsonl`. Same join key.
2. **Retro-rubric job.** Once ~20 posted clips exist across runs, `clipper retro` reads the four logs (`ranker`, `shape`, `render`, `posted` — plus `metrics` once that brick lands), groups by `shaped_candidate_index`, and has Claude produce a rubric appendix. Writes to `clipper/prompts/vN+1/rubric_appendix.md`. Bumps `PROMPT_VERSION`.
3. **Learned ranker.** Not before hundreds of outcomes exist. `VISION.md:75` is explicit.

Each brick is independently valuable and preserves the seam discipline. This plan covers brick 1 only.
