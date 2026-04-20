# Pass 2b — Rendering: next implementation step

## Context

[VISION.md](VISION.md) describes a four-stage pipeline (Pass 1 → global ranker → Pass 2a shaping → Pass 2b rendering) plus platform profiles and a feedback loop. Updated audit:

| VISION stage | Status | Evidence |
|---|---|---|
| Pass 1 — Candidate identification | **Built** | [clipper/pass1.py](clipper/pass1.py), [schemas.py](clipper/schemas.py) (`Candidate`, `Pass1Result`), [prompts/v2/candidate_identification.md](clipper/prompts/v2/candidate_identification.md) |
| Global ranker | Not implemented | No `RankedCandidates` schema; no ranker module |
| Pass 2a — Shaping | **Built** | [clipper/pass2a.py](clipper/pass2a.py), [schemas.py](clipper/schemas.py) (`ShapedClip`, `Pass2aResult`), [prompts/v1/shaping.md](clipper/prompts/v1/shaping.md) |
| Pass 2b — Rendering | Not implemented | No reframer, captioner, loudness step, or rendering deps in [pyproject.toml](pyproject.toml) |
| Platform profiles | Not implemented | No `profiles/` directory |
| Feedback loop | Not implemented | `.cache/logs/pass2a.jsonl` exists but nothing reads it; `ShapedClip` carries provenance already |

Per [VISION.md:81](VISION.md#L81) the stated build order starts with Pass 2b — *"unlocks shipping anything at all."* That framing is load-bearing: every rubric tuned before 2b exists (Pass 1 and Pass 2a) has been tuned against scores and text, not against watchable output. Pass 2b is the stage that closes the evaluation loop for everything upstream. The ranker ([§VISION.md Stage map](VISION.md#L38)) is still more conceptually important, but without 2b we cannot tell whether ranker taste is correct when it lands.

## Goal

Implement a **deterministic** render pipeline that consumes a `ShapedClip`, a **pinned transcript artifact**, and the source video, and produces a 9:16 MP4 with burned word-by-word captions, loudness-normalized audio, and an optional hook overlay. Output: a typed `Pass2bResult` JSON at the seam plus one rendered MP4 per kept `ShapedClip`. No Claude call. No STT call. Seam-clean: Pass 2b re-reads nothing upstream except the shaped JSON, the transcript JSON, and the source video file — all three are hashed and recorded per render.

## Scope

**In scope:** schemas, render module, ffmpeg-driven pipeline, MediaPipe-based face tracker, word-by-word ASS captions, EBU R128 loudness normalization, text hook overlay, CLI wiring, per-render JSONL log, a bundled caption font, a persisted transcript artifact produced by the upstream `shape` CLI, [docs/technical-decisions.md](docs/technical-decisions.md) entry for library/parameter choices, tests including a tiny fixture-video integration test.
**Out of scope:** multi-subject segmentation (explicit non-goal per [VISION.md:94](VISION.md#L94)), zoom-based hook emphasis (text overlay only in v1), B-roll / music / synthetic voiceover, auto-publishing to platforms, per-platform profile *content* (the schema reserves the injection point — see §9). These stay deferred and must not silently leak in.

## Design

### 1. Schemas — add to [clipper/schemas.py](clipper/schemas.py)

```python
class RenderSettings(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_codec: str = "libx264"
    video_crf: int = 20
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    loudness_target_lufs: float = -14.0         # TikTok / Instagram target
    loudness_true_peak_dbfs: float = -1.0
    face_tracker: str = "mediapipe"             # rationale in docs/technical-decisions.md
    crop_smoothing_halflife_s: float = 0.5

class CaptionStyle(BaseModel):
    font_family: str = "Inter"                  # bundled, see §4 / pyproject package-data
    font_file: Path | None = None               # override for tests; None → use bundled asset
    font_size: int = 64
    active_color: str = "#FFD400"
    inactive_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    position: Literal["top_third", "middle", "bottom_third"] = "bottom_third"
    words_visible: int = 3

class RenderedClip(BaseModel):
    shaped_candidate_index: int                  # joins back to ShapedClip
    source_start: float
    source_end: float
    trim_start: float
    trim_end: float
    output_path: Path
    output_sha256: str                           # post-render; join key for outcome data
    input_fingerprint: str                       # pre-render; §3; drives skip_existing
    output_duration_seconds: float               # ffprobe-measured
    measured_integrated_lufs: float              # ffmpeg loudnorm-measured post-apply
    face_coverage: float = Field(ge=0, le=1)     # fraction of frames a face was tracked
    hook_overlay_text: str | None = None
    settings: RenderSettings
    caption_style: CaptionStyle
    render_wall_seconds: float
    # Provenance chain — everything needed to replay or join to outcomes.
    render_version: str
    source_video_sha256: str
    source_transcript_sha256: str
    font_file_sha256: str
    source_shaped_prompt_version: str
    source_pass1_prompt_version: str

class Pass2bResult(BaseModel):
    rendered: list[RenderedClip]
    render_version: str
    source_video_sha256: str
    source_transcript_sha256: str
    source_shaped_prompt_version: str
    source_pass1_prompt_version: str
    renders_completed: int = 0
    renders_skipped_kill: int = 0                # ShapedClip.verdict == "kill"
    renders_skipped_fingerprint: int = 0         # prior render matched input_fingerprint
    renders_failed: int = 0
```

**Versioning rule mirrors `PROMPT_VERSION`.** `RENDER_VERSION = "v1"` is pinned in `render.py`; any change to the visible output (caption typography, crop algorithm, loudness target, codec params, **bundled font**) bumps to `v2`. Old `rendered.json` files stay joinable by the feedback loop. [CLAUDE.md](CLAUDE.md)'s "prompt versioning is load-bearing" invariant applies equally here, for the same reason: outcome data gets keyed by version. `RENDER_VERSION` is also folded into `input_fingerprint` so a version bump forces a re-render even when every other input byte is unchanged.

### 2. New module — [clipper/render.py](clipper/render.py)

Single file in v1; split only when it wants to split. Mirrors the [pass2a.py](clipper/pass2a.py) shape where analogous (config dataclass, `RENDER_VERSION` constant, JSONL log helper) and diverges where it must (no Claude client, no rubric, no retry-on-JSON).

```python
RENDER_VERSION = "v1"
DEFAULT_LOG_PATH = Path(".cache/logs/pass2b.jsonl")

@dataclass
class RenderConfig:
    settings: RenderSettings = field(default_factory=RenderSettings)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    out_dir: Path = Path("clips")
    log_path: Path | None = DEFAULT_LOG_PATH
    skip_existing: bool = True       # fingerprint-keyed resume; see §3 fingerprint rules
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

def run_pass2b(
    shaped: Pass2aResult,
    transcript: Transcript,                # loaded by the caller; never re-transcribed here
    transcript_sha256: str,                # hash of the on-disk transcript artifact
    source_video: Path,
    cfg: RenderConfig | None = None,
) -> Pass2bResult

def _render_one(shaped, transcript, source, source_sha, transcript_sha, font_sha, cfg) -> RenderedClip
def _compute_input_fingerprint(shaped_clip, source_sha, transcript_sha, font_sha, settings, caption_style) -> str
def _prior_manifest_for_skip(out_dir) -> dict[int, RenderedClip]   # read rendered.json if present
def _measure_loudness(source, trim_start, trim_end, cfg) -> dict    # two-pass loudnorm measurement
def _compute_crop_path(source, trim_start, trim_end, cfg) -> list[CropPoint]
def _smooth_crop_path(pts, halflife_s, fps) -> list[CropPoint]
def _write_ass_file(words, trim_start, trim_end, style, out_path) -> None
def _render_ffmpeg(source, trim_start, trim_end, crop_path, ass_path, font_dir, hook, measured, cfg, out_path) -> None
def _probe_duration_and_lufs(path, cfg) -> tuple[float, float]
def _sha256_file(path) -> str
def _resolve_font(cfg) -> tuple[Path, str]                          # returns (font_path, sha256); uses importlib.resources when cfg.font_file is None
def _append_log(path, rendered) -> None
```

**Atomic output write.** Render to `clips/clip-<idx>.mp4.tmp` then `os.replace`, matching the [ingest.py](clipper/ingest.py) cache-write discipline ([CLAUDE.md invariant](CLAUDE.md#L48)). `test_render_pipeline.py` asserts no partial `.mp4` remains on forced failure.

**Kill verdicts are skipped**, not rendered. They increment `renders_skipped_kill`. The loop still writes a log line with `"skipped_reason": "kill"` so the feedback loop sees the kill.

### 3. Pipeline — single ffmpeg graph per clip, two-pass for loudness

EBU R128 loudnorm needs to measure first, then apply with measured values:

0. **Compute `input_fingerprint`** *before* any ffmpeg work.
   `input_fingerprint = sha256_hex(canonical_json({
        "render_version": RENDER_VERSION,
        "source_video_sha256": ...,            # hashed once per run
        "source_transcript_sha256": ...,        # sha of the on-disk transcript artifact
        "font_file_sha256": ...,                # sha of the resolved font file
        "shaped_clip": shaped.model_dump(mode="json"),
        "render_settings": cfg.settings.model_dump(mode="json"),
        "caption_style": cfg.caption_style.model_dump(mode="json"),
   }))`
   If `cfg.skip_existing`, load `rendered.json` from `cfg.out_dir`; for the current `shaped_candidate_index`, skip only when **all three** hold: stored `input_fingerprint` matches, `output_path` exists on disk, and `_sha256_file(output_path) == stored output_sha256`. Any mismatch falls through to a full re-render. Skips increment `renders_skipped_fingerprint` and emit a log line with `"skipped_reason": "fingerprint_match"`. Existence-only or filename-only skip is explicitly disallowed — this is the rollback/idempotency guardrail against stale MP4s surviving a `RENDER_VERSION`, caption-style, loudness, or transcript change.
1. **Measure loudness** — run `ffmpeg -ss trim_start -to trim_end -i source -af loudnorm=print_format=json -f null -` and parse the last JSON block for `input_i`, `input_lra`, `input_tp`, `input_thresh`, `target_offset`.
2. **Caption timeline** — build `captions.ass` from the `TranscriptWord`s whose midpoint lies in `[trim_start, trim_end]`. Shift timestamps to clip-local (`t' = t - trim_start`). Emit two layered events per word: the active-colored word for its own `[t'_start, t'_end]`, plus the surrounding `words_visible - 1` words in inactive color. Pure-Python generation, no new dep. Reference the font by family name resolved from the bundled font file (§4); the file's directory is passed to ffmpeg's `subtitles` filter as `fontsdir=` so libass does not hit the host fontconfig.
3. **Crop path** — sample frames at 5 fps across `[trim_start, trim_end]` with OpenCV; run MediaPipe Face Detection on each; record `(t, face_cx, face_cy)` or `None`. EMA-smooth the (cx, cy) series with the configured halflife. Clamp the smoothed crop window inside the source frame. Fill `None` runs by holding the last valid center; if the entire clip has no detections, fall back to center-crop and record `face_coverage = 0.0`. Emit as a list of `sendcmd` entries for ffmpeg's `crop` filter, or — simpler for v1 — pre-compute a static crop rect by taking the median of the smoothed series. **Start with static median-crop; escalate to sendcmd only if v1 footage shows unacceptable head drift.**
4. **Single render call** — one ffmpeg invocation: `-ss trim_start -to trim_end -i source`, then filter chain `crop=...,scale=1080:1920,subtitles=captions.ass:fontsdir=<bundled-font-dir>` on video, `loudnorm=I=-14:TP=-1:LRA=11:measured_I=...:...` on audio, optional `drawtext=fontfile=<bundled-font-path>:enable='between(t,0,0.8)'` for the hook overlay. Deterministic params: `-preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart -video_track_timescale 30000 -ar 48000 -c:v libx264 -c:a aac -b:a 128k`. No frame drops, no VFR.
5. **Probe + hash** — ffprobe for output duration and a second loudnorm-measurement pass for `measured_integrated_lufs`; sha256 the resulting file for `output_sha256`.

**Determinism test** (§6.4): same `ShapedClip` + same source bytes + same transcript artifact + same bundled font → identical `input_fingerprint` and identical `output_sha256`. Enforce by fixing all filter params, writing the ASS file with stable key order, seeding nothing, and resolving the font from the bundled asset (never the host fontconfig). The determinism guarantee is scoped to a machine with a libx264/ffmpeg build matching the project's pinned system-dep expectations; we do not promise bit-identical output across libx264 ABI changes, and [docs/technical-decisions.md](docs/technical-decisions.md) says so.

### 4. Library / parameter choices — [docs/technical-decisions.md](docs/technical-decisions.md)

[VISION.md:3](VISION.md#L3) refers to this file; it doesn't exist yet. Create it as part of this plan with entries for:

- **Face tracker: MediaPipe Face Detection.** Cheapest dependency that works on the single-face case (podcasts / interviews — the common footage for this project). Pure-pip install, deterministic, no runtime model download. Limitation: multi-subject scenes get a compromise crop. Revisit when multi-subject becomes in-scope.
- **Loudness target: −14 LUFS integrated, −1 dBTP true peak, 11 LU range.** Matches TikTok / Instagram / YouTube Shorts normalization. Overridable per-profile when profiles land.
- **Caption renderer: pure-Python ASS file + ffmpeg `subtitles` filter with `fontsdir=<bundled>`.** No external library; burn-in at render time; libass never reads host fontconfig.
- **Bundled caption font: Inter (SIL OFL, shipped under `clipper/assets/fonts/Inter-Regular.ttf`).** Chosen because (a) SIL OFL lets us redistribute inside the wheel, (b) it's a sans designed for screens and reads cleanly at caption size, (c) pinning the font file is the only way to make `output_sha256` portable across developer machines, CI, and production workers. Font sha256 is folded into `input_fingerprint`, so a font swap forces a re-render.
- **Determinism scope.** Byte-identical output is guaranteed when `input_fingerprint` is constant *and* the host ffmpeg/libx264 build matches the version range declared in this file. Cross-version libx264 reproducibility is explicitly not promised; `output_sha256` is still a valid join key within a pinned environment and across resumes on the same machine.
- **Video codec: libx264 CRF 20, preset medium, yuv420p, faststart.** Standard deterministic set; balances size and compatibility.

This file is revisable and changes faster than [VISION.md](VISION.md), which is exactly [VISION.md:52](VISION.md#L52)'s stated purpose for it.

### 5. CLI — edit [clipper/cli.py](clipper/cli.py)

Add one subcommand and one optional flag on `process`, mirroring the existing `shape` pattern ([cli.py:127](clipper/cli.py#L127)):

- **`clipper render <video> --shaped shaped.json --transcript transcript.json --out-dir clips/ [--manifest rendered.json]`** — loads `Pass2aResult` and the pinned `Transcript` artifact from disk, hashes the video + transcript + bundled font, and calls `run_pass2b`. **Pass 2b never calls `extract_audio` or `transcribe`.** If `--transcript` is omitted the command errors with a message pointing at the `shape` subcommand's sidecar path. Re-runnable: fingerprint-keyed skip (§3) avoids re-rendering unchanged shapes; a mismatch on any of video-sha, transcript-sha, font-sha, render-version, settings, or caption-style forces a full re-render.
- **`clipper shape`** (existing) gains one behavior change: it writes a sibling `<shaped.json>.transcript.json` containing the `Transcript` it used, so the downstream `render` call consumes bit-identical word timings. This is a prerequisite seam for Pass 2b — small diff in [clipper/cli.py](clipper/cli.py), called out explicitly in "Critical files to modify".
- **`clipper process ... [--render/--no-render] [--render-dir clips/]`** — default **off** initially; requires `--shape`. `--render` consumes the in-memory `Transcript` produced earlier in the run (no reload round-trip) and also persists the sidecar transcript for later standalone `render` invocations. Once stable, flip default to on once `--shape` is also on by default.

### 6. Package data / dependencies — [pyproject.toml](pyproject.toml)

Add to `dependencies`:
- `mediapipe>=0.10` — face tracker
- `opencv-python-headless>=4.9` — frame reading for the tracker (headless avoids Qt/X11 pull-in on servers / CI)

Add to `package-data`:
- `clipper/assets/fonts/*.ttf` — bundled caption font; resolved via `importlib.resources` exactly like [CLAUDE.md](CLAUDE.md)'s rubric-loading invariant.

System deps unchanged: `ffmpeg` + `ffprobe` are already required by Pass 1's ingest. Declare a minimum ffmpeg major version in [docs/technical-decisions.md](docs/technical-decisions.md) to narrow the determinism promise.

No prompt package data — `prompts/*/*.md` globbing at [pyproject.toml:29](pyproject.toml#L29) is Pass-1/2a concern; Pass 2b ships no prompts.

### 7. Per-render JSONL log — `.cache/logs/pass2b.jsonl`

Mirrors the Pass 2a JSONL discipline at [pass2a.py:390-416](clipper/pass2a.py#L390-L416). One line appended per render (completed, skipped, or failed):

```json
{"ts": "2026-04-20T17:02:11Z", "stage": "pass2b", "render_version": "v1",
 "source_pass1_prompt_version": "v2", "source_shaped_prompt_version": "v1",
 "shaped_candidate_index": 0, "trim_start": 121.1, "trim_end": 136.8,
 "output_path": "clips/clip-0.mp4",
 "input_fingerprint": "f0e1...", "output_sha256": "a1b2...",
 "source_video_sha256": "9c3d...", "source_transcript_sha256": "77ab...",
 "font_file_sha256": "4e4e...",
 "output_duration": 15.7, "measured_lufs": -14.03,
 "hook_overlay_text": "Most founders undercharge by a factor of three.",
 "face_coverage": 0.92, "render_wall_seconds": 12.4,
 "skipped_reason": null, "failed": false}
```

`face_coverage` is a noted future signal for the retro rubric — low coverage may correlate with worse retention on face-centric platforms. No rotation, no indexing, atomic append. Worth factoring the log helper into a shared `clipper/_log.py` used by both [pass2a.py](clipper/pass2a.py) and `render.py`; if factoring feels like over-reach for two call sites, copy.

### 8. Rationale against leaking shaping into rendering

Two specific temptations to name and refuse now, because both would collapse the seam:

- **Snapping trims to non-silence boundaries at render time.** The shaper already chose the trim; if its trims bleed, the fix is in shaping, not rendering.
- **Rewriting the hook overlay text based on what the tracker sees.** The overlay text is `ShapedClip.hook.text`. If it doesn't render well, fix the shaper's prompt.

Pass 2b is deterministic and dumb. That is its feature.

### 9. Seam notes for future stages

- **Global ranker lands** → its filtered output still produces a `Pass2aResult` downstream; Pass 2b's input type does not change.
- **Platform profiles land** → `RenderSettings` and `CaptionStyle` become per-profile overrides. `RenderConfig` already separates them from module-level defaults, so profile loading is config-construction, not render-logic. The [SOUL.md:96-102](SOUL.md#L96-L102) authority order (feedback loop > profile > craft) is enforced here by the render pipeline staying dumb — it applies whatever `RenderSettings` / `CaptionStyle` it's handed. A profile change flips `input_fingerprint`, which forces re-renders — intentional.
- **Feedback loop lands** → `output_sha256` + `rendered.json` + the per-render JSONL are the join key to outcome data. The `published_clips` table ([VISION.md:71](VISION.md#L71)) keys on `output_sha256`. `input_fingerprint` is the secondary key used by the render-side resume; outcome data doesn't need it.
- **`hook.kind == "rewrite"`** can't be synthesized by Pass 2b (would require TTS). v1 renders the clip without overlay in that case and logs `hook_overlay_text: null`. A future stage that re-records or splices hooks would sit between Pass 2a and Pass 2b — new seam, not a render change.

## Critical files to modify

- [clipper/schemas.py](clipper/schemas.py) — add `RenderSettings`, `CaptionStyle`, `RenderedClip` (incl. `input_fingerprint`, `source_video_sha256`, `source_transcript_sha256`, `font_file_sha256`), `Pass2bResult`
- [clipper/render.py](clipper/render.py) — new module
- [clipper/cli.py](clipper/cli.py) — new `render` subcommand requiring `--transcript`; edit `shape` to persist the transcript sidecar; optional `--render` on `process`
- [clipper/assets/fonts/Inter-Regular.ttf](clipper/assets/fonts/Inter-Regular.ttf) — new (SIL OFL, vendored into the wheel)
- [pyproject.toml](pyproject.toml) — add `mediapipe`, `opencv-python-headless`; add `clipper/assets/fonts/*.ttf` to `package-data`
- [docs/technical-decisions.md](docs/technical-decisions.md) — new; record face tracker, loudness target, codec params, bundled font, determinism scope / ffmpeg version floor
- [tests/test_render_schemas.py](tests/test_render_schemas.py) — new
- [tests/test_render_captions.py](tests/test_render_captions.py) — new (pure ASS-string tests, no ffmpeg)
- [tests/test_render_crop.py](tests/test_render_crop.py) — new (synthetic face tracks, smoothing, fallback)
- [tests/test_render_fingerprint.py](tests/test_render_fingerprint.py) — new (input-fingerprint stability + stale-skip guardrail; no ffmpeg)
- [tests/test_render_pipeline.py](tests/test_render_pipeline.py) — new (tiny fixture video; runs ffmpeg; asserts dims, duration, LUFS, sha-determinism)

Reuse (do not reimplement):
- `_append_log` pattern from [pass2a.py:390](clipper/pass2a.py#L390) — factor into `clipper/_log.py` or duplicate for two call sites.
- Atomic-write discipline from [ingest.py](clipper/ingest.py) (`.tmp` then `os.replace`) — applied to the output MP4 *and* to the transcript sidecar written by `shape`.
- `importlib.resources` font-loading pattern mirroring rubric loading — one helper in `render.py`.

## Verification

1. **Unit — schemas:** `pytest tests/test_render_schemas.py` — required fields, `face_coverage ∈ [0,1]`, `render_version` / `input_fingerprint` / `source_transcript_sha256` preserved through round-trip.
2. **Unit — ASS generation:** given a fixed list of `TranscriptWord`s + `CaptionStyle`, the generated ASS file is byte-identical across runs; one word is active at a time; words outside the trim are dropped; words straddling a boundary are truncated to the trim.
3. **Unit — crop path smoothing:** synthetic face track with a single spike is damped by the EMA; an all-None track yields a center-crop and `face_coverage == 0.0`; a partial-coverage track computes the correct fraction.
4. **Unit — input fingerprint:** `_compute_input_fingerprint` is stable across key order and equivalent dumps; changing any of `RENDER_VERSION`, source sha, transcript sha, font sha, shaped clip, settings, or caption style changes the fingerprint; renaming the output path does **not** change it.
5. **Unit — skip guardrail:** with `skip_existing=True`, a prior manifest that matches the current fingerprint **and** whose MP4 exists **and** whose on-disk sha matches the stored `output_sha256` triggers a skip; any one of those three mismatching forces a full re-render (stale-skip regression test: mutate `RENDER_VERSION`, mutate caption color, mutate transcript bytes — each must re-render).
6. **Integration — tiny fixture:** a 5-second 720p synthetic clip with a face and a short transcript JSON; render one `ShapedClip`; assert output dims = (1080, 1920), duration within 0.1s of trim, measured LUFS within 0.5 of target, and **output sha256 is identical across two renders** on the same machine (determinism invariant, scoped per §4).
7. **Integration — kill skip:** a `Pass2aResult` with one `kill` and one `keep` produces exactly one MP4 and `renders_skipped_kill == 1`; the kill emits a `"skipped_reason": "kill"` log line.
8. **Integration — fingerprint skip:** re-running the fixture render with unchanged inputs produces `renders_skipped_fingerprint == 1` and no ffmpeg invocation; deleting the MP4 between runs forces a re-render even with a matching fingerprint.
9. **Integration — atomic write:** a simulated mid-render failure leaves no `clip-*.mp4` (only a `.tmp` file if anything).
10. **Regression:** existing `pytest` suite (14 tests) remains green — Pass 1, Pass 2a, ingest all untouched except for the `shape` CLI's new transcript-sidecar write (covered by an updated pass2a CLI test).
11. **Manual E2E** (requires `ffmpeg` + `mediapipe` installable, no Claude call, no STT call): run end-to-end on `test.webm`:
    ```sh
    clipper process test.webm --shape --render --max-windows 3
    clipper render test.webm --shaped shaped.json --transcript shaped.json.transcript.json
    ```
    Then inspect `clips/*.mp4` in a player. Checklist: captions land on words; no audio clipping; 9:16 framing holds the speaker; first-0.8s hook overlay is readable. The second invocation should be a pure fingerprint-skip.
12. **Feedback-loop readiness check:** `jq '.[] | {shaped_candidate_index, input_fingerprint, output_sha256, source_transcript_sha256, measured_lufs, face_coverage}' rendered.json` produces one row per clip with all provenance fields populated. If the feedback loop lands tomorrow, this is what it joins on.

## What this plan does not do

- Does not build the global ranker. That's a separate plan when it comes due. Pass 2b runs on `Pass2aResult` whether it came from a ranked or unranked pipeline.
- Does not build platform profiles. The schema leaves `RenderSettings` and `CaptionStyle` overridable; profile *content* is future work.
- Does not build the retro rubric or outcome-joining infrastructure. It writes the provenance fields those future stages will read, and stops there.
- Does not synthesize audio for `hook.kind == "rewrite"`. That's a new stage, not a Pass 2b feature.
- Does not promise bit-identical output across libx264/ffmpeg ABI changes. Determinism is scoped to a pinned system-dep range, documented in [docs/technical-decisions.md](docs/technical-decisions.md); `input_fingerprint` + `output_sha256` remain valid join keys within that range.
