# Fix face-driven framing in Pass 2b renders

## Context

`clips/clip-0.mp4` (the user's `clip-0-review.mp4`) was rendered with `face_coverage: 0.0` — the OpenCV Haar cascade in `clipper/render.py:_compute_crop_path` found zero faces across every sampled frame, even though faces are clearly on screen. With no face data, `_build_crop_filter` falls back to a dead-center crop of the 9:16 window, which cuts speakers off when they're not centered in the source frame.

Haar's failure here matches the known limitation documented in `docs/technical-decisions.md:13`: *"Haar is less robust than a modern DL detector on profile faces and heavy motion blur."* The same file already names the fix as the upgrade path: swap to MediaPipe's Tasks-API `FaceDetector` with a bundled `blaze_face_short_range.task` model, and bump `RENDER_VERSION` to `v2`.

The user chose the MediaPipe upgrade over the cheaper Haar-tuning option, accepting the re-render cost for every prior clip in exchange for robustness.

## Goal

Replace the Haar detector with MediaPipe BlazeFace (short-range), vendored as an asset. Keep every other Pass 2b invariant intact (single-subject largest-detection wins, EMA smoothing, median static crop, fingerprint-keyed skip, atomic write). Bump `RENDER_VERSION` to `v2` so old renders are segregated from new ones in the feedback-loop join keys.

## Design

### 1. Vendor the model file

- New path: `clipper/assets/models/blaze_face_short_range.task` (~230KB, Apache 2.0).
- Source: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task`. Matches the `technical-decisions.md:13` reference and the Tasks-API examples in upstream MediaPipe docs.
- If the build environment blocks the fetch (same failure mode that forced DejaVu over Inter per `docs/technical-decisions.md:33`), the user drops the file in manually. Plan does **not** add a build-time downloader — the whole point of vendoring is zero-download at runtime.
- Update `pyproject.toml` `[tool.setuptools.package-data]` (currently line 31):
  ```toml
  clipper = ["prompts/*/*.md", "assets/fonts/*.ttf", "assets/models/*.task"]
  ```

### 2. Rewrite `_compute_crop_path` in `clipper/render.py` (lines 440–497)

Replace the Haar block with a MediaPipe Tasks `FaceDetector` call. Keep the function signature, return type (`list[CropPoint]`), and the `CropPoint` dataclass unchanged — every downstream helper (`_smooth_crop_path`, `_face_coverage`, `_median_crop_center`, `_build_crop_filter`) stays untouched.

Concrete shape:

```python
def _compute_crop_path(source, trim_start, trim_end, cfg) -> list[CropPoint]:
    try:
        import cv2
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp
    except ImportError as e:
        log.warning("mediapipe/opencv unavailable (%s); falling back to center crop", e)
        return [CropPoint(t=trim_start, cx=None, cy=None)]

    model_path = _resolve_face_model()          # importlib.resources → concrete Path
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,           # BlazeFace short-range default
    )
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        cap = cv2.VideoCapture(str(source))
        ...
        for f_idx in range(start_frame, end_frame + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            t = f_idx / src_fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(round(t * 1000))        # monotonically increasing required
            result = detector.detect_for_video(mp_image, ts_ms)
            if result.detections:
                # Largest detection wins (single-subject case — same rule as Haar).
                h, w = frame_rgb.shape[:2]
                best = max(result.detections,
                           key=lambda d: d.bounding_box.width * d.bounding_box.height)
                bb = best.bounding_box
                cx = (bb.origin_x + bb.width / 2) / w
                cy = (bb.origin_y + bb.height / 2) / h
                points.append(CropPoint(t=t, cx=cx, cy=cy))
            else:
                points.append(CropPoint(t=t, cx=None, cy=None))
        cap.release()
    return points or [CropPoint(t=trim_start, cx=None, cy=None)]
```

Key details:
- **Running mode = VIDEO**, not IMAGE — BlazeFace in video mode uses inter-frame continuity and accepts monotonic timestamps via `detect_for_video(image, timestamp_ms)`. We already sample at `CROP_SAMPLE_FPS = 5.0`, so timestamps derived from `f_idx / src_fps` are naturally monotonic.
- **BGR→RGB conversion** is mandatory. OpenCV reads BGR; MediaPipe expects SRGB.
- **Largest detection wins** — same single-subject rule as the Haar path. Multi-subject framing stays explicitly out of scope per `VISION.md:94`.
- **`min_detection_confidence=0.5`** is the BlazeFace default. Tune only if the first real render shows false positives; don't pre-optimize.
- The try/except ImportError fallback returning `[CropPoint(t=trim_start, cx=None, cy=None)]` preserves the current "library missing → center crop" safety net.

### 3. Add `_resolve_face_model` helper in `clipper/render.py`

Mirrors `_resolve_font` (already in the file per `plan_pass2b.md:131`). Uses `importlib.resources` so the installed console script works from any cwd, matching the `CLAUDE.md` rubric-loading invariant.

```python
def _resolve_face_model() -> Path:
    # importlib.resources.as_file → concrete path usable by mediapipe
    ref = importlib.resources.files("clipper.assets.models") / "blaze_face_short_range.task"
    with importlib.resources.as_file(ref) as p:
        return Path(p)
```

Needs `clipper/assets/models/__init__.py` (empty file) so `importlib.resources.files("clipper.assets.models")` resolves as a package. `assets/` and `assets/fonts/` currently aren't packages either — verify the existing `_resolve_font` pattern and copy it exactly.

### 4. Bump `RENDER_VERSION`

- `clipper/render.py`: `RENDER_VERSION = "v1"` → `"v2"`.
- `clipper/schemas.py:RenderSettings.face_tracker` default: `"opencv_haar"` → `"mediapipe_blazeface_short_range"`. This string gets serialized into every `rendered.json`, so outcome data is keyed by which detector produced the crop. Also folded into `input_fingerprint` via `settings.model_dump()`, so a detector swap alone forces a re-render.
- Also fold `font_file_sha256` equivalent for the model: add `model_file_sha256` to `RenderedClip` in `schemas.py` and into the fingerprint input dict (`_compute_input_fingerprint` in `render.py`). This matches how the bundled font was handled. Without it, a future model file swap wouldn't invalidate prior renders. Log it in `.cache/logs/pass2b.jsonl` alongside `font_file_sha256`.

### 5. Update `docs/technical-decisions.md`

Rewrite the `Face tracker` section (lines 7–15) in place — technical-decisions is the living/revisable doc (per its own preamble). New content:

- Detector: MediaPipe Tasks API `FaceDetector` with bundled `blaze_face_short_range.task`.
- Running mode: VIDEO; `min_detection_confidence=0.5`; largest detection wins.
- Model file sha256 folded into `input_fingerprint` and `rendered.json`.
- Determinism: MediaPipe Tasks is deterministic for a fixed model file + input frame; we pin the model sha, not the mediapipe library version alone.
- Old Haar entry kept as a "Retired in v2" note for archaeology.

### 6. Update tests

- **`tests/test_render_crop.py`** — no changes needed. The tests operate on `CropPoint` series directly and never invoke the detector. They test smoothing, coverage, and the crop filter math — all detector-agnostic.
- **`tests/test_render_schemas.py`** — bump any pinned `face_tracker` / `render_version` / added-field assertions to match `"mediapipe_blazeface_short_range"` / `"v2"` / `model_file_sha256`.
- **`tests/test_render_fingerprint.py`** — add one regression test: mutating `model_file_sha256` alone flips `input_fingerprint`. Symmetric to the existing font-sha mutation test.
- **`tests/test_render_pipeline.py`** — currently uses a synthetic teal video with no faces; it already asserts the zero-face fallback path. Keep as-is but verify the end-to-end run succeeds with the new detector (no ImportError, no MediaPipe init failure). If MediaPipe init is slow, either mark the test slow or mock the detector — do not regress pipeline-test runtime.
- **New `tests/test_render_detector_smoke.py`** — skip-if-model-missing smoke test: given a synthetic frame with a drawn face-like blob, assert `_compute_crop_path` returns at least one point with `cx, cy` inside [0, 1]. Doesn't assert detection accuracy — just that the detector wiring runs end-to-end without raising. Guard with `pytest.importorskip("mediapipe")` and `pytest.skip` if the `.task` file isn't vendored.

### 7. One-time re-render

After the code change lands, the user runs `clipper render test.webm --shaped shaped.json --transcript shaped.json.transcript.json` (or `clipper process ... --render`). Fingerprint mismatch (every field derived from `RENDER_VERSION`, `face_tracker` string, and `model_file_sha256` has changed) forces a full re-render of every keep. The existing skip-guardrail in `render.py:~153` already handles this correctly — no code needed to trigger re-render.

## Critical files to modify

- `clipper/render.py` — rewrite `_compute_crop_path` (lines 440–497); add `_resolve_face_model` helper near `_resolve_font`; bump `RENDER_VERSION` to `"v2"`; extend `_compute_input_fingerprint` dict to include `model_file_sha256`; extend the JSONL log line in `_append_log` or its call site to include `model_file_sha256`.
- `clipper/schemas.py` — change `RenderSettings.face_tracker` default; add `model_file_sha256: str` to `RenderedClip` and `Pass2bResult` (mirrors `font_file_sha256`).
- `clipper/assets/models/blaze_face_short_range.task` — new vendored model file (~230KB).
- `clipper/assets/models/__init__.py` — new empty file so `importlib.resources.files("clipper.assets.models")` resolves.
- `pyproject.toml:31` — extend `package-data` glob to include `assets/models/*.task`.
- `docs/technical-decisions.md` — rewrite the Face tracker section.
- `tests/test_render_schemas.py` — update pinned defaults / new field.
- `tests/test_render_fingerprint.py` — add `model_file_sha256` mutation test.
- `tests/test_render_detector_smoke.py` — new; skip-if-missing smoke test.

Reuse (do not reimplement):
- `_resolve_font` pattern in `render.py` for `_resolve_face_model`. Same `importlib.resources.as_file` idiom.
- Font-sha fingerprint wiring in `_compute_input_fingerprint` — copy the exact shape for `model_file_sha256`.
- `CropPoint`, `_smooth_crop_path`, `_face_coverage`, `_median_crop_center`, `_build_crop_filter` — all untouched. Detector swap is surgical.

## Verification

1. **Vendored file present:** `ls clipper/assets/models/blaze_face_short_range.task` resolves and is ~230KB.
2. **Unit (schema round-trip):** `pytest tests/test_render_schemas.py` — `RenderSettings.face_tracker == "mediapipe_blazeface_short_range"`, `RenderedClip.model_file_sha256` preserved through JSON round-trip.
3. **Unit (fingerprint):** `pytest tests/test_render_fingerprint.py` — mutating `model_file_sha256` alone changes `input_fingerprint`; mutating `RENDER_VERSION` to `v2` flips it relative to a synthetic v1 fingerprint.
4. **Unit (crop math unchanged):** `pytest tests/test_render_crop.py` — stays green with zero edits (detector-agnostic).
5. **Smoke (detector wiring):** `pytest tests/test_render_detector_smoke.py` — returns `len(points) > 0` on a synthetic frame, skips cleanly if model file absent.
6. **Integration (fixture pipeline):** `pytest tests/test_render_pipeline.py` — ffmpeg still produces a 1080×1920 MP4, loudness within 0.5 LUFS of target. Asserts `face_coverage == 0.0` on the teal synthetic (still true — no face on screen).
7. **Regression:** full `pytest` stays green.
8. **Manual E2E (the actual fix for the reported bug):**
   ```sh
   clipper render test.webm --shaped shaped.json --transcript shaped.json.transcript.json --out-dir clips
   ```
   Fingerprint-mismatch forces a full re-render.
   - `jq '.rendered[0].face_coverage' clips/rendered.json` should be **well above 0.0** (expect >0.5 for an interview clip with visible faces).
   - `jq '.rendered[0].settings.face_tracker' clips/rendered.json` should report `"mediapipe_blazeface_short_range"`.
   - `jq '.rendered[0].render_version' clips/rendered.json` should report `"v2"`.
   - Open `clips/clip-0.mp4` in a player: speaker's face should be centered in the 9:16 crop (not dead-center of the source frame).
9. **Idempotency:** re-run the same command immediately — `renders_skipped_fingerprint == 1`, no ffmpeg invocation.

## What this plan does not do

- Does not add multi-subject framing (explicitly out of scope per `VISION.md:94`). Largest detection wins, same as before.
- Does not add zoom/pan via ffmpeg `sendcmd` — static median-crop per-clip stays the v1/v2 choice (`plan_pass2b.md:156`). Escalate only if smoothed crop still shows head drift on real footage.
- Does not add a build-time model downloader. Vendoring is the whole point of the zero-download-at-runtime policy in `technical-decisions.md`. If the model file can't be fetched in this environment, the user drops it in manually once.
- Does not change caption/loudness/codec pipeline. Detector swap is surgical.
- Does not re-render from the Haar-era baseline automatically — the fingerprint-mismatch guardrail already re-renders on next invocation; the user triggers that when ready.
