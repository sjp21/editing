# Technical Decisions

Living record of library and parameter choices for the clipper pipeline. Revisable — changes faster than VISION.md.

## Pass 2b — Rendering

### Face tracker: MediaPipe Tasks `FaceDetector` (BlazeFace short-range)

Detector: MediaPipe Tasks API `FaceDetector` running the bundled `blaze_face_short_range.tflite` model, shipped under `clipper/assets/models/`. Running mode is VIDEO (`detect_for_video`) — inter-frame continuity, monotonic timestamp input — with `min_detection_confidence=0.5`. Largest detection wins (single-subject case — interviews and podcasts).

Determinism: MediaPipe Tasks produces deterministic outputs for a fixed model file + input frame, so we pin the model sha256, not just the mediapipe library version. The model file hash is folded into `input_fingerprint` and serialized onto every `RenderedClip.model_file_sha256`, so a model swap forces a re-render for every clip.

Multi-subject framing stays a compromise crop regardless of detector; that non-goal is owned by VISION.

**Retired in v2 (Haar cascade).** v1 used OpenCV's bundled `haarcascade_frontalface_default.xml`. In practice Haar missed faces in real footage (see `clips/rendered.json` v1 render with `face_coverage: 0.0` on a clip with visible faces), so the fallback center crop was being hit for every frame. The BlazeFace upgrade is a `RENDER_VERSION = v1 → v2` change: `face_tracker` default flipped from `"opencv_haar"` to `"mediapipe_blazeface_short_range"`, `model_file_sha256` was added to the fingerprint inputs and the schema, and every prior `rendered.json` entry deterministically mismatches the new fingerprint and re-renders on next invocation.

### Loudness target: −14 LUFS integrated, −1 dBTP true peak, 11 LU range

Matches TikTok / Instagram Reels / YouTube Shorts normalization. Clips authored hotter than this are attenuated by the platforms; clips quieter than this play at lower apparent loudness than other users'. Two-pass EBU R128 (`ffmpeg loudnorm`) — measure first, then apply with measured values.

Overridable per-profile when platform profiles land; `RenderSettings` already carries the fields.

### Caption renderer: pure-Python ASS + ffmpeg `subtitles` filter

Word-by-word highlight built by emitting layered Dialogue events in an ASS file. Burned in at render time via ffmpeg's `subtitles` filter with `fontsdir=` pointing at the bundled font directory — libass never reads host fontconfig, so the same input bytes render identically across machines.

### Bundled caption font: DejaVu Sans

Shipped under `clipper/assets/fonts/DejaVuSans.ttf`. Chosen because (a) its license permits redistribution inside the wheel, (b) it renders cleanly at caption size, (c) it is present on the reference dev machine's system-font path, so the vendored copy is a known-good copy of what the developer sees.

Font sha256 is folded into `input_fingerprint`, so a font swap forces a re-render for every clip.

(Plan originally specified Inter; the network policy in the current dev environment blocks fetching Inter's release asset at build time, so DejaVu Sans is the concrete bundled choice. A future bump to Inter is a `RENDER_VERSION=v2` change — `input_fingerprint` will flip, old `rendered.json` stays joinable, and every keep re-renders.)

### Video codec: libx264 CRF 20, preset medium, yuv420p, faststart

Standard deterministic set. CRF 20 balances size against visible quality at 1080×1920; `yuv420p` keeps the output playable on consumer platforms that reject 4:4:4 / 4:2:2; `+faststart` moves the moov atom to the front so web players can start playing before download completes. Frame rate pinned to 30fps and `-video_track_timescale 30000` disables VFR.

### Determinism scope

Byte-identical output is guaranteed when `input_fingerprint` is constant *and* the host ffmpeg / libx264 build matches the pinned version range below. `input_fingerprint` and `output_sha256` remain valid join keys within that range.

Pinned reference environment:
- **ffmpeg**: >= 6.1 (x264-enabled). Tested on ffmpeg 6.1.1 (Ubuntu 24.04 apt build).
- **libx264**: whatever ships in the above. We do not promise bit-identical output across libx264 ABI changes.
- **opencv-python-headless**: >= 4.9 (video frame decoding + color conversion feeding MediaPipe).
- **mediapipe**: >= 0.10 (Tasks `FaceDetector` on the render hot path).

Cross-version libx264 reproducibility is explicitly not promised. If you change ffmpeg / libx264 / mediapipe major versions, bump `RENDER_VERSION` so the feedback loop can segregate outcomes by rendering stack.

### Non-goals (explicit)

- Multi-subject segmentation — VISION explicitly defers this.
- Zoom-based hook emphasis — text overlay only in v1.
- Hook TTS for `hook.kind == "rewrite"` — v1 renders the clip with no overlay in that case.
- B-roll, music, synthetic voiceover, platform auto-publish.
