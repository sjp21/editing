# Technical Decisions

Living record of library and parameter choices for the clipper pipeline. Revisable — changes faster than VISION.md.

## Pass 2b — Rendering

### Face tracker: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`)

The cascade XML ships inside the opencv-python-headless wheel (at `cv2.data.haarcascades`), so no runtime model download is required and determinism is scoped to the opencv build, not to network state.

Parameters: `detectMultiScale(scaleFactor=1.2, minNeighbors=5, minSize=(max(30, w/20), max(30, h/20)))`. Largest detection wins (single-subject case — interviews and podcasts).

Limitation: Haar is less robust than a modern DL detector on profile faces and heavy motion blur. It is the cheapest dependency available in the current build environment with zero-download semantics. Upgrade path: swap in MediaPipe's Tasks API `FaceDetector` with a bundled `blaze_face_short_range.task` model — that's a `RENDER_VERSION=v2` change (fingerprint flips, old `rendered.json` stays joinable, every keep re-renders). The plan originally specified MediaPipe; the `mediapipe>=0.10.15` release removed the legacy `mp.solutions.face_detection` module, and the replacement Tasks API needs a downloaded model that isn't vendored in this tree yet.

Multi-subject framing stays a compromise crop regardless of detector; that non-goal is owned by VISION.

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
- **opencv-python-headless**: >= 4.9 (ships the Haar cascade we use).
- **mediapipe**: >= 0.10 (installed for a future `RENDER_VERSION=v2` face-tracker swap; not on the hot path in v1).

Cross-version libx264 reproducibility is explicitly not promised. If you change ffmpeg / libx264 / mediapipe major versions, bump `RENDER_VERSION` so the feedback loop can segregate outcomes by rendering stack.

### Non-goals (explicit)

- Multi-subject segmentation — VISION explicitly defers this.
- Zoom-based hook emphasis — text overlay only in v1.
- Hook TTS for `hook.kind == "rewrite"` — v1 renders the clip with no overlay in that case.
- B-roll, music, synthetic voiceover, platform auto-publish.
