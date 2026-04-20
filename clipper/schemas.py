"""Pydantic schemas for the clipper pipeline.

All intermediate outputs are structured so downstream feedback loops
(human labels, metrics) can be attached later without schema changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class TranscriptWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    """A Whisper segment after post-processing / grouping."""

    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    language: str | None = None
    duration: float
    segments: list[TranscriptSegment]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class AudioFeature(BaseModel):
    """A single timestamped audio observation."""

    time: float
    kind: Literal["volume_spike", "silence_gap", "speech_rate_change"]
    value: float = 0.0
    note: str | None = None


class TextFeature(BaseModel):
    """A transcript-derived energy signal."""

    time: float
    kind: Literal[
        "exclamation",
        "interruption",
        "repeated_word",
        "length_spike",
        "sentence_boundary",
    ]
    value: float = 0.0
    note: str | None = None


class Signals(BaseModel):
    audio: list[AudioFeature] = Field(default_factory=list)
    text: list[TextFeature] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Frames & Windows
# ---------------------------------------------------------------------------


class Frame(BaseModel):
    time: float
    path: Path
    trigger: Literal["base", "audio_spike", "text_peak", "sentence_boundary"] = "base"


class Window(BaseModel):
    """A candidate region handed to Pass 1."""

    index: int
    start: float
    end: float
    transcript: str
    segments: list[TranscriptSegment]
    frames: list[Frame] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Pass 1 output
# ---------------------------------------------------------------------------


class Scores(BaseModel):
    hook_strength: float = Field(ge=0, le=10)
    emotional_peak: float = Field(ge=0, le=10)
    quotability: float = Field(ge=0, le=10)
    self_contained: float = Field(ge=0, le=10)
    visual_interest: float = Field(ge=0, le=10)
    reaction_worthy: float = Field(ge=0, le=10)


class Candidate(BaseModel):
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    scores: Scores
    overall_score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    rationale: str
    suggested_hook: str

    # Provenance — enables the future feedback loop.
    window_index: int | None = None
    prompt_version: str | None = None
    model: str | None = None

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration(cls, v: float) -> float:
        if not (15.0 <= v <= 20.0):
            raise ValueError(f"duration_seconds must be in [15, 20], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_timing(self) -> "Candidate":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) must be > start_seconds ({self.start_seconds})"
            )
        span = self.end_seconds - self.start_seconds
        if abs(self.duration_seconds - span) > 0.1:
            raise ValueError(
                f"duration_seconds ({self.duration_seconds}) does not match "
                f"end - start ({span:.3f})"
            )
        return self


class WindowDiagnostic(BaseModel):
    """Per-window telemetry. One entry per window attempted, pass or fail.

    Surfaced so that patterns like "all rejections are pure-filler" or
    "quiet windows have zero text features" become inspectable without
    reading logs.
    """

    index: int
    start: float
    end: float
    num_text_features: int = 0
    num_audio_features: int = 0
    num_frames: int = 0
    num_event_frames: int = 0
    candidates_returned: int = 0
    filter_reason: str | None = None
    call_succeeded: bool = True


class Pass1Result(BaseModel):
    candidates: list[Candidate]
    # Observability — lets us reproduce and diff runs later.
    model: str
    prompt_version: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    windows_processed: int = 0
    windows_rejected: int = 0
    window_diagnostics: list[WindowDiagnostic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass 2a — Shaping output
# ---------------------------------------------------------------------------


class ShapedTrim(BaseModel):
    start_seconds: float
    end_seconds: float
    start_word: str
    end_word: str
    duration_seconds: float

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration(cls, v: float) -> float:
        if not (12.0 <= v <= 22.0):
            raise ValueError(f"duration_seconds must be in [12, 22], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_timing(self) -> "ShapedTrim":
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) must be > start_seconds ({self.start_seconds})"
            )
        span = self.end_seconds - self.start_seconds
        if abs(self.duration_seconds - span) > 0.1:
            raise ValueError(
                f"duration_seconds ({self.duration_seconds}) does not match "
                f"end - start ({span:.3f})"
            )
        return self


class HookTreatment(BaseModel):
    kind: Literal["rewrite", "overlay", "none"]
    text: str | None = None
    rationale: str

    @model_validator(mode="after")
    def _validate_text(self) -> "HookTreatment":
        if self.kind != "none" and not (self.text and self.text.strip()):
            raise ValueError(f"hook text required when kind={self.kind!r}")
        return self


class ShapedClip(BaseModel):
    candidate_index: int
    source_start: float
    source_end: float
    verdict: Literal["keep", "kill"]
    kill_reason: str | None = None
    trim: ShapedTrim | None = None
    hook: HookTreatment | None = None
    reasoning: str
    model: str
    prompt_version: str

    @model_validator(mode="after")
    def _check_verdict(self) -> "ShapedClip":
        if self.verdict == "keep":
            if self.trim is None:
                raise ValueError("verdict=keep requires a trim")
            if self.hook is None:
                raise ValueError("verdict=keep requires a hook")
        else:  # kill
            if not (self.kill_reason and self.kill_reason.strip()):
                raise ValueError("verdict=kill requires a kill_reason")
        return self


class Pass2aResult(BaseModel):
    shaped: list[ShapedClip]
    model: str
    prompt_version: str
    source_pass1_prompt_version: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    candidates_shaped: int = 0
    candidates_killed: int = 0
    candidates_failed: int = 0


# ---------------------------------------------------------------------------
# Pass 2b — Rendering
# ---------------------------------------------------------------------------


class RenderSettings(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_codec: str = "libx264"
    video_crf: int = 20
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    loudness_target_lufs: float = -14.0
    loudness_true_peak_dbfs: float = -1.0
    loudness_range_lu: float = 11.0
    face_tracker: str = "mediapipe_blazeface_short_range"
    crop_smoothing_halflife_s: float = 0.5


class CaptionStyle(BaseModel):
    font_family: str = "DejaVu Sans"
    font_file: Path | None = None
    font_size: int = 64
    active_color: str = "#FFD400"
    inactive_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    position: Literal["top_third", "middle", "bottom_third"] = "bottom_third"
    words_visible: int = 3


class RenderedClip(BaseModel):
    shaped_candidate_index: int
    source_start: float
    source_end: float
    trim_start: float
    trim_end: float
    output_path: Path
    output_sha256: str
    input_fingerprint: str
    output_duration_seconds: float
    measured_integrated_lufs: float
    face_coverage: float = Field(ge=0, le=1)
    hook_overlay_text: str | None = None
    settings: RenderSettings
    caption_style: CaptionStyle
    render_wall_seconds: float
    render_version: str
    source_video_sha256: str
    source_transcript_sha256: str
    font_file_sha256: str
    model_file_sha256: str = ""
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
    renders_skipped_kill: int = 0
    renders_skipped_fingerprint: int = 0
    renders_failed: int = 0
