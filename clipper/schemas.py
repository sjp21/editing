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
    """A candidate region handed to the scorer."""

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
# Score output
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
    # Stable identity assigned after the scorer's dedupe+sort. The ranker
    # reorders/filters candidates; this field lets downstream stages (and
    # their JSONL logs) join back to the original scorer identity. Nullable
    # for back-compat with candidates.json files written before this field
    # existed.
    score_index: int | None = None

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


class ScoreResult(BaseModel):
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
# Shape output
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


class ShapeResult(BaseModel):
    shaped: list[ShapedClip]
    model: str
    prompt_version: str
    source_score_prompt_version: str
    # Set when shape's input was filtered by the ranker. None when shape
    # consumes a raw ScoreResult. Lets JSONL rows join back to the ranker call.
    source_ranker_prompt_version: str | None = None
    profile: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    candidates_shaped: int = 0
    candidates_killed: int = 0
    candidates_failed: int = 0


# ---------------------------------------------------------------------------
# Render output
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
    source_score_prompt_version: str


class RenderResult(BaseModel):
    rendered: list[RenderedClip]
    render_version: str
    source_video_sha256: str
    source_transcript_sha256: str
    source_shaped_prompt_version: str
    source_score_prompt_version: str
    renders_completed: int = 0
    renders_skipped_kill: int = 0
    renders_skipped_fingerprint: int = 0
    renders_failed: int = 0


# ---------------------------------------------------------------------------
# Posted output (manual feedback-loop entry)
# ---------------------------------------------------------------------------


class PostedClip(BaseModel):
    ts: str
    stage: Literal["post"] = "post"
    # Join key back to ShapedClip.candidate_index / RenderedClip.shaped_candidate_index.
    shaped_candidate_index: int
    video_stem: str
    source_video_sha256: str
    output_path: str
    output_sha256: str
    render_version: str
    source_shaped_prompt_version: str
    source_score_prompt_version: str
    platform: str
    post_id: str
    post_url: str | None = None
    title: str | None = None
    posted_at: str | None = None
    notes: str | None = None

    @field_validator("platform", "post_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Global ranker output
# ---------------------------------------------------------------------------


class RankedClip(BaseModel):
    candidate_index: int  # points at Candidate.score_index
    rank: int = Field(ge=1)
    hook_quote: str
    payoff_in_window: bool
    payoff_reason: str
    comparative_reason: str
    profile_conflict_note: str | None = None


class DroppedClip(BaseModel):
    candidate_index: int
    reason_kind: Literal[
        "semantic_duplicate",
        "weak_hook",
        "payoff_absent",
        "generic",
        "two_ideas",
        "opens_on_conjunction",
        "other",
    ]
    reason_detail: str
    comparative_reason: str
    duplicate_of_candidate_index: int | None = None
    hook_quote: str | None = None


class RankedCandidates(BaseModel):
    ranked: list[RankedClip]
    dropped: list[DroppedClip]
    requested_k: int = Field(ge=1)
    truncation_note: str | None = None
    profile: str | None = None
    profile_version: str | None = None
    model: str
    prompt_version: str
    source_score_prompt_version: str
    call_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @model_validator(mode="after")
    def _validate_ranking(self) -> "RankedCandidates":
        if len(self.ranked) > self.requested_k:
            raise ValueError(
                f"ranked count ({len(self.ranked)}) exceeds requested_k ({self.requested_k})"
            )

        ranks = sorted(c.rank for c in self.ranked)
        if ranks != list(range(1, len(self.ranked) + 1)):
            raise ValueError(
                f"ranks must be 1..N contiguous with no duplicates, got {ranks}"
            )

        ranked_ids = {c.candidate_index for c in self.ranked}
        dropped_ids = {c.candidate_index for c in self.dropped}
        if len(ranked_ids) != len(self.ranked):
            raise ValueError("duplicate candidate_index within ranked")
        if len(dropped_ids) != len(self.dropped):
            raise ValueError("duplicate candidate_index within dropped")
        if ranked_ids & dropped_ids:
            raise ValueError(
                f"candidate appears in both ranked and dropped: "
                f"{sorted(ranked_ids & dropped_ids)}"
            )

        truncated = len(self.ranked) < self.requested_k
        if truncated and not (self.truncation_note and self.truncation_note.strip()):
            raise ValueError(
                "truncation_note required when len(ranked) < requested_k"
            )
        if not truncated and self.truncation_note:
            raise ValueError(
                "truncation_note must be absent when len(ranked) == requested_k"
            )

        for d in self.dropped:
            if d.duplicate_of_candidate_index is None:
                continue
            if d.duplicate_of_candidate_index not in ranked_ids:
                raise ValueError(
                    f"dropped candidate {d.candidate_index} references "
                    f"duplicate_of_candidate_index={d.duplicate_of_candidate_index} "
                    f"which is not in ranked"
                )
        return self


# ---------------------------------------------------------------------------
# Hand-labeled eval (clipper eval)
# ---------------------------------------------------------------------------


class LabeledClip(BaseModel):
    start: float
    end: float
    tier: Literal["must_ship", "nice", "marginal", "rejected"]
    reason: str

    @model_validator(mode="after")
    def _validate_timing(self) -> "LabeledClip":
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be > start ({self.start})"
            )
        return self


class HumanLabels(BaseModel):
    video_stem: str
    labeler: str
    labeled_at: str
    labels: list[LabeledClip]
    schema_version: Literal[1] = 1


class LabelMatch(BaseModel):
    label_index: int
    tier: Literal["must_ship", "nice", "marginal", "rejected"]
    reason: str
    matched_score_index: int | None = None
    iou: float | None = None
    # None when the ranker did not run for this video.
    in_ranker_topk: bool | None = None
    # `keep`/`kill` come from ShapedClip.verdict on the matched candidate.
    # `missing` = matched candidate was filtered by the ranker (never reached shape).
    # `failed`  = shape ran but did not emit a result for this candidate.
    # None      = no matched candidate, or shape did not run.
    shape_verdict: Literal["keep", "kill", "missing", "failed"] | None = None


class LabelScorecard(BaseModel):
    video_stem: str
    eval_version: Literal[1] = 1
    iou_threshold: float = 0.5
    scorer_recall: float
    ranker_inclusion: float
    shape_alignment: float
    matches: list[LabelMatch]
    misses: list[int]
    false_positives: list[int]
    timestamp: str
