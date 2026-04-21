"""Pure helpers for `clipper post` — path → identity resolution + append.

Kept outside ``cli.py`` so tests can drive the logic directly, matching the
``_profiles.py`` / ``_atomic_write_json`` pattern used elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from clipper.schemas import PostedClip, RenderedClip, RenderResult


def resolve_identity(
    clip_path: Path, manifest_path: Path
) -> tuple[RenderedClip, Path]:
    """Match ``clip_path`` against a ``rendered.json`` manifest.

    Returns ``(matched_clip, stem_root)`` where ``stem_root`` is
    ``manifest_path.parent.parent`` — the per-video tree root that holds
    ``logs/posted.jsonl`` by default.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no rendered.json at {manifest_path}; "
            f"pass --manifest if your layout is non-default"
        )
    result = RenderResult.model_validate_json(manifest_path.read_text())
    target = clip_path.resolve()
    for c in result.rendered:
        if Path(c.output_path).resolve() == target:
            return c, manifest_path.parent.parent
    listing = ", ".join(str(c.output_path) for c in result.rendered) or "(empty)"
    raise ValueError(
        f"clip {clip_path} not in {manifest_path}; manifest contains: {listing}"
    )


def append_posted(log_path: Path, row: PostedClip) -> None:
    """Append one JSONL row. Creates parent directories as needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
