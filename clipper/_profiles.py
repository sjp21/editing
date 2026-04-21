"""Shared platform-profile loader + rubric injector.

Profiles (``clipper/prompts/{vN}/profiles/{name}.md``) are a taste layer
above the editor persona: ranker and shaper both load the active profile
and inject it into the ``<!-- [PLATFORM_PROFILE] -->`` marker in their
rubric. The logic is identical — this module is the one copy both stages
call, so drift between them is impossible.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


PROFILE_MARKER = "<!-- [PLATFORM_PROFILE] -->"


def load_profile(
    name: str | None,
    profiles_dir: Path | None,
    prompt_version: str,
) -> str:
    """Load a platform profile fragment. Empty string when name is None.

    Raises FileNotFoundError with a clear message when name is set but the
    file is missing. No silent fallback — a profile typo must not pass
    through as "use defaults."
    """
    if name is None:
        return ""

    if profiles_dir is not None:
        path = Path(profiles_dir) / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(
                f"profile {name!r} not found at {path}. "
                f"Profiles live alongside prompts in <profiles-dir>/<name>.md."
            )
        return path.read_text(encoding="utf-8")

    res = resources.files("clipper").joinpath(
        f"prompts/{prompt_version}/profiles/{name}.md"
    )
    if not res.is_file():
        raise FileNotFoundError(
            f"profile {name!r} not found at clipper/prompts/{prompt_version}/profiles/{name}.md"
        )
    return res.read_text(encoding="utf-8")


def inject_profile(rubric: str, profile_text: str) -> str:
    """Substitute the [PLATFORM_PROFILE] slot in the rubric."""
    if not profile_text:
        return rubric
    # str.replace (not re.sub) so backslash sequences in the profile —
    # Windows paths, regex examples, LaTeX — aren't interpreted as
    # replacement-template backreferences and crash startup.
    return rubric.replace(PROFILE_MARKER, profile_text, 1)
