"""Ensure the score rubric loads regardless of cwd or install location."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clipper.score import PROMPT_VERSION, ScoreConfig, _load_rubric


def test_default_rubric_loads_via_package_resources(tmp_path, monkeypatch):
    # Simulate being invoked from an arbitrary cwd (what happens when the
    # installed `clipper` command runs outside the repo root).
    monkeypatch.chdir(tmp_path)

    rubric = _load_rubric(ScoreConfig().prompt_path)
    assert isinstance(rubric, str)
    assert rubric.strip(), "rubric must not be empty"


def test_explicit_prompt_path_honored(tmp_path):
    custom = tmp_path / "custom.md"
    custom.write_text("CUSTOM RUBRIC")

    rubric = _load_rubric(custom)
    assert rubric == "CUSTOM RUBRIC"


def test_prompt_bundled_in_package():
    from importlib import resources

    path = resources.files("clipper").joinpath(
        f"prompts/{PROMPT_VERSION}/candidate_identification.md"
    )
    assert path.is_file()


def test_prompt_version_is_v2():
    """PROMPT_VERSION is pinned to v2 (cache-threshold-crossing rubric)."""
    assert PROMPT_VERSION == "v2"


def test_v1_prompt_still_bundled():
    """v1 must stay shipped so old candidates.json files remain joinable."""
    from importlib import resources

    path = resources.files("clipper").joinpath("prompts/v1/candidate_identification.md")
    assert path.is_file()


# ---------------------------------------------------------------------------
# Ranker prompt + SOUL.md loading from the installed package
# ---------------------------------------------------------------------------


def test_ranking_prompt_bundled_in_package():
    from importlib import resources

    path = resources.files("clipper").joinpath("prompts/v1/ranking.md")
    assert path.is_file()


def test_soul_bundled_in_package(tmp_path, monkeypatch):
    from importlib import resources

    # Resolve from an arbitrary cwd — same guardrail as the candidate
    # identification prompt: the installed CLI must find SOUL.md regardless
    # of where it was invoked.
    monkeypatch.chdir(tmp_path)
    path = resources.files("clipper").joinpath("SOUL.md")
    assert path.is_file()


def test_shorts_profile_bundled_in_package():
    """The one shipped platform profile must travel with the installed wheel."""
    from importlib import resources

    path = resources.files("clipper").joinpath("prompts/v1/profiles/shorts.md")
    assert path.is_file()


def test_ranking_prompt_mentions_required_fields():
    """Regex guard against drift between ranker.py/schemas.py and the rubric."""
    from importlib import resources

    body = resources.files("clipper").joinpath("prompts/v1/ranking.md").read_text(
        encoding="utf-8"
    )
    for token in [
        "hook_quote",
        "payoff_in_window",
        "comparative_reason",
        "reason_kind",
        "[PLATFORM_PROFILE]",
    ]:
        assert token in body, f"ranking.md missing required token: {token!r}"
