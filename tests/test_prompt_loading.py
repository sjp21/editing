"""Ensure the Pass 1 rubric loads regardless of cwd or install location."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clipper.pass1 import PROMPT_VERSION, Pass1Config, _load_rubric


def test_default_rubric_loads_via_package_resources(tmp_path, monkeypatch):
    # Simulate being invoked from an arbitrary cwd (what happens when the
    # installed `clipper` command runs outside the repo root).
    monkeypatch.chdir(tmp_path)

    rubric = _load_rubric(Pass1Config().prompt_path)
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
