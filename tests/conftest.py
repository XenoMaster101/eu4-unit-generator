"""Shared fixtures: a synthetic EU4 install, a project pointing at a temp mod folder, MAC."""
from __future__ import annotations

from pathlib import Path

import pytest

from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.library.scanner import GameFiles, build_index
from fakes import build_fake_install


@pytest.fixture
def fake_install(tmp_path: Path) -> Path:
    return build_fake_install(tmp_path / "eu4_full")


@pytest.fixture
def game(fake_install: Path):
    files = GameFiles(fake_install)
    yield files, build_index(files)
    files.close()


@pytest.fixture
def mod_root(tmp_path: Path) -> Path:
    return tmp_path / "mod" / "hellenistic_revolution"


@pytest.fixture
def project(fake_install: Path, mod_root: Path) -> Project:
    return Project(mod_root=mod_root, eu4_path=fake_install)


@pytest.fixture
def nation() -> Nation:
    return Nation("MAC", "hellenicgfx", name="Macedon", adjective="Macedonian", colour=(120, 30, 140))
