"""Shared fixtures: a synthetic EU4 install, a project pointing at a temp mod folder, MAC."""
from __future__ import annotations

import os

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
    return Nation("MAC", "hellenicgfx", name="Macedon", adjective="Macedonian", colour=(120, 30, 140),
                  unit_colours=((150, 20, 30), (210, 170, 40), (25, 35, 90)))


@pytest.fixture(autouse=True)
def _private_config(tmp_path, monkeypatch):
    """Tests never read or write the user's real config folder (their own vocabulary.json)."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # nor the installed app's root (core.layout): the checkout's .eu4unitgen-root points at the real one
    from eu4unitgen.core import layout
    real = layout.blender_exe() if not os.environ.get("BLENDER_PATH") else None
    monkeypatch.delenv("EU4UNITGEN_ROOT", raising=False)
    monkeypatch.setattr(layout, "_checkout", lambda: None)
    if real is not None and real.is_file():  # its portable Blender only (the tests that run real Blender jobs)
        monkeypatch.setenv("BLENDER_PATH", str(real))
