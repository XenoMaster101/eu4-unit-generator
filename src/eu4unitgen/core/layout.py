"""The app's one root folder (Track P5): everything the installed app needs lives under a single folder chosen at
install (default ``D:\\EU4 Unit Generator``); nothing else on the PC but the shortcuts and the Apps-list entry.

  app\\       the program and its own signed Python (``app\\layout.json`` marks the root)
  tools\\     portable Blender 4.2 with io_pdx_mesh (Blender's ``portable`` folder keeps its settings inside)
  ai\\        the Windows AI environment, the TRELLIS.2 code, the models and weights, the Hugging Face cache
  wsl\\       the WSL distribution TRELLIS.2 runs in (its disk), registered as ``EU4UnitGen``
  work\\      references, projects, result sheets, exported mods
  settings\\  settings.json, the window's state (gui.ini), logs, and the temp folder (tmp\\: emptied at start)

The root is found, in order: ``EU4UNITGEN_ROOT``; the installed app's own folder (a parent holding ``app\\layout.json``);
a source checkout's ``.eu4unitgen-root`` file (one line: the root), so the developer's checkout uses the same tools,
AI parts and settings. None: no root (an older install, Linux, the tests), and the paths fall back to ``core.config``'s.

Two things can't live in the root, both Windows' own: the WSL feature itself and ``%USERPROFILE%\\.wslconfig`` (WSL reads
it only there). The marker records whether the app wrote ``.wslconfig`` and registered the distribution, so the
uninstaller removes only what the app made.
"""
from __future__ import annotations

import json
import os
import shutil
import string
import tempfile
from pathlib import Path

MARKER = "layout.json"  # in <root>/app
POINTER = ".eu4unitgen-root"  # in a source checkout
FOLDERS = ("app", "tools", "ai", "wsl", "work", "settings")
DEFAULT_NAME = "EU4 Unit Generator"
DISTRO = "EU4UnitGen"


def _checkout() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "eu4unitgen").is_dir() else None


def root() -> Path | None:
    env = os.environ.get("EU4UNITGEN_ROOT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "app" / MARKER).is_file():
            return p
    co = _checkout()
    if co is not None and (co / POINTER).is_file():
        line = (co / POINTER).read_text(encoding="utf-8").strip()
        if line:
            return Path(line)
    return None


def folder(name: str) -> Path | None:
    r = root()
    return r / name if r is not None else None


def marker(r: Path | None = None) -> dict:
    r = r or root()
    try:
        return json.loads((r / "app" / MARKER).read_text(encoding="utf-8")) if r else {}
    except (OSError, ValueError):
        return {}


def update_marker(**values) -> dict:
    """Record what the app made outside its files (the WSL distribution, ``.wslconfig``) for the uninstaller."""
    r = root()
    if r is None:
        return {}
    m = {**marker(r), **values}
    p = r / "app" / MARKER
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def default_root() -> Path:
    """Where a new install goes by default: ``D:\\EU4 Unit Generator`` when there is a D: drive, else the fixed drive
    with the most free space; ``~/EU4 Unit Generator`` off Windows."""
    if os.name != "nt":
        return Path.home() / DEFAULT_NAME
    if Path("D:/").exists():
        return Path("D:/") / DEFAULT_NAME
    best = max((Path(f"{c}:/") for c in string.ascii_uppercase if Path(f"{c}:/").exists()),
               key=lambda p: shutil.disk_usage(p).free, default=Path("C:/"))
    return best / DEFAULT_NAME


def blender_exe(r: Path | None = None) -> Path | None:
    r = r or root()
    if r is None:
        return None
    return r / "tools" / "blender" / ("blender.exe" if os.name == "nt" else "blender")


def use_root_temp() -> Path | None:
    """Point this process's and its children's temp folder, pip's and Hugging Face's caches into the root, so the app
    leaves nothing in %TEMP% or the user profile. Called once at start. -> the temp folder or None (no root)."""
    r = root()
    if r is None:
        return None
    tmp = r / "settings" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    import time
    for p in tmp.iterdir():  # what earlier runs left (a day old: another window's running job keeps its own)
        try:
            if time.time() - p.stat().st_mtime > 86400:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
        except OSError:
            pass
    for k in ("TEMP", "TMP", "TMPDIR"):
        os.environ[k] = str(tmp)
    os.environ.setdefault("PIP_CACHE_DIR", str(tmp / "pip"))
    os.environ.setdefault("HF_HOME", str(r / "ai" / "hf_home"))
    tempfile.tempdir = str(tmp)
    return tmp
