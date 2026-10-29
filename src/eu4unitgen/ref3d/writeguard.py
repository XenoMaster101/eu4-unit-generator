"""Where Track M runs may write: the project folder and the AI root (user rule, 2026-10-20, after a
test run wrote to ``C:\\c\\Users\\...``: a Git Bash path ``/c/Users/...`` given to a Windows python
resolves to a folder off the drive root).

Every output path a ref3d or AI job is given goes through ``check_writes`` before the job starts.
The roots: the project (EU4UNITGEN_PROJECT_ROOT, else the checkout this file is in) and the AI root
(EU4UNITGEN_AI_ROOT, else the folder holding the AI environment: ``<root>/<env>/Scripts/python.exe``
on Windows, ``<root>/<env>/bin/python`` on Linux).

Blender jobs too (the user, 2026-10-29, after 50 stray renders in ``C:\assets_in``: a run started as
``flow go assets_in/<name>`` handed Blender relative output paths, which it resolved against the drive's root).
``BlenderRunner.run_job`` makes every relative path in a job's arguments absolute (``absolute``) and checks its
write paths (``is_write_key``: ``WRITE_KEYS`` and any ``*_out``, ``out_*`` or ``*_dir``) against ``app_roots``: the
install root, the workspace and its projects, references, result sheets and exports, the EU4 mod folder, a source
checkout, the settings folder and the system's temporary folder.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# argument keys that name something a job writes (the job's own result file included)
WRITE_KEYS = {"out_dir", "output", "out", "result_path", "out_path", "out_glb", "work_dir"}


def is_write_key(k: str) -> bool:
    return k in WRITE_KEYS or k.endswith("_out") or k.endswith("_dir") or k.startswith("out_")


class WriteGuardError(RuntimeError):
    pass


def project_root() -> Path:
    env = os.environ.get("EU4UNITGEN_PROJECT_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[3]


def ai_root(python: Path | str | None = None) -> Path | None:
    env = os.environ.get("EU4UNITGEN_AI_ROOT")
    if env:
        return Path(env).resolve()
    return Path(python).resolve().parents[2] if python else None


def allowed_roots(python: Path | str | None = None) -> list[Path]:
    """The project, the workspace (settings: where an installed app keeps its runs) and the AI root."""
    from eu4unitgen.core import config
    roots = [project_root(), config.load().work(), ai_root(python)]
    out = []
    for r in roots:
        if r is not None and r not in out:
            out.append(r)
    return out


def app_roots() -> list[Path]:
    """Every folder the app writes in (module doc)."""
    from eu4unitgen.core import config, layout
    roots = [project_root(), Path(tempfile.gettempdir()), config.config_dir()]
    try:
        cfg = config.load()
        roots += [cfg.work(), cfg.projects(), cfg.references(), cfg.sheets(), cfg.exports(), cfg.mods()]
    except Exception:  # noqa: BLE001 - no settings yet: the other folders still count
        pass
    roots += [p for p in (layout.root(), config.checkout_root()) if p is not None]
    out = []
    for r in roots:
        r = Path(r).resolve()
        if r not in out:
            out.append(r)
    return out


def _pathlike(v: str) -> bool:
    return ("/" in v or "\\" in v) and "://" not in v and "\n" not in v and len(v) < 1024


def absolute(args):
    """``args`` with every relative path (a string holding a path separator) made absolute against the app's working
    folder, nested dicts and lists included."""
    if isinstance(args, dict):
        return {k: absolute(v) for k, v in args.items()}
    if isinstance(args, list):
        return [absolute(v) for v in args]
    if isinstance(args, str) and _pathlike(args) and not Path(args).is_absolute():
        return str(Path(args).resolve())
    return args


def is_inside(path: Path | str, roots: list[Path]) -> bool:
    p = Path(path).resolve()
    return any(p == r or r in p.parents for r in (Path(q).resolve() for q in roots))


def check_writes(args: dict, roots: list[Path], where: str = "") -> None:
    """Raise if any write path in ``args`` (WRITE_KEYS, nested dicts included) is outside ``roots``."""
    bad = []

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            elif is_write_key(str(k)) and isinstance(v, (str, Path)) and not is_inside(v, roots):
                bad.append(f"{prefix}{k} = {Path(v).resolve()}")

    walk(args)
    if bad:
        raise WriteGuardError(f"{where or 'job'} would write outside the allowed folders "
                              f"({', '.join(str(r) for r in roots)}): " + "; ".join(bad))
