"""Where Track M runs may write: the project folder and the AI root (user rule, 2026-10-20, after a
test run wrote to ``C:\\c\\Users\\...``: a Git Bash path ``/c/Users/...`` given to a Windows python
resolves to a folder off the drive root).

Every output path a ref3d or AI job is given goes through ``check_writes`` before the job starts.
The roots: the project (EU4UNITGEN_PROJECT_ROOT, else the checkout this file is in) and the AI root
(EU4UNITGEN_AI_ROOT, else the folder holding the AI environment: ``<root>/<env>/Scripts/python.exe``
on Windows, ``<root>/<env>/bin/python`` on Linux).
"""
from __future__ import annotations

import os
from pathlib import Path

# argument keys that name something a job writes (the job's own result file included)
WRITE_KEYS = {"out_dir", "output", "out", "result_path", "out_path", "out_glb", "work_dir"}


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
    return [r for r in (project_root(), ai_root(python)) if r is not None]


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
            elif k in WRITE_KEYS and isinstance(v, (str, Path)) and not is_inside(v, roots):
                bad.append(f"{prefix}{k} = {Path(v).resolve()}")

    walk(args)
    if bad:
        raise WriteGuardError(f"{where or 'job'} would write outside the allowed folders "
                              f"({', '.join(str(r) for r in roots)}): " + "; ".join(bad))
