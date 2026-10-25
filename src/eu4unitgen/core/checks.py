"""Checks before a run (Track P2.4): each says what is wrong and how to fix it, in plain words.

  ram        free RAM against what TRELLIS.2 needs in WSL (about 19 GB at its peak with 20 GB given to WSL)
  eu4        EU4 must be closed: GPU jobs never run while it is open (it holds the GPU and RAM)
  disk       free space where runs are kept and on C: (WSL's disk lives there)
  blender    Blender 4.2 LTS with io_pdx_mesh
  eu4_path   the EU4 install
  ai_env     the Windows AI environment's python (cut-out, SAM 2)
  weights    the AI weights under the AI root (TRELLIS.2, its decoder, DINOv3, SAM 2.1, BiRefNet)
  wsl        WSL installed, the distribution there, and the TRELLIS.2 environment's python inside it

``run_all()`` -> [Check]; ``blocking(checks)`` the ones that stop a run. Standard library only.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from eu4unitgen.core import config

NEED_RAM_GB = 14.0  # free host RAM for a TRELLIS.2 run (WSL takes up to 20 GB; Windows keeps ~2 GB)
NEED_DISK_GB = {"work": 3.0, "c": 20.0}
WEIGHTS = {"TRELLIS.2-4B": "TRELLIS.2 (the 3D model, MIT)", "TRELLIS-image-large": "TRELLIS's decoder (MIT)",
           "dinov3-vitl16-pretrain-lvd1689m": "DINOv3 (the image encoder; gated: accept its licence on Hugging Face)",
           "sam2.1-hiera-large": "SAM 2.1 (part cut-outs, Apache-2.0)"}
EU4_EXE = ("eu4.exe", "eu4")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Check:
    key: str
    ok: bool
    what: str  # what was found, in plain words
    fix: str = ""  # what to do when it isn't ok
    blocks: bool = True  # a failed check stops the run (else a warning)

    def as_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 20) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, str(exc)
    out = r.stdout or b""
    # wsl.exe answers in UTF-16
    text = out.decode("utf-16-le", errors="ignore") if out[:2] in (b"\xff\xfe",) or b"\x00" in out[:20] else out.decode(errors="ignore")
    return r.returncode, text


def top_programs(n: int) -> list[tuple[str, float]]:
    """The ``n`` programs using the most RAM (summed by name), or [] if the system won't say."""
    use: dict[str, float] = {}
    try:
        if sys.platform == "win32":
            r = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=20,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in r.stdout.splitlines():
                f = [c.strip('"') for c in line.split('","')]
                if len(f) >= 5:
                    kb = int("".join(ch for ch in f[4] if ch.isdigit()) or 0)
                    use[f[0]] = use.get(f[0], 0) + kb / 2 ** 20
        else:
            r = subprocess.run(["ps", "-eo", "rss=,comm="], capture_output=True, text=True, timeout=20)
            for line in r.stdout.splitlines():
                kb, _, name = line.strip().partition(" ")
                if kb.isdigit():
                    use[name] = use.get(name, 0) + int(kb) / 2 ** 20
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return sorted(use.items(), key=lambda kv: -kv[1])[:n]


def eu4_running() -> bool:
    if sys.platform == "win32":
        _, out = _run(["tasklist", "/fo", "csv", "/nh"])
        return any(line.split(",")[0].strip('"').lower() == "eu4.exe" for line in out.splitlines() if line)
    _, out = _run(["ps", "-eo", "comm="])
    return any(line.strip().lower() in EU4_EXE for line in out.splitlines())


def check_ram() -> Check:
    from eu4unitgen.ref3d import memory
    total, free = memory.ram_gb()
    if free >= NEED_RAM_GB:
        return Check("ram", True, f"{free:.1f} of {total:.1f} GB RAM free.")
    top = ", ".join(f"{n} ({gb:.1f} GB)" for n, gb in top_programs(4))
    return Check("ram", False, f"Only {free:.1f} of {total:.1f} GB RAM is free; a run needs about {NEED_RAM_GB:.0f} GB.",
                 "Close programs you don't need" + (f" (the largest now: {top})" if top else "") + ", then check again.",
                 blocks=False)


def check_eu4() -> Check:
    if eu4_running():
        return Check("eu4", False, "EU4 is running.", "Close EU4 first: the 3D steps need the graphics card and RAM it holds. "
                     "A run already going waits until EU4 closes.")
    return Check("eu4", True, "EU4 is closed.")


def check_disk() -> list[Check]:
    out = []
    work = config.load().work()
    for key, path, need in (("disk_work", work, NEED_DISK_GB["work"]), ("disk_c", Path("C:/") if sys.platform == "win32" else Path.home(), NEED_DISK_GB["c"])):
        p = path
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            free = shutil.disk_usage(p).free / 2 ** 30
        except OSError:
            continue
        label = "where runs are kept" if key == "disk_work" else "on C: (WSL's disk)"
        if free >= need:
            out.append(Check(key, True, f"{free:.0f} GB free {label}."))
        else:
            out.append(Check(key, False, f"Only {free:.1f} GB free {label}; about {need:.0f} GB is wanted.",
                             "Free some space (empty the Recycle Bin, remove old runs from the app's list)."))
    return out


def check_blender() -> Check:
    b = config.load().blender()
    if b and Path(b).is_file():
        return Check("blender", True, f"Blender found ({b}).")
    return Check("blender", False, "Blender 4.2 LTS wasn't found.",
                 "Install Blender 4.2 LTS from blender.org (the installer's default folder), or set its path in Settings. "
                 "The Setup screen then installs the io_pdx_mesh add-on.")


def check_eu4_path() -> Check:
    p = config.load().eu4()
    if p and (Path(p) / "common").is_dir():
        return Check("eu4_path", True, f"EU4 install found ({p}).")
    return Check("eu4_path", False, "The EU4 install isn't set.",
                 "In Settings, point to the folder that holds eu4.exe (Steam: steamapps/common/Europa Universalis IV).")


def check_ai() -> list[Check]:
    cfg = config.load()
    py = cfg.ai()
    out = []
    if not py or not Path(py).is_file():
        out.append(Check("ai_env", False, "The AI environment wasn't found.",
                         "Run the Setup screen's \"Set up the AI environment\" (or point Settings to its python.exe)."))
        return out
    out.append(Check("ai_env", True, f"AI environment found ({py})."))
    root = cfg.ai_root_path()
    missing = [desc for name, desc in WEIGHTS.items() if not (root / "weights" / name).is_dir()] if root else list(WEIGHTS.values())
    if missing:
        out.append(Check("weights", False, "Missing AI weights: " + "; ".join(missing) + ".",
                         "Open the Setup screen and press \"Download weights\". DINOv3 needs you to accept its licence on "
                         "Hugging Face first (the screen shows the link)."))
    else:
        out.append(Check("weights", True, "All AI weights are present."))
    return out


def check_wsl() -> Check:
    from eu4unitgen.ref3d.trellis import sh_python
    if sys.platform != "win32":
        return Check("wsl", True, "Linux: TRELLIS.2 runs natively.", blocks=False)
    cfg = config.load()
    rc, out = _run(["wsl.exe", "--status"], 30)
    if rc != 0 and "default" not in out.lower():
        return Check("wsl", False, "WSL isn't installed.",
                     "Open the Setup screen and press \"Install WSL\" (it asks for administrator rights; Windows then "
                     "needs a restart).")
    rc, out = _run(cfg.wsl_base() + ["-e", "bash", "-lc", f"test -x {sh_python(cfg.wsl_python)} && echo ok"], 60)
    if "ok" not in out:
        return Check("wsl", False, "WSL is there, but the TRELLIS.2 environment isn't set up in it.",
                     "Open the Setup screen and press \"Set up TRELLIS.2\" (about 40 minutes; it asks for your Linux "
                     "password once).")
    return Check("wsl", True, "WSL and the TRELLIS.2 environment are ready.")


def run_all(quick: bool = False) -> list[Check]:
    """Every check (``quick``: skip WSL, the slowest)."""
    out = [check_eu4_path(), check_blender(), *check_ai(), check_eu4(), check_ram(), *check_disk()]
    if not quick:
        out.append(check_wsl())
    return out


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok and c.blocks]
