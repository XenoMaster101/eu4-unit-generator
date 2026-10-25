"""Build the Windows package (Track P4): ``dist/eu4unitgen-<version>-windows.zip``.

Inside: the app's wheel (pip wheel, no dependencies: PySide6, Jinja2 and Pillow come from PyPI at install time),
Install.cmd and install.ps1 (Python found or installed, a venv, shortcuts), Uninstall.cmd and uninstall.ps1,
README.txt (the install steps) and LICENSE (GPL-3.0). No Paradox files, no AI weights, no third-party binaries.

usage: python packaging/windows/build_package.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def version() -> str:
    return re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)


def readme_txt(v: str) -> str:
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    install = md.split("## Install", 1)[1].split("\n## ", 1)[0].strip()
    need = md.split("## What you need", 1)[1].split("\n## ", 1)[0].strip()
    return (f"EU4 Unit Generator {v}\n\nWhat you need\n-------------\n{need}\n\nInstall\n-------\n"
            "0. If Windows marks the download as from the internet: right-click the zip, Properties, tick Unblock, OK "
            "(before unzipping).\n" + install + "\n\nTo remove the app: Uninstall.cmd (in %LOCALAPPDATA%\\eu4unitgen\\app, or "
            "in this folder). Your settings, runs and the AI setup are kept.\n\nMore: README.md in the source, "
            "docs/DEVELOPER.md, and the app's About and licences page.\n")


def main() -> Path:
    v = version()
    dist = ROOT / "dist"
    stage = dist / f"eu4unitgen-{v}-windows"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    wheels = dist / "wheel"
    if wheels.exists():
        shutil.rmtree(wheels)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w",
                    str(wheels)], check=True)
    wheel = next(wheels.glob(f"eu4unitgen-{v}-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    for must in ("eu4unitgen/setup_scripts/trellis2_setup.sh", "eu4unitgen/ai/jobs/trellis2_generate.py",
                 "eu4unitgen/blender/jobs/fresh_horse.py", "eu4unitgen/library/base_models.json"):
        assert must in names, f"{must} is missing from the wheel"
    assert not any(n.lower().endswith((".dds", ".mesh", ".anim")) for n in names), "game files in the wheel"
    shutil.copy(wheel, stage / wheel.name)
    for f in ("Install.cmd", "install.ps1", "Uninstall.cmd", "uninstall.ps1"):
        shutil.copy(HERE / f, stage / f)
    shutil.copy(ROOT / "LICENSE", stage / "LICENSE.txt")
    (stage / "README.txt").write_text(readme_txt(v), encoding="utf-8")
    out = dist / f"eu4unitgen-{v}-windows.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            z.write(p, Path(stage.name) / p.relative_to(stage))
    print(out, f"{out.stat().st_size / 2 ** 20:.1f} MB", [p.name for p in stage.iterdir()])
    return out


if __name__ == "__main__":
    main()
