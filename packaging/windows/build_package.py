"""Build the Windows package (Track P4): ``dist/eu4unitgen-<version>-windows.zip``.

Inside: the app's wheel (PySide6, Jinja2 and Pillow come from PyPI at install time), python.org's signed NuGet Python
(the app's own Python: none needed on the PC), Install.cmd and install.ps1 (a wizard window: licence, location,
shortcuts, progress, an entry in Windows' Apps list), Uninstall.cmd and uninstall.ps1 (asks before deleting your work
or the AI parts), the icon, README.txt and LICENSE.txt (GPL-3.0). No Paradox files, no AI weights.

Why not one setup.exe: a self-built .exe is unsigned, and Smart App Control (on, on the author's PC) refuses
unsigned programs; PowerShell and python.org's Python are signed. The PyInstaller/Inno Setup scripts for a signed
build are in packaging/windows/exe/ (SPEC: the signing options).

usage: python packaging/windows/build_package.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = "3.12.10"  # python.org's NuGet build: signed by the Python Software Foundation, with venv and pip
PYTHON_URL = f"https://api.nuget.org/v3-flatcontainer/python/{PYTHON}/python.{PYTHON}.nupkg"


def version() -> str:
    return re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)


def readme_txt(v: str) -> str:
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    install = md.split("## Install", 1)[1].split("\n## ", 1)[0].strip()
    need = md.split("## What you need", 1)[1].split("\n## ", 1)[0].strip()
    return (f"EU4 Unit Generator {v}\n\nWhat you need\n-------------\n{need}\n\nInstall\n-------\n" + install
            + "\n\nTo remove the app: Windows Settings > Apps > EU4 Unit Generator > Uninstall. It asks before deleting "
            "your work or the AI parts.\n\nMore: the app's About and licences page; docs/DEVELOPER.md in the source.\n")


def python_package(dist: Path) -> Path:
    """python.org's NuGet Python (downloaded once into dist/), refused unless its python.exe is validly signed."""
    p = dist / f"python.{PYTHON}.nupkg"
    if not p.is_file():
        dist.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PYTHON_URL, p)
    if sys.platform == "win32":
        tmp = dist / "python_check"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(p) as z:
            z.extract("tools/python.exe", tmp)
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(Get-AuthenticodeSignature '{tmp / 'tools' / 'python.exe'}').Status"],
                           capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
        if r.stdout.strip() != "Valid":
            raise SystemExit(f"{p.name}: python.exe isn't validly signed ({r.stdout.strip()}); not packaged")
    return p


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
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(wheels)], check=True)
    wheel = next(wheels.glob(f"eu4unitgen-{v}-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    for must in ("eu4unitgen/setup_scripts/trellis2_setup.sh", "eu4unitgen/ai/jobs/trellis2_generate.py",
                 "eu4unitgen/blender/jobs/fresh_horse.py", "eu4unitgen/library/base_models.json",
                 "eu4unitgen/gui/icon.png", "eu4unitgen/gui/eu4unitgen.ico"):
        assert must in names, f"{must} is missing from the wheel"
    assert not any(n.lower().endswith((".dds", ".mesh", ".anim")) for n in names), "game files in the wheel"
    shutil.copy(wheel, stage / wheel.name)
    for f in ("Install.cmd", "install.ps1", "Uninstall.cmd", "uninstall.ps1"):
        shutil.copy(HERE / f, stage / f)
    shutil.copy(ROOT / "src" / "eu4unitgen" / "gui" / "eu4unitgen.ico", stage / "eu4unitgen.ico")
    shutil.copy(python_package(dist), stage / f"python.{PYTHON}.nupkg")
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
