"""Build the Windows downloads (Track P5): ``dist/eu4unitgen-<version>-windows.zip`` and ``dist/eu4unitgen-setup.exe``.

The zip unzips to one folder holding only:
  Install EU4 Unit Generator.cmd   the one thing to click (opens files\\install.ps1's wizard window)
  README.txt                       the README
  LICENSE.txt                      GPL-3.0
  files\\                           the app's wheel, python.org's signed NuGet Python (the app's own: none needed on the
                                   PC), install.ps1, uninstall.ps1, the icon
No Paradox files, no AI weights.

eu4unitgen-setup.exe: Inno Setup's wizard around the same files and install.ps1 (setup.iss); built when Inno Setup's
ISCC is found (``.tools/innosetup`` in the checkout, or on PATH). It isn't code-signed: Smart App Control refuses it.

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
TOP = "EU4 Unit Generator"  # the folder the zip unzips to
CLICK = "Install EU4 Unit Generator.cmd"
PYTHON = "3.12.10"  # python.org's NuGet build: signed by the Python Software Foundation, with venv and pip
PYTHON_URL = f"https://api.nuget.org/v3-flatcontainer/python/{PYTHON}/python.{PYTHON}.nupkg"


def version() -> str:
    return re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)


def readme_txt(v: str) -> str:
    """The README as plain text (it is written to read well as is), with the version on top."""
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    return f"EU4 Unit Generator {v}\n\n" + md.split("\n", 1)[1].lstrip()


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


def iscc() -> Path | None:
    for c in (ROOT / ".tools" / "innosetup" / "ISCC.exe", shutil.which("ISCC") and Path(shutil.which("ISCC"))):
        if c and Path(c).is_file():
            return Path(c)
    return None


def main() -> dict:
    v = version()
    dist = ROOT / "dist"
    stage = dist / "stage" / TOP
    if stage.parent.exists():
        shutil.rmtree(stage.parent)
    files = stage / "files"
    files.mkdir(parents=True)
    wheels = dist / "wheel"
    if wheels.exists():
        shutil.rmtree(wheels)
    # the starter library from the author's finished runs (raw TRELLIS.2 models, tags, rights; not in git)
    starter = ROOT / "src" / "eu4unitgen" / "library" / "starter"
    if starter.exists():
        shutil.rmtree(starter)
    sys.path.insert(0, str(ROOT / "src"))
    from eu4unitgen.core import library
    packed = library.pack_starters(starter)
    print("starter library:", packed)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(wheels)], check=True)
    wheel = next(wheels.glob(f"eu4unitgen-{v}-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    for must in ("eu4unitgen/setup_scripts/trellis2_setup.sh", "eu4unitgen/ai/jobs/trellis2_generate.py",
                 "eu4unitgen/blender/jobs/fresh_horse.py", "eu4unitgen/library/base_models.json",
                 "eu4unitgen/gui/icon.png", "eu4unitgen/gui/eu4unitgen.ico", "eu4unitgen/core/layout.py",
                 "eu4unitgen/core/example_pictures/RIGHTS.txt"):
        assert must in names, f"{must} is missing from the wheel"
    # the only pictures: the app's icon, the Example's three and the starter library's (the author's, bundling
    # allowed 2026-10-28: their pictures, cut-outs, part crops and raw renders)
    pics = {n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            and not n.startswith("eu4unitgen/library/starter/")}
    assert pics == {"eu4unitgen/gui/icon.png"} | {f"eu4unitgen/core/example_pictures/{n}" for n in (
        "cataphract_rider.png", "cataphract_side_clean.png", "horse2_light_cavalry.png")}, sorted(pics)
    assert not any(n.lower().endswith((".dds", ".mesh", ".anim", ".asset", ".gfx")) for n in names), "game files in the wheel"
    got = {n.split("/")[3] for n in names if n.startswith("eu4unitgen/library/starter/") and n.endswith("/part.json")}
    assert got == {k for k, v in packed.items() if v == "packed"}, f"starter parts in the wheel: {sorted(got)}"
    shutil.copy(wheel, files / wheel.name)
    for f in ("install.ps1", "uninstall.ps1"):
        shutil.copy(HERE / f, files / f)
    shutil.copy(ROOT / "src" / "eu4unitgen" / "gui" / "eu4unitgen.ico", files / "eu4unitgen.ico")
    shutil.copy(ROOT / "src" / "eu4unitgen" / "gui" / "icon.png", files / "icon.png")
    shutil.copy(python_package(dist), files / f"python.{PYTHON}.nupkg")
    shutil.copy(HERE / CLICK, stage / CLICK)
    shutil.copy(ROOT / "LICENSE", stage / "LICENSE.txt")
    (stage / "README.txt").write_text(readme_txt(v), encoding="utf-8")
    assert sorted(p.name for p in stage.iterdir()) == sorted([CLICK, "README.txt", "LICENSE.txt", "files"])
    out = dist / f"eu4unitgen-{v}-windows.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            z.write(p, Path(TOP) / p.relative_to(stage))
    built = {"zip": out}
    print(out, f"{out.stat().st_size / 2 ** 20:.1f} MB")
    exe = iscc()
    if exe is None:
        print("Inno Setup's ISCC not found: eu4unitgen-setup.exe not built")
    else:
        subprocess.run([str(exe), "/Q", f"/DAppVersion={v}", f"/DFilesDir={files}", f"/DLicense={stage / 'LICENSE.txt'}",
                        f"/DOutDir={dist}", str(HERE / "setup.iss")], check=True)
        built["setup"] = dist / "eu4unitgen-setup.exe"
        print(built["setup"], f"{built['setup'].stat().st_size / 2 ** 20:.1f} MB")
    return built


if __name__ == "__main__":
    main()
