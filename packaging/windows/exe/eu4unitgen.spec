# PyInstaller spec for a signed eu4unitgen.exe (Track P4). NOT used today: Smart App Control refuses unsigned .exe
# files, so the shipped package is install.ps1 with python.org's signed Python (packaging/windows/build_package.py).
# Use this once there is a code-signing certificate (SPEC §P4: the signing options).
#
# Build (from the repo root, in the app's venv with PyInstaller installed):
#   pyinstaller packaging/windows/exe/eu4unitgen.spec --noconfirm
# Output: dist/eu4unitgen/ (a folder: eu4unitgen.exe and _internal/). Sign eu4unitgen.exe, then build the installer
# with Inno Setup (eu4unitgen.iss) and sign that too.
#
# Before shipping a frozen build, two things to change (TODO):
#   * core/setup.app_python() makes the Windows AI environment from sys.executable; in a frozen build that's
#     eu4unitgen.exe, which can't make a venv. Ship python.org's NuGet Python next to the exe (as the package does)
#     and point app_python() at it.
#   * Blender jobs, AI jobs and the WSL setup scripts are run as files: they are collected as data below, so they
#     sit under _internal/eu4unitgen/ and Path(__file__) lookups keep working (check with a test run).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).resolve().parents[2]
SRC = ROOT / "src"

datas = collect_data_files("eu4unitgen", include_py_files=True, excludes=["**/__pycache__/**"])

a = Analysis(
    [str(Path(SPECPATH) / "launch.py")],
    pathex=[str(SRC)],
    datas=datas,
    hiddenimports=["eu4unitgen.gui.app", "eu4unitgen.gui.setup_wizard"],
    excludes=["tkinter", "numpy", "torch"],  # the AI side runs in its own environments, never in the app
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="eu4unitgen",
    icon=str(SRC / "eu4unitgen" / "gui" / "eu4unitgen.ico"),
    console=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="eu4unitgen")
