"""Setting up a new machine (Track P4): the steps the first-launch setup wizard runs (gui/setup_wizard.py), each a
list of commands with a plain title, run with their output shown in the wizard: nothing is typed in a terminal.

  blender      Blender 4.2 LTS: found, or its official installer downloaded and run (Windows asks for administrator
               rights)
  io_pdx_mesh  the Blender extension that reads and writes EU4 .mesh files, downloaded and installed into Blender
  ai_folder    where the AI parts go (~60 GB): the drive with the most free space by default
  wsl          Windows' Linux layer (administrator rights; Windows restarts afterwards)
  ubuntu       Ubuntu 26.04 in WSL, set up to run as its root user (no Linux user or password to make)
  trellis_env  TRELLIS.2's code (commit 75fbf01) and its environment in Ubuntu: CUDA 12.9, then
               setup_scripts/trellis2_setup.sh (about 40 minutes)
  ai_env       the Windows AI environment (BiRefNet, SAM 2), made with the app's own Python
  weights      the AI weights (TRELLIS.2, its decoder, DINOv3 with the user's Hugging Face token, SAM 2.1) and
               BiRefNet (the URL rembg's own code downloads)

Pinned to what works on the author's machine (2026-10-25): Ubuntu 26.04, CUDA 12.9, TRELLIS.2 75fbf01, torch 2.14.1
cu126 for the Windows AI environment, Blender 4.2.23 LTS, io_pdx_mesh 0.91.
"""
from __future__ import annotations

import os
import shutil
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core import config

TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
DISTRO = "Ubuntu-26.04"
WEIGHTS = {"TRELLIS.2-4B": "microsoft/TRELLIS.2-4B", "TRELLIS-image-large": "microsoft/TRELLIS-image-large",
           "dinov3-vitl16-pretrain-lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
           "sam2.1-hiera-large": "facebook/sam2.1-hiera-large"}
GATED = {"dinov3-vitl16-pretrain-lvd1689m": "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"}
BIREFNET = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"  # rembg's own
AI_PACKAGES = ["torch==2.14.1", "torchvision==0.29.1"]
AI_PACKAGES_PLAIN = ["transformers==5.20.1", "rembg==2.0.69", "onnxruntime-gpu==1.23.2", "scipy==1.15.3",
                     "opencv-python-headless==5.0.0.93", "numpy==2.2.6", "pillow", "huggingface_hub==1.33.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
BLENDER_MSI = "https://download.blender.org/release/Blender4.2/blender-4.2.23-windows-x64.msi"  # checked 2026-10-25
IO_PDX_MESH = "https://github.com/ross-g/io_pdx_mesh/releases/download/0.91/blender-io_pdx_mesh.zip"  # checked
WSLCONFIG = "[wsl2]\nmemory=20GB\nswap=16GB\n\n[experimental]\nautoMemoryReclaim=dropCache\n"
ENV_NAME = "ai_env"
NEED_GB = 60


@dataclass
class Cmd:
    title: str
    argv: list[str] = field(default_factory=list)
    admin: bool = False  # run elevated (Windows asks); its output isn't shown, only whether it worked
    download: tuple[str, str] | None = None  # (url, file): done by the app with a progress bar, not a command
    env: dict | None = None


@dataclass
class Step:
    key: str
    title: str
    what: str
    needs: str = ""
    minutes: int = 1
    restart: bool = False


def drives() -> list[tuple[Path, float]]:
    """Fixed drives with their free space (GB), most free first."""
    out = []
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            p = Path(f"{letter}:/")
            try:
                if p.exists():
                    out.append((p, shutil.disk_usage(p).free / 2 ** 30))
            except OSError:
                pass
    else:
        out.append((Path.home(), shutil.disk_usage(Path.home()).free / 2 ** 30))
    return sorted(out, key=lambda t: -t[1])


def default_ai_root() -> Path:
    """The AI folder: the settings', else ``eu4unitgen_ai`` on the drive with the most free space."""
    r = config.load().ai_root_path()
    if r:
        return r
    d = drives()
    return (d[0][0] if d else Path.home()) / "eu4unitgen_ai"


def ai_root() -> Path:
    return default_ai_root()


def steps() -> list[Step]:
    return [
        Step("blender", "Blender 4.2 LTS", "The free 3D program the app runs in the background for every mesh step.",
             "administrator rights if it has to be installed (about 400 MB)", 5),
        Step("io_pdx_mesh", "The io_pdx_mesh add-on", "Lets Blender read and write EU4's .mesh files.", "", 1),
        Step("ai_folder", "Where the AI parts go", f"About {NEED_GB} GB: TRELLIS.2, its weights, DINOv3, SAM 2.1.", "", 0),
        Step("wsl", "Windows Subsystem for Linux", "Windows' own Linux layer; TRELLIS.2 runs in it.",
             "administrator rights; Windows restarts afterwards", 5, restart=True),
        Step("ubuntu", "Ubuntu in WSL", "Ubuntu 26.04, set up to run as its own administrator: no Linux user or "
             "password to make.", "about 1 GB", 5),
        Step("trellis_env", "TRELLIS.2", "Downloads TRELLIS.2 and builds its environment in Ubuntu (CUDA 12.9, PyTorch, "
             "its GPU extensions).", "about 40 minutes, 16 GB", 40),
        Step("ai_env", "The Windows AI environment", "Runs the cut-out (BiRefNet) and the part cut-outs (SAM 2).",
             "about 10 minutes, 5 GB", 10),
        Step("weights", "The AI weights", "TRELLIS.2, its decoder, DINOv3, SAM 2.1 and BiRefNet (about 20 GB).",
             "DINOv3 is gated: accept its licence on Hugging Face and paste a read token", 30),
    ]


def wsl_path(p: Path) -> str:
    from eu4unitgen.ref3d.trellis import wsl_path as wp
    return wp(p)


def app_python() -> str:
    """The Python the app runs on (the installed app's own; a checkout's venv)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").is_file():
        return str(exe.parent / "python.exe")
    return str(exe)


def blender_installed() -> Path | None:
    from eu4unitgen.core import install
    exe, _ = install.find_blender()
    return exe if exe and Path(exe).is_file() else None


def plan(key: str, token: str | None = None, root: Path | None = None) -> list[Cmd]:
    """The commands for one step (Windows)."""
    cfg = config.load()
    root = Path(root or default_ai_root())
    tmp = Path(os.environ.get("TEMP") or "/tmp")
    distro = cfg.wsl_distro or DISTRO
    wsl = ["wsl.exe", "-d", distro]
    scripts = Path(__file__).resolve().parents[1] / "setup_scripts"
    if key == "blender":
        msi = tmp / "blender-4.2.23-windows-x64.msi"
        return [Cmd("Download Blender 4.2.23 LTS", download=(BLENDER_MSI, str(msi))),
                Cmd("Install Blender (Windows asks for permission)", ["msiexec", "/i", str(msi), "/passive", "/norestart"],
                    admin=True)]
    if key == "io_pdx_mesh":
        z = tmp / "blender-io_pdx_mesh.zip"
        b = cfg.blender() or blender_installed()
        return [Cmd("Download io_pdx_mesh 0.91", download=(IO_PDX_MESH, str(z))),
                Cmd("Install it into Blender", [str(b), "--background", "--command", "extension", "install-file", "-r",
                                                 "user_default", "-e", str(z)])]
    if key == "wsl":
        return [Cmd("Install WSL (Windows asks for permission)", ["wsl.exe", "--install", "--no-distribution"], admin=True)]
    if key == "ubuntu":
        p = Path.home() / ".wslconfig"
        cmds = [Cmd(f"Install {distro}", ["wsl.exe", "--install", "-d", distro, "--no-launch"]),
                Cmd("Start it as its administrator (no user to make)", wsl + ["-u", "root", "-e", "true"])]
        if not p.is_file():
            cmds.insert(0, Cmd("Give WSL 20 GB of memory (.wslconfig)", ["powershell", "-NoProfile", "-Command",
                               f"Set-Content -Path '{p}' -Value \"{WSLCONFIG.replace(chr(10), '`n')}\""]))
        return cmds
    if key == "trellis_env":
        r = wsl_path(root)
        root_step = ("apt-get update && apt-get install -y build-essential gcc-13 g++-13 git libjpeg-dev curl && "
                     "(dpkg -s cuda-keyring >/dev/null 2>&1 || (curl -fsSLo /tmp/cuda-keyring.deb "
                     "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && "
                     "dpkg -i /tmp/cuda-keyring.deb && apt-get update)) && apt-get install -y cuda-toolkit-12-9 && "
                     f"bash '{wsl_path(scripts)}/cuda_glibc_patch.sh'")
        return [Cmd("Download TRELLIS.2 (commit 75fbf01)", wsl + ["-u", "root", "-e", "bash", "-lc",
                    f"mkdir -p '{r}' && cd '{r}' && ([ -d TRELLIS.2 ] || git clone {TRELLIS_REPO} TRELLIS.2) && cd TRELLIS.2 "
                    f"&& git fetch -q origin && git checkout -q {TRELLIS_COMMIT} && echo TRELLIS.2 ready"]),
                Cmd("Install the build tools and CUDA 12.9", wsl + ["-u", "root", "-e", "bash", "-lc", root_step]),
                Cmd("Build the TRELLIS.2 environment (about 40 minutes)", wsl + ["-e", "bash", "-lc",
                    f"AI_WIN='{r}' bash '{wsl_path(scripts)}/trellis2_setup.sh'"])]
    if key == "ai_env":
        env = root / ENV_NAME
        py = env / "Scripts" / "python.exe"
        return [Cmd("Make the environment", [app_python(), "-m", "venv", str(env)]),
                Cmd("Update pip", [str(py), "-m", "pip", "install", "--upgrade", "pip"]),
                Cmd("Install PyTorch with CUDA (about 3 GB)", [str(py), "-m", "pip", "install", *AI_PACKAGES,
                                                                "--index-url", TORCH_INDEX]),
                Cmd("Install the cut-out and SAM 2 libraries", [str(py), "-m", "pip", "install", *AI_PACKAGES_PLAIN])]
    if key == "weights":
        py = Path(cfg.ai() or root / ENV_NAME / "Scripts" / "python.exe")
        job = Path(__file__).resolve().parents[1] / "ai" / "jobs" / "download_weights.py"
        return [Cmd("Download BiRefNet (the cut-out model)", download=(BIREFNET, str(root / "rembg_models" / "birefnet-general.onnx"))),
                Cmd("Download TRELLIS.2, its decoder, DINOv3 and SAM 2.1 (about 19 GB)",
                    [str(py), str(job), str(root)] + [f"{n}={r}" for n, r in WEIGHTS.items()],
                    env={"HF_TOKEN": token} if token else None)]
    raise KeyError(key)


def state() -> dict[str, bool]:
    """Which steps look done."""
    from eu4unitgen.core import checks
    cfg = config.load()
    root = default_ai_root()
    out = {}
    b = cfg.blender() or blender_installed()
    out["blender"] = bool(b) and Path(b).is_file()
    ext = Path(os.environ.get("APPDATA", "")) / "Blender Foundation" / "Blender" / "4.2" / "extensions" / "user_default" / "io_pdx_mesh"
    out["io_pdx_mesh"] = ext.is_dir()
    out["ai_folder"] = bool(cfg.ai_root)
    wsl = checks.check_wsl()
    out["wsl"] = "isn't installed" not in wsl.what
    out["ubuntu"] = wsl.ok or "environment isn't set up" in wsl.what
    out["trellis_env"] = wsl.ok and (root / "TRELLIS.2" / "trellis2").is_dir()
    py = cfg.ai() or str(root / ENV_NAME / "Scripts" / "python.exe")
    out["ai_env"] = Path(py).is_file()
    out["weights"] = all((root / "weights" / n).is_dir() for n in WEIGHTS) and (root / "rembg_models").is_dir()
    return out


def adopt(root: Path) -> dict:
    """After a step: what it made written into the settings (the AI folder, its environment, Blender)."""
    found = {"ai_root": str(root)}
    py = root / ENV_NAME / "Scripts" / "python.exe"
    if py.is_file() and not config.load().ai():
        found["ai_python"] = str(py)
    b = blender_installed()
    if b and not config.load().blender_path:
        found["blender_path"] = str(b)
    config.update(**found)
    return found


def adopt_ai_env() -> str | None:
    """The AI environment this setup made, into the settings when none is set. -> the python or None."""
    py = default_ai_root() / ENV_NAME / "Scripts" / "python.exe"
    if py.is_file() and not config.load().ai():
        config.update(ai_python=str(py), ai_root=str(default_ai_root()))
        return str(py)
    return None
