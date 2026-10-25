"""Setting up the AI side on a new machine (Track P4): each step a command the Setup screen runs in a window of its
own, so the user sees progress and never types a command. Steps that need administrator rights or a restart say so
first; the only things typed are the Linux user's password (Ubuntu asks for it) and, for DINOv3, a Hugging Face
token pasted into the app.

Pinned to what works on the author's machine (2026-10-24):
  * the Windows AI environment: Python 3.10 (uv), torch 2.14.1+cu126, torchvision 0.29.1, transformers 5.20.1,
    rembg 2.0.69, onnxruntime 1.23.2, scipy 1.15.3, opencv-python 5.0.0, numpy 2.2.6, huggingface_hub 1.33.0;
  * TRELLIS.2 at commit 75fbf0183001ed9876c8dbb35de6b68552ee08bd, its environment in WSL from
    tools/wsl/trellis2_setup.sh (CUDA 12.9, torch 2.7.0 cu128);
  * the weights: microsoft/TRELLIS.2-4B, microsoft/TRELLIS-image-large, facebook/dinov3-vitl16-pretrain-lvd1689m
    (gated), facebook/sam2.1-hiera-large; BiRefNet-general (rembg fetches it on first use).

``steps()`` -> [Step]; ``command(step)`` -> the argv the Setup screen starts; ``state()`` which steps look done.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from eu4unitgen.core import config

TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
WEIGHTS = {"TRELLIS.2-4B": "microsoft/TRELLIS.2-4B", "TRELLIS-image-large": "microsoft/TRELLIS-image-large",
           "dinov3-vitl16-pretrain-lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
           "sam2.1-hiera-large": "facebook/sam2.1-hiera-large"}
GATED = {"dinov3-vitl16-pretrain-lvd1689m": "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"}
AI_PACKAGES = ["torch==2.14.1", "torchvision==0.29.1"]
AI_PACKAGES_PLAIN = ["transformers==5.20.1", "rembg==2.0.69", "onnxruntime==1.23.2", "scipy==1.15.3",
                     "opencv-python==5.0.0", "numpy==2.2.6", "pillow", "huggingface_hub==1.33.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
IO_PDX_MESH = "https://github.com/ross-g/io_pdx_mesh/releases/download/0.91/blender-io_pdx_mesh.zip"  # checked: the 0.91 release's Blender asset
WSLCONFIG = "[wsl2]\nmemory=20GB\nswap=16GB\n\n[experimental]\nautoMemoryReclaim=dropCache\n"
ENV_NAME = "ai_env"


@dataclass
class Step:
    key: str
    title: str
    what: str  # what it does, in plain words
    needs: str = ""  # administrator rights, a restart, a password, a token, time
    minutes: int = 1


def ai_root() -> Path:
    cfg = config.load()
    r = cfg.ai_root_path()
    if r:
        return r
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_DATA_HOME")
    return Path(base or Path.home() / ".local" / "share") / "eu4unitgen" / "ai"


def steps() -> list[Step]:
    return [
        Step("wsl", "Install WSL", "Installs the Windows Subsystem for Linux (Windows' own Linux layer).",
             "administrator rights; Windows restarts afterwards", 5),
        Step("ubuntu", "Install Ubuntu", "Installs Ubuntu in WSL. A window asks you to make a Linux user name and password.",
             "a Linux password (you make it up; it's asked again once later)", 5),
        Step("wslconfig", "Give WSL enough memory", "Writes %USERPROFILE%\\.wslconfig: 20 GB for WSL, the rest kept for "
             "Windows (TRELLIS.2 peaks near 19 GB).", "", 1),
        Step("code", "Get TRELLIS.2", f"Downloads Microsoft's TRELLIS.2 code (commit {TRELLIS_COMMIT[:7]}) into the AI folder.", "", 3),
        Step("trellis_env", "Set up TRELLIS.2 in WSL", "Installs CUDA 12.9 and builds TRELLIS.2's environment in Ubuntu "
             "(tools/wsl/trellis2_setup.sh).", "your Linux password once; about 40 minutes", 40),
        Step("ai_env", "Set up the Windows AI environment", "A Python environment for the cut-out (BiRefNet) and the part "
             "crops (SAM 2).", "about 10 minutes, 5 GB", 10),
        Step("weights", "Download the AI weights", "TRELLIS.2, its decoder, DINOv3 and SAM 2.1 (about 20 GB).",
             "DINOv3 is gated: accept its licence on Hugging Face and paste your token", 30),
        Step("io_pdx_mesh", "Install the io_pdx_mesh add-on", "The Blender add-on that reads and writes EU4 .mesh files.",
             "Blender 4.2 LTS installed first", 1),
    ]


def _ps(script: str, admin: bool = False) -> list[str]:
    if admin:
        return ["powershell", "-NoProfile", "-Command",
                f"Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-NoExit','-Command','{script}'"]
    return ["powershell", "-NoProfile", "-NoExit", "-Command", script]


def wsl_path(p: Path) -> str:
    from eu4unitgen.ref3d.trellis import wsl_path as wp
    return wp(p)


def command(key: str, token: str | None = None) -> list[str]:
    """The argv for a step (Windows); started in a new console window by the Setup screen."""
    cfg = config.load()
    root = ai_root()
    repo = config.checkout_root() or Path(sys.prefix)
    wsl = " ".join(cfg.wsl_base())
    if key == "wsl":
        return _ps("wsl --install --no-distribution; Read-Host Done. Press Enter, then restart Windows and open the app again",
                   admin=True)
    if key == "ubuntu":
        return _ps("wsl --install -d Ubuntu-24.04")
    if key == "wslconfig":
        p = Path.home() / ".wslconfig"
        body = WSLCONFIG.replace("\n", "`n")
        return _ps(f"if (Test-Path '{p}') {{ Write-Host 'Already there:'; Get-Content '{p}' }} else "
                   f"{{ Set-Content -Path '{p}' -Value \"{body}\"; wsl --shutdown; Write-Host 'Written; WSL restarted.' }}")
    if key == "code":
        return _ps(f"New-Item -ItemType Directory -Force '{root}' | Out-Null; "
                   f"{wsl} -e bash -lc \"cd '{wsl_path(root)}' && ([ -d TRELLIS.2 ] || git clone {TRELLIS_REPO} TRELLIS.2) && "
                   f"cd TRELLIS.2 && git fetch -q origin && git checkout -q {TRELLIS_COMMIT} && echo TRELLIS.2 ready\"")
    if key == "trellis_env":
        tools = wsl_path(Path(repo) / "tools" / "wsl")
        root_step = ("sudo apt-get update && sudo apt-get install -y build-essential gcc-13 g++-13 git libjpeg-dev curl && "
                     "(dpkg -s cuda-keyring >/dev/null 2>&1 || (curl -fsSLo /tmp/cuda-keyring.deb "
                     "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && "
                     "sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update)) && sudo apt-get install -y cuda-toolkit-12-9 && "
                     f"sudo bash {tools}/cuda_glibc_patch.sh")
        return _ps(f"{wsl} -e bash -lc \"{root_step} && AI_WIN='{wsl_path(root)}' bash {tools}/trellis2_setup.sh\"")
    if key == "ai_env":
        env = root / ENV_NAME
        py = env / "Scripts" / "python.exe"
        return _ps(f"New-Item -ItemType Directory -Force '{root}' | Out-Null; "
                   "if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { powershell -ExecutionPolicy Bypass -c "
                   "\"irm https://astral.sh/uv/install.ps1 | iex\"; $env:Path = \"$env:USERPROFILE\\.local\\bin;$env:Path\" }; "
                   f"uv python install 3.10; uv venv --python 3.10 '{env}'; "
                   f"uv pip install --python '{py}' {' '.join(AI_PACKAGES)} --index-url {TORCH_INDEX}; "
                   f"uv pip install --python '{py}' {' '.join(AI_PACKAGES_PLAIN)}; Write-Host 'AI environment ready.'")
    if key == "weights":
        py = Path(cfg.ai() or root / ENV_NAME / "Scripts" / "python.exe")
        job = Path(__file__).resolve().parents[1] / "ai" / "jobs" / "download_weights.py"
        pairs = " ".join(f"{n}={r}" for n, r in WEIGHTS.items())
        env = f"$env:HF_TOKEN='{token}'; " if token else ""
        return _ps(f"{env}& '{py}' '{job}' '{root}' {pairs}")
    if key == "io_pdx_mesh":
        b = cfg.blender()
        return _ps(f"$z = Join-Path $env:TEMP 'blender-io_pdx_mesh.zip'; Invoke-WebRequest '{IO_PDX_MESH}' -OutFile $z; "
                   f"& '{b}' --command extension install-file -r user_default -e $z; Write-Host 'io_pdx_mesh installed.'")
    raise KeyError(key)


def state() -> dict[str, bool]:
    """Which steps look done (cheap checks: no WSL calls but one)."""
    from eu4unitgen.core import checks
    cfg = config.load()
    root = ai_root()
    out = {}
    wsl = checks.check_wsl()
    out["wsl"] = "isn't installed" not in wsl.what
    out["ubuntu"] = out["wsl"] and "environment isn't set up" in wsl.what or wsl.ok
    out["wslconfig"] = (Path.home() / ".wslconfig").is_file()
    out["code"] = (root / "TRELLIS.2" / "trellis2").is_dir()
    out["trellis_env"] = wsl.ok
    out["ai_env"] = bool(cfg.ai()) and Path(cfg.ai()).is_file()
    out["weights"] = all((root / "weights" / n).is_dir() for n in WEIGHTS)
    b = cfg.blender()
    ext = Path(os.environ.get("APPDATA", "")) / "Blender Foundation" / "Blender" / "4.2" / "extensions" / "user_default" / "io_pdx_mesh"
    out["io_pdx_mesh"] = bool(b) and ext.is_dir()
    return out


def adopt_ai_env() -> str | None:
    """After the AI environment step: its python in the settings (when none is set). -> the python or None."""
    py = ai_root() / ENV_NAME / "Scripts" / "python.exe"
    if py.is_file() and not config.load().ai():
        config.update(ai_python=str(py), ai_root=str(ai_root()))
        return str(py)
    return None
