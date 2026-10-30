"""Setting up a new machine (Track P4/P5): the steps the first-launch setup wizard runs (gui/setup_wizard.py), each a
list of commands with a plain title, run with their output shown in the wizard: nothing is typed in a terminal.

With a root (core.layout: the installed app), everything goes inside it:
  blender      portable Blender 4.2 LTS: its official zip (checksum checked) unpacked into ``tools\\blender`` with a
               ``portable`` folder, so its settings and add-ons stay there too
  io_pdx_mesh  the Blender extension that reads and writes EU4 .mesh files, installed into that Blender
  wsl          Windows' Linux layer (administrator rights; Windows restarts afterwards)
  ubuntu       Ubuntu 26.04 installed into ``wsl\\`` as the distribution ``EU4UnitGen``, set up to run as its root user
               (no Linux user or password to make)
  trellis_env  TRELLIS.2's code (commit 75fbf01) into ``ai\\`` and its environment inside that distribution: CUDA 12.9,
               then setup_scripts/trellis2_setup.sh (about 40 minutes)
  ai_env       the Windows AI environment (BiRefNet, SAM 2) in ``ai\\ai_env``, made with the app's own Python
  weights      the AI weights into ``ai\\weights`` (TRELLIS.2, its decoder, DINOv3 with the user's Hugging Face token,
               SAM 2.1; Hugging Face's cache in ``ai\\hf_home``) and BiRefNet (the URL rembg's own code downloads)
Without a root (a source checkout with no ``.eu4unitgen-root``) the older choices stay: Blender's MSI, an AI folder
picked in the wizard (``ai_folder``), WSL's ``Ubuntu-26.04``.

Pinned to what works on the author's machine (2026-10-25): Ubuntu 26.04, CUDA 12.9, TRELLIS.2 75fbf01, torch 2.14.1
cu126 for the Windows AI environment, Blender 4.2.23 LTS, io_pdx_mesh 0.91.

``python -m eu4unitgen.core.setup unzip-blender <zip> <dest>`` and ``... wslconfig`` are the wizard's own helper steps.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import string
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core import config, layout

TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
DISTRO = "Ubuntu-26.04"  # WSL's name for the image; installed under layout.DISTRO when there is a root
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
BLENDER_ZIP = "https://download.blender.org/release/Blender4.2/blender-4.2.23-windows-x64.zip"  # checked 2026-10-25
BLENDER_ZIP_SHA256 = "82e791475779a7342424a480bdde9a20b43710da9264c60346125aa16cd910cb"  # blender-4.2.23.sha256
IO_PDX_MESH = "https://github.com/ross-g/io_pdx_mesh/releases/download/0.91/blender-io_pdx_mesh.zip"  # checked
WSLCONFIG = "[wsl2]\nmemory=20GB\nswap=16GB\n\n[experimental]\nautoMemoryReclaim=dropCache\n"


@dataclass(frozen=True)
class Download:
    """One external download (Release 1.0 R3: every one shown before anything starts): the step that fetches it, what
    it is and why the app needs it, its size, licence, where it comes from and goes, and whether the app needs it."""
    step: str
    name: str
    why: str
    size_mb: int
    licence: str
    source: str
    where: str
    required: bool


# the two groups the setup wizard shows (the user, 2026-10-30: seven rows each saying "optional" read as unorganized)
GROUPS = (
    (True, "Needed for everything",
     "The 3D program the app works in. Enough to build units from the library and the example, import 3D models and "
     "add units to a mod."),
    (False, "Optional: the AI pack (make parts from your own pictures)",
     "Turns a picture of a soldier, horse or gun into a 3D model. The model, TRELLIS.2 (Microsoft Research), only runs "
     "on Linux, so the pack installs Windows' own Linux layer (WSL) with Ubuntu inside it, NVIDIA's CUDA, and the AI "
     "models. Needs an NVIDIA graphics card (8 GB or more), 32 GB of RAM, about 40 GB of disk and 1 to 2 hours. Skip it "
     "if you only use the library and imported models; you can add it any time from Setup."),
)
AI_STEPS = ("wsl", "ubuntu", "trellis_env", "ai_env", "weights")
DOWNLOADS = (
    Download("installer", "PySide6, Pillow, Jinja2", "the app's window, pictures and DDS textures, script templates",
             260, "LGPL-3.0 / MIT-CMU / BSD-3", "pypi.org (by the installer)", r"app\python", True),
    Download("blender", "Blender 4.2.23 LTS (portable)", "every 3D step: fitting, rigging, rendering the previews",
             380, "GPL-2.0-or-later", "download.blender.org", r"tools\blender (about 940 MB unpacked)", True),
    Download("io_pdx_mesh", "io_pdx_mesh 0.91", "lets Blender read and write EU4's .mesh files", 1, "GPL-3.0-or-later",
             "github.com/ross-g/io_pdx_mesh", r"tools\blender\portable", True),
    Download("wsl", "Windows Subsystem for Linux", "Windows' own Linux layer; TRELLIS.2 runs in it", 150,
             "Microsoft", "Microsoft (Windows' own installer)", "Windows", False),
    Download("ubuntu", "Ubuntu 26.04", "the Linux TRELLIS.2 runs in", 1000, "free software (Canonical)",
             "Microsoft's WSL catalogue", r"wsl\ (a Linux disk that grows to about 25 GB)", False),
    Download("trellis_env", "TRELLIS.2 code (commit 75fbf01)", "turns a cut-out picture into a 3D model",
             40, "MIT", "github.com/microsoft/TRELLIS.2", r"ai\TRELLIS.2", False),
    Download("trellis_env", "CUDA 12.9 and build tools", "builds and runs TRELLIS.2 on your NVIDIA card",
             7500, "NVIDIA CUDA EULA / GPL", "developer.download.nvidia.com, Ubuntu's archive", r"wsl\ (the Linux disk)",
             False),
    Download("trellis_env", "TRELLIS.2's environment (PyTorch and its GPU extensions)", "TRELLIS.2's libraries",
             7000, "BSD / MIT / Apache-2.0", "pypi.org, download.pytorch.org, github.com", r"wsl\ (the Linux disk)",
             False),
    Download("ai_env", "PyTorch 2.14 (CUDA)", "runs the cut-out and the part cut-outs on your card", 3000,
             "BSD-3", "download.pytorch.org", r"ai\ai_env", False),
    Download("ai_env", "transformers, rembg, onnxruntime-gpu, SciPy, OpenCV", "the cut-out and SAM 2 libraries",
             2000, "Apache-2.0 / MIT / BSD", "pypi.org", r"ai\ai_env", False),
    Download("weights", "BiRefNet-general", "cuts the subject out of your picture", 930, "MIT",
             "github.com/danielgatis/rembg", r"ai\rembg_models", False),
    Download("weights", "TRELLIS.2-4B", "the 3D model's weights", 16000, "MIT",
             "huggingface.co/microsoft", r"ai\weights", False),
    Download("weights", "TRELLIS-image-large (decoder)", "TRELLIS.2's shape decoder", 140, "MIT",
             "huggingface.co/microsoft", r"ai\weights", False),
    Download("weights", "DINOv3 ViT-L/16", f"TRELLIS.2's picture reader; gated: a free Hugging Face account, "
             "its licence accepted, a read token", 1200, "DINOv3 License", "huggingface.co/facebook", r"ai\weights", False),
    Download("weights", "SAM 2.1 hiera-large", "cuts each part (helmet, shield, weapon) out of the picture",
             860, "Apache-2.0", "huggingface.co/facebook", r"ai\weights", False),
)


def downloads(step: str | None = None) -> list[Download]:
    return [d for d in DOWNLOADS if step is None or d.step == step]


def size_text(mb: float) -> str:
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def _table(rows: list[Download]) -> str:
    head = ("<tr><th align=left>What</th><th align=left>Why the app needs it</th><th align=right>Size</th>"
            "<th align=left>Licence</th><th align=left>From</th><th align=left>Goes to</th></tr>")
    body = "".join(f"<tr><td><b>{d.name}</b></td><td>{d.why}</td><td align=right>{size_text(d.size_mb)}</td>"
                   f"<td>{d.licence}</td><td>{d.source}</td><td>{d.where}</td></tr>" for d in rows)
    return f"<table cellspacing=0 cellpadding=4 border=1>{head}{body}</table>"


def downloads_html(step: str | None = None) -> str:
    """The downloads as HTML: one step's table, the AI pack's (``step="ai_pack"``), or all of them in their two groups
    with each group's total (the setup wizard's overview, the About page)."""
    if step == "ai_pack":
        rows = [d for d in DOWNLOADS if not d.required]
        return _table(rows) + f"<p><b>The AI pack in all: {size_text(sum(d.size_mb for d in rows))}</b>.</p>"
    if step is not None:
        rows = downloads(step)
        return _table(rows) if rows else "<p>Nothing is downloaded in this step.</p>"
    out = []
    for required, title, text in GROUPS:
        rows = [d for d in DOWNLOADS if d.required == required]
        out.append(f"<h3>{title}: {size_text(sum(d.size_mb for d in rows))}</h3><p>{text}</p>{_table(rows)}")
    return "".join(out) + "<p>Everything goes inside the app's folder.</p>"


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
    """The AI folder: the settings', else the root's ``ai``, else ``eu4unitgen_ai`` on the drive with most free space."""
    r = config.load().ai_root_path()
    if r:
        return r
    if layout.root() is not None:
        return layout.root() / "ai"
    d = drives()
    return (d[0][0] if d else Path.home()) / "eu4unitgen_ai"


def ai_root() -> Path:
    return default_ai_root()


def distro() -> str:
    """The WSL distribution the AI side uses: the settings', the root's (``EU4UnitGen``), else ``Ubuntu-26.04``."""
    cfg = config.load()
    if cfg.wsl_distro:
        return cfg.wsl_distro
    if layout.root() is not None:
        return layout.marker().get("wsl_distro") or layout.DISTRO
    return DISTRO


def steps() -> list[Step]:
    rooted = layout.root() is not None
    out = [
        Step("blender", "Blender 4.2 LTS", "The free 3D program the app runs in the background for every mesh step.",
             "about 380 MB to download" if rooted else "administrator rights if it has to be installed (about 400 MB)", 5),
        Step("io_pdx_mesh", "The io_pdx_mesh add-on", "Lets Blender read and write EU4's .mesh files.", "", 1),
        Step("ai_folder", "Where the AI parts go", f"About {NEED_GB} GB: TRELLIS.2, its weights, DINOv3, SAM 2.1.", "", 0),
        Step("wsl", "Windows Subsystem for Linux", "Windows' own Linux layer; TRELLIS.2 runs in it.",
             "administrator rights; Windows restarts afterwards", 5, restart=True),
        Step("ubuntu", "Ubuntu in WSL", "Ubuntu 26.04, set up to run as its own administrator: no Linux user or "
             "password to make." + (" It goes in the app's wsl folder." if rooted else ""), "about 1 GB", 5),
        Step("trellis_env", "TRELLIS.2", "Downloads TRELLIS.2 and builds its environment in Ubuntu (CUDA 12.9, PyTorch, "
             "its GPU extensions).", "about 40 minutes, 16 GB", 40),
        Step("ai_env", "The Windows AI environment", "Runs the cut-out (BiRefNet) and the part cut-outs (SAM 2).",
             "about 10 minutes, 5 GB", 10),
        Step("weights", "The AI weights", "TRELLIS.2, its decoder, DINOv3, SAM 2.1 and BiRefNet (about 20 GB).",
             "DINOv3 is gated: accept its licence on Hugging Face and paste a read token", 30),
    ]
    return [s for s in out if not (rooted and s.key == "ai_folder")]  # with a root the AI folder is fixed: ai\


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
    b = layout.blender_exe()
    if b is not None and b.is_file():
        return b
    from eu4unitgen.core import install
    exe, _ = install.find_blender()
    return exe if exe and Path(exe).is_file() else None


def io_pdx_mesh_dir() -> Path:
    """Where Blender keeps io_pdx_mesh: inside the portable Blender's ``portable`` folder, else the user's profile."""
    b = layout.blender_exe()
    if b is not None and b.is_file():
        return b.parent / "portable" / "extensions" / "user_default" / "io_pdx_mesh"
    return Path(os.environ.get("APPDATA", "")) / "Blender Foundation" / "Blender" / "4.2" / "extensions" / "user_default" / "io_pdx_mesh"


def _helper(*args: str) -> list[str]:
    return [app_python(), "-m", "eu4unitgen.core.setup", *args]


def plan(key: str, token: str | None = None, root: Path | None = None) -> list[Cmd]:
    """The commands for one step (Windows)."""
    cfg = config.load()
    root = Path(root or default_ai_root())
    top = layout.root()
    tmp = Path(os.environ.get("TEMP") or "/tmp")
    d = distro()
    wsl = ["wsl.exe", "-d", d]
    scripts = Path(__file__).resolve().parents[1] / "setup_scripts"
    q = shlex.quote
    if key == "blender":
        if top is not None:
            z = top / "settings" / "tmp" / "blender-4.2.23-windows-x64.zip"
            return [Cmd("Download Blender 4.2.23 LTS (portable)", download=(BLENDER_ZIP, str(z))),
                    Cmd("Unpack it into the app's tools folder", _helper("unzip-blender", str(z), str(top / "tools" / "blender")))]
        msi = tmp / "blender-4.2.23-windows-x64.msi"
        return [Cmd("Download Blender 4.2.23 LTS", download=(BLENDER_MSI, str(msi))),
                Cmd("Install Blender (Windows asks for permission)", ["msiexec", "/i", str(msi), "/passive", "/norestart"],
                    admin=True)]
    if key == "io_pdx_mesh":
        z = (top / "settings" / "tmp" if top is not None else tmp) / "blender-io_pdx_mesh.zip"
        b = cfg.blender() or blender_installed()
        return [Cmd("Download io_pdx_mesh 0.91", download=(IO_PDX_MESH, str(z))),
                Cmd("Install it into Blender", [str(b), "--background", "--command", "extension", "install-file", "-r",
                                                 "user_default", "-e", str(z)])]
    if key == "wsl":
        return [Cmd("Install WSL (Windows asks for permission)", ["wsl.exe", "--install", "--no-distribution"], admin=True)]
    if key == "ubuntu":
        if top is not None:
            install = ["wsl.exe", "--install", "-d", DISTRO, "--location", str(top / "wsl"), "--name", d, "--no-launch"]
        else:
            install = ["wsl.exe", "--install", "-d", d, "--no-launch"]
        cmds = [Cmd(f"Install Ubuntu 26.04 as {d}", install),
                Cmd("Start it as its administrator (no user to make)", wsl + ["-u", "root", "-e", "true"])]
        if not (Path.home() / ".wslconfig").is_file():
            cmds.insert(0, Cmd("Give WSL 20 GB of memory (.wslconfig)", _helper("wslconfig")))
        return cmds
    if key == "trellis_env":
        r = wsl_path(root)
        root_step = ("apt-get update && apt-get install -y build-essential gcc-13 g++-13 git libjpeg-dev curl && "
                     "(dpkg -s cuda-keyring >/dev/null 2>&1 || (curl -fsSLo /tmp/cuda-keyring.deb "
                     "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && "
                     "dpkg -i /tmp/cuda-keyring.deb && apt-get update)) && apt-get install -y cuda-toolkit-12-9 && "
                     f"bash {q(wsl_path(scripts) + '/cuda_glibc_patch.sh')}")
        return [Cmd("Download TRELLIS.2 (commit 75fbf01)", wsl + ["-u", "root", "-e", "bash", "-lc",
                    f"mkdir -p {q(r)} && cd {q(r)} && ([ -d TRELLIS.2 ] || git clone {TRELLIS_REPO} TRELLIS.2) && cd TRELLIS.2 "
                    f"&& git fetch -q origin && git checkout -q {TRELLIS_COMMIT} && echo TRELLIS.2 ready"]),
                Cmd("Install the build tools and CUDA 12.9", wsl + ["-u", "root", "-e", "bash", "-lc", root_step]),
                Cmd("Build the TRELLIS.2 environment (about 40 minutes)", wsl + ["-e", "bash", "-lc",
                    f"AI_WIN={q(r)} bash {q(wsl_path(scripts) + '/trellis2_setup.sh')}"])]
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
        env = {"HF_HOME": str(root / "hf_home")}
        if token:
            env["HF_TOKEN"] = token
        return [Cmd("Download BiRefNet (the cut-out model)", download=(BIREFNET, str(root / "rembg_models" / "birefnet-general.onnx"))),
                Cmd("Download TRELLIS.2, its decoder, DINOv3 and SAM 2.1 (about 19 GB)",
                    [str(py), str(job), str(root)] + [f"{n}={r}" for n, r in WEIGHTS.items()], env=env)]
    raise KeyError(key)


def state() -> dict[str, bool]:
    """Which steps look done."""
    from eu4unitgen.core import checks
    cfg = config.load()
    root = default_ai_root()
    out = {}
    b = cfg.blender() or blender_installed()
    out["blender"] = bool(b) and Path(b).is_file()
    out["io_pdx_mesh"] = io_pdx_mesh_dir().is_dir()
    out["ai_folder"] = bool(cfg.ai_root) or layout.root() is not None
    wsl = checks.check_wsl()
    out["wsl"] = "isn't installed" not in wsl.what
    out["ubuntu"] = wsl.ok or "environment isn't set up" in wsl.what
    out["trellis_env"] = wsl.ok and (root / "TRELLIS.2" / "trellis2").is_dir()
    py = cfg.ai() or str(root / ENV_NAME / "Scripts" / "python.exe")
    out["ai_env"] = Path(py).is_file()
    out["weights"] = all((root / "weights" / n).is_dir() for n in WEIGHTS) and (root / "rembg_models").is_dir()
    return out


def adopt(root: Path) -> dict:
    """After a step: what it made written into the settings (the AI folder, its environment, Blender) and, with a
    root, the WSL distribution into the marker (the uninstaller removes only what the app registered)."""
    found = {"ai_root": str(root)}
    py = root / ENV_NAME / "Scripts" / "python.exe"
    if py.is_file() and not config.load().ai():
        found["ai_python"] = str(py)
    b = blender_installed()
    if b and not config.load().blender_path:
        found["blender_path"] = str(b)
    config.update(**found)
    if layout.root() is not None and not layout.marker().get("wsl_distro") and _registered(distro()):
        layout.update_marker(wsl_distro=distro())
    return found


def _registered(name: str) -> bool:
    if os.name != "nt":
        return False
    import subprocess
    try:
        out = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    text = out.decode("utf-16-le", errors="ignore") if b"\x00" in out[:40] else out.decode(errors="ignore")
    return name in {ln.strip() for ln in text.splitlines()}


def adopt_ai_env() -> str | None:
    """The AI environment this setup made, into the settings when none is set. -> the python or None."""
    py = default_ai_root() / ENV_NAME / "Scripts" / "python.exe"
    if py.is_file() and not config.load().ai():
        config.update(ai_python=str(py), ai_root=str(default_ai_root()))
        return str(py)
    return None


# ---------------------------------------------------------------------------- the wizard's helper steps
def unzip_blender(zip_path: Path, dest: Path) -> Path:
    """Blender's portable zip -> ``dest`` (its one top folder flattened), checksum checked, with a ``portable``
    folder so Blender keeps its settings and add-ons beside itself. -> blender.exe"""
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != BLENDER_ZIP_SHA256:
        raise SystemExit(f"{zip_path.name}: the checksum doesn't match Blender's published one; download it again")
    dest = Path(dest)
    stage = dest.with_name(dest.name + ".unpacking")
    shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(stage)
    inner = [p for p in stage.iterdir()]
    src = inner[0] if len(inner) == 1 and inner[0].is_dir() else stage
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    shutil.rmtree(stage, ignore_errors=True)
    (dest / "portable").mkdir(exist_ok=True)
    Path(zip_path).unlink(missing_ok=True)
    print(f"Blender unpacked into {dest}")
    return dest / "blender.exe"


def write_wslconfig() -> Path:
    """``%USERPROFILE%\\.wslconfig`` (WSL reads it only there), when there is none; recorded so uninstall removes it."""
    p = Path.home() / ".wslconfig"
    if p.exists():
        print(f"{p} is already there: left as it is")
        return p
    p.write_text(WSLCONFIG, encoding="utf-8")
    layout.update_marker(wslconfig=True)
    print(f"wrote {p}")
    return p


def main(argv: list[str]) -> int:
    if argv[:1] == ["unzip-blender"] and len(argv) == 3:
        unzip_blender(Path(argv[1]), Path(argv[2]))
        return 0
    if argv[:1] == ["wslconfig"]:
        write_wslconfig()
        return 0
    print("usage: python -m eu4unitgen.core.setup unzip-blender <zip> <dest> | wslconfig")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
