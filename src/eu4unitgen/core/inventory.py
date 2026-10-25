"""Everything installed for the project, measured on the author's PC (2026-10-25): what each piece is for, where it
lives, its size, its licence and whether the pipeline needs it (Track P, the user's inventory request). Shown on the
About page and in SPEC. Nothing here is removed by the app; "safe" marks what can go without breaking the pipeline.

status: "required" (the pipeline uses it), "optional" (a cache, a rebuild tool, a speed-up), "unused" (an abandoned
approach), "never installed" (proposed in SPEC Track M, never approved or installed).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    name: str
    step: str
    where: str
    size_gb: float
    licence: str
    status: str
    safe_to_remove: bool = False
    note: str = ""


AI = r"D:\eu4unitgen_ai"
WSL = "WSL ~/eu4unitgen_ai"
ITEMS = (
    # ---------------------------------------------------------------- the app itself
    Item("Python 3.14 (python.org)", "runs the app from the source checkout", r"C:\Python314", 0.17, "PSF", "required",
         note="the installed app brings its own Python"),
    Item("Python 3.12.10 (python.org NuGet build, signed)", "the installed app's own Python; also makes the Windows AI "
         "environment", r"<app>\python", 0.04, "PSF", "required", note="installed by install.ps1"),
    Item("PySide6 6.12, Pillow, Jinja2", "the app's window, images and DDS, script templates",
         r"<app>\python\Lib\site-packages (dev: .venv)",
         0.70, "LGPL-3.0 / MIT-CMU / BSD-3", "required"),
    Item("Blender 4.2 LTS", "every mesh job, run as a separate program", r"C:\Program Files\Blender Foundation\Blender 4.2",
         0.95, "GPL-2.0-or-later", "required"),
    Item("io_pdx_mesh 0.91 (Blender extension)", "reads and writes EU4 .mesh files", r"%APPDATA%\Blender Foundation\Blender\4.2"
         r"\extensions\user_default\io_pdx_mesh", 0.002, "GPL-3.0-or-later", "required"),
    # ---------------------------------------------------------------- WSL: TRELLIS.2
    Item("WSL2 + Ubuntu 26.04", "where TRELLIS.2 runs", r"%LOCALAPPDATA%\wsl\...\ext4.vhdx", 38.0, "Microsoft / open source",
         "required", note="the 38 GB disk holds the three items below; it doesn't shrink by itself when files inside are deleted"),
    Item("CUDA toolkit 12.9", "builds TRELLIS.2's GPU extensions", "WSL /usr/local/cuda-12.9", 7.3, "NVIDIA CUDA EULA",
         "required", note="needed to rebuild the extensions; whether TRELLIS.2 needs it while running is UNVERIFIED, so keep it"),
    Item("TRELLIS.2 environment (torch 2.7 cu128, xformers, transformers 4.57.1, CuMesh, FlexGEMM, o-voxel, utils3d)",
         "image to 3D model", f"{WSL}/trellis2_env", 7.2, "BSD / MIT / Apache-2.0", "required"),
    Item("TRELLIS.2 extension build tree", "the sources CuMesh, FlexGEMM and o-voxel were built from", f"{WSL}/ext", 1.1,
         "MIT / MPL-2.0 (Eigen)", "optional", True, "installed as copies: only a rebuild needs it (setup re-downloads it)"),
    Item("uv cache in WSL", "package download cache", f"{WSL}/uv_cache", 0.76, "-", "optional", True,
         "frees space inside the WSL disk, not on C: until the disk is compacted"),
    # ---------------------------------------------------------------- the Windows AI folder
    Item("TRELLIS.2 code (commit 75fbf01)", "TRELLIS.2's model code, read by the WSL environment", rf"{AI}\TRELLIS.2", 0.04,
         "MIT", "required"),
    Item("TRELLIS.2-4B weights", "the 3D model", rf"{AI}\weights\TRELLIS.2-4B", 16.0, "MIT", "required"),
    Item("TRELLIS-image-large (decoder)", "TRELLIS.2's sparse-structure decoder", rf"{AI}\weights\TRELLIS-image-large", 0.14,
         "MIT", "required"),
    Item("DINOv3 ViT-L/16", "TRELLIS.2's image encoder", rf"{AI}\weights\dinov3-vitl16-pretrain-lvd1689m", 1.2,
         "DINOv3 License (gated)", "required"),
    Item("SAM 2.1 hiera-large", "cuts each part out of the reference", rf"{AI}\weights\sam2.1-hiera-large", 0.86, "Apache-2.0",
         "required"),
    Item("BiRefNet-general (ONNX, via rembg)", "cuts the subject out of the image", rf"{AI}\rembg_models", 0.93, "MIT",
         "required"),
    Item("Windows AI environment \"triposg_env\" (torch 2.14 cu126, transformers 5.20, rembg, onnxruntime-gpu, scipy, "
         "opencv)", "runs BiRefNet and SAM 2", rf"{AI}\triposg_env", 5.1, "BSD / Apache-2.0 / MIT", "required",
         note="misnamed: made for TripoSG; it also carries packages only TripoSG used (diffusers, peft, pymeshlab, numba, "
              "accelerate). A clean environment would be ~3 GB"),
    Item("CPython 3.10 (uv)", "the base Python of the Windows AI environment", rf"{AI}\python\cpython-3.10.19-windows-x86_64-none",
         0.07, "PSF", "required"),
    Item("uv", "makes Python environments (setup)", rf"{AI}\tools\uv", 0.06, "MIT / Apache-2.0", "optional"),
    Item("uv cache (Windows)", "package download cache", rf"{AI}\uv_cache", 0.18, "-", "optional", True),
    # ---------------------------------------------------------------- no longer used
    Item("Windows TRELLIS.2 environment", "the first TRELLIS.2 build, on Windows (blocked by Smart App Control)",
         rf"{AI}\trellis2_env", 6.6, "mixed", "unused", True,
         "also holds nvdiffrast and nvdiffrec (NVIDIA: research or evaluation only), which the pipeline never uses"),
    Item("CPython 3.11 (uv)", "base of the Windows TRELLIS.2 environment", rf"{AI}\python\cpython-3.11.14-windows-x86_64-none",
         0.08, "PSF", "unused", True),
    Item("TRELLIS.2 Windows wheels", "the Windows build's wheels", rf"{AI}\wheels", 0.006, "MIT", "unused", True),
    Item("TripoSG weights and code", "the hand-built master pipeline's 3D shells (archived)", rf"{AI}\weights\TripoSG, {AI}\TripoSG",
         7.53, "MIT", "unused", True),
    Item("SDXL inpainting", "filled hidden views for the master pipeline (archived)", rf"{AI}\weights\sdxl_inpaint", 6.5,
         "CreativeML Open RAIL++-M", "unused", True),
    Item("SigLIP 2 so400m", "labelled parts for the self-reading pipeline (archived)", rf"{AI}\weights\siglip2-so400m-patch14-384",
         4.3, "Apache-2.0", "unused", True),
    Item("LPIPS", "perceptual metric, in TRELLIS.2's requirement list", "not in any environment now", 0.0, "BSD-2",
         "unused", note="not found in either environment"),
    Item("ControlNet depth SDXL, IP-Adapter SDXL, DINOv2, cma (CMA-ES)", "proposed for the master pipeline (SPEC M1-M2)",
         "-", 0.0, "RAIL++ / Apache-2.0 / Apache-2.0 / BSD", "never installed"),
    Item("texconv (DirectXTex)", "optional DDS encoder", "-", 0.0, "MIT", "never installed", note="Pillow writes the DDS files"),
)


def freed_gb() -> float:
    return round(sum(i.size_gb for i in ITEMS if i.safe_to_remove), 1)


def _size(gb: float) -> str:
    return "-" if gb == 0 else f"{gb * 1024:.0f} MB" if gb < 1 else f"{gb:.1f} GB"


def markdown() -> str:
    rows = ["| Tool, model or library | For | Where | Size | Licence | Status | Safe to remove |", "|---|---|---|---|---|---|---|"]
    for i in ITEMS:
        note = f" ({i.note})" if i.note else ""
        rows.append(f"| {i.name} | {i.step}{note} | `{i.where}` | {_size(i.size_gb)} | {i.licence} | {i.status} | "
                    f"{'yes' if i.safe_to_remove else 'no'} |")
    rows.append("")
    rows.append(f"Safe to remove in all: about {freed_gb()} GB (nothing is removed until the user confirms).")
    return "\n".join(rows)


def html() -> str:
    rows = "".join(f"<tr><td>{i.name}</td><td>{i.step}</td><td>{_size(i.size_gb)}</td><td>{i.licence}</td>"
                   f"<td>{i.status}{' (safe to remove)' if i.safe_to_remove else ''}</td></tr>" for i in ITEMS)
    return (f"<table cellspacing='6'><tr><th align='left'>Piece</th><th align='left'>For</th><th align='left'>Size</th>"
            f"<th align='left'>Licence</th><th align='left'>Status</th></tr>{rows}</table>"
            f"<p>Measured on the author's PC, 2026-10-25. About {freed_gb()} GB is safe to remove there.</p>")
