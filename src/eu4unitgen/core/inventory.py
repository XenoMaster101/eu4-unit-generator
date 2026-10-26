"""Everything installed for the project, measured on the author's PC (2026-10-25): what each piece is for, where it
lives, its size, its licence and whether the pipeline needs it (Track P, the user's inventory request). Shown on the
About page and in SPEC. Nothing here is removed by the app; "safe" marks what can go without breaking the pipeline.

status: "required" (the pipeline uses it), "optional" (a cache, a rebuild tool, a speed-up), "unused" (an abandoned
approach or an old copy), "development" (only for working on the app), "never installed" (proposed in SPEC Track M,
never approved or installed). ``<root>`` is the install folder (core.layout).
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


R = "<root>"  # the install folder (core.layout; on the author's PC D:\EU4 Unit Generator)
AI = R + r"\ai"
WSL = "Linux disk ~/eu4unitgen_ai"
OLD = r"D:\eu4unitgen_ai"  # the scattered copies left after the move into the root (2026-10-25)
ITEMS = (
    # ---------------------------------------------------------------- the app itself
    Item("Python 3.12.10 (python.org NuGet build, signed)", "the app's own Python; also makes the Windows AI environment",
         R + r"\app\python", 0.04, "PSF", "required", note="installed by the installer"),
    Item("PySide6 6.12, Pillow, Jinja2", "the app's window, images and DDS, script templates",
         R + r"\app\python\Lib\site-packages", 0.70, "LGPL-3.0 / MIT-CMU / BSD-3", "required"),
    Item("Blender 4.2.23 LTS (portable)", "every mesh job, run as a separate program", R + r"\tools\blender", 0.94,
         "GPL-2.0-or-later", "required", note=r"its settings and add-ons in tools\blender\portable"),
    Item("io_pdx_mesh 0.91 (Blender extension)", "reads and writes EU4 .mesh files",
         R + r"\tools\blender\portable\extensions\user_default\io_pdx_mesh", 0.002, "GPL-3.0-or-later", "required"),
    # ---------------------------------------------------------------- WSL: TRELLIS.2
    Item("WSL2 + Ubuntu 26.04 (distribution EU4UnitGen)", "where TRELLIS.2 runs", R + r"\wsl\ext4.vhdx", 37.4,
         "Microsoft / open source", "required",
         note="the disk holds the five items below; it doesn't shrink by itself when files inside are deleted"),
    Item("CUDA toolkit 12.9", "builds TRELLIS.2's GPU extensions; they load its runtime", "Linux disk /usr/local/cuda-12.9",
         7.3, "NVIDIA CUDA EULA", "required"),
    Item("CUDA toolkit 12.8", "an earlier install; nothing links to it (/usr/local/cuda points at 12.9)",
         "Linux disk /usr/local/cuda-12.8", 6.7, "NVIDIA CUDA EULA", "unused", True),
    Item("TRELLIS.2 environment (torch 2.7 cu128, xformers, transformers 4.57.1, CuMesh, FlexGEMM, o-voxel, utils3d)",
         "image to 3D model", f"{WSL}/trellis2_env", 7.2, "BSD / MIT / Apache-2.0", "required"),
    Item("TRELLIS.2 extension build tree", "the sources CuMesh, FlexGEMM and o-voxel were built from", f"{WSL}/ext", 1.1,
         "MIT / MPL-2.0 (Eigen)", "optional", True, "installed as copies: only a rebuild needs it (setup re-downloads it)"),
    Item("uv cache in WSL", "package download cache", f"{WSL}/uv_cache", 7.8, "-", "optional", True),
    # ---------------------------------------------------------------- the AI folder
    Item("TRELLIS.2 code (commit 75fbf01)", "TRELLIS.2's model code, read by the WSL environment", AI + r"\TRELLIS.2", 0.04,
         "MIT", "required"),
    Item("TRELLIS.2-4B weights", "the 3D model", AI + r"\weights\TRELLIS.2-4B", 16.0, "MIT", "required"),
    Item("TRELLIS-image-large (decoder)", "TRELLIS.2's sparse-structure decoder", AI + r"\weights\TRELLIS-image-large", 0.14,
         "MIT", "required"),
    Item("DINOv3 ViT-L/16", "TRELLIS.2's image encoder", AI + r"\weights\dinov3-vitl16-pretrain-lvd1689m", 1.2,
         "DINOv3 License (gated)", "required"),
    Item("SAM 2.1 hiera-large", "cuts each part out of the reference", AI + r"\weights\sam2.1-hiera-large", 0.86, "Apache-2.0",
         "required"),
    Item("BiRefNet-general (ONNX, via rembg)", "cuts the subject out of the image", AI + r"\rembg_models", 0.93, "MIT",
         "required"),
    Item("Windows AI environment (torch 2.14 cu126, transformers 5.20, rembg, onnxruntime-gpu, scipy, opencv)",
         "runs BiRefNet and SAM 2", AI + r"\ai_env", 5.1, "BSD / Apache-2.0 / MIT", "required",
         note="the author's was made for TripoSG and still carries packages only TripoSG used (diffusers, peft, pymeshlab, "
              "numba, accelerate); a new install's is ~3 GB"),
    Item("CPython 3.10 (uv's build)", "the base Python of the author's AI environment", AI + r"\python\cpython-3.10.19-windows-x86_64-none",
         0.07, "PSF", "required", note="a new install bases it on the app's own Python instead"),
    # ---------------------------------------------------------------- left in the old AI folder: no longer used
    Item("Windows TRELLIS.2 environment", "the first TRELLIS.2 build, on Windows (blocked by Smart App Control)",
         OLD + r"\trellis2_env", 6.6, "mixed", "unused", True,
         "also holds nvdiffrast and nvdiffrec (NVIDIA: research or evaluation only), which the pipeline never uses"),
    Item("uv and its cache (Windows)", "made the first Python environments", OLD + r"\tools, " + OLD + r"\uv_cache", 11.8,
         "MIT / Apache-2.0", "unused", True),
    Item("CPython 3.11 (uv) and the TRELLIS.2 Windows wheels", "base of the Windows TRELLIS.2 environment",
         OLD + r"\python, " + OLD + r"\wheels", 0.09, "PSF / MIT", "unused", True),
    Item("TripoSG weights and code", "the hand-built master pipeline's 3D shells (archived)", OLD + r"\weights\TripoSG, " + OLD + r"\TripoSG",
         7.53, "MIT", "unused", True),
    Item("SDXL inpainting", "filled hidden views for the master pipeline (archived)", OLD + r"\weights\sdxl_inpaint", 6.5,
         "CreativeML Open RAIL++-M", "unused", True),
    Item("SigLIP 2 so400m", "labelled parts for the self-reading pipeline (archived)", OLD + r"\weights\siglip2-so400m-patch14-384",
         4.3, "Apache-2.0", "unused", True),
    # ---------------------------------------------------------------- development only (not part of the app)
    Item("Python 3.14 (python.org)", "runs the app and tests from the source checkout", r"C:\Python314", 0.17, "PSF",
         "development"),
    Item("Inno Setup 6.7.3 (portable)", "builds eu4unitgen-setup.exe", r"<checkout>\.tools\innosetup", 0.02,
         "Inno Setup License (free)", "development", note="portable: no Apps entry, no registry"),
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
