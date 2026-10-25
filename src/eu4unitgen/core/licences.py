"""Every model and tool the app uses, with its licence (Track P3.5): one list for SPEC §9 and the About page.

Checked from the files themselves where they are installed (2026-10-22 to 2026-10-24; SPEC M21): the TRELLIS.2
checkout's LICENSE, the weights' model cards and LICENSE.md, the io_pdx_mesh manifest, TripoSG's LICENSE.
``use``: what the app does with it. ``shipped``: whether the app's package includes it (nothing marked False is
ever redistributed: each user installs or downloads it under their own acceptance).
"""
from __future__ import annotations

LICENCES = [
    # name, licence, use, shipped, note
    ("eu4unitgen (this app)", "GPL-3.0", "the app", True, ""),
    ("Python packages: PySide6", "LGPL-3.0", "the window (Qt)", False, "installed from PyPI by the installer"),
    ("Pillow", "MIT-CMU (HPND)", "images and DDS", False, "installed from PyPI"),
    ("Jinja2", "BSD-3-Clause", "script templates", False, "installed from PyPI"),
    ("Blender 4.2 LTS", "GPL-2.0-or-later", "mesh jobs, run only as a separate program", False, "the user installs it"),
    ("io_pdx_mesh 0.91 (ross-g)", "GPL-3.0-or-later", "reads and writes EU4 .mesh files inside Blender", False,
     "Blender extension"),
    ("TRELLIS.2 code (Microsoft)", "MIT", "image to 3D model, in WSL", False, ""),
    ("TRELLIS.2-4B weights", "MIT", "the 3D model", False, "downloaded by the user"),
    ("TRELLIS-image-large (sparse-structure decoder)", "MIT", "part of TRELLIS.2", False, "downloaded by the user"),
    ("DINOv3 ViT-L/16 (Meta)", "DINOv3 License (2025-08-19)", "TRELLIS.2's image encoder", False,
     "gated: each user accepts it on Hugging Face; use and derivatives allowed, outputs are the user's (SPEC M21)"),
    ("SAM 2.1 hiera-large (Meta)", "Apache-2.0", "cuts parts out of the reference", False, "downloaded by the user"),
    ("BiRefNet-general (through rembg)", "MIT", "cuts the subject out of the image", False, "downloaded by the user"),
    ("rembg", "MIT", "runs BiRefNet", False, ""),
    ("PyTorch, torchvision", "BSD-3-Clause", "runs the AI models", False, ""),
    ("transformers (Hugging Face)", "Apache-2.0", "runs SAM 2", False, ""),
    ("xformers", "BSD-3-Clause", "attention in TRELLIS.2", False, ""),
    ("CuMesh, FlexGEMM (J. Xiang)", "MIT", "TRELLIS.2's mesh and sparse ops", False, ""),
    ("o-voxel (TRELLIS.2)", "MIT", "TRELLIS.2's export", False, "Eigen inside it: MPL-2.0"),
    ("our UV rasteriser (ai/jobs/_uv_raster.py)", "GPL-3.0", "replaces nvdiffrast in TRELLIS.2's texture bake", True,
     "nvdiffrast and nvdiffrec (NVIDIA, research or evaluation only) are never installed"),
    ("CUDA toolkit 12.9 (NVIDIA)", "NVIDIA CUDA EULA", "builds TRELLIS.2's extensions inside WSL", False,
     "installed by the setup script from NVIDIA's own repository"),
    ("WSL2 and Ubuntu", "Microsoft / various open-source", "where TRELLIS.2 runs", False, ""),
    ("TripoSG (VAST, optional)", "MIT", "the old hand-built pipeline only (archived)", False, "not needed by the app"),
    ("SigLIP 2 (Google, optional)", "Apache-2.0", "the old self-reading pipeline only (archived)", False, ""),
    ("texconv (Microsoft DirectXTex, optional)", "MIT", "DDS encoding on Windows if present", False, ""),
    ("Europa Universalis IV files (Paradox)", "the game's EULA", "read from your own install at run time", False,
     "never copied into the app or the repo (rule 1); DLC files never redistributed"),
]
NOT_USED = [("briaai/RMBG-2.0", "CC BY-NC 4.0 (non-commercial)", "TRELLIS.2's default background remover: replaced by BiRefNet"),
            ("nvdiffrast, nvdiffrec", "NVIDIA Source Code License (research or evaluation only)", "replaced by our rasteriser")]


def markdown() -> str:
    rows = ["| Model or tool | Licence | What it does here | In the package |", "|---|---|---|---|"]
    rows += [f"| {n} | {lic} | {use}{' (' + note + ')' if note else ''} | {'yes' if shipped else 'no'} |"
             for n, lic, use, shipped, note in LICENCES]
    rows += ["", "Not used: " + "; ".join(f"{n} ({lic}): {why}" for n, lic, why in NOT_USED) + "."]
    return "\n".join(rows)


def html() -> str:
    rows = "".join(f"<tr><td>{n}</td><td>{lic}</td><td>{use}{'<br><i>' + note + '</i>' if note else ''}</td></tr>"
                   for n, lic, use, _, note in LICENCES)
    nu = "".join(f"<li>{n} ({lic}): {why}</li>" for n, lic, why in NOT_USED)
    return (f"<table cellspacing='6'><tr><th align='left'>Model or tool</th><th align='left'>Licence</th>"
            f"<th align='left'>What it does here</th></tr>{rows}</table><p><b>Not used</b></p><ul>{nu}</ul>")
