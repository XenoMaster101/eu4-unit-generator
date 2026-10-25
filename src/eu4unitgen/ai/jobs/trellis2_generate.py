"""AI job (M19 bake-off): one image -> a textured mesh with TRELLIS.2 (microsoft/TRELLIS.2-4B, MIT).

Runs in its own environment (Python 3.11, torch 2.7 cu128; inside WSL2, ``~/eu4unitgen_ai/trellis2_env`` from
``tools/wsl/trellis2_setup.sh``) on local weights only:
  * the cut-out is ours (BiRefNet, ref3d_mask): the image comes in RGBA, so TRELLIS.2's own background
    remover (briaai/RMBG-2.0, non-commercial) is never loaded (a stand-in takes its place);
  * the pipeline config is rewritten to local folders: TRELLIS.2-4B, the sparse-structure decoder from
    TRELLIS-image-large (MIT), DINOv3 ViT-L/16 (the DINOv3 License: use and derivatives allowed, no attribution
    needed for outputs, never redistributed by us; gated, fetched by the user's own Hugging Face account; SPEC M21);
  * the texture bake rasterises with ``_uv_raster`` (ours), so neither nvdiffrast nor nvdiffrec (NVIDIA Source
    Code License: research or evaluation only) is loaded or installed (SPEC M21);
  * low-VRAM mode (each stage's model moved to the GPU only while it runs) for a 12 GB card; on an
    out-of-memory error the next lower ``pipeline_type`` is tried (not "512" after a larger one); only the
    flow models those types use are loaded (1024_cascade: about 13.4 GB of system RAM, not 16).

args: {"image": RGBA png, "output": .glb, "repo": TRELLIS.2 checkout, "weights": TRELLIS.2-4B folder,
       "ss_decoder": TRELLIS-image-large folder, "dinov3": DINOv3 folder, "seed", "pipeline_type"
       ("1024_cascade" default; then "1024"), "decimation_target" (faces, default 300000),
       "texture_size" (default 2048), "result_path"}
       or, for several images on one model load (Track P1.2's part crops), "batch": [{"image", "output",
       "decimation_target", "texture_size"}, ...] in place of "image" and "output".
Result: seconds (load, generate, export), peak VRAM (GB), the pipeline type used, faces and vertices; with
"batch", {"load": seconds, "items": [that, per image]}.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "xformers")  # flash-attn isn't built
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ORDER = ["1536_cascade", "1024_cascade", "1024", "512"]
# flow models each pipeline type samples with (trellis2_image_to_3d.run); the decoders are always loaded
USES = {"512": {"shape_slat_flow_model_512", "tex_slat_flow_model_512"},
        "1024": {"shape_slat_flow_model_1024", "tex_slat_flow_model_1024"},
        "1024_cascade": {"shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"},
        "1536_cascade": {"shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}}


def fallbacks(first: str) -> list[str]:
    """``first`` and the smaller types tried on out-of-memory; "512" only when asked for, so a larger run never
    loads its 2.6 GB texture flow (every model sits in system RAM in low-VRAM mode)."""
    chain = ORDER[ORDER.index(first):]
    return chain if first == "512" else [t for t in chain if t != "512"]


def local_config(weights: Path, ss_decoder: Path, dinov3: Path, types: list[str]) -> str:
    """pipeline.json with every model path local and only the flow models ``types`` use; written beside the
    original (inside the AI root)."""
    cfg = json.loads((weights / "pipeline.json").read_text(encoding="utf-8"))
    m = cfg["args"]["models"]
    used = set().union(*(USES[t] for t in types))
    for k in [k for k in m if "_flow_model_" in k and k not in used]:
        del m[k]
    for k, v in m.items():
        if v.startswith("microsoft/TRELLIS-image-large/"):
            m[k] = str(ss_decoder / v.split("microsoft/TRELLIS-image-large/", 1)[1]).replace("\\", "/")
    cfg["args"]["image_cond_model"]["args"]["model_name"] = str(dinov3)
    cfg["args"]["rembg_model"] = {"name": "NoRembg", "args": {}}
    name = "pipeline_eu4unitgen.json"
    (weights / name).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return name


def main(args_path: str) -> None:
    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    repo = Path(a["repo"])
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    import _uv_raster
    _uv_raster.install()  # the texture bake's rasteriser is ours; nvdiffrast is never loaded
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    class NoRembg:  # the image arrives cut out; RMBG-2.0 is never loaded
        def __init__(self, **_):
            pass

        def to(self, *_):
            return self

        cpu = cuda = to

        def __call__(self, image):
            raise RuntimeError("the image must come in RGBA, cut out (ref3d_mask)")

    rembg.NoRembg = NoRembg
    t0 = time.time()
    weights = Path(a["weights"])
    types = fallbacks(a.get("pipeline_type", "1024_cascade"))
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(weights), local_config(weights, Path(a["ss_decoder"]),
                                                                                Path(a["dinov3"]), types))
    pipe.low_vram = True
    pipe.cuda()
    t_load = time.time() - t0
    if a.get("batch"):
        items = []
        for it in a["batch"]:
            r = one(pipe, types, {**a, **it}, torch, Image, o_voxel, 0.0)
            items.append(r)
            print(json.dumps(r), flush=True)
        Path(a["result_path"]).write_text(json.dumps({"load": round(t_load, 1), "items": items}, indent=1), encoding="utf-8")
        return
    res = one(pipe, types, a, torch, Image, o_voxel, t_load)
    Path(a["result_path"]).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)


def one(pipe, types, a, torch, Image, o_voxel, t_load):
    """One image through the loaded pipeline -> its .glb; the result entry."""
    image = Image.open(a["image"]).convert("RGBA")
    tried, mesh, used = [], None, None
    t1 = time.time()
    for ptype in types:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            mesh = pipe.run(image, seed=int(a.get("seed", 42)), pipeline_type=ptype)[0]
            used = ptype
            break
        except torch.cuda.OutOfMemoryError as exc:
            tried.append({"pipeline_type": ptype, "error": str(exc).splitlines()[0][:200]})
            mesh = None
    if mesh is None:
        raise SystemExit(f"out of GPU memory at every size: {tried}")
    t_gen = time.time() - t1
    peak_gen = torch.cuda.max_memory_allocated() / 2 ** 30
    t2 = time.time()
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(a.get("decimation_target", 300000)), texture_size=int(a.get("texture_size", 2048)),
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    out = Path(a["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out))  # PNG textures (Blender's glTF importer reads them without the WebP extension)
    res = {"output": str(out), "pipeline_type": used, "oom_tried": tried,
           "seconds": {"load": round(t_load, 1), "generate": round(t_gen, 1), "export": round(time.time() - t2, 1)},
           "peak_vram_gb": round(max(peak_gen, torch.cuda.max_memory_allocated() / 2 ** 30), 2),
           "gpu": torch.cuda.get_device_name(0), "rasteriser": sys.modules["nvdiffrast.torch"].rasterize.__module__,
           "faces": int(len(glb.faces)), "vertices": int(len(glb.vertices))}
    return res


if __name__ == "__main__":
    main(sys.argv[1])
