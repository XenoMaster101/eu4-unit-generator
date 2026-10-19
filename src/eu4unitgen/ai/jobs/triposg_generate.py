"""TripoSG job: one image -> one raw mesh. Runs ONLY in the TripoSG environment (Python 3.10,
torch + CUDA), started by eu4unitgen.ai.triposg.TripoSGRunner; never imported by the app.

Deviations from TripoSG's own scripts/inference_triposg.py (both deliberate):
- no RMBG-1.4: the input already has its background removed (an alpha channel), and TripoSG's
  load_image uses that alpha as it is; RMBG-1.4's licence is non-commercial;
- no diso: it ships as source only and needs the CUDA toolkit to build on Windows. diso serves
  only the "flash" decoder, so the pipeline runs with use_flash_decoder=False (scikit-image
  marching cubes), and a stub stands in for the import TripoSG makes at module load.
"""
import json
import sys
import time
import types
from pathlib import Path


def _stub_diso() -> None:
    stub = types.ModuleType("diso")

    class DiffDMC:  # only the flash decoder uses it
        def __init__(self, *a, **k):
            raise RuntimeError("diso isn't installed; run with use_flash_decoder=False")
    stub.DiffDMC = DiffDMC
    sys.modules["diso"] = stub


def main(args_path: str) -> None:
    args = json.loads(Path(args_path).read_text(encoding="utf-8"))
    repo = Path(args["repo"])
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    _stub_diso()

    import numpy as np
    import torch
    import trimesh
    from image_process import prepare_image
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    if not torch.cuda.is_available():
        raise SystemExit("TripoSG needs a CUDA GPU; torch sees none")
    t0 = time.time()
    pipe = TripoSGPipeline.from_pretrained(args["weights"]).to("cuda", torch.float16)
    t_load = time.time() - t0
    image = prepare_image(args["image"], bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
    prepared = Path(args["output"]).with_suffix(".input.png")
    image.save(prepared)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    out = pipe(image=image, generator=torch.Generator(device="cuda").manual_seed(int(args.get("seed", 42))),
               num_inference_steps=int(args.get("steps", 50)), guidance_scale=float(args.get("guidance", 7.0)),
               use_flash_decoder=False, hierarchical_octree_depth=int(args.get("octree_depth", 9)))
    vertices, faces = out.samples[0]
    mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
    Path(args["output"]).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args["output"])
    result = {"output": args["output"], "prepared_input": str(prepared), "vertices": int(len(mesh.vertices)),
              "faces": int(len(mesh.faces)), "bounds": mesh.bounds.tolist(), "watertight": bool(mesh.is_watertight),
              "components": int(len(mesh.split(only_watertight=False))), "seconds_load": round(t_load, 1),
              "seconds_generate": round(time.time() - t1, 1),
              "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
              "torch": torch.__version__}
    Path(args["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
