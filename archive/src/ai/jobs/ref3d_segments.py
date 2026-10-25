"""Track M S1 job: shape segments of a reference view with SAM 2.1 (hiera-large, Apache-2.0; weights
only, run through transformers' Sam2Model). Runs ONLY in the AI environment.

args: {"images": {name: path}, "weights": the model folder, "points_per_crop" (grid side, default 32),
       "crops_n_layers" (default 0), "points_per_batch" (default 64), "pred_iou_thresh" (default 0.88),
       "stability_score_thresh" (default 0.95), "out_dir", "result_path"}
Automatic mask generation (a grid of point prompts, no hand input). The masks are painted largest
first, so a pixel ends in the smallest segment covering it; 0 = no segment. Writes
``<name>_segments.npy`` (int32 ids) and ``<name>_segments.png`` (random colours, to look at).
Result: {name: {"npy", "png", "segments", "covered"}, "seconds", "peak_vram_gb"}.
"""
import json
import sys
import time
from pathlib import Path


def main(args_path):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import pipeline

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    gen = pipeline("mask-generation", model=a["weights"], device=0)  # fp32: fp16 breaks the NMS post-process
    res = {}
    for name, path in a["images"].items():
        img = Image.open(path).convert("RGB")
        r = gen(img, points_per_batch=int(a.get("points_per_batch", 64)),
                points_per_crop=int(a.get("points_per_crop", 32)), crops_n_layers=int(a.get("crops_n_layers", 0)),
                pred_iou_thresh=float(a.get("pred_iou_thresh", 0.88)),
                stability_score_thresh=float(a.get("stability_score_thresh", 0.95)))
        masks = sorted((np.asarray(m, dtype=bool) for m in r["masks"]), key=lambda m: -int(m.sum()))
        seg = np.zeros(img.size[::-1], np.int32)
        for i, m in enumerate(masks, 1):
            seg[m] = i
        np.save(out / f"{name}_segments.npy", seg)
        rng = np.random.default_rng(0)
        pal = rng.integers(40, 255, (len(masks) + 1, 3), dtype=np.uint8)
        pal[0] = 0
        Image.fromarray(pal[seg]).save(out / f"{name}_segments.png")
        res[name] = {"npy": str(out / f"{name}_segments.npy"), "png": str(out / f"{name}_segments.png"),
                     "segments": len(masks), "covered": round(float((seg > 0).mean()), 4)}
        print(name, res[name], flush=True)
    res["seconds"] = round(time.time() - t, 1)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
