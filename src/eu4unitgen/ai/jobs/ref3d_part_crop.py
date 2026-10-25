"""AI job (Track P1.2): each part of a reference cut out with SAM 2.1 (hiera-large, Apache-2.0; through transformers'
Sam2Model), prompted by where the part lies (ref3d.part_gen.locate projects the model's split part into the image).
Runs ONLY in the AI environment.

args: {"image": the reference (RGB), "fg": the subject's mask (optional: the part is kept inside it),
       "parts": {key: {"box": [x0, y0, x1, y1], "points": [[x, y], ...] (on the part), "negative": [[x, y], ...]
                       (off it), "hint": a mask png of the projected part (optional: picks among SAM's masks),
                       "clip": px (optional: the mask kept within this distance of the hint; a hand gripping a shield
                       is cut off)}},
       "weights": the SAM 2.1 folder, "crop_size" (default 1024), "out_dir", "result_path"}
Per part: SAM's three masks for the box and points; the one that best overlaps the hint (else SAM's best score),
inside the subject's mask, its largest piece with the holes filled. Writes ``<key>_mask.png`` (full size) and
``<key>_crop.png``: the part alone on transparency, its box with a margin, centred in a square ``crop_size`` image
(TRELLIS.2 takes it as it is). Result: {key: {"crop", "mask", "box", "area", "score", "hint_iou"}}.
"""
import json
import sys
from pathlib import Path


def largest_filled(m):
    import numpy as np
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(keep)


def main(args_path):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(a["image"]).convert("RGB")
    W, H = image.size
    fg = np.array(Image.open(a["fg"]).convert("L").resize((W, H))) > 127 if a.get("fg") else np.ones((H, W), dtype=bool)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam2Model.from_pretrained(a["weights"]).to(dev).eval()
    proc = Sam2Processor.from_pretrained(a["weights"])
    size = int(a.get("crop_size", 1024))
    res = {}
    for key, p in a["parts"].items():
        pts = [list(map(float, q)) for q in p.get("points", [])] + [list(map(float, q)) for q in p.get("negative", [])]
        labels = [1] * len(p.get("points", [])) + [0] * len(p.get("negative", []))
        kw = {"input_boxes": [[list(map(float, p["box"]))]]}
        if pts:
            kw["input_points"] = [[pts]]
            kw["input_labels"] = [[labels]]
        inputs = proc(images=image, return_tensors="pt", **kw).to(dev)
        with torch.no_grad():
            o = model(**inputs, multimask_output=True)
        masks = proc.post_process_masks(o.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0].numpy() > 0
        scores = o.iou_scores.cpu().numpy().reshape(-1)
        hint = np.array(Image.open(p["hint"]).convert("L").resize((W, H))) > 127 if p.get("hint") else None
        best, best_v, hint_iou = 0, -1.0, None
        for k, m in enumerate(masks):
            m = m & fg
            if hint is not None:
                v = float((m & hint).sum() / max((m | hint).sum(), 1))
            else:
                v = float(scores[k])
            if v > best_v:
                best, best_v = k, v
        m = largest_filled(masks[best] & fg)
        if hint is not None and p.get("clip"):
            from scipy import ndimage
            near = ndimage.distance_transform_edt(~hint) <= float(p["clip"])
            m = largest_filled(m & near)
        if hint is not None:
            hint_iou = round(float((m & hint).sum() / max((m | hint).sum(), 1)), 3)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            res[key] = {"error": "empty mask"}
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{key}_mask.png")
        rgba = np.dstack([np.array(image), (m * 255).astype(np.uint8)])[y0:y1, x0:x1]
        crop = Image.fromarray(rgba, "RGBA")
        side = max(x1 - x0, y1 - y0)
        k = size * 0.9 / side
        crop = crop.resize((max(1, round((x1 - x0) * k)), max(1, round((y1 - y0) * k))), Image.LANCZOS)
        sq = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sq.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
        sq.save(out / f"{key}_crop.png")
        res[key] = {"crop": str(out / f"{key}_crop.png"), "mask": str(out / f"{key}_mask.png"), "box": [x0, y0, x1, y1],
                    "area": int(m.sum()), "score": round(float(scores[best]), 3), "hint_iou": hint_iou}
        print(key, res[key], flush=True)
    Path(a["result_path"]).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
