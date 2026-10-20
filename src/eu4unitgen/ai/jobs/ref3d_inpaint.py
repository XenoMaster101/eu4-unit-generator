"""Track M S1 job: remove things from a reference crop with SDXL inpainting 0.1 (CreativeML Open
RAIL++-M; user-approved). Runs ONLY in the AI environment (eu4unitgen.ai.env.AIJobRunner).

args: {"weights": the diffusers folder, "image": crop PNG, "mask": PNG (white = remove), "out_dir",
       "name", "seeds": [int], "prompt", "negative", "steps", "guidance", "strength", "long_side",
       "grow": px the mask grows by, "feather": px blur of the paste-back edge, "prefill": bool (default
       true: the masked area is first filled by OpenCV's Telea inpaint, so the removed object's shape
       and colours don't leak into SDXL's fill; test S1 v001 redrew the rider in 6 of 8 seeds without
       it), "result_path"}
Per seed: the crop is scaled so its long side is ``long_side`` (multiples of 8), inpainted, scaled
back, and pasted onto the ORIGINAL through the grown, feathered mask: outside it every pixel is the
original's. Writes ``<name>_seed<k>.png`` and ``<name>_mask_used.png``.
Result: {"seeds": {k: {"path", "seam"}}, "mask_share", "seconds", "peak_vram_gb"}.
seam: mean gradient magnitude on the mask's edge band over that just outside it (1 = no visible
seam; higher = an edge the inpaint left). A ranking aid only: whether an inpaint invented
structure is the user's call at H1.
"""
import json
import sys
import time
from pathlib import Path


def seam_score(img, mask):
    import cv2
    import numpy as np
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = (mask > 127).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    band = cv2.dilate(m, k, iterations=2) & (1 - cv2.erode(m, k, iterations=2))
    outer = cv2.dilate(m, k, iterations=8) & (1 - cv2.dilate(m, k, iterations=3))
    return float(mag[band > 0].mean() / max(mag[outer > 0].mean(), 1e-6))


def main(args_path):
    import cv2
    import numpy as np
    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image, ImageFilter

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(a["image"]).convert("RGB")
    mask = Image.open(a["mask"]).convert("L")
    m = np.array(mask) > 127
    grow = int(a.get("grow", 8))
    if grow:
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow) > 0
    used = Image.fromarray((m * 255).astype(np.uint8))
    used.save(out / f"{a['name']}_mask_used.png")
    paste = used.filter(ImageFilter.GaussianBlur(int(a.get("feather", 3))))
    if a.get("prefill", True):  # S1 lesson: the removed object's shape and colours leaked into SDXL's fill
        arr = cv2.inpaint(np.array(img), (m * 255).astype(np.uint8), 9, cv2.INPAINT_TELEA)
        img_in = Image.fromarray(arr)
        img_in.save(out / f"{a['name']}_prefill.png")
    else:
        img_in = img
    w, h = img.size
    s = float(a.get("long_side", 1024)) / max(w, h)
    W, H = max(8, int(round(w * s / 8)) * 8), max(8, int(round(h * s / 8)) * 8)
    t = time.time()
    pipe = AutoPipelineForInpainting.from_pretrained(a["weights"], dtype=torch.float16, variant="fp16",
                                                     local_files_only=True)
    pipe.enable_model_cpu_offload()
    res = {"seeds": {}, "mask_share": round(float(m.mean()), 4), "size": [W, H]}
    for seed in a["seeds"]:
        r = pipe(prompt=a["prompt"], negative_prompt=a.get("negative"), image=img_in.resize((W, H), Image.LANCZOS),
                 mask_image=used.resize((W, H), Image.NEAREST), width=W, height=H,
                 num_inference_steps=int(a.get("steps", 30)), guidance_scale=float(a.get("guidance", 7.0)),
                 strength=float(a.get("strength", 1.0)), generator=torch.Generator("cpu").manual_seed(int(seed))).images[0]
        r = r.resize((w, h), Image.LANCZOS)
        final = Image.composite(r, img, paste)
        p = out / f"{a['name']}_seed{seed}.png"
        final.save(p)
        res["seeds"][str(seed)] = {"path": str(p), "seam": round(seam_score(np.array(final), np.array(used)), 3)}
        print("seed", seed, res["seeds"][str(seed)], flush=True)
    res["seconds"] = round(time.time() - t, 1)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
