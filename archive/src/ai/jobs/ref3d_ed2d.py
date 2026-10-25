"""Track M spike E-D in 2D (SPEC M9, user decision 2026-10-20): do the M5 gates catch deliberate
breaks? The cleaned reference stands in for a perfect master's render. Runs ONLY in the AI
environment.

Variants per view, all in the reference's pixels:
  unbroken_k: the reference plus render-like differences, seeded: a camera residual (shift and scale
    up to ``residual`` of the body length, a small roll), a smooth warp (``warp_px``), a lighting
    field (``light``), anti-alias blur and noise; landmarks moved with it plus annotation jitter.
    Every gate must pass.
  breaks (each made on the reference, then given unbroken_0's perturbation):
    trim_removed: trim pixels inpainted from their surroundings, relabelled as the nearest class;
    caparison_short: every hanging panel's lowest 10% of the caparison height replaced by what is
      just below it (the hem landmarks move up);
    plates_bigger: each scale area's texture zoomed 1.3x about its centroid;
    head_5deg: the head (head zone polygons) turned 5 degrees nose-down about the poll;
    wrong_red: cloth hue turned ``red_shift`` degrees towards orange.
  Every break must fail at least one gate (in some view).

args: {"views": {name: {"image", "labels" (class-colour PNG), "fg", "inpainted", "never_use":
       [polygon] (crop px), "landmarks": {name: [x, y]} (crop px), "body_length", "head": [polygon]
       | null, "pivot": [x, y] | null, "details": [{"name", "xy", "size"}]}}, "classes": {name: rgb},
       "regions": {region: [class]}, "gates": {...}, "unbroken": n, "seed", "residual", "warp_px",
       "light", "red_shift", "out_dir", "result_path"}
Writes per view and variant ``<view>_<variant>.png`` (+ ``_cls.png``), and metrics in the result.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def decode(lab_rgb, classes):
    import numpy as np
    cls = np.full(lab_rgb.shape[:2], -1, np.int32)
    for i, rgb in enumerate(classes.values()):
        cls[np.all(lab_rgb == np.array(rgb, np.uint8), axis=-1)] = i
    return cls


def poly_mask(shape, polys):
    import cv2
    import numpy as np
    m = np.zeros(shape, np.uint8)
    for p in polys or []:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m > 0


def smooth_field(rng, shape, sigma):
    import cv2
    import numpy as np
    f = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return f / max(float(np.abs(f).max()), 1e-6)


def perturb(v, rng, bl, a):
    """Render-like differences: returns a new variant dict (image, cls, fg, landmarks)."""
    import cv2
    import numpy as np
    H, W = v["fg"].shape
    s = 1 + rng.uniform(-a["residual"], a["residual"])
    ang = rng.uniform(-0.3, 0.3)
    dx, dy = rng.uniform(-a["residual"], a["residual"], 2) * bl
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
    M[:, 2] += (dx, dy)
    warp = lambda im, interp: cv2.warpAffine(im, M, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    img = warp(v["image"], cv2.INTER_LINEAR)
    cls = warp(v["cls"].astype(np.float32), cv2.INTER_NEAREST).astype(np.int32)
    fg = warp(v["fg"].astype(np.uint8), cv2.INTER_NEAREST) > 0
    ux, uy = smooth_field(rng, (H, W), 25) * a["warp_px"], smooth_field(rng, (H, W), 25) * a["warp_px"]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx, my = gx + ux, gy + uy
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cls = cv2.remap(cls.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
    fg = cv2.remap(fg.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
    light = (1 + a["light"] * smooth_field(rng, (H, W), 80)) * rng.uniform(0.92, 1.08)
    img = img.astype(np.float32) * light[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 1.5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    lm = {}
    for k, (x, y) in v["landmarks"].items():
        px, py = M @ np.array([x, y, 1.0])
        xi, yi = int(np.clip(round(px), 0, W - 1)), int(np.clip(round(py), 0, H - 1))
        lm[k] = (float(px - ux[yi, xi] + rng.normal(0, 0.003 * bl)), float(py - uy[yi, xi] + rng.normal(0, 0.003 * bl)))
    return {"image": img, "cls": cls, "fg": fg, "landmarks": lm}


def br_trim_removed(v, classes, a):
    import cv2
    import numpy as np
    from scipy import ndimage as ndi
    t = classes.index("trim")
    m = v["cls"] == t
    img = cv2.inpaint(v["image"], cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)
    keep = v["fg"] & ~m & (v["cls"] >= 0)
    _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
    cls = v["cls"].copy()
    cls[m] = cls[iy[m], ix[m]]
    return {**v, "image": img, "cls": cls}


def br_caparison_short(v, classes, a, frac=0.10):
    import numpy as np
    cap = np.isin(v["cls"], [classes.index("cloth"), classes.index("trim")]) & v["fg"]
    ys, xs = np.nonzero(cap)
    if not len(ys):
        return v
    top, bot = ys.min(), ys.max()
    cut = int(round(frac * (bot - top)))
    img, cls, fg = v["image"].copy(), v["cls"].copy(), v["fg"].copy()
    H = fg.shape[0]
    for x in np.unique(xs):
        b = ys[xs == x].max()
        if b < top + 0.5 * (bot - top) or b + cut >= H:
            continue
        img[b - cut + 1:b + 1, x] = v["image"][b + 1:b + cut + 1, x]
        cls[b - cut + 1:b + 1, x] = v["cls"][b + 1:b + cut + 1, x]
        fg[b - cut + 1:b + 1, x] = v["fg"][b + 1:b + cut + 1, x]
    lm = {k: ((x, y - cut) if k.startswith("hem_") else (x, y)) for k, (x, y) in v["landmarks"].items()}
    return {**v, "image": img, "cls": cls, "fg": fg, "landmarks": lm, "_cut_px": cut}


def br_plates_bigger(v, classes, a, zoom=1.3):
    import cv2
    import numpy as np
    from scipy import ndimage as ndi

    import _ref3d_metrics as metrics
    sc = (metrics.coarse(v["cls"], len(classes), a["label_scale_px"]) == classes.index("scale")) & v["fg"]
    lab, n = ndi.label(sc)
    img = v["image"].copy()
    H, W = sc.shape
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 2000:
            continue
        cy, cx = ndi.center_of_mass(m)
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        z = cv2.warpAffine(v["image"], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img[m] = z[m]
    return {**v, "image": img}


def br_head_5deg(v, classes, a, deg=5.0):
    import cv2
    import numpy as np
    if v.get("head") is None or v.get("pivot") is None:
        return None
    H, W = v["fg"].shape
    head = poly_mask((H, W), v["head"]) & v["fg"]
    px, py = v["pivot"]
    # the head faces left (muzzle at small x): a positive angle turns it counter-clockwise on screen,
    # which brings the nose down
    M = cv2.getRotationMatrix2D((px, py), deg, 1.0)
    rot = lambda im, interp, val: cv2.warpAffine(im, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=val)
    hm = rot(head.astype(np.uint8), cv2.INTER_NEAREST, 0) > 0
    img, cls, fg = v["image"].copy(), v["cls"].copy(), v["fg"].copy()
    gone = head & ~hm  # where the head was and no longer is: background behind it
    img = cv2.inpaint(img, gone.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    cls[gone], fg[gone] = -1, False
    ri = rot(v["image"], cv2.INTER_LINEAR, 0)
    rc = rot(v["cls"].astype(np.float32), cv2.INTER_NEAREST, -1).astype(np.int32)
    img[hm], cls[hm], fg[hm] = ri[hm], rc[hm], True
    in_head = poly_mask((H, W), v["head"])  # landmarks inside the head zone turn with it
    lm = {}
    for k, (x, y) in v["landmarks"].items():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W and in_head[yi, xi]:
            nx, ny = M @ np.array([x, y, 1.0])
            lm[k] = (float(nx), float(ny))
        else:
            lm[k] = (x, y)
    return {**v, "image": img, "cls": cls, "fg": fg, "landmarks": lm}


def br_wrong_red(v, classes, a):
    import cv2
    import numpy as np
    m = (v["cls"] == classes.index("cloth")) & v["fg"]
    hsv = cv2.cvtColor(v["image"], cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hsv[..., 0][m] = (hsv[..., 0][m] + a["red_shift"] * 256 / 360) % 256
    img = v["image"].copy()
    img[m] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)[m]
    return {**v, "image": img}


BREAKS = {"trim_removed": br_trim_removed, "caparison_short": br_caparison_short, "plates_bigger": br_plates_bigger,
          "head_5deg": br_head_5deg, "wrong_red": br_wrong_red}


def main(args_path):
    import numpy as np
    from PIL import Image

    import _ref3d_metrics as M

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    classes = list(a["classes"])
    pal = np.array([a["classes"][c] for c in classes] + [[0, 0, 0]], np.uint8)
    res = {"views": {}}
    for name, v in a["views"].items():
        img = np.array(Image.open(v["image"]).convert("RGB"))
        cls = decode(np.array(Image.open(v["labels"]).convert("RGB")), a["classes"])
        fg = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127
        never = poly_mask(fg.shape, v.get("never_use"))
        ref = {"image": img, "cls": np.where(fg & ~never, cls, -1), "fg": fg & ~never, "valid": ~never,
               "obs": fg & ~inp & ~never & (cls >= 0), "landmarks": {k: tuple(p) for k, p in v["landmarks"].items()},
               "head": v.get("head"), "pivot": v.get("pivot")}
        cfg = {"classes": classes, "regions": a["regions"], "body_length": v["body_length"],
               "details": v["details"], "gates": a["gates"]}
        variants = {}
        rngs = [np.random.default_rng(a["seed"] + k) for k in range(a["unbroken"])]
        for k in range(a["unbroken"]):
            variants[f"unbroken_{k}"] = perturb(ref, rngs[k], v["body_length"], a)
        for b, fn in BREAKS.items():
            broken = fn(ref, classes, {**a, "label_scale_px": a["gates"]["label_scale"] * v["body_length"]})
            if broken is None:
                continue
            variants[b] = perturb(broken, np.random.default_rng(a["seed"]), v["body_length"], a)
            if "_cut_px" in broken:
                variants[b]["_cut_px"] = broken["_cut_px"]
        vr = {}
        for vn, c in variants.items():
            mets, gates = M.evaluate(ref, c, cfg)
            p = out / f"{name}_{vn}.png"
            Image.fromarray(c["image"]).save(p)
            Image.fromarray(pal[np.where(c["cls"] >= 0, c["cls"], len(classes))]).save(out / f"{name}_{vn}_cls.png")
            vr[vn] = {"image": str(p), "metrics": mets, "gates": gates, "landmarks": c["landmarks"],
                      "failed": [g for g, r in gates.items() if not r["pass"]], **({"cut_px": c["_cut_px"]} if "_cut_px" in c else {})}
            print(name, vn, "failed:", vr[vn]["failed"], flush=True)
        res["views"][name] = vr
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
