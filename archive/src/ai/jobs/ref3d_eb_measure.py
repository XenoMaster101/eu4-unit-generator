"""Track M spike E-B: score a plate render in view L's solved camera against the cleaned reference
(SPEC M5 metrics 4 and the plate period). Runs ONLY in the AI environment (numpy, OpenCV).

The render comes from the Blender job eb_plates: ``lit`` (Workbench, plate edges shaded) and ``id``
(flat: red = plates and backing, green = the body inside E-B's region, blue = the rest of the body).
  coverage:     IoU (within the label tolerance) of the scale class against the rendered plates,
                inside E-B's region, on observed reference pixels that aren't straps or trim
                (those lie over the armour: unknown);
  plate period: the M5 metric (gate), plus the period along each image axis (x: plate width,
                y: row pitch), each the candidate's lag over the reference's, for calibrating the
                plate size.

args: {"image", "labels", "fg", "inpainted", "classes": {name: rgb}, "occluders": [class], "lit",
       "id", "label_scale", "body_length", "result_path"}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def axis_periods(ref_img, cand_img, ref_cls, cand_cls, obs, scale_id, n=64, stride=16, lo=4, hi=30, min_scale=0.85):
    """Reference and candidate lags along x and y over the tiles that are scale in both maps."""
    import _ref3d_metrics as M
    ra, rb = (ref_cls == scale_id) & obs, cand_cls == scale_id
    H, W = ra.shape
    tiles = [(y, x) for y in range(0, H - n, stride) for x in range(0, W - n, stride)
             if ra[y:y + n, x:x + n].mean() >= min_scale and rb[y:y + n, x:x + n].mean() >= min_scale]
    out = {"tiles": len(tiles)}
    if not tiles:
        return out
    aa, ab = M._mean_acorr(M._gray(ref_img), tiles, n), M._mean_acorr(M._gray(cand_img), tiles, n)
    for ax, (pa, pb) in {"x": (aa[n, n:], ab[n, n:]), "y": (aa[n:, n], ab[n:, n])}.items():
        lag, prom = M._first_peak(pa, lo, hi)
        if lag is None:
            out[ax] = None
            continue
        k0, k1 = max(2, int(np.floor(0.6 * lag))), min(len(pb) - 2, int(np.ceil(1.7 * lag)))
        k = k0 + int(np.argmax(pb[k0:k1 + 1]))
        cand = M._refine(pb, k)
        out[ax] = {"ref_px": float(lag), "cand_px": float(cand), "ratio": float(cand / lag), "prominence": prom}
    return out


def main(args_path):
    from PIL import Image

    import _ref3d_metrics as M
    import ref3d_ed2d as E
    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    classes = a["classes"]
    names = list(classes)
    img = np.array(Image.open(a["image"]).convert("RGB"))
    cls = E.decode(np.array(Image.open(a["labels"]).convert("RGB")), classes)
    fg = np.array(Image.open(a["fg"]).convert("L")) > 127
    inp = np.array(Image.open(a["inpainted"]).convert("L")) > 127
    lit = np.array(Image.open(a["lit"]).convert("RGB"))
    idm = np.array(Image.open(a["id"]).convert("RGB")).astype(int)
    scale_id = names.index("scale")
    plates = (idm[..., 0] > 127) & (idm[..., 1] < 100) & (idm[..., 2] < 100)
    zone_body = (idm[..., 1] > 127) & (idm[..., 0] < 100)
    zone = plates | zone_body
    occl = np.isin(cls, [names.index(c) for c in a["occluders"] if c in names])
    obs = fg & ~inp
    valid = zone & obs & ~occl
    tol = max(1, int(round(a["label_scale"] * a["body_length"])))
    ref_scale = (cls == scale_id) & valid
    cand_scale = plates & valid
    cov = M.tol_iou(cand_scale, ref_scale, tol)
    cand_cls = np.where(plates, scale_id, -1)
    period = M.plate_period(img, lit, cls, cand_cls, obs, scale_id)
    res = {"coverage_iou": cov, "plate_period": period,
           "axes": axis_periods(img, lit, cls, cand_cls, obs, scale_id),
           "pixels": {"zone": int(zone.sum()), "valid": int(valid.sum()), "ref_scale": int(ref_scale.sum()),
                      "cand_scale": int(cand_scale.sum())}}
    # overlay: reference scale (green), plates (red), both (yellow), on the darkened reference
    ov = (img * 0.4).astype(np.uint8)
    ov[ref_scale & ~cand_scale] = (40, 220, 60)
    ov[cand_scale & ~ref_scale] = (230, 50, 50)
    ov[cand_scale & ref_scale] = (240, 220, 60)
    p = Path(a["lit"]).with_name("coverage_overlay.png")
    Image.fromarray(ov).save(p)
    res["overlay"] = str(p)
    Path(a["result_path"]).write_text(json.dumps(res, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
