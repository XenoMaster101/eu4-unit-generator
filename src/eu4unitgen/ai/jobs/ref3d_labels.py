"""Track M S1 job: proposed region labels for a cleaned reference view. Runs ONLY in the AI
environment (numpy, OpenCV).

args: {"views": {name: {"image", "fg" (horse mask), "inpainted" (mask of inpainted pixels) | null,
       "zones": {zone: [polygon | [polygon, ...]]} (crop px)}}, "rules": [{"class", "h", "s", "v"}],
       "zone_map": {zone: {class: zone class}}, "colours": {class: [r, g, b]}, "out_dir", "result_path"}
Each foreground pixel gets the first HSV rule it matches (h in degrees, a negative lower bound wraps
round 0; s, v 0-1), else "unclassified"; inside a zone a class is renamed by zone_map (scale inside
the head zone = chanfron). Writes ``<name>_labels.png`` (class colours), ``<name>_labels_overlay.png``.
Result per view: {"shares": {class: share of the horse}, "palette": {class: {"observed": [r, g, b]
median, "inpainted": [...] | null, "pixels"}}, "inpainted_share", "unclassified_share"}.
"""
import json
import sys
from pathlib import Path


def _polys(z):
    return [z] if z and isinstance(z[0][0], (int, float)) else z


def main(args_path):
    import cv2
    import numpy as np
    from PIL import Image

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    colours = {k: np.array(v, dtype=np.uint8) for k, v in a["colours"].items()}
    res = {}
    for name, v in a["views"].items():
        rgb = np.array(Image.open(v["image"]).convert("RGB"))
        fg = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127 if v.get("inpainted") else np.zeros_like(fg)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
        H, S, V = hsv[..., 0] * 360 / 256, hsv[..., 1] / 255, hsv[..., 2] / 255
        cls = np.full(fg.shape, "unclassified", dtype=object)
        free = fg.copy()
        for r in a["rules"]:
            h0, h1 = r["h"]
            hm = ((H >= h0 % 360) | (H <= h1)) if h0 < 0 else ((H >= h0) & (H <= h1))
            hit = free & hm & (S >= r["s"][0]) & (S <= r["s"][1]) & (V >= r["v"][0]) & (V <= r["v"][1])
            cls[hit] = r["class"]
            free &= ~hit
        for zone, polys in (v.get("zones") or {}).items():
            zm = np.zeros(fg.shape, np.uint8)
            for p in _polys(polys):
                cv2.fillPoly(zm, [np.array(p, dtype=np.int32)], 1)
            for c_from, c_to in a["zone_map"].get(zone, {}).items():
                cls[(zm > 0) & fg & (cls == c_from)] = c_to
        lab = np.zeros_like(rgb)
        shares, palette = {}, {}
        n = max(int(fg.sum()), 1)
        for c in colours:
            m = fg & (cls == c)
            lab[m] = colours[c]
            if not m.any():
                continue
            shares[c] = round(float(m.sum() / n), 4)
            ob, ip = m & ~inp, m & inp
            palette[c] = {"observed": np.median(rgb[ob], axis=0).astype(int).tolist() if ob.any() else None,
                          "inpainted": np.median(rgb[ip], axis=0).astype(int).tolist() if ip.any() else None,
                          "pixels": int(m.sum())}
        Image.fromarray(lab).save(out / f"{name}_labels.png")
        over = rgb.copy()
        over[fg] = (0.45 * rgb[fg] + 0.55 * lab[fg]).astype(np.uint8)
        over[~fg] = (0.35 * rgb[~fg]).astype(np.uint8)
        edges = cv2.Canny((inp * 255).astype(np.uint8), 50, 150) > 0
        over[edges] = (255, 255, 0)  # the inpainted area's outline
        Image.fromarray(over).save(out / f"{name}_labels_overlay.png")
        res[name] = {"shares": shares, "palette": palette, "inpainted_share": round(float((inp & fg).sum() / n), 4),
                     "unclassified_share": shares.get("unclassified", 0.0),
                     "labels": str(out / f"{name}_labels.png"), "overlay": str(out / f"{name}_labels_overlay.png")}
        print(name, shares, flush=True)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
