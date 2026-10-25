"""Track M S1 job: proposed region labels for a cleaned reference view. Runs ONLY in the AI
environment (numpy, OpenCV, SciPy).

args: {"views": {name: {"image", "fg" (horse mask), "inpainted" (mask of inpainted pixels) | null,
       "zones": {zone: [polygon | [polygon, ...]]} (crop px), "never_use": [polygon, ...] (crop px),
       "segments": SAM 2 segment ids (.npy, job ref3d_segments) | null}},
       "rules": [{"class", "h", "s", "v", "only_in": [zone, ...] (optional)}], "zone_map": {zone: {class:
       zone class}}, "colours": {class: [r, g, b]}, "segment_vote": {"ambiguous": [class, ...],
       "abstain": [class, ...], "min_confident": share} | null, "neighbour_radius": px (default 0), "out_dir", "result_path"}
Each foreground pixel gets the first HSV rule it matches (h in degrees, a negative lower bound wraps
round 0; s, v 0-1; a rule with ``only_in`` applies only inside those zones). Segment vote (S1 v004:
shaded gold scale edges and leather straps share hue, saturation and value): inside each SAM 2
segment, the pixels of an ambiguous class or no class take the majority class of the segment's
confident pixels (matched, not ambiguous) when those are at least ``min_confident`` of it; a segment
that is mostly ambiguous (a strap, the saddle) keeps its colours. An ``abstain`` class (coat: allowed
only inside its zones) keeps its pixels but never votes and never spreads to neighbours (v005: a
segment joining the tail and the haunch voted the haunch into coat). A pixel still unmatched
takes the class of the nearest matched pixel when that is at most ``neighbour_radius`` px away
(S1 v003: in shade, cloth and coat share hue 0-28, so dark pixels are labelled by where they are,
not by colour), else "unclassified". Then inside a zone a class is renamed by zone_map (scale inside
the head zone = chanfron). Pixels inside ``never_use`` are "excluded": invented parts that later
stages must not use; they are left out of the shares and palettes.
Writes ``<name>_labels.png`` (class colours), ``<name>_labels_overlay.png`` (yellow outline:
inpainted; orange outline: labelled by segment vote; cyan outline: labelled by neighbour;
excluded hatched).
Result per view: {"shares": {class: share of the usable horse}, "palette": {class: {"observed":
[r, g, b] median, "inpainted": [...] | null, "pixels"}}, "inpainted_share", "unclassified_share",
"segment_share", "neighbour_share", "excluded_share" (of the whole horse)}.
"""
import json
import sys
from pathlib import Path


def _polys(z):
    return [z] if z and isinstance(z[0][0], (int, float)) else z


def _fill(shape, polys):
    import cv2
    import numpy as np
    m = np.zeros(shape, np.uint8)
    for p in _polys(polys):
        cv2.fillPoly(m, [np.array(p, dtype=np.int32)], 1)
    return m > 0


def main(args_path):
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    colours = {k: np.array(v, dtype=np.uint8) for k, v in a["colours"].items()}
    radius = float(a.get("neighbour_radius", 0))
    res = {}
    for name, v in a["views"].items():
        rgb = np.array(Image.open(v["image"]).convert("RGB"))
        fg_all = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127 if v.get("inpainted") else np.zeros_like(fg_all)
        never = _fill(fg_all.shape, v["never_use"]) if v.get("never_use") else np.zeros_like(fg_all)
        fg = fg_all & ~never
        zones = {z: _fill(fg.shape, p) for z, p in (v.get("zones") or {}).items()}
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
        H, S, V = hsv[..., 0] * 360 / 256, hsv[..., 1] / 255, hsv[..., 2] / 255
        cls = np.full(fg.shape, "unclassified", dtype=object)
        free = fg.copy()
        for r in a["rules"]:
            h0, h1 = r["h"]
            hm = ((H >= h0 % 360) | (H <= h1)) if h0 < 0 else ((H >= h0) & (H <= h1))
            hit = free & hm & (S >= r["s"][0]) & (S <= r["s"][1]) & (V >= r["v"][0]) & (V <= r["v"][1])
            if "only_in" in r:
                hit &= np.logical_or.reduce([zones[z] for z in r["only_in"] if z in zones] or [np.zeros_like(fg)])
            cls[hit] = r["class"]
            free &= ~hit
        voted = np.zeros_like(fg)
        sv = a.get("segment_vote")
        if sv and v.get("segments"):
            seg = np.load(v["segments"])
            amb = fg & (free | np.isin(cls, sv["ambiguous"]))
            conf = fg & ~amb & ~np.isin(cls, sv.get("abstain", []))
            for sid in np.unique(seg[amb]):
                if sid == 0:
                    continue
                m = seg == sid
                cm = conf & m
                if cm.sum() < sv["min_confident"] * (m & fg).sum():
                    continue
                names, counts = np.unique(cls[cm], return_counts=True)
                take = amb & m
                cls[take] = names[counts.argmax()]
                voted |= take
            free &= ~voted
        neighbour = np.zeros_like(fg)
        source = fg & ~free & ~np.isin(cls, (sv or {}).get("abstain", []))
        if radius > 0 and free.any() and source.any():
            dist, (iy, ix) = distance_transform_edt(~source, return_indices=True)
            neighbour = free & (dist <= radius)
            cls[neighbour] = cls[iy[neighbour], ix[neighbour]]
        for zone, zm in zones.items():
            for c_from, c_to in a["zone_map"].get(zone, {}).items():
                cls[zm & fg & (cls == c_from)] = c_to
        lab = np.zeros_like(rgb)
        shares, palette = {}, {}
        n = max(int(fg.sum()), 1)
        for c in colours:
            if c == "excluded":
                continue
            m = fg & (cls == c)
            lab[m] = colours[c]
            if not m.any():
                continue
            shares[c] = round(float(m.sum() / n), 4)
            ob, ip = m & ~inp & ~neighbour & ~voted, m & (inp | neighbour | voted)
            palette[c] = {"observed": np.median(rgb[ob], axis=0).astype(int).tolist() if ob.any() else None,
                          "inpainted": np.median(rgb[ip], axis=0).astype(int).tolist() if ip.any() else None,
                          "pixels": int(m.sum())}
        ex = fg_all & never
        yy, xx = np.indices(fg.shape)
        hatch = ((xx + yy) // 4) % 2 == 0
        lab[ex] = np.where(hatch[ex][:, None], colours.get("excluded", np.array([70, 70, 70], np.uint8)), 0)
        Image.fromarray(lab).save(out / f"{name}_labels.png")
        over = rgb.copy()
        over[fg] = (0.45 * rgb[fg] + 0.55 * lab[fg]).astype(np.uint8)
        over[~fg_all] = (0.35 * rgb[~fg_all]).astype(np.uint8)
        over[ex & hatch] = (0.3 * rgb[ex & hatch]).astype(np.uint8)
        over[cv2.Canny((inp * 255).astype(np.uint8), 50, 150) > 0] = (255, 255, 0)  # the inpainted area
        over[cv2.Canny((voted * 255).astype(np.uint8), 50, 150) > 0] = (255, 140, 0)  # labelled by segment vote
        over[cv2.Canny((neighbour * 255).astype(np.uint8), 50, 150) > 0] = (0, 230, 255)  # labelled by neighbour
        Image.fromarray(over).save(out / f"{name}_labels_overlay.png")
        res[name] = {"shares": shares, "palette": palette, "inpainted_share": round(float((inp & fg).sum() / n), 4),
                     "unclassified_share": shares.get("unclassified", 0.0),
                     "segment_share": round(float(voted.sum() / n), 4),
                     "neighbour_share": round(float(neighbour.sum() / n), 4),
                     "excluded_share": round(float(ex.sum() / max(int(fg_all.sum()), 1)), 4),
                     "labels": str(out / f"{name}_labels.png"), "overlay": str(out / f"{name}_labels_overlay.png")}
        print(name, shares, flush=True)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
