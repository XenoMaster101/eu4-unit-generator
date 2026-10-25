"""Parts from the reference itself (Track P1.2; the user's core rule for weapons, shields, helmets, plumes, gear).

"Never a placeholder primitive with flat colour. Every part matches the reference's shape and carries texture +
normal map from the reference." The preferred method, per part:
  1. ``locate``: the split part's points (fresh_rider's ``fitted_points.json``: the model placed, not yet slimmed or
     posed, as the reference shows it) projected into the reference: the model's front silhouette framed onto the
     subject's mask (mirrored or not, whichever overlaps better) -> a box, points on the part, points off it, and
     a hint mask;
  2. ``crop``: SAM 2 cuts the part out (ai job ``ref3d_part_crop``); bits hidden behind a hand or a lance are filled
     from the part's mirror side, inside its convex hull (``fill_occlusion``: shields and helmets are symmetric);
  3. ``generate``: TRELLIS.2 on the crops in one model load, cached by the crop's hash (the regression set reuses
     them);
  4. ``build``: Blender job ``fresh_part`` turns the model to face as the crop does, closes and collapses it; then
     ``bake_colour`` (its colour; the field on the country mask) and ``bake_horse`` (its normal map) from it;
  5. ``check``: the finished part drawn from the front next to its crop: silhouette IoU and the mean colour
     distance (Lab); under ``CHECK`` it is flagged and the fallback is used.
Fallback, still from the crop: ``core.gear`` geometry (a shield's outline traced from the crop, its emblem traced
from the crop, its colours the crop's moved onto the unit's palette).

Plumes, crests and manes always use the horsehair blade (the user's rule after test 32), coloured from their crop
(``hair_colours``). Long thin weapons (lances, spears) keep the raw model's own split geometry (their crop is a
line a few pixels wide), with a baked normal map. Pure Python and Pillow here (the app has no numpy); the jobs
run in Blender and the AI environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from eu4unitgen.core import palette as pal

CHECK = {"iou": 0.78, "colour_de": 18.0}  # a part under these (IoU at least, Lab distance at most) is flagged
# the emblem crisp enough (the user, 2026-10-25: the TRELLIS.2 shield passed shape and colour with a broken-up cross):
# the emblem traced from the part's front view against the crop's, overlap at least this
EMBLEM_MIN = 0.5
PART_TRIS = {"shield": 600, "sword": 300, "helmet": 600}


# ------------------------------------------------------------------ 1. where each part is in the reference
def _bbox(pts2):
    xs, ys = [p[0] for p in pts2], [p[1] for p in pts2]
    return min(xs), min(ys), max(xs), max(ys)


def _raster(pts2, box, size, radius=1):
    """Points (x, y up) drawn as dots in a ``size`` image framed on ``box`` (the image's y down)."""
    x0, y0, x1, y1 = box
    w, h = size
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for x, y in pts2:
        px = (x - x0) / max(x1 - x0, 1e-9) * (w - 1)
        py = (y1 - y) / max(y1 - y0, 1e-9) * (h - 1)
        d.ellipse([px - radius, py - radius, px + radius, py + radius], fill=255)
    return im


def _iou(a: Image.Image, b: Image.Image) -> float:
    a1, b1 = a.point(lambda v: 255 if v > 127 else 0), b.point(lambda v: 255 if v > 127 else 0)
    inter = ImageChops.multiply(a1, b1).histogram()[255]
    union = ImageChops.lighter(a1, b1).histogram()[255]
    return inter / max(union, 1)


def locate(points: dict, fg: Image.Image, keys=("shield", "helmet", "crest", "lance")) -> dict:
    """``points`` (fitted_points.json) projected onto the reference whose subject mask is ``fg`` (L): per key
    {"box", "points", "negative", "hint" (an L image the reference's size)}, plus "mirror" and "fit_iou"."""
    fgb = fg.point(lambda v: 255 if v > 127 else 0)
    W, H = fg.size
    fx0, fy0, fx1, fy1 = fgb.getbbox()
    best = None
    for mirror in (False, True):
        sgn = -1.0 if mirror else 1.0
        allp = [(sgn * p[0], p[1]) for p in points["all"]]
        box = _bbox(allp)
        # the model's front silhouette, framed on the subject's box, at a working size
        ww, hh = 160, max(2, round(160 * (fy1 - fy0) / max(fx1 - fx0, 1)))
        sil = _raster(allp, box, (ww, hh), 2).filter(ImageFilter.MaxFilter(3))
        ref = fgb.crop((fx0, fy0, fx1, fy1)).resize((ww, hh))
        v = _iou(sil, ref)
        if best is None or v > best[0]:
            best = (v, mirror, box)
    v, mirror, (bx0, by0, bx1, by1) = best
    sgn = -1.0 if mirror else 1.0

    def to_px(p):
        return (fx0 + (sgn * p[0] - bx0) / max(bx1 - bx0, 1e-9) * (fx1 - fx0),
                fy0 + (by1 - p[1]) / max(by1 - by0, 1e-9) * (fy1 - fy0))
    out = {"mirror": mirror, "fit_iou": round(v, 3)}
    body = [to_px(p) for p in points["all"][:: max(1, len(points["all"]) // 400)]]
    for key in keys:
        pts = points.get(key) or []
        if len(pts) < 20:
            continue
        px = [to_px(p) for p in pts]
        x0, y0, x1, y1 = _bbox(px)
        mx, my = (x1 - x0) * 0.08 + 4, (y1 - y0) * 0.08 + 4
        cx, cy = sorted(p[0] for p in px)[len(px) // 2], sorted(p[1] for p in px)[len(px) // 2]
        hint = Image.new("L", (W, H), 0)
        hd = ImageDraw.Draw(hint)
        r = max(2.0, 0.012 * (fy1 - fy0))
        for x, y in px:
            hd.ellipse([x - r, y - r, x + r, y + r], fill=255)
        # off the part: body points far from it (two, the nearest such to its box)
        far = sorted((q for q in body if not (x0 - mx <= q[0] <= x1 + mx and y0 - my <= q[1] <= y1 + my)),
                     key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)[:2]
        out[key] = {"box": [max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)],
                    "points": [[cx, cy]], "negative": [list(q) for q in far], "hint": hint}
    return out


# ------------------------------------------------------------------ 2. the crop, its hidden bits filled
def _hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def fill_occlusion(crop: Image.Image) -> tuple[Image.Image, float]:
    """The crop (RGBA, its alpha the part) with the bits hidden behind something filled from the part's mirror
    image across its long axis, only inside the part's convex hull (a foreshortened side stays as seen). -> the
    filled crop and the share of the part's area added."""
    a = crop.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    w, h = a.size
    small = a.resize((w // 4, h // 4))
    sp = small.load()
    pts = [(x, y) for y in range(small.height) for x in range(small.width) if sp[x, y]]
    if len(pts) < 20:
        return crop, 0.0
    n = len(pts)
    cx, cy = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxx = sum((p[0] - cx) ** 2 for p in pts) / n
    syy = sum((p[1] - cy) ** 2 for p in pts) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts) / n
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)  # the long axis
    ux, uy = math.cos(th), math.sin(th)
    cx, cy = cx * 4 + 2, cy * 4 + 2
    r00, r01, r11 = 2 * ux * ux - 1, 2 * ux * uy, 2 * uy * uy - 1
    coeff = (r00, r01, cx - r00 * cx - r01 * cy, r01, r11, cy - r01 * cx - r11 * cy)
    mirror = crop.transform(crop.size, Image.AFFINE, coeff, resample=Image.BILINEAR)
    hull = Image.new("L", (w, h), 0)
    ImageDraw.Draw(hull).polygon([(x * 4 + 2, y * 4 + 2) for x, y in _hull(pts)], fill=255)
    hull = hull.filter(ImageFilter.MinFilter(5))
    ma = mirror.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    add = ImageChops.multiply(ImageChops.multiply(ImageChops.invert(a), ma), hull)
    added = add.histogram()[255] / max(a.histogram()[255], 1)
    out = crop.copy()
    out.paste(mirror, (0, 0), add)
    return out, round(added, 3)


def trim_foreign(crop: Image.Image, band: float = 0.15, max_de: float = 28.0, work: int = 256) -> tuple[Image.Image, float]:
    """A painted part's crop without what covers its edge (a hand gripping a shield's top): pixels in the outer
    ``band`` of its box whose colour is far (Lab ``max_de``) from all of the part's own main colours (three, from the
    middle of the box), in pieces that touch the outside. -> (the crop, the share of its area removed);
    ``fill_occlusion`` then fills the gap from the mirror side."""
    full = crop.size
    a_full = crop.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    box = a_full.getbbox()
    if box is None:
        return crop, 0.0
    k = work / max(full)
    im = crop.resize((max(1, round(full[0] * k)), max(1, round(full[1] * k))), Image.BILINEAR)
    w, h = im.size
    px = im.load()
    x0, y0, x1, y1 = (round(v * k) for v in box)
    bw, bh = x1 - x0, y1 - y0
    mids = [pal.lab(px[x, y][:3]) for y in range(y0 + bh // 5, y1 - bh // 5, 2) for x in range(x0 + bw // 5, x1 - bw // 5, 2)
            if px[x, y][3] > 200]
    if len(mids) < 30:
        return crop, 0.0
    cs = [mids[0], mids[len(mids) // 2], mids[-1]]
    for _ in range(10):
        groups = [[], [], []]
        for q in mids:
            groups[min(range(3), key=lambda i: sum((q[j] - cs[i][j]) ** 2 for j in range(3)))].append(q)
        cs = [tuple(sum(q[j] for q in g) / len(g) for j in range(3)) if g else cs[i] for i, g in enumerate(groups)]
    far = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            if px[x, y][3] <= 127:
                continue
            edge = min(x - x0, x1 - x, y - y0, y1 - y) < band * max(bw, bh)
            if edge:
                q = pal.lab(px[x, y][:3])
                if min(sum((q[j] - c[j]) ** 2 for j in range(3)) ** 0.5 for c in cs) > max_de:
                    far[y * w + x] = 1
    removed = Image.new("L", (w, h), 0)
    rp = removed.load()
    seen = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            if not far[y * w + x] or seen[y * w + x]:
                continue
            stack, piece, touches = [(x, y)], [], False
            seen[y * w + x] = 1
            while stack:
                cx, cy = stack.pop()
                piece.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < w and 0 <= ny < h) or px[nx, ny][3] <= 127:
                        touches = True
                        continue
                    if far[ny * w + nx] and not seen[ny * w + nx]:
                        seen[ny * w + nx] = 1
                        stack.append((nx, ny))
            if touches and len(piece) > 4:
                for cx, cy in piece:
                    rp[cx, cy] = 255
    removed = removed.filter(ImageFilter.MaxFilter(3)).resize(full, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    n = ImageChops.multiply(removed, a_full).histogram()[255]
    if n == 0:
        return crop, 0.0
    out = crop.copy()
    a = ImageChops.subtract(a_full, removed)
    out.putalpha(a)
    return out, round(n / max(a_full.histogram()[255], 1), 3)


def deskew(crop: Image.Image) -> tuple[Image.Image, float]:
    """The crop turned so its part's long axis is upright (the reference's shield leans): (the crop, the turn in
    degrees), recentred in a square of the same size."""
    a = crop.getchannel("A").point(lambda v: 255 if v > 127 else 0).resize((max(1, crop.width // 4), max(1, crop.height // 4)))
    ap = a.load()
    pts = [(x, y) for y in range(a.height) for x in range(a.width) if ap[x, y]]
    if len(pts) < 20:
        return crop, 0.0
    n = len(pts)
    cx, cy = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxx = sum((p[0] - cx) ** 2 for p in pts) / n
    syy = sum((p[1] - cy) ** 2 for p in pts) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts) / n
    th = math.degrees(0.5 * math.atan2(2 * sxy, sxx - syy))  # the long axis from image x (y down)
    turn = th - 90.0 if th > 0 else th + 90.0  # to vertical, the smaller way
    if abs(turn) < 2.0 or sxx > syy:  # upright already, or wider than tall (not a long part)
        return crop, 0.0
    r = crop.rotate(turn, resample=Image.BICUBIC, expand=True)
    box = r.getchannel("A").point(lambda v: 255 if v > 127 else 0).getbbox()
    part = r.crop(box)
    k = min(1.0, 0.9 * crop.width / max(part.size))
    part = part.resize((max(1, round(part.width * k)), max(1, round(part.height * k))), Image.LANCZOS)
    sq = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    sq.paste(part, ((crop.width - part.width) // 2, (crop.height - part.height) // 2))
    return sq, round(turn, 1)


def crops(reference: Path, fg: Path, located: dict, out: Path, ai_python: str, weights: str,
          keys=("shield", "helmet", "crest", "lance")) -> dict:
    """SAM 2 cuts each located part out of ``reference`` -> {key: {"crop", "mask", "box", "added", ...}}."""
    from eu4unitgen.ai.env import AIJobRunner
    out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for key in keys:
        if key not in located:
            continue
        loc = located[key]
        hint = out / f"{key}_hint.png"
        loc["hint"].save(hint)
        p = {"box": loc["box"], "points": loc["points"], "negative": loc["negative"], "hint": str(hint.resolve())}
        if key == "helmet" and "crest" in located:  # the steel, not the plume on it
            p["negative"] = p["negative"] + located["crest"]["points"]
        if key in ("shield", "helmet"):  # kept near where the part projects (a hand gripping its top is cut off)
            x0, y0, x1, y1 = loc["box"]
            p["clip"] = round(0.04 * max(x1 - x0, y1 - y0), 1)
        parts[key] = p
    if not parts:
        return {}
    res = AIJobRunner(ai_python).run_job("ref3d_part_crop", {"image": str(Path(reference).resolve()),
                                                             "fg": str(Path(fg).resolve()), "parts": parts,
                                                             "weights": weights, "out_dir": str(out.resolve())}).result
    for key, r in res.items():
        if r.get("crop") and key in ("shield", "helmet"):
            im = Image.open(r["crop"]).convert("RGBA")
            if key == "shield":  # what covers its edge goes first (the gripping hand)
                im, r["trimmed"] = trim_foreign(im)
            filled, added = fill_occlusion(im)
            r["added"] = added
            if key == "shield":  # upright, so its traced outline and emblem stand straight
                filled, r["deskew_deg"] = deskew(filled)
            filled.save(r["crop"])
    (out / "crops.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------ 3. TRELLIS.2 on the crops (cached)
def crop_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def generate(items: dict, cache: Path, ai_root: Path) -> dict:
    """TRELLIS.2 on each crop (``items``: {key: crop png}) in one model load; a crop seen before (same bytes) comes
    from ``cache``. -> {key: {"glb", "cached", ...}}."""
    import subprocess

    from eu4unitgen.core import config
    from eu4unitgen.ref3d import trellis
    cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for key, crop in items.items():
        glb = cache / f"{crop_hash(crop)}.glb"
        if glb.is_file():
            out[key] = {"glb": str(glb), "cached": True}
        else:
            todo.append((key, crop, glb))
    if todo:
        cfg = config.load()
        w = {**trellis.WSL, "python": cfg.wsl_python or trellis.WSL["python"]}
        job = Path(trellis.__file__).resolve().parents[1] / "ai" / "jobs" / "trellis2_generate.py"
        stem = cache / f"batch_{todo[0][2].stem}"
        result = stem.with_suffix(".result.json")
        args = {"batch": [{"image": trellis.wsl_path(Path(c).resolve()), "output": trellis.wsl_path(g.resolve()),
                           "decimation_target": 100000, "texture_size": 1024} for _, c, g in todo],
                "result_path": trellis.wsl_path(result.resolve()),
                "repo": trellis.wsl_path(ai_root / w["repo"]), "weights": trellis.wsl_path(ai_root / w["weights"]),
                "ss_decoder": trellis.wsl_path(ai_root / w["ss_decoder"]), "dinov3": trellis.wsl_path(ai_root / w["dinov3"]),
                "seed": w["seed"], "pipeline_type": w["pipeline_type"]}
        args_path = stem.with_suffix(".args.json")
        args_path.write_text(json.dumps(args, indent=1), encoding="utf-8")
        log = stem.with_suffix(".log")
        cmd = f"cd /tmp && {w['python']} {trellis.wsl_path(job)} {trellis.wsl_path(args_path.resolve())} > {trellis.wsl_path(log.resolve())} 2>&1"
        rc = subprocess.run(cfg.wsl_base() + ["-e", "bash", "-lc", cmd]).returncode
        if rc != 0 or not result.is_file():
            raise RuntimeError(f"TRELLIS.2 failed on the part crops (exit {rc}); see {log}")
        r = json.loads(result.read_text(encoding="utf-8"))
        for (key, _, glb), item in zip(todo, r["items"]):
            out[key] = {"glb": str(glb), "cached": False, **{k: item.get(k) for k in ("seconds", "faces", "peak_vram_gb")}}
    return out


# ------------------------------------------------------------------ 4. the part built, baked, placed
def build(br, glb: Path, crop: Path, out: Path, key: str, size: float, country_mask: bool, common: dict) -> dict:
    """Blender ``fresh_part`` then the colour and normal bakes -> {"geometry" (front frame: x the crop's right, y up,
    z towards the viewer), "diffuse", "specular", "normal", "view", "default_primary"}."""
    from eu4unitgen.ref3d import trellis
    out.mkdir(parents=True, exist_ok=True)
    basecolor = trellis.glb_basecolor(glb, out / f"{key}_gen_basecolor.png")
    fp = br.run_job("fresh_part", {"source": str(Path(glb).resolve()), "crop": str(Path(crop).resolve()),
                                   "size": float(size), "target_tris": PART_TRIS.get(key, 400),
                                   "close_voxel": max(0.03, size / 90.0), "closing": max(0.05, size / 56.0),
                                   "geometry_out": str(out / f"{key}_gen_low.json"),
                                   "textured_out": str(out / f"{key}_gen_textured.obj")}).result
    bc = br.run_job("bake_colour", {**common, "image": str(basecolor), "size": 512, "country_mask": bool(country_mask),
                                    "low": str(out / f"{key}_gen_low.json"), "high": str(out / f"{key}_gen_textured.obj"),
                                    "out_dir": str(out), "name": f"{key}_gen"}).result
    nb = br.run_job("bake_horse", {"low": str(out / f"{key}_gen_low.json"), "high": [str(out / f"{key}_gen_textured.obj")],
                                   "diffuse": bc["diffuse"], "size": 512, "cage": 0.05, "distance": 0.2,
                                   "high_smooth": 0, "strength": 0.8, "samples": 16, "ao_strength": 0.25,
                                   "preview": False, "out_dir": str(out), "name": f"{key}_genb"}).result
    return {"geometry": str(out / f"{key}_gen_low.json"), "diffuse": str(out / f"{key}_genb_diffuse.png"),
            "specular": bc["specular"], "normal": str(out / f"{key}_genb_normal.png"), "view": fp["view"],
            "tris": fp["tris"], "default_primary": bc.get("default_primary"), "normal_tilt": nb.get("tilt")}


def place(geometry: Path, frame_axes: dict, centre, height: float, out: Path) -> dict:
    """A front-frame part (x right, y up, z towards the viewer) into a node frame: its up along ``up``, its face
    along ``out``, its width along ``across``, ``height`` tall, centred at ``centre``. -> the part geometry json
    (p, tri, uv) for a loadout ``mesh`` part."""
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    p = g["p"]
    ys = p[1::3]
    lo, hi = min(ys), max(ys)
    k = height / max(hi - lo, 1e-9)
    mids = [(min(p[i::3]) + max(p[i::3])) / 2 for i in range(3)]
    ac, up, ou = (frame_axes[n] for n in ("across", "up", "out"))
    q = []
    for i in range(0, len(p), 3):
        x, y, z = ((p[i + j] - mids[j]) * k for j in range(3))
        q.extend(centre[j] + ac[j] * x + up[j] * y + ou[j] * z for j in range(3))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"p": [round(v, 5) for v in q], "tri": g["tri"], "uv": g["u0"]}), encoding="utf-8")
    return {"scale": round(k, 4)}


# ------------------------------------------------------------------ 5. the check: the part next to its crop
def draw_front(geometry: dict, diffuse: Image.Image, size: int = 256, axes=(0, 1, 2)) -> Image.Image:
    """The part drawn from the front (axes: right, up, towards the viewer), textured per pixel (its uv interpolated
    across each triangle), nearer surfaces over farther ones; RGBA framed by its box (aspect kept)."""
    p, tri, uv = geometry["p"], geometry["tri"], geometry.get("u0") or geometry.get("uv")
    ax, ay, az = axes
    xs, ys = p[ax::3], p[ay::3]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) or 1.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out = im.load()
    depth = [[-1e30] * size for _ in range(size)]
    dif = diffuse.convert("RGB")
    dw, dh = dif.size
    dp = dif.load()
    k = size * 0.96 / side
    for t in range(0, len(tri), 3):
        ids = tri[t:t + 3]
        sx = [(p[3 * i + ax] - cx) * k + size / 2 for i in ids]
        sy = [(cy - p[3 * i + ay]) * k + size / 2 for i in ids]
        sz = [p[3 * i + az] for i in ids]
        den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
        if abs(den) < 1e-9:
            continue
        us = [uv[2 * i] for i in ids]
        vs = [uv[2 * i + 1] for i in ids]
        for py in range(max(0, int(min(sy))), min(size, int(max(sy)) + 1)):
            for px in range(max(0, int(min(sx))), min(size, int(max(sx)) + 1)):
                qx, qy = px + 0.5, py + 0.5
                w0 = ((sy[1] - sy[2]) * (qx - sx[2]) + (sx[2] - sx[1]) * (qy - sy[2])) / den
                w1 = ((sy[2] - sy[0]) * (qx - sx[2]) + (sx[0] - sx[2]) * (qy - sy[2])) / den
                w2 = 1 - w0 - w1
                if min(w0, w1, w2) < -1e-6:
                    continue
                z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2]
                if z <= depth[py][px]:
                    continue
                depth[py][px] = z
                u = w0 * us[0] + w1 * us[1] + w2 * us[2]
                v = w0 * vs[0] + w1 * vs[1] + w2 * vs[2]
                out[px, py] = dp[min(dw - 1, max(0, int(u * dw))), min(dh - 1, max(0, int(v * dh)))] + (255,)
    return im


def _framed(im: Image.Image, size: int = 256) -> Image.Image:
    a = im.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    box = a.getbbox()
    if box is None:
        return Image.new("RGBA", (size, size))
    c = im.crop(box)
    k = size * 0.96 / max(c.size)
    c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (size, size))
    out.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return out


def check(front: Image.Image, crop: Image.Image) -> dict:
    """Silhouette IoU and mean Lab colour distance between the part drawn from the front and its crop, both framed
    by their boxes. -> {"iou", "colour_de", "ok"}."""
    a, b = _framed(front), _framed(crop)
    am, bm = a.getchannel("A"), b.getchannel("A")
    iou = _iou(am, bm)
    pa, pb, ma, mb = a.load(), b.load(), am.load(), bm.load()
    des = []
    for y in range(0, 256, 4):
        for x in range(0, 256, 4):
            if ma[x, y] > 127 and mb[x, y] > 127:
                des.append(pal.delta_e(pa[x, y], pb[x, y]))
    de = sum(des) / len(des) if des else 100.0
    return {"iou": round(iou, 3), "colour_de": round(de, 1), "ok": iou >= CHECK["iou"] and de <= CHECK["colour_de"]}


def check_turned(geometry: dict, diffuse: Image.Image, crop: Image.Image, yaws=range(-75, 76, 15)):
    """``check`` from the best of several views turned about the part's up axis (the reference may show a shield
    turned from the camera; the part is built facing it). -> (the result with its "yaw", the front image)."""
    p = geometry["p"]
    best = None
    for yaw in yaws:
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        q = []
        for i in range(0, len(p), 3):
            x, y, z = p[i:i + 3]
            q += [c * x + s_ * z, y, -s_ * x + c * z]
        im = draw_front({**geometry, "p": q}, diffuse)
        r = check(im, crop)
        if best is None or r["iou"] - r["colour_de"] / 100 > best[0]["iou"] - best[0]["colour_de"] / 100:
            best = ({**r, "yaw": yaw}, im)
    return best


def _part_frame(mask: Image.Image, part_alpha: Image.Image, size: int = 256) -> Image.Image:
    """``mask`` (L) cropped to the part's box (``part_alpha``) and set in a square, aspect kept."""
    box = part_alpha.point(lambda v: 255 if v > 127 else 0).getbbox()
    if box is None:
        return Image.new("L", (size, size))
    c = mask.convert("L").crop(box)
    k = size * 0.96 / max(c.size)
    c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.NEAREST)
    out = Image.new("L", (size, size))
    out.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return out


def emblem_match(front: Image.Image, crop: Image.Image) -> float | None:
    """How well the part's emblem matches the crop's: each traced (``two_colours``) and framed by its part's box;
    their overlap (IoU). None when the crop shows no emblem."""
    b = two_colours(crop)
    a = two_colours(front)
    if b is None or a is None:
        return None
    ma = _part_frame(a[2], front.getchannel("A"))
    mb = _part_frame(b[2], crop.getchannel("A"))
    if not mb.getbbox() or not ma.getbbox():
        return None
    # the best over small shifts and scales: an emblem placed a little differently is still crisp; a broken one
    # never lines up
    best = 0.0
    S = ma.width
    for k in (0.9, 0.95, 1.0, 1.05, 1.1):
        m = ma.resize((round(S * k), round(S * k)), Image.NEAREST)
        base = Image.new("L", (S, S))
        base.paste(m, ((S - m.width) // 2, (S - m.height) // 2))
        for dx in range(-16, 17, 4):
            for dy in range(-16, 17, 4):
                sh = ImageChops.offset(base, dx, dy)
                inter = ImageChops.multiply(sh, mb).histogram()[255]
                union = ImageChops.lighter(sh, mb).histogram()[255]
                best = max(best, inter / max(union, 1))
    return round(best, 3)


def check_sheet(rows: list, path: Path) -> Path:
    """Each part next to its crop: rows of (label, crop, front, result)."""
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    S = 256
    sheet = Image.new("RGB", (2 * S + 20, max(1, len(rows)) * (S + 44)), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    for i, (label, crop, front, r) in enumerate(rows):
        y = i * (S + 44)
        for j, im in enumerate((crop, front)):
            cell = Image.new("RGBA", (S, S), (232, 232, 236, 255))
            cell.alpha_composite(_framed(im))
            sheet.paste(cell.convert("RGB"), (j * (S + 20), y + 40))
        verdict = "OK" if r.get("ok") else "FLAGGED"
        d.text((4, y + 4), f"{label}: crop (left) and the part (right)", fill=(20, 20, 24), font=font)
        d.text((4, y + 22), f"IoU {r.get('iou')}, colour distance {r.get('colour_de')}: {verdict}", fill=(20, 20, 24), font=font)
    sheet.save(path)
    return path


# ------------------------------------------------------------------ colours read from a crop
def crop_pixels(crop: Image.Image, step: int = 3) -> list:
    im = crop.convert("RGBA")
    px = im.load()
    return [px[x, y][:3] for y in range(0, im.height, step) for x in range(0, im.width, step) if px[x, y][3] > 200]


def hair_colours(crop: Image.Image) -> tuple[tuple, tuple] | None:
    """Horsehair's dark base and light strands from its crop (what the user had sampled by hand for test 32): the
    median of the 20-50th and of the 80-95th percentiles by lightness, among the hair's own saturated pixels (pale
    highlights and the background's fringe left out)."""
    allpx = crop_pixels(crop)
    sat = [c for c in allpx if _hcl(c)[1] > 0.45]
    px = sorted(sat if len(sat) > 0.3 * len(allpx) else allpx, key=pal.lum)
    if len(px) < 50:
        return None

    def med(a, b):
        s = px[int(len(px) * a): max(int(len(px) * a) + 1, int(len(px) * b))]
        return tuple(sorted(c[k] for c in s)[len(s) // 2] for k in range(3))
    return med(0.2, 0.5), med(0.8, 0.95)


def two_colours(crop: Image.Image, inset: float = 0.08, work: int = 384):
    """A painted face's field and emblem colours (a shield's): k-means with two centres over the pixels inside its
    rim (the outline shrunk by ``inset`` of its height), on a copy ``work`` px across. -> (field, emblem, the emblem
    mask L, the crop's size) or None. The field is the class covering more of the face."""
    full = crop.size
    im = crop.convert("RGBA")
    if max(im.size) > work:
        im = im.resize((round(im.width * work / max(im.size)), round(im.height * work / max(im.size))), Image.LANCZOS)
    a = im.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    box = a.getbbox()
    if box is None:
        return None
    # the rim: ``inset`` of the height, but never more than a sixth of the width (a narrow or foreshortened face)
    r = max(1, int(min((box[3] - box[1]) * inset / 2, (box[2] - box[0]) / 6)))
    inner = a
    for _ in range(max(1, r // 2)):
        inner = inner.filter(ImageFilter.MinFilter(5))
    px = im.load()
    ip = inner.load()
    pts = [px[x, y][:3] for y in range(0, im.height, 3) for x in range(0, im.width, 3) if ip[x, y] > 127]
    if len(pts) < 40:
        return None
    labs = [pal.lab(c) for c in pts]
    cs = [min(labs, key=lambda q: q[0]), max(labs, key=lambda q: q[0])]
    for _ in range(12):
        groups = [[], []]
        for q in labs:
            groups[0 if sum((q[k] - cs[0][k]) ** 2 for k in range(3)) <= sum((q[k] - cs[1][k]) ** 2 for k in range(3)) else 1].append(q)
        cs = [tuple(sum(q[k] for q in g) / len(g) for k in range(3)) if g else cs[i] for i, g in enumerate(groups)]
    field_i = 0 if len(groups[0]) >= len(groups[1]) else 1
    field, emb = pal.rgb(cs[field_i]), pal.rgb(cs[1 - field_i])
    mask = Image.new("L", im.size, 0)
    mp = mask.load()
    for y in range(im.height):
        for x in range(im.width):
            if ip[x, y] > 127:
                q = pal.lab(px[x, y][:3])
                if sum((q[k] - cs[1 - field_i][k]) ** 2 for k in range(3)) < sum((q[k] - cs[field_i][k]) ** 2 for k in range(3)):
                    mp[x, y] = 255
    mask = drop_specks(mask.filter(ImageFilter.MedianFilter(3)), 0.01)
    mask = mask.resize(full, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    return field, emb, mask


def drop_specks(mask: Image.Image, share: float) -> Image.Image:
    """``mask`` (L) without its white pieces smaller than ``share`` of its white area (rivets, stray texels)."""
    w, h = mask.size
    px = mask.load()
    seen = bytearray(w * h)
    pieces = []
    for y in range(h):
        for x in range(w):
            if px[x, y] > 127 and not seen[y * w + x]:
                stack, piece = [(x, y)], []
                seen[y * w + x] = 1
                while stack:
                    cx, cy = stack.pop()
                    piece.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx] and px[nx, ny] > 127:
                            seen[ny * w + nx] = 1
                            stack.append((nx, ny))
                pieces.append(piece)
    total = sum(len(p) for p in pieces)
    out = mask.copy()
    op = out.load()
    for p in pieces:
        if len(p) < share * total:
            for x, y in p:
                op[x, y] = 0
    return out


def outline(crop: Image.Image, n: int = 48) -> list[tuple[float, float]]:
    """A part's outline from its crop's alpha (x right, y up, about its box's centre, height 1): the farthest mask
    pixel from the centre along each of 180 directions (a star-shaped outline: shields), resampled to ``n`` points
    evenly along the perimeter (anticlockwise from the bottom)."""
    star = _star(crop, 180)
    per = [0.0]
    for a, b in zip(star, star[1:] + star[:1]):
        per.append(per[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = per[-1] or 1.0
    out, j = [], 0
    for k in range(n):
        d = total * k / n
        while per[j + 1] < d:
            j += 1
        a, b = star[j], star[(j + 1) % len(star)]
        t = (d - per[j]) / max(per[j + 1] - per[j], 1e-9)
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _star(crop: Image.Image, n: int) -> list[tuple[float, float]]:
    a = crop.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    box = a.getbbox()
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = y1 - y0
    px = a.load()
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n - math.pi / 2  # from the bottom, anticlockwise
        dx, dy = math.cos(t), -math.sin(t)
        r, best = 0.0, 0.0
        while True:
            x, y = int(cx + dx * r), int(cy + dy * r)
            if not (0 <= x < a.width and 0 <= y < a.height):
                break
            if px[x, y]:
                best = r
            r += 1.0
        out.append((dx * best / h, -dy * best / h))
    return out


def snap_image(im: Image.Image, mapping: list[tuple[tuple, tuple]], radius: float = 30.0) -> Image.Image:
    """``im`` with each pixel near (Lab ``radius``) a ``mapping`` source colour shifted by that pair's Lab difference
    (the shading kept), fading out with distance: a part's own colours moved onto the unit's palette."""
    labs = [(pal.lab(s), pal.lab(t)) for s, t in mapping]
    rgb = im.convert("RGB")
    a = im.getchannel("A") if im.mode == "RGBA" else None
    out = Image.new("RGB", rgb.size)
    cache = {}
    src = rgb.load()
    dst = out.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            c = src[x, y]
            key = (c[0] >> 2, c[1] >> 2, c[2] >> 2)
            if key not in cache:
                q = pal.lab(c)
                best = None
                for ls, lt in labs:
                    dd = sum((q[k] - ls[k]) ** 2 for k in range(3)) ** 0.5
                    if dd < radius and (best is None or dd < best[0]):
                        best = (dd, ls, lt)
                if best is None:
                    cache[key] = c
                else:
                    f = 1.0 - (best[0] / radius) ** 2
                    cache[key] = pal.rgb(tuple(q[k] + f * (best[2][k] - best[1][k]) for k in range(3)))
            dst[x, y] = cache[key]
    if a is not None:
        out = out.convert("RGBA")
        out.putalpha(a)
    return out


def copy_part(entry: dict, dest: Path, prefix: str) -> dict:
    """A part entry's files copied under ``dest`` as ``<prefix>_<kind>``; the entry pointing at the copies."""
    dest.mkdir(parents=True, exist_ok=True)
    mesh = dict(entry["mesh"])
    for k, v in list(mesh.items()):
        q = dest / f"{prefix}_{k}{Path(v).suffix}"
        if Path(v).resolve() != q.resolve():
            shutil.copyfile(v, q)
        mesh[k] = str(q)
    return {**entry, "mesh": mesh}


# ------------------------------------------------------------------ a part's texture made part of the design
def finish_texture(diffuse: Path, specular: Path, mapping: list, primary, cloth, darken: float, out_prefix: Path,
                   gloss: dict | None = None) -> dict:
    """The part's baked colour moved onto the unit's palette (``mapping``: (its colour, the palette's)), ``darken``
    (the user, test 34: "a little darker overall so it doesn't stand out more than the armour"), and with a
    ``primary`` the texels near the palette's ``cloth`` put on the country mask (a grey whose product with the
    unit's primary gives the cloth back; spec R 255). -> {"diffuse", "specular", "mask_share"}."""
    d = snap_image(Image.open(diffuse).convert("RGB"), mapping)
    if darken:
        d = d.point(lambda v: int(v * (1.0 - darken)))
    sp = Image.open(specular).convert("RGBA")
    if sp.size != d.size:
        sp = sp.resize(d.size, Image.NEAREST)
    n = 0
    if primary is not None and cloth is not None:
        lp = pal.lum(primary) / 255.0
        cl = pal.lab(cloth)
        dp, ss = d.load(), sp.load()
        cache = {}
        for y in range(d.height):
            for x in range(d.width):
                c = dp[x, y]
                key = (c[0] >> 2, c[1] >> 2, c[2] >> 2)
                if key not in cache:
                    q = pal.lab(c)
                    cache[key] = sum((q[k] - cl[k]) ** 2 for k in range(3)) ** 0.5 < 22.0
                if cache[key]:
                    g = int(round(min(1.0, pal.lum(c) / 255.0 / max(lp, 1e-6)) * 255))
                    dp[x, y] = (g, g, g)
                    _, g_, b_, a_ = ss[x, y]
                    ss[x, y] = (255, g_, b_, (gloss or {}).get("matte", a_))
                    n += 1
    paths = {"diffuse": str(out_prefix) + "_diffuse.png", "specular": str(out_prefix) + "_specular.png"}
    d.convert("RGBA").save(paths["diffuse"])
    sp.save(paths["specular"])
    return {**paths, "mask_share": round(n / (d.width * d.height), 3)}


def project_emblem(geometry: Path, diffuse: Path, emblem: Image.Image, inner: Image.Image, box, yaw: float, colour,
                   field, out: Path) -> dict:
    """A crisp emblem on a front-frame part's texture (the user: emblems as crisp decals; TRELLIS.2 blurs them),
    seen as the crop sees the part (turned ``yaw`` degrees about its up axis; ``check_turned`` finds it; the part's
    box framed on ``box``, its box in the crop): texels on faces turned to the viewer whose point falls on the crop's ``emblem`` mask are painted
    ``colour``; the rest of the face inside its rim (``inner``: the crop's mask shrunk past the rim) turns ``field``,
    keeping its own light and dark (the blurred emblem TRELLIS.2 baked goes). -> {"emblem_texels", "field_texels"}."""
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    p0, tri, uv = g["p"], g["tri"], g["u0"]
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    p = []
    for i in range(0, len(p0), 3):
        x, y, z = p0[i:i + 3]
        p += [c * x + s_ * z, y, -s_ * x + c * z]
    xs, ys = p[0::3], p[1::3]
    pcx, pcy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    pside = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    em, inn = emblem.convert("L"), inner.convert("L")
    bx0, by0, bx1, by1 = box
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    k = max(bx1 - bx0, by1 - by0) / pside
    ep, ip = em.load(), inn.load()
    ew, eh = em.size
    img = Image.open(diffuse).convert("RGBA")
    W, H = img.size
    px = img.load()
    flum = max(1.0, sum(field) / 3)
    seen_lum = []
    jobs = []
    for t in range(0, len(tri), 3):
        i, j, kk = tri[t:t + 3]
        a, b, cc = (p[3 * q:3 * q + 3] for q in (i, j, kk))
        if (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0]) <= 0:
            continue  # facing away from the viewer: the back and the edges
        jobs.append((i, j, kk, a, b, cc))
    hits = []
    for i, j, kk, a, b, cc in jobs:
        ua = [(uv[2 * q] * W, uv[2 * q + 1] * H) for q in (i, j, kk)]
        den = (ua[1][1] - ua[2][1]) * (ua[0][0] - ua[2][0]) + (ua[2][0] - ua[1][0]) * (ua[0][1] - ua[2][1])
        if abs(den) < 1e-9:
            continue
        for ty in range(max(0, int(min(u[1] for u in ua))), min(H, int(max(u[1] for u in ua)) + 2)):
            for tx in range(max(0, int(min(u[0] for u in ua))), min(W, int(max(u[0] for u in ua)) + 2)):
                qx, qy = tx + 0.5, ty + 0.5
                w0 = ((ua[1][1] - ua[2][1]) * (qx - ua[2][0]) + (ua[2][0] - ua[1][0]) * (qy - ua[2][1])) / den
                w1 = ((ua[2][1] - ua[0][1]) * (qx - ua[2][0]) + (ua[0][0] - ua[2][0]) * (qy - ua[2][1])) / den
                w2 = 1 - w0 - w1
                if min(w0, w1, w2) < -1e-6:
                    continue
                X = w0 * a[0] + w1 * b[0] + w2 * cc[0]
                Y = w0 * a[1] + w1 * b[1] + w2 * cc[1]
                ex, ey = int(bcx + (X - pcx) * k), int(bcy - (Y - pcy) * k)
                if 0 <= ex < ew and 0 <= ey < eh and ip[ex, ey] > 127:
                    hits.append((tx, ty, ep[ex, ey] > 127))
                    if ep[ex, ey] <= 127:
                        seen_lum.append(sum(px[tx, ty][:3]) / 3)
    mean = sorted(seen_lum)[len(seen_lum) // 2] if seen_lum else flum
    ne = nf = 0
    for tx, ty, on in hits:
        if on:
            px[tx, ty] = tuple(int(v) for v in colour) + (255,)
            ne += 1
        else:
            f = max(0.7, min(1.2, (sum(px[tx, ty][:3]) / 3) / max(mean, 1.0)))
            px[tx, ty] = tuple(min(255, int(v * f)) for v in field) + (255,)
            nf += 1
    img.save(out)
    return {"emblem_texels": ne, "field_texels": nf}


def palette_crop(crop: Image.Image, mapping: list, darken: float) -> Image.Image:
    """The crop as the finished part means to look: its colours moved onto the palette and darkened (what the
    check compares the part with)."""
    out = snap_image(crop, mapping)
    if darken:
        a = out.getchannel("A")
        out = out.convert("RGB").point(lambda v: int(v * (1.0 - darken))).convert("RGBA")
        out.putalpha(a)
    return out


def emblem_box(mask: Image.Image):
    """The part's box in its crop (alpha or mask, L)."""
    return mask.convert("L").point(lambda v: 255 if v > 127 else 0).getbbox()


def inner_face(crop: Image.Image, inset: float = 0.08) -> Image.Image:
    """The crop's mask shrunk past the rim (``two_colours``' rule), full size."""
    a = crop.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    box = a.getbbox()
    small = a.resize((max(1, a.width // 4), max(1, a.height // 4)))
    r = max(1, int(min((box[3] - box[1]) * inset / 2, (box[2] - box[0]) / 6) / 4))
    for _ in range(max(1, r // 2)):
        small = small.filter(ImageFilter.MinFilter(5))
    inner = small.resize(a.size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    return inner


def _is_gold(c) -> bool:
    r, g, b = (v / 255.0 for v in c[:3])
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 0.25 or (mx - mn) / max(mx, 1e-6) < 0.3:
        return False
    return 28 < math.degrees(math.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)) % 360 < 58


def _hcl(c):
    r, g, b = (v / 255.0 for v in c[:3])
    mx, mn = max(r, g, b), min(r, g, b)
    return (math.degrees(math.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)) % 360, (mx - mn) / max(mx, 1e-6), mx)


BRONZE = (17.0, 60.0)  # gold to bronze hues on a reference (the cataphract's finial is 21-29; the bakes' gold 28-58)


def read_helmet(crop: Image.Image) -> dict:
    """What gold the reference's helmet shows (its crop, front view). The plume's colour (the crop's dominant
    saturated non-gold hue) is ignored, as is everything under the brow but the nasal: the brow band is the row of
    the most gold in the middle of the steel's height; the dome the steel above it; a finial is gold above the
    dome's top at the centre; ribs are gold columns up the dome (the centre one too); a nasal guard is gold at the
    centre just under the brow. -> the flags, the gold's share of the helmet and the brow's row."""
    im = crop.convert("RGBA")
    im = im.resize((128, max(1, round(128 * im.height / im.width))))
    w0, h0 = im.size
    px = im.load()
    hues = {}
    for y in range(h0):
        for x in range(w0):
            if px[x, y][3] > 127:
                hue, ch, mx = _hcl(px[x, y])
                if ch > 0.45 and mx > 0.2 and not (BRONZE[0] < hue < BRONZE[1]):
                    hues[int(hue // 15)] = hues.get(int(hue // 15), 0) + 1
    plume = (max(hues, key=hues.get) + 0.5) * 15 if hues else None

    def kind(c):
        if c[3] <= 127:
            return None
        hue, ch, mx = _hcl(c)
        if plume is not None and ch > 0.45 and abs((hue - plume + 180) % 360 - 180) < 22:
            return None  # the plume
        if mx > 0.25 and ch > 0.3 and BRONZE[0] < hue < BRONZE[1]:
            return "gold"
        return "steel"
    grid = [[kind(px[x, y]) for x in range(w0)] for y in range(h0)]
    rows = [(y, [g for g in grid[y] if g]) for y in range(h0)]
    rows = [(y, r) for y, r in rows if len(r) > 2]
    if not rows:
        return {"gold_share": 0.0}
    y0, y1 = rows[0][0], rows[-1][0]
    h = max(1, y1 - y0)
    mid = [(y, sum(1 for g in r if g == "gold") / len(r)) for y, r in rows if y0 + 0.3 * h <= y <= y0 + 0.8 * h]
    by, brow = max(mid, key=lambda t: t[1]) if mid else (y0 + int(0.6 * h), 0.0)
    xs = [x for y in range(y0, by) for x in range(w0) if grid[y][x]]
    cx = (min(xs) + max(xs)) / 2 if xs else w0 / 2
    hw = (max(xs) - min(xs)) / 2 if xs else w0 / 4
    # the dome's top: the first row where steel spans a third of the helmet's width
    dome_top = next((y for y in range(y0, by) if sum(1 for g in grid[y] if g == "steel") > 0.66 * hw), y0)

    def share(xa, xb, ya, yb):
        cells = [grid[y][x] for y in range(max(0, ya), min(h0, yb)) for x in range(max(0, int(xa)), min(w0, int(xb))) if grid[y][x]]
        return sum(1 for g in cells if g == "gold") / max(len(cells), 1), len(cells)
    fin, fin_n = share(cx - 0.3 * hw, cx + 0.3 * hw, y0, dome_top)
    nas, _ = share(cx - 0.12 * hw, cx + 0.12 * hw, by + 2, by + 2 + int(0.25 * h))
    dome_y = (dome_top + int(0.15 * (by - dome_top)), by - 2)
    cols = [share(x, x + 1, *dome_y)[0] for x in range(int(cx - hw), int(cx + hw) + 1)]
    ribs = sum(1 for i, v in enumerate(cols) if v > 0.5 and (i == 0 or cols[i - 1] <= 0.5))
    gold_all = sum(1 for r in grid for g in r if g == "gold")
    any_all = sum(1 for r in grid for g in r if g)
    return {"gold_share": round(gold_all / max(any_all, 1), 3), "finial": fin > 0.35 and fin_n > 15,
            "brow": brow > 0.4, "nasal": nas > 0.35, "ribs": ribs >= 2, "rib_columns": ribs,
            "brow_row": round((by - y0) / h, 2)}