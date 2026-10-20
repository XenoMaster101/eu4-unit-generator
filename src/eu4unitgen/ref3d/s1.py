"""Track M stage S1: reference analysis, up to the user's checkpoint H1 (SPEC Track M).

From a subject config (``assets_in/<subject>/subject.json``; every coordinate in the original's
pixels) and the original reference, all of it reproducible:
  1. the views and close-ups cropped (the original is never written);
  2. per view, a removal mask from the config's polygons (rider, lances, reins ...);
  3. SDXL inpainting, one image per seed (AI job ref3d_inpaint), ranked by seam score;
  4. horse masks (BiRefNet; AI job ref3d_mask) of the originals and of every cleaned image;
  5. the rider-gone check: inside each view's ``background_above`` polygon (above the horse's back
     line, clear of the saddle) the cleaned image must be background (BiRefNet foreground share at
     most RIDER_GONE), else the seed is rejected whatever its seam score (S1 v001: SDXL drew a new
     rider in 6 of 8 seeds, and the seam score proposed one of them); then proposed region labels
     and palettes on the chosen seed (AI job ref3d_labels): the config's ``chosen`` seed per view,
     else the passing seed with the lowest seam score (a proposal, not a decision; none when no
     seed passes);
  6. the H1 sheets: cleaning (original, mask, every seed, horse mask), labels, landmarks,
     conflicts between the views (the original's crops side by side).
Outputs go to ``<subject>/refs/clean/vNNN/`` and ``<subject>/h1/vNNN/`` (a new number every run;
nothing is overwritten) with a manifest: input hashes, settings, seeds, results.

usage: python -m eu4unitgen.ref3d.s1 <subject.json> --ai-python <env python> --sdxl <weights dir>
       --u2net <rembg model dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEMPLATES = Path(__file__).with_name("templates")


def load_subject(path: Path) -> dict:
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "template", "original", "views"):
        if key not in s:
            raise ValueError(f"subject config {path}: missing {key!r}")
    for name, v in s["views"].items():
        x0, y0, x1, y1 = v.get("crop", (0, 0, 0, 0))
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"view {name}: crop must be [x0, y0, x1, y1] with x1 > x0, y1 > y0")
    s["_root"] = Path(path).resolve().parent
    s["_template"] = json.loads((TEMPLATES / f"{s['template']}.json").read_text(encoding="utf-8"))
    unknown = {k for v in s["views"].values() for k in v.get("landmarks", {})} - set(s["_template"]["landmarks"])
    if unknown:
        raise ValueError(f"landmarks not in the {s['template']} template: {sorted(unknown)}")
    return s


RIDER_GONE = 0.10  # assumed: at most this share of a view's background_above region may be foreground


def fg_share(fg_path, region: Image.Image):
    """Share of ``region`` (an L mask) that the foreground mask at ``fg_path`` covers."""
    fg = Image.open(fg_path).convert("L").point(lambda x: 255 if x > 127 else 0)
    reg = region.convert("L").point(lambda x: 255 if x > 127 else 0)
    inside = reg.histogram()[255]
    both = Image.composite(fg, Image.new("L", fg.size, 0), reg).histogram()[255]
    return round(both / inside, 4) if inside else None


def to_crop(points, crop):
    return [(float(x) - crop[0], float(y) - crop[1]) for x, y in points]


def polygons(z):
    """A zone's value: one polygon or a list of them -> list of polygons."""
    return [z] if z and isinstance(z[0][0], (int, float)) else list(z or [])


def polygon_mask(size, polys) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon([tuple(q) for q in p], fill=255)
    return m


def next_version(folder: Path) -> Path:
    n = 1
    while (folder / f"v{n:03d}").exists():
        n += 1
    out = folder / f"v{n:03d}"
    out.mkdir(parents=True)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_views(s: dict, out: Path) -> dict:
    sheet = Image.open(s["_root"] / s["original"]).convert("RGB")
    paths = {}
    for name, v in s["views"].items():
        p = out / f"{name}.png"
        sheet.crop(tuple(v["crop"])).save(p)
        paths[name] = p
    return paths


def removal_masks(s: dict, crops: dict, out: Path) -> dict:
    paths = {}
    for name, v in s["views"].items():
        if not v.get("remove"):
            continue
        size = Image.open(crops[name]).size
        polys = [to_crop(p, v["crop"]) for group in v["remove"].values() for p in polygons(group)]
        p = out / f"{name}_remove.png"
        polygon_mask(size, polys).save(p)
        paths[name] = p
    return paths


def zones_in_crop(v: dict) -> dict:
    return {z: [to_crop(p, v["crop"]) for p in polygons(polys)] for z, polys in (v.get("zones") or {}).items()}


# ---- sheets (Pillow only) ----

def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit(im: Image.Image, w: int, h: int) -> Image.Image:
    im = im.convert("RGB")
    s = min(w / im.size[0], h / im.size[1])
    small = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS)
    tile = Image.new("RGB", (w, h), (28, 28, 28))
    tile.paste(small, ((w - small.size[0]) // 2, (h - small.size[1]) // 2))
    return tile


def tint(img: Image.Image, mask: Image.Image, rgb=(255, 40, 40), alpha=0.55) -> Image.Image:
    over = Image.new("RGB", img.size, rgb)
    return Image.composite(Image.blend(img.convert("RGB"), over, alpha), img.convert("RGB"), mask.convert("L"))


def grid_sheet(rows, title, path, tile=(300, 360), notes=()):
    """rows: [(row label, [(caption, image, highlight)])] -> one labelled sheet."""
    tw, th = tile
    cols = max(len(r[1]) for r in rows)
    cap, lab, top = 40, 150, 46
    H = top + len(rows) * (th + cap) + 22 * len(notes) + 16
    sheet = Image.new("RGB", (lab + cols * tw, H), (36, 36, 36))
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), title, fill=(240, 240, 240), font=_font(22))
    f, fs = _font(15), _font(13)
    for i, (label, tiles) in enumerate(rows):
        y = top + i * (th + cap)
        d.multiline_text((10, y + 10), label, fill=(220, 220, 220), font=f)
        for j, (caption, im, hi) in enumerate(tiles):
            x = lab + j * tw
            sheet.paste(_fit(im, tw - 6, th), (x + 3, y))
            if hi:
                d.rectangle((x + 2, y - 1, x + tw - 4, y + th), outline=(90, 230, 90), width=3)
            d.multiline_text((x + 4, y + th + 3), caption, fill=(90, 230, 90) if hi else (210, 210, 210), font=fs)
    for k, line in enumerate(notes):
        d.text((10, top + len(rows) * (th + cap) + 22 * k), line, fill=(220, 220, 220), font=f)
    sheet.save(path)
    return path


def landmark_image(img: Image.Image, marks: dict, crop) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = _font(11)
    for name, (x, y) in marks.items():
        x, y = x - crop[0], y - crop[1]
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 0), width=2)
        d.text((x + 6, y - 6), name, fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def conflict_sheet(s: dict, path: Path, cleaned: dict):
    sheet_img = Image.open(s["_root"] / s["original"]).convert("RGB")
    rows = []
    for c in s.get("conflicts", []):
        tiles = []
        for view in ("L", "R"):
            box = c.get(view)
            if box:
                tiles.append((f"view {view}, original", sheet_img.crop(tuple(box)), False))
                v = s["views"][view]
                if view in cleaned:
                    cx = [box[0] - v["crop"][0], box[1] - v["crop"][1], box[2] - v["crop"][0], box[3] - v["crop"][1]]
                    tiles.append((f"view {view}, cleaned", Image.open(cleaned[view]).crop(tuple(cx)), False))
        label = f"{c['id']}\nrule: {c['rule']}\nproposal:\n{c['proposal']}"
        rows.append((_wrap(label, 20), tiles))
    notes = [f"{c['id']}: {c['what']}" for c in s.get("conflicts", [])]
    return grid_sheet(rows, f"H1 conflicts between the views: {s['name']}", path, tile=(300, 300), notes=notes)


def _wrap(text, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            if len(line) + len(word) + 1 > width and line:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return "\n".join(out)


# ---- the stage ----

def run(subject_path: Path, ai_python: str, sdxl: str, u2net: str) -> dict:
    from eu4unitgen.ai.env import AIJobRunner

    s = load_subject(subject_path)
    root = s["_root"]
    clean = next_version(root / "refs" / "clean")
    h1 = root / "h1" / clean.name
    h1.mkdir(parents=True)
    t0 = time.time()
    manifest = {"stage": "S1", "subject": s["name"], "version": clean.name,
                "inputs": {"original": {"path": s["original"], "sha256": sha256(root / s["original"])},
                           "subject": {"path": str(subject_path), "sha256": sha256(subject_path)}},
                "inpaint": s.get("inpaint"), "views": {}}
    crops = crop_views(s, clean)
    masks = removal_masks(s, crops, clean)
    ai = AIJobRunner(ai_python, env={"U2NET_HOME": u2net})
    ip = s["inpaint"]
    seeds = {}
    for name, mask in masks.items():
        r = ai.run_job("ref3d_inpaint", {"weights": sdxl, "image": str(crops[name]), "mask": str(mask), "out_dir": str(clean),
                                         "name": name, **{k: ip[k] for k in ("seeds", "prompt", "negative", "steps",
                                                                            "guidance", "strength", "long_side", "grow",
                                                                            "feather", "prefill") if k in ip}}).result
        seeds[name] = r
    images = {f"{n}_original": str(p) for n, p in crops.items() if n in masks}
    images.update({f"{n}_seed{k}": v["path"] for n, r in seeds.items() for k, v in r["seeds"].items()})
    fg = ai.run_job("ref3d_mask", {"images": images, "out_dir": str(clean)}).result
    chosen = {}
    for name, r in seeds.items():
        v = s["views"][name]
        region = polygon_mask(Image.open(crops[name]).size, [to_crop(p, v["crop"]) for p in polygons(v.get("background_above"))])
        for k, sd in r["seeds"].items():
            sd["fg_above"] = fg_share(fg[f"{name}_seed{k}"]["path"], region) if v.get("background_above") else None
            sd["rider_gone"] = sd["fg_above"] is None or sd["fg_above"] <= RIDER_GONE
        want = (s.get("chosen") or {}).get(name)
        ok = [k for k in r["seeds"] if r["seeds"][k]["rider_gone"]]
        chosen[name] = str(want) if want is not None else (min(ok, key=lambda k: r["seeds"][k]["seam"]) if ok else None)
        r["passing"] = len(ok)
    t = s["_template"]
    lab = ai.run_job("ref3d_labels", {
        "views": {n: {"image": seeds[n]["seeds"][k]["path"], "fg": fg[f"{n}_seed{k}"]["path"],
                      "inpainted": str(clean / f"{n}_mask_used.png"), "zones": zones_in_crop(s["views"][n])}
                  for n, k in chosen.items() if k is not None},
        "rules": s["colour_classes"]["rules"], "zone_map": s["colour_classes"]["zone_map"],
        "colours": t["classes"], "out_dir": str(clean)}).result
    # sheets
    rows = []
    for name, r in seeds.items():
        v = s["views"][name]
        orig = Image.open(crops[name])
        tiles = [("original (kept untouched)", orig, False),
                 (f"removal mask\n{r['mask_share'] * 100:.1f}% of the crop", tint(orig, Image.open(clean / f"{name}_mask_used.png")), False)]
        if v.get("background_above"):
            tiles.append(("rider-gone check region\n(must be background)", tint(orig, polygon_mask(
                orig.size, [to_crop(p, v["crop"]) for p in polygons(v["background_above"])]), (60, 120, 255)), False))
        for k, sd in r["seeds"].items():
            verdict = "rider gone" if sd["rider_gone"] else "REJECTED: rider still there"
            fa = "" if sd["fg_above"] is None else f" ({sd['fg_above'] * 100:.0f}% fg)"
            tiles.append((f"seed {k}: seam {sd['seam']:.2f}\n{verdict}{fa}" + ("\nPROPOSED" if k == chosen[name] else ""),
                          Image.open(sd["path"]), k == chosen[name]))
        if chosen[name] is not None:
            cimg = Image.open(r["seeds"][chosen[name]]["path"])
            tiles.append(("horse mask (BiRefNet),\nproposed seed", tint(cimg, Image.open(fg[f"{name}_seed{chosen[name]}"]["path"]), (40, 220, 90), 0.45), False))
        rows.append((_wrap(f"view {name}\n{v['role']}", 18), tiles))
    closeups = [(f"{n}: {v['role']}", Image.open(crops[n]), False) for n, v in s["views"].items() if n not in masks]
    if closeups:
        rows.append(("close-ups\n(not cleaned)", closeups))
    grid_sheet(rows, f"H1 cleaning: {s['name']} {clean.name} (green = proposed; the original is never changed)",
               h1 / "h1_cleaning.png",
               notes=["seam = gradient on the inpaint edge / just outside it (1 = no visible edge): a ranking aid only.",
                      "Inpainted pixels are not verified information: they are kept apart in every later stage."])
    lrows = []
    for name, k in chosen.items():
        if k is None:
            continue
        v, L = s["views"][name], lab[name]
        cimg = Image.open(seeds[name]["seeds"][k]["path"])
        shares = ", ".join(f"{c} {x * 100:.0f}%" for c, x in sorted(L["shares"].items(), key=lambda kv: -kv[1]))
        lrows.append((_wrap(f"view {name}\nseed {k}", 18),
                      [("labels over the cleaned view\n(yellow line: inpainted area)", Image.open(L["overlay"]), False),
                       ("label map", Image.open(L["labels"]), False),
                       (f"landmarks ({len(v.get('landmarks', {}))})", landmark_image(cimg, v.get("landmarks", {}), v["crop"]), False),
                       (_wrap(shares, 40) + f"\ninpainted: {L['inpainted_share'] * 100:.0f}% of the horse", Image.new("RGB", (10, 10), (36, 36, 36)), False)]))
    legend = "  ".join(f"{c}: {tuple(rgb)}" for c, rgb in t["classes"].items())
    grid_sheet(lrows, f"H1 labels and landmarks: {s['name']} {clean.name} (proposed; you approve or correct)",
               h1 / "h1_labels.png", tile=(380, 460), notes=["classes (label colours): " + legend])
    conflict_sheet(s, h1 / "h1_conflicts.png", {n: seeds[n]["seeds"][k]["path"] for n, k in chosen.items() if k is not None})
    manifest["views"] = {n: {"crop": s["views"][n]["crop"], "inpaint": seeds.get(n), "chosen_seed": chosen.get(n),
                             "labels": lab.get(n)} for n in s["views"]}
    manifest["masks"] = fg
    manifest["sheets"] = {k: str(h1 / f"h1_{k}.png") for k in ("cleaning", "labels", "conflicts")}
    manifest["seconds"] = round(time.time() - t0, 1)
    (clean / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("subject", type=Path)
    p.add_argument("--ai-python", required=True)
    p.add_argument("--sdxl", required=True)
    p.add_argument("--u2net", required=True)
    a = p.parse_args(argv)
    m = run(a.subject, a.ai_python, a.sdxl, a.u2net)
    print(json.dumps({k: m[k] for k in ("version", "sheets", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
