"""The quality report for a new model, in plain language (Track P1.5; pure Python and Pillow: the app shows it).

Five measures, each with a verdict (good / ok / check) and a sentence:
  silhouette   the game version's outline against the reference's (its side for a horse, its front for a rider):
               both framed by their boxes, mirrored or not (whichever fits), overlap as IoU
  deformation  folded-over triangles (flips) and the worst stretch in idle, moving and attack, against vanilla's
  parts        kept (the reference's own), rebuilt (our geometry from the reference: tail, plume, shield) or lost
               (too thin for the budget: reins, stirrup leathers), with each part's check against its crop
  texture      how much of the texture the model's UVs use, and how much the bake reached
  map zoom     the unit's brightness and contrast at map zoom against vanilla's, from the gallery's map row
  animation    a soldier on foot: the animation review's flags (ref3d.review: bends, clipping, floating gear against
               a base-game unit playing the same animations)
  joints       a soldier: the skeleton's joints inside his body (each spine bone's distance from where vanilla's
               body has it in depth, each leg's from vanilla's at the knee and ankle; the joint fit's own measure)

``report(...)`` -> {"measures": {name: {"verdict", "text", ...numbers}}, "summary"}; ``markdown(report)``.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

GOOD, OK, CHECK = "good", "ok", "check"


def _mask(im: Image.Image, size: int = 200) -> Image.Image | None:
    """An image's subject (alpha, else what differs from its corner) cropped to its box and set in a square."""
    im = im.convert("RGBA")
    a = im.getchannel("A")
    if a.getextrema()[0] > 250:  # no transparency: the subject is what isn't the corner's colour
        bg = im.getpixel((2, 2))
        diff = ImageChops.difference(im.convert("RGB"), Image.new("RGB", im.size, bg[:3])).convert("L")
        a = diff.point(lambda v: 255 if v > 18 else 0)
    a = a.point(lambda v: 255 if v > 127 else 0)
    box = a.getbbox()
    if box is None:
        return None
    c = a.crop(box)
    k = size / max(c.size)
    c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))))
    out = Image.new("L", (size, size), 0)
    out.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return out


def silhouette_iou(model: Image.Image, reference_fg: Image.Image, band: float | None = None) -> float:
    """Overlap of two outlines framed by their boxes, the better of as-is and mirrored. ``band``: only the central
    columns, that share of the frame (a rider: its head, torso and legs; its arms are posed to the hand nodes)."""
    a, b = _mask(model), _mask(reference_fg)
    if a is None or b is None:
        return 0.0
    if band:
        keep = Image.new("L", a.size, 0)
        x0 = int(a.width * (0.5 - band / 2))
        keep.paste(255, (x0, 0, a.width - x0, a.height))
        a, b = ImageChops.multiply(a, keep), ImageChops.multiply(b, keep)
    best = 0.0
    for m in (a, a.transpose(Image.FLIP_LEFT_RIGHT)):
        inter = ImageChops.multiply(m, b).histogram()[255]
        union = ImageChops.lighter(m, b).histogram()[255]
        best = max(best, inter / max(union, 1))
    return round(best, 3)


def unit_light(path: Path) -> dict | None:
    """Mean brightness (0-255) and contrast (the standard deviation of brightness) of a gallery image's unit pixels
    (what differs from the background)."""
    im = Image.open(path).convert("RGB")
    m = _raw_mask(im)
    if m is None:
        return None
    st = ImageStat.Stat(im.convert("L"), m)
    return {"mean": round(st.mean[0], 1), "contrast": round(st.stddev[0], 1), "pixels": int(m.histogram()[255])}


def _raw_mask(im: Image.Image) -> Image.Image | None:
    bg = im.getpixel((2, 2))
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg)).convert("L").point(lambda v: 255 if v > 18 else 0)
    return diff if diff.getbbox() else None


def _verdict(v, good, ok, higher=True):
    if v is None:
        return CHECK
    if higher:
        return GOOD if v >= good else OK if v >= ok else CHECK
    return GOOD if v <= good else OK if v <= ok else CHECK


def deformation(deform: dict, kind: str) -> dict:
    ours, van = deform.get("ours") or {}, deform.get("vanilla") or {}
    order = {GOOD: 0, OK: 1, CHECK: 2}
    rows, worst = [], CHECK if not ours else GOOD
    for anim in ("idle", "moving", "attack"):
        o, v = ours.get(anim) or {}, van.get(anim) or {}
        if not o:
            continue
        f, vf = o.get("flips", 0), v.get("flips", 0)
        # flips: vanilla's own plus a margin that grows with it; the M23 rider sits near 3x vanilla's
        # ok up to the approved cataphract rider's level (test 35: 33 in moving against vanilla's 0)
        verdict = GOOD if f <= 1.5 * vf + 10 else OK if f <= 3.5 * vf + 35 else CHECK
        if o.get("max", 0) > max(2 * v.get("max", 0), 6.0):
            verdict = CHECK
        rows.append((anim, f, vf, o.get("max"), v.get("max"), verdict))
        if order[verdict] > order[worst]:
            worst = verdict
    who = (f"vanilla's infantry, {deform['stance'].replace('_', ' ')}" if deform.get("stance")
           else "vanilla's rider" if kind == "rider" else "the vanilla horse")
    text = ("In idle, moving and attack, " + ", ".join(f"{f}" for _, f, _, _, _, _ in rows)
            + f" triangles fold over ({who}: " + ", ".join(f"{vf}" for _, _, vf, _, _, _ in rows) + "); the worst stretch is "
            + ", ".join(f"{m}" for _, _, _, m, _, _ in rows) + " (" + ", ".join(f"{vm}" for _, _, _, _, vm, _ in rows) + ").")
    if worst == CHECK:
        text += " Watch it move in game: some triangles may visibly fold or stretch."
    return {"verdict": worst, "text": text, "rows": [list(r) for r in rows]}


# the joint fit (the user, T38): limits on the fit's own measure (fresh_rider.joints_report) of the low mesh; the
# fitted English soldier 0.17 / 0.15, the cataphract 0.14 / 0.54 (his right knee inside his skirt); 1.0-1.7 before
JOINTS_SPINE = (0.25, 0.4)
JOINTS_LEGS = (0.35, 0.6)


def joints(manifest: dict) -> dict | None:
    """The joints measure (module doc) from the build's ``fresh.joints``; None for builds made before the joint fit."""
    j = (manifest.get("fresh") or {}).get("joints")
    if not j:
        return None
    spine = max((j.get("spine_off") or {}).values(), default=0.0)
    legs = max((v for d in (j.get("legs_off") or {}).values() for v in d), default=0.0)
    verdict = (GOOD if spine <= JOINTS_SPINE[0] and legs <= JOINTS_LEGS[0]
               else OK if spine <= JOINTS_SPINE[1] and legs <= JOINTS_LEGS[1] else CHECK)
    text = (f"The skeleton's joints sit inside the body as on the game's own: the spine within {spine:.2f}, the knees "
            f"and ankles within {legs:.2f} of where the game's body has them")
    text += "." if verdict != CHECK else ": the joints sit outside the body, so it will fold and squash when it moves."
    return {"verdict": verdict, "text": text, "spine": spine, "legs": legs}


ANIMATION_FLAGS = (0, 6)  # good with none, ok up to this many (of 15 poses x 3 measures)


def animation(rev: dict | None) -> dict | None:
    """The animation measure from ``ref3d.review``'s result (None: not reviewed)."""
    if not rev:
        return None
    n = (rev.get("summary") or {}).get("flags", 0)
    flags = [f"{st}: {f['flag']}" for st, v in (rev.get("states") or {}).items() for f in v.get("flags") or []]
    verdict = GOOD if n <= ANIMATION_FLAGS[0] else OK if n <= ANIMATION_FLAGS[1] else CHECK
    text = (f"Through idle, walk and attack against the game's {rev.get('against')}: "
            + ("nothing worse than the game's own." if not n else f"{n} flagged pose{'s' if n != 1 else ''} ("
               + "; ".join(flags[:3]) + ("; ..." if n > 3 else "") + "). See the review sheet."))
    return {"verdict": verdict, "text": text, "flags": n, "sheet": rev.get("sheet")}


def parts_summary(manifest: dict, kind: str) -> dict:
    kept, rebuilt, lost, flagged = [], [], [], []
    if kind == "horse":
        hair = (manifest.get("fresh") or {}).get("hair") or {}
        if hair.get("tube"):
            rebuilt.append("tail (our tube on the tail bones, painted from the model's)")
        if hair.get("mane"):
            rebuilt.append("mane (smoothed into a solid crest)")
        col = manifest.get("colour") or {}
        if (col.get("head_band") or {}).get("band"):
            rebuilt.append("head band (painted)")
        if col.get("head_marks"):
            rebuilt.append("head tassels and plume (painted marks: too small for geometry)")
        reins = (manifest.get("fresh") or {}).get("reins") or {}
        if reins.get("built"):
            rebuilt.append("reins (our strap, coloured from the model's)")
        elif reins.get("found") is False or not reins:
            lost.append("reins and stirrup leathers (too thin for about 2,900 triangles)")
    else:
        checks = manifest.get("part_checks") or {}
        for key, p in (manifest.get("parts") or {}).items():
            if key == "lance":
                kept.append("lance (the model's own, baked colour and normal map)")
            elif key == "shield":
                how = (checks.get("shield") or {}).get("method", "procedural")
                rebuilt.append(f"shield ({how})")
                tried = (checks.get("shield") or {}).get("tried") or []
                for t in tried:
                    if t.get("ok") is False:
                        flagged.append(f"shield via {t['method']}: outline {t.get('iou')}, colour distance {t.get('colour_de')}")
            elif key == "plume":
                src = (checks.get("plume") or {}).get("from", "default")
                rebuilt.append(f"plume (horsehair blade, colours from the {'reference' if src == 'crop' else 'defaults'}) with its gold finial")
            elif key == "sword":
                rebuilt.append("sword (our own, textured)")
    text = []
    if kept:
        text.append("Kept: " + "; ".join(kept) + ".")
    if rebuilt:
        text.append("Rebuilt: " + "; ".join(rebuilt) + ".")
    if lost:
        text.append("Lost: " + "; ".join(lost) + ".")
    if flagged:
        text.append("Checks that failed (the next method was used): " + "; ".join(flagged) + ".")
    verdict = GOOD if not lost else OK
    return {"verdict": verdict, "text": " ".join(text) or "No separate parts.", "kept": kept, "rebuilt": rebuilt,
            "lost": lost, "flagged": flagged}


def texture(manifest: dict, kind: str) -> dict:
    col = manifest.get("colour") or {}
    used = col.get("covered")
    reached = col.get("reached")
    v = _verdict(used, 0.6, 0.45)
    text = (f"The UVs use {used:.0%} of the texture" if used is not None else "Texture use unknown")
    if reached is not None:
        text += f"; the colour bake reached {reached:.0%} of it from the model"
    text += "." + ("" if v != CHECK else " Low use means a blurrier texture in game.")
    return {"verdict": v, "text": text, "used": used, "reached": reached}


def map_zoom(ours: Path | None, vanilla: Path | None) -> dict:
    if not ours or not vanilla or not Path(ours).is_file() or not Path(vanilla).is_file():
        return {"verdict": OK, "text": "Not measured (no map-zoom renders yet)."}
    o, v = unit_light(ours), unit_light(vanilla)
    if not o or not v:
        return {"verdict": CHECK, "text": "The unit wasn't found in the map-zoom render."}
    db = o["mean"] / max(v["mean"], 1e-6)
    dc = o["contrast"] / max(v["contrast"], 1e-6)
    verdict = GOOD if 0.8 <= db <= 1.2 and 0.7 <= dc <= 1.4 else OK if 0.65 <= db <= 1.35 and 0.5 <= dc <= 1.8 else CHECK
    word = "brighter" if db > 1 else "darker"
    text = (f"At map zoom it is {abs(db - 1):.0%} {word} than vanilla, with {dc:.1f} times its contrast"
            + ("." if verdict != CHECK else ": it may stand out or vanish among other units."))
    return {"verdict": verdict, "text": text, "ours": o, "vanilla": v, "brightness_ratio": round(db, 3),
            "contrast_ratio": round(dc, 3)}


def report(manifest: dict, kind: str, reference_fg: Path | None, game_view: Path | None,
           map_ours: Path | None = None, map_vanilla: Path | None = None, raw_view: Path | None = None,
           legs: dict | None = None, exclude: list | None = None, on_foot: bool = False, review: dict | None = None) -> dict:
    """The five measures (module doc). ``game_view``: the game version rendered from the reference's view
    (compare/game_side.png for a horse, compare/game_front.png for a rider); ``raw_view``: TRELLIS.2's own model
    from the same camera. The outline's verdict is relative to the raw model's own overlap (the approved horses'
    raw models overlap their references by 72-74%: camera and perspective differ from the drawing). ``exclude``:
    masks of parts cut off the body (a rider's lance, shield, plume) taken out of the reference first. ``on_foot``:
    a soldier as infantry (its deformation under its stance's animations; nothing about seating)."""
    m = {}
    if reference_fg and game_view and Path(reference_fg).is_file() and Path(game_view).is_file():
        fg = Image.open(reference_fg).convert("L")
        for mp in exclude or []:
            if Path(mp).is_file():
                fg = ImageChops.subtract(fg, Image.open(mp).convert("L").resize(fg.size).point(lambda v: 255 if v > 127 else 0))
        band = 0.3 if kind == "rider" else None
        iou = silhouette_iou(Image.open(game_view), fg, band)
        raw = silhouette_iou(Image.open(raw_view), fg, band) if raw_view and Path(raw_view).is_file() else None
        rel = iou / raw if raw else iou / 0.73
        # a rider is slimmed to vanilla's rider (the user, test 29): the approved cataphract keeps 63% of its raw
        # model's overlap; a horse 88-92% (slimmed only when over 1.15x vanilla)
        v = _verdict(rel, 0.58, 0.48) if band else _verdict(rel, 0.88, 0.78)
        text = f"Its outline{' (head, body and legs)' if band else ''} overlaps the reference's by {iou:.0%}"
        if raw:
            text += f" (TRELLIS.2's own model: {raw:.0%}; the game version is slimmed and posed to vanilla's build)"
        text += "." if v != CHECK else ": compare the sheet; the shape may be off."
        m["silhouette"] = {"verdict": v, "iou": iou, "raw_iou": raw, "text": text}
    else:
        m["silhouette"] = {"verdict": OK, "text": "Not measured (no reference outline)."}
    m["deformation"] = deformation(manifest.get("deform") or {}, kind)
    m["parts"] = parts_summary(manifest, kind)
    m["texture"] = texture(manifest, kind)
    m["map_zoom"] = map_zoom(map_ours, map_vanilla)
    if kind == "rider" and (jm := joints(manifest)):
        m["joints"] = jm
    if (am := animation(review)):
        m["animation"] = am
    if (kind == "rider" and not on_foot) or legs:
        m["seating"] = clearance((legs or {}).get("ours"), (legs or {}).get("vanilla"))
    order = {GOOD: 0, OK: 1, CHECK: 2}
    worst = max((x["verdict"] for x in m.values()), key=lambda v: order[v])
    summary = {GOOD: "Looks good.", OK: "Usable; see the notes.", CHECK: "Check the flagged items before using it."}[worst]
    return {"measures": m, "summary": summary, "verdict": worst}


TITLES = {"checks": "Checked before showing", "silhouette": "Outline vs the reference", "deformation": "Deformation vs vanilla", "parts": "Parts",
          "texture": "Texture", "map_zoom": "At map zoom", "seating": "Seated on the horse",
          "joints": "Joints inside the body", "animation": "In its animations", "muzzle": "The muzzle", "wheels": "The wheels", "budget": "Size"}  # the last three: a gun's (Track A)


def markdown(rep: dict) -> str:
    lines = [f"**{rep['summary']}**", ""]
    for k, t in TITLES.items():
        x = rep["measures"].get(k)
        if x:
            lines.append(f"- {t} ({x['verdict']}): {x['text']}")
    return "\n".join(lines)


def save(rep: dict, folder: Path) -> Path:
    p = Path(folder) / "quality.json"
    p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (Path(folder) / "quality.md").write_text(markdown(rep), encoding="utf-8")
    return p


# ------------------------------------------------------------------ the rider's legs clear of the horse
def inside_count(points: list, tris_p: list, tri: list, cell: float = 0.5) -> int:
    """How many ``points`` lie inside the closed mesh (``tris_p`` flat xyz, ``tri`` indices): rays along +x, the
    crossings counted (odd: inside); triangles bucketed by their (y, z) box so each ray meets few of them."""
    P = [tris_p[i:i + 3] for i in range(0, len(tris_p), 3)]
    buckets: dict = {}
    faces = []
    for t in range(0, len(tri), 3):
        a, b, c = P[tri[t]], P[tri[t + 1]], P[tri[t + 2]]
        faces.append((a, b, c))
        y0, y1 = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        z0, z1 = min(a[2], b[2], c[2]), max(a[2], b[2], c[2])
        for iy in range(int(y0 // cell), int(y1 // cell) + 1):
            for iz in range(int(z0 // cell), int(z1 // cell) + 1):
                buckets.setdefault((iy, iz), []).append(len(faces) - 1)
    n = 0
    for x, y, z in points:
        y, z = y + 1.2345e-6, z + 2.3456e-6  # off any shared edge (a ray along one would cross twice)
        crossings = 0
        for fi in buckets.get((int(y // cell), int(z // cell)), ()):
            a, b, c = faces[fi]
            # where the line (y, z) meets the triangle's projection onto the y-z plane (barycentric)
            d = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])
            if abs(d) < 1e-12:
                continue
            u = ((y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (z - a[2])) / d
            v = ((b[1] - a[1]) * (z - a[2]) - (y - a[1]) * (b[2] - a[2])) / d
            if u < 0 or v < 0 or u + v > 1:
                continue
            xs = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
            if xs > x:
                crossings += 1
        n += crossings % 2
    return n


def legs_clear(posed: Path, below: float = 1.5) -> dict | None:
    """A unit's rider against its horse in the posed meshes (``Session.gallery(dump_dir=)``): the rider's points
    lower than ``below`` over the saddle's top that are inside the horse. -> {"inside", "points", "share"} or None
    when the dump has no rider."""
    d = json.loads(Path(posed).read_text(encoding="utf-8"))
    horse = next((v for k, v in d.items() if "rider" not in k.lower()), None)
    rider = next((v for k, v in d.items() if "rider" in k.lower()), None)
    if horse is None or rider is None:
        return None
    hp = horse["p"]
    rp = [rider["p"][i:i + 3] for i in range(0, len(rider["p"]), 3)]
    near = [hp[i + 1] for i in range(0, len(hp), 3) if abs(hp[i]) < 0.6 and abs(hp[i + 2]) < 1.0]
    seat = max(near) if near else max(hp[1::3])
    legs = [q for q in rp if q[1] < seat + below]
    n = inside_count(legs, hp, horse["tri"])
    return {"inside": n, "points": len(legs), "share": round(n / max(len(legs), 1), 3), "seat": round(seat, 2)}


def clearance(ours: dict | None, vanilla: dict | None) -> dict:
    """The verdict for ``legs_clear`` against vanilla's rider on vanilla's horse."""
    if not ours:
        return {"verdict": OK, "text": "Not measured (no rider)."}
    vs = (vanilla or {}).get("share", 0.0)
    v = GOOD if ours["share"] <= vs + 0.03 else OK if ours["share"] <= vs + 0.1 else CHECK
    text = (f"Seated, {ours['share']:.0%} of the rider's lower body is inside the horse (vanilla's rider: {vs:.0%})"
            + ("." if v != CHECK else ": the legs cut into the horse's barrel."))
    return {"verdict": v, "text": text, "ours": ours, "vanilla": vanilla}
