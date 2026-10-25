"""The simple edits after a run (Track P2.2): the common tweaks without code, each rebuilding only what it needs.

  plume         on/off (a rider's plume part)                               finish: the part dropped
  tassels       on/off (a horse's painted head tassels)                     build: the bake re-run
  emblem        none, a library decal or an uploaded mask (white on black)  build
  country_mask  the cloth as the country colour's mask, on/off              build
  cloth_colour  [r, g, b]: with the mask, the unit's primary colour; without it, the cloth's texels moved there
                                                                            finish
  brightness    a factor on every diffuse texture (1 = as built)            finish
  shield_side   "auto", "left", "right" (a rider: the weapon to the other hand, mirrored)
                                                                            build
  hem           1 = as built; under 1 shortens the cloth's hang below the body (a horse's caparison skirt, a rider's
                skirt and cloak)                                            finish

The TRELLIS.2 model is never re-made: "build" re-runs the processing (about 1.5 minutes for a horse, 5 for a rider);
"finish" works on a copy of the built version (seconds). ``stage_for(old, new)`` says which; ``finish`` writes the
copy. Pure Python and Pillow.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

DEFAULTS = {"plume": True, "tassels": True, "emblem": None, "country_mask": None, "cloth_colour": None,
            "brightness": 1.0, "shield_side": "auto", "hem": 1.0}
BUILD = ("tassels", "emblem", "country_mask", "shield_side")
FINISH = ("plume", "cloth_colour", "brightness", "hem")


def merged(edits: dict | None) -> dict:
    return {**DEFAULTS, **{k: v for k, v in (edits or {}).items() if k in DEFAULTS}}


def stage_for(old: dict | None, new: dict | None) -> str | None:
    """The first stage an edit sends the run back to: "build", "finish" or None (nothing changed)."""
    a, b = merged(old), merged(new)
    changed = [k for k in DEFAULTS if a[k] != b[k]]
    if any(k in BUILD for k in changed):
        return "build"
    if changed:
        return "finish"
    return None


def build_options(edits: dict | None, kind: str) -> dict:
    """The processing's options from the edits (trellis.process / process_rider keyword arguments)."""
    e = merged(edits)
    if kind == "horse":
        out = {"cfg": {"tassels": bool(e["tassels"])}}
        if e["country_mask"] is not None:
            out["country_mask"] = bool(e["country_mask"])
        if e["emblem"]:
            out["emblem"] = e["emblem"]
        return out
    out = {"shield_side": e["shield_side"], "plume": "part"}
    if e["country_mask"] is not None:
        out["country_mask"] = bool(e["country_mask"])
    return out


def emblem_file(choice: str | None, folder: Path) -> Path | None:
    """A library decal's name (textures.decals) drawn into ``folder``, or an uploaded mask's path as it is."""
    if not choice:
        return None
    from eu4unitgen.textures import decals
    if choice in decals.DECALS:
        return decals.DECALS[choice](Path(folder) / f"emblem_{choice}.png")
    return Path(choice)


def _scale(path: Path, k: float, out: Path) -> None:
    im = Image.open(path)
    a = im.getchannel("A") if im.mode == "RGBA" else None
    rgb = im.convert("RGB").point(lambda v: max(0, min(255, int(round(v * k)))))
    if a is not None:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(a)
    rgb.save(out)


def _recolour(path: Path, cloth, target, out: Path) -> None:
    from eu4unitgen.ref3d.part_gen import snap_image
    snap_image(Image.open(path), [(tuple(cloth), tuple(target))], radius=26.0).save(out)


def _hem(geometry: Path, diffuse: Path, cloth, k: float, line: float, out: Path) -> int:
    """Cloth below ``line`` (its texel's colour near ``cloth``, or on the country mask's grey) lifted towards it:
    y' = line - (line - y) * k. -> vertices moved."""
    from eu4unitgen.core import palette as pal
    g = json.loads(Path(geometry).read_text(encoding="utf-8"))
    im = Image.open(diffuse).convert("RGB")
    W, H = im.size
    px = im.load()
    uv = g.get("u0") or g.get("uv")
    n = 0
    for i in range(len(g["p"]) // 3):
        y = g["p"][3 * i + 1]
        if y >= line:
            continue
        c = px[min(W - 1, int(uv[2 * i] * W)), min(H - 1, int(uv[2 * i + 1] * H))]
        grey = abs(c[0] - c[1]) < 6 and abs(c[1] - c[2]) < 6 and 40 < c[0] < 230
        if (cloth is not None and pal.delta_e(c, cloth) < 22) or grey:
            g["p"][3 * i + 1] = line - (line - y) * k
            n += 1
    Path(out).write_text(json.dumps(g), encoding="utf-8")
    return n


HEM_LINE = {"horse": 8.6, "rider": 9.0}  # the cloth's top of hang: under a horse's belly, a rider's hips


def finish(vdir: Path, edits: dict | None, kind: str, out: Path) -> Path:
    """The built version ``vdir`` copied to ``out`` with the finish edits applied; its manifest points at the copies.
    -> ``out``."""
    e = merged(edits)
    vdir, out = Path(vdir), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    m = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    cloth = ((m.get("palette") or {}).get("cloth") or {}).get("colour")
    if cloth is None:  # a horse: its cloth as the bake read it (the default primary is its brightest-channel form)
        prim = (m.get("colour") or {}).get("default_primary")
        cloth = None
        if prim:
            from eu4unitgen.core import palette as pal
            p = pal.classes(m["textures"]["diffuse"], m["textures"]["specular"], prim)
            cloth = (p.get("cloth") or {}).get("colour")
    masked = bool((m.get("colour") or {}).get("default_primary"))
    log = {}

    def texture(src: str, name: str) -> str:
        q = out / name
        shutil.copyfile(src, q)
        if e["cloth_colour"] and not masked and cloth:
            _recolour(q, cloth, e["cloth_colour"], q)
        if abs(float(e["brightness"]) - 1.0) > 1e-3:
            _scale(q, float(e["brightness"]), q)
        return str(q)
    tex = dict(m["textures"])
    tex["diffuse"] = texture(tex["diffuse"], "diffuse.png")
    for k in ("specular", "normal"):
        shutil.copyfile(tex[k], out / f"{k}.png")
        tex[k] = str(out / f"{k}.png")
    geo = out / "low_geometry.json"
    shutil.copyfile(m["geometry"], geo)
    if float(e["hem"]) < 0.999:
        log["hem_vertices"] = _hem(geo, Path(m["textures"]["diffuse"]), cloth, float(e["hem"]), HEM_LINE[kind], geo)
    parts = {}
    for key, p in (m.get("parts") or {}).items():
        if key == "plume" and not e["plume"]:
            continue
        q = json.loads(json.dumps(p))
        if "mesh" in q and "diffuse" in q["mesh"]:
            q["mesh"]["diffuse"] = texture(q["mesh"]["diffuse"], f"{key}_diffuse.png")
        parts[key] = q
    m.update({"textures": tex, "geometry": str(geo), "parts": parts, "edits": e, "edited_from": str(vdir), "edit_log": log})
    if e["cloth_colour"] and masked:  # the game tints the mask: the colour is the unit's primary
        m.setdefault("colour", {})["default_primary"] = list(e["cloth_colour"])
    (out / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return out
