"""Track L3: a unit put together from library parts (a run of kind ``unit``; ``core.library.new_unit`` writes its
request). No AI: the parts' built files go into a test-mod spec; Blender only draws the previews.

  infantry    each tier's soldier on foot: its standing fit (its stance's animations, its shield strapped to the
              forearm through them)
  cavalry     each tier's horse, and its soldier as the rider: its seated fit (``finished_seated``; older soldiers:
              the seated hand parts, ``seated_geometry``) and the legs spread for that horse's width
              (``trellis.leg_spread``; geometry only, on a copy)
  artillery   each tier's crew is the tier's infantry entity (the base game attaches ``<TAG>_INFANTRY_<n>`` at the
              gun's ``infantry_node``), so the crew soldier is that tier's infantry too; the gun is a gun part (its
              own cannon entity, Track A) or the game's own

  stages      assemble (the spec, staged and validated: rule 4)  ->  preview (each tier through the game's camera, a
              turntable of the first tier, a sheet)

A tier with no parts keeps the game's own look. The preview stands on the HDA test set (``unitstage.TEST``) like a
part's preview; Add to mod writes the real tag's set from the same spec.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from eu4unitgen.ref3d import unitstage

HAND_KINDS = ("mesh_weapon", "mesh_shield", "mesh_weapon_left", "mesh_shield_right")


def part_dir(ref: str) -> tuple[Path, dict]:
    """A part's built folder (its manifest) and its subject (rights, original)."""
    from eu4unitgen.core import library
    from eu4unitgen.ref3d import flow
    kind, name = ref.split(":", 1)
    if kind == "run":
        f = flow.folder_of(name)
        fin = f / "finished"
        if not (fin / "manifest.json").is_file():
            raise FileNotFoundError(f"the part {library.display(name)} isn't finished yet")
        return fin, json.loads((f / "subject.json").read_text(encoding="utf-8"))
    if kind == "starter":
        return library.fitted(name)
    raise ValueError(f"not a part: {ref}")


def seated_geometry(part: dict) -> str:
    """A soldier's hand part as it sits in the rider's hand: recorded by ``trellis.to_foot`` (``seated_geometry``),
    or for versions built before, the part file beside the on-foot one with the same triangles (``foot_part`` only
    turns and moves a part)."""
    mesh = part.get("mesh") or {}
    if mesh.get("seated_geometry"):
        return mesh["seated_geometry"]
    g = Path(mesh["geometry"])
    if not g.name.endswith("_foot.json"):
        return str(g)
    tri = json.loads(g.read_text(encoding="utf-8"))["tri"]
    for cand in sorted(g.parent.glob("*_part.json")):
        try:
            if json.loads(cand.read_text(encoding="utf-8")).get("tri") == tri:
                return str(cand)
        except (OSError, ValueError):
            continue
    raise FileNotFoundError(f"no seated version of {g.name} beside it")


def rider_for(soldier: Path, subject: dict, horse: Path | None, out: Path) -> tuple[dict, list]:
    """The soldier (a built folder) as a rider of ``horse``: a copy of its manifest with the seated hand parts and,
    for a horse wider than the game's, its legs spread (on a copy of its geometry). -> rider_entry's pair."""
    from eu4unitgen.ref3d import trellis
    out.mkdir(parents=True, exist_ok=True)
    seated_fit = soldier.parent / "finished_seated"  # the soldier's seated fit (the standing one is for foot)
    if (seated_fit / "manifest.json").is_file():
        soldier = seated_fit
    m = json.loads((soldier / "manifest.json").read_text(encoding="utf-8"))
    for p in (m.get("parts") or {}).values():
        if p.get("kind") in HAND_KINDS and "mesh" in p:
            p["mesh"]["geometry"] = seated_geometry(p)
            p["mesh"].pop("seated_geometry", None)
    geo = out / "low_geometry.json"
    shutil.copyfile(m["geometry"], geo)
    m["geometry"] = str(geo)
    if horse is not None:
        hm = json.loads((horse / "manifest.json").read_text(encoding="utf-8"))
        hp = json.loads(Path(hm["geometry"]).read_text(encoding="utf-8"))["p"]
        from eu4unitgen.core.fresh_mesh import base_arrays
        vp = base_arrays((trellis.eu4() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes())["p"]
        d = trellis.leg_spread(hp, vp)
        m["leg_spread"] = {"by": d, "vertices": trellis.spread_legs(geo, d)}
    (out / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return unitstage.rider_entry(out, subject)


def own_colours(entry: dict, parts: list, primary, out: Path) -> None:
    """Country colours off for this unit (the user, T38: "Let each country paint its cloth" switchable per unit): each
    texture set's masked cloth given back its own colour (the default primary, as ``trellis.tinted`` draws it) and
    the mask cleared (the specular's red to 0), on copies under ``out``."""
    from PIL import Image

    from eu4unitgen.ref3d import trellis
    out.mkdir(parents=True, exist_ok=True)

    def one(tex: dict, stem: str) -> None:
        if not tex.get("diffuse") or not tex.get("specular") or not primary:
            return
        d = trellis.tinted(Path(tex["diffuse"]), Path(tex["specular"]), primary, out / f"{stem}_diffuse.png")
        s = Image.open(tex["specular"]).convert("RGBA")
        r, g, b, a = s.split()
        Image.merge("RGBA", (r.point(lambda _: 0), g, b, a)).save(out / f"{stem}_specular.png")
        tex["diffuse"], tex["specular"] = str(d), str(out / f"{stem}_specular.png")
    one(entry["textures"], "body")
    entry.pop("unit_colours", None)
    for i, p in enumerate(parts):
        if "mesh" in p:
            one(p["mesh"], f"part{i}")
        p.pop("colours", None)


def spec(folder: Path, req: dict) -> dict:
    """The test-mod spec of the whole unit: every tier with parts, the game's own elsewhere."""
    utype = req["type"]
    test = unitstage.TEST
    loadout: dict = {"schema": 1}
    colours = test["unit_colours"]
    for t, slots in sorted(req["tiers"].items(), key=lambda kv: int(kv[0])):
        t = int(t)
        if utype == "cavalry":
            hdir, hsub = part_dir(slots["horse"])
            he = unitstage.horse_entry(hdir, hsub)
            if not req.get("country_colours", True):
                own_colours(he, [], _primary(hdir), Path(folder) / "assembled" / f"horse_{t}")
            loadout.setdefault("horses", {})[str(t)] = he
            colours = he.get("unit_colours") or colours
            sdir, ssub = part_dir(slots["soldier"])
            re_, parts = rider_for(sdir, ssub, hdir, Path(folder) / "assembled" / f"rider_{t}")
            if not req.get("country_colours", True):
                own_colours(re_, parts, _primary(sdir), Path(folder) / "assembled" / f"rider_{t}" / "own")
            loadout.setdefault("riders", {})[str(t)] = re_
            if parts:
                loadout.setdefault("slots", {})[f"{t}/rider"] = {"parts": parts}
        else:  # infantry, and artillery's crew: the tier's infantry on foot
            sdir, ssub = part_dir(slots["soldier"])
            se, parts = unitstage.rider_entry(sdir, ssub)
            if not req.get("country_colours", True):
                own_colours(se, parts, _primary(sdir), Path(folder) / "assembled" / f"soldier_{t}")
            loadout.setdefault("soldiers", {})[str(t)] = se
            if parts:
                loadout.setdefault("slots", {})[f"{t}/infantry"] = {"parts": parts}
            if slots.get("gun"):  # Track A: the tier's own cannon entity wears it (UNVERIFIED in game: test 37)
                gdir, gsub = part_dir(slots["gun"])
                loadout.setdefault("guns", {})[str(t)] = unitstage.gun_entry(gdir, gsub)
    return {"schema": 1, "name": f"{req['name']}: {utype}, tiers {', '.join(sorted(req['tiers']))}",
            "set": {"tags": [test["tag"]], "gfx": test["gfx"], "set_name": test["set_name"], "base_gfx": test["base_gfx"],
                    "unit_colours": {test["tag"]: colours}, "unit_names": unitstage.unit_names(req["name"])},
            "loadout": loadout, "test": {"sprite_techs": test["sprite_techs"]}}


def _primary(vdir: Path):
    """A built part's cloth colour as it was baked (the default primary its mask carries)."""
    m = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    return (m.get("colour") or {}).get("default_primary")


def stage_assemble(folder: Path, req: dict, cfg) -> dict:
    pv = Path(folder) / "preview"
    if pv.exists():
        shutil.rmtree(pv)
    st = unitstage.stage(pv, spec(folder, req), "preview")
    if st["exit"] != 0:
        raise RuntimeError(f"the parts didn't pass the game checks; see {pv / 'log.txt'}")
    return {"tiers": sorted(req["tiers"])}


def stage_preview(folder: Path, req: dict, cfg) -> dict:
    """Each tier with parts through the game's camera, the first tier turning, and a sheet of the tiers."""
    pv = Path(folder) / "preview"
    tiers = sorted(int(t) for t in req["tiers"])
    utype = req["type"]
    g = unitstage.gallery(pv, "preview", tiers, {t: f"tier {t}" for t in tiers}, cfg.eu4(), utype=utype,
                          rows=("close", "map"), views=("three_quarter", "side"))
    try:
        frames = len(unitstage.turntable(pv, "preview", tiers[0], cfg.eu4(), utype))
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal: the unit is usable without it
        frames = f"failed: {type(exc).__name__}: {exc}"
    sheet = pv / "gallery" / "sheet.png"
    from eu4unitgen.core import config
    sheets = config.load().sheets()
    sheets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sheet, sheets / f"{req['name']}.png")
    q = {"summary": f"Put together from your parts ({', '.join(f'tier {t}' for t in tiers)}); each part's own report "
                    "is on its page.", "verdict": "good", "measures": {}}
    (pv / "quality.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    return {"sheet": str(sheet), "quality": q["summary"], "verdict": "good", "turntable": frames,
            "images": len(g["images"])}


RUN = {"assemble": stage_assemble, "preview": stage_preview}
