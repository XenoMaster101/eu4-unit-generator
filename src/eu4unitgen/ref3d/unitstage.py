"""A generated unit staged and shown (Track P2; the general form of the per-test staging scripts tests 25-35 used).

A *unit* is what the app makes from images: a horse (game version), a soldier (game version with its parts: on
foot it is the infantry, and the gun crew; on a horse the rider), or a horse and its rider, for one unit slot
(infantry or cavalry, tier 1-4). The vanilla unit stands at the next tier for comparison.

  entries     a game version folder -> the loadout's horse or rider entry (geometry, textures, rights, source)
  spec        the test-mod spec: the unit at its tier, vanilla next, unit names from the subject's name, colours
              that keep a masked cloth's look (its red as the primary; the set's tags without colours get it)
  stage       ``eu4unitgen.cli test-mod --spec`` into ``<folder>/stage`` (validated there: rule 4)
  gallery     the staged units rendered through the game's camera: close, near and map zoom; three views each;
              a sheet; the images the quality report reads (map zoom brightness and contrast)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

GOLD, WHITE = (212, 175, 55), (230, 225, 210)  # a masked unit's secondary and tertiary colours
# the preview's test mod: HDA, the only 1444 tag no sprite pack touches (in-game tests 3-35), on a new culture
TEST = {"tag": "HDA", "gfx": "hellenicgfx", "set_name": "cataphract_test", "base_gfx": "westerngfx",
        "sprite_techs": [2, 3, 4], "unit_colours": [[40, 60, 160], [220, 190, 40], [230, 225, 210]]}
ROWS = {"close": {}, "near": {"ortho_k": 0.6, "target": [0, 0, 4.5]},
        "map": {"ortho_k": 3.0, "width": 600, "height": 420}}
VIEWS = {"side": (-1.0, 0.0), "three_quarter": (-0.7071, 0.7071), "front": (0.0, 1.0)}


def _abs(p) -> str:
    return str(Path(p).resolve())


def horse_entry(vdir: Path, subject: dict) -> dict:
    m = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    e = {"base": "cavalry_horse", "geometry": _abs(m["geometry"]),
         "textures": {k: _abs(v) for k, v in m["textures"].items()}, "rights": subject["rights"],
         "source": f"TRELLIS.2 from {subject.get('original', '?')}, game {Path(vdir).name}"}
    if (m.get("colour") or {}).get("default_primary"):
        e["unit_colours"] = [m["colour"]["default_primary"], list(GOLD), list(WHITE)]
    return e


def gun_entry(vdir: Path, subject: dict) -> dict:
    """The loadout's gun entry (Track A): as a horse's, on the base game's gun (artillery_mesh)."""
    e = horse_entry(vdir, subject)
    e.pop("unit_colours", None)  # a gun has no country colour (the base game's is PdxMeshStandard)
    return {**e, "base": "artillery_mesh"}


def rider_entry(vdir: Path, subject: dict) -> tuple[dict, list]:
    """(the loadout's rider entry, its parts for the slot)."""
    m = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    e = {"base": "western_generic_01_type1", "geometry": _abs(m["geometry"]),
         "textures": {k: _abs(v) for k, v in m["textures"].items()}, "rights": subject["rights"],
         "source": f"TRELLIS.2 from {subject.get('original', '?')}, game {Path(vdir).name}"}
    parts = []
    for p in (m.get("parts") or {}).values():
        q = json.loads(json.dumps(p))
        if "mesh" in q:  # the loadout's four keys (the seated version is ref3d.assemble's to use)
            q["mesh"] = {k: _abs(v) for k, v in q["mesh"].items() if k != "seated_geometry"}
        parts.append(q)
    return e, parts


def unit_names(name: str) -> dict:
    """The units' names in game from the subject's name ("cataphract_rider" -> "Cataphract Rider Cavalry")."""
    pretty = " ".join(w.capitalize() for w in name.replace("-", "_").split("_") if w)
    return {u: f"{pretty} {u.title()}" for u in ("cavalry", "infantry", "artillery")}


def spec(title: str, tier: int, horse: dict | None, rider: tuple | None, names: dict, test: dict,
         vanilla_next: bool = True, soldier: tuple | None = None, gun: dict | None = None) -> dict:
    """The test-mod spec: the unit at ``tier`` (its horse and/or rider, or its soldier on foot: ``rider_entry``'s
    pair as the infantry; the vanilla rest), vanilla at the next tier."""
    horses = {str(tier): horse} if horse else {}
    riders = {str(tier): rider[0]} if rider else {}
    slots = {f"{tier}/rider": {"parts": rider[1]}} if rider and rider[1] else {}
    if soldier and soldier[1]:
        slots[f"{tier}/infantry"] = {"parts": soldier[1]}
    colours = test["unit_colours"]
    if horse and horse.get("unit_colours"):
        colours = horse["unit_colours"]
    loadout = {"schema": 1}
    if horses:
        loadout["horses"] = horses
    if riders:
        loadout["riders"] = riders
    if soldier:
        loadout["soldiers"] = {str(tier): soldier[0]}
    if gun:
        loadout["guns"] = {str(tier): gun}
    if slots:
        loadout["slots"] = slots
    sprite_techs = [t for t in test["sprite_techs"] if t > tier] or test["sprite_techs"]
    return {"schema": 1, "name": title,
            "set": {"tags": [test["tag"]], "gfx": test["gfx"], "set_name": test["set_name"], "base_gfx": test["base_gfx"],
                    "unit_colours": {test["tag"]: colours}, "unit_names": names},
            "loadout": loadout, "test": {"sprite_techs": sprite_techs if vanilla_next else test["sprite_techs"]}}


def stage(folder: Path, spec_: dict, name: str) -> dict:
    """The spec written to ``folder/<name>.json`` and staged into ``folder/stage`` (validated by test-mod)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sp = folder / f"{name}.json"
    sp.write_text(json.dumps(spec_, indent=1), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "eu4unitgen.cli", "test-mod", "--spec", str(sp), "--stage-dir",
                        str(folder / "stage")], capture_output=True, text=True)
    (folder / "log.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    return {"spec": str(sp), "stage": str(folder / "stage"), "exit": r.returncode}


def gallery(folder: Path, name: str, tiers: list[int], labels: dict, eu4: Path, rows=("close", "near", "map"),
            views=("side", "three_quarter", "front"), utype: str = "cavalry") -> dict:
    """The staged units through the game's camera (export.gallery.GAME_CLOSE): each row x view x tier an image
    under ``folder/gallery``, and ``sheet.png``. -> {"sheet", "images": {"<row>_<view>_<tier>": path}}."""
    from PIL import Image, ImageDraw, ImageFont

    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.export.gallery import GAME_CLOSE
    from eu4unitgen.export.pipeline import Session, Settings
    folder = Path(folder)
    ch = CharacterSpec.load(folder / f"{name}.json")
    s = Session(Settings(eu4_path=Path(eu4), owned_dlc=[], prefix="hr_", stage_dir=folder / "gallery_stage"))
    v = GAME_CLOSE["view"]
    el = math.atan2(-v[2], math.hypot(v[0], v[1]))
    out = folder / "gallery"
    out.mkdir(exist_ok=True)
    images = {}
    for row in rows:
        r = ROWS[row]
        cam0 = dict(GAME_CLOSE, ortho_scale=GAME_CLOSE["ortho_scale"] * r.get("ortho_k", 1.0),
                    **{k: r[k] for k in ("target", "width", "height") if k in r})
        for view in views:
            hx, hy = VIEWS[view]
            cam = dict(cam0, view=[hx * math.cos(el), hy * math.cos(el), -math.sin(el)])
            g = s.gallery(ch.set_spec(), units=[(utype, t) for t in tiers], camera=cam, lit={"normal_maps": True})
            for (_, tier), im in g.images.items():
                p = out / f"{row}_{view}_{tier}.png"
                im.save(p)
                images[f"{row}_{view}_{tier}"] = str(p)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    cell = 360
    cols = [(row, view) for row in rows for view in views]
    sheet = Image.new("RGB", (len(cols) * cell, len(tiers) * (cell + 28) + 4), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    for ci, (row, view) in enumerate(cols):
        for ti, tier in enumerate(tiers):
            p = images.get(f"{row}_{view}_{tier}")
            if not p:
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            y = ti * (cell + 28) + 28
            sheet.paste(im, (ci * cell + (cell - im.width) // 2, y))
            d.text((ci * cell + 6, y - 24), f"{labels.get(tier, tier)}: {row}, {view.replace('_', ' ')}", fill=(20, 20, 24), font=font)
    sheet.save(out / "sheet.png")
    return {"sheet": str(out / "sheet.png"), "images": images}


def action_sheet(folder: Path, name: str, tiers: list[int], labels: dict, eu4: Path, utype: str,
                 states=(("idle", 1), ("walk", 8), ("attack", 1), ("attack", 12)), view: str = "three_quarter") -> dict:
    """The staged units through the game's close camera at chosen frames of their states (the units' own animations:
    infantry, and artillery with its crew), each tier side by side (ours, the game's), and a sheet of them. ->
    {"sheet", "images": {"<state>_<frame>_<tier>": path}}."""
    from PIL import Image, ImageDraw, ImageFont

    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.export.gallery import GAME_CLOSE
    from eu4unitgen.export.pipeline import Session, Settings
    folder = Path(folder)
    ch = CharacterSpec.load(folder / f"{name}.json")
    s = Session(Settings(eu4_path=Path(eu4), owned_dlc=[], prefix="hr_", stage_dir=folder / "action_stage"))
    v = GAME_CLOSE["view"]
    el = math.atan2(-v[2], math.hypot(v[0], v[1]))
    hx, hy = VIEWS[view]
    cam = dict(GAME_CLOSE, view=[hx * math.cos(el), hy * math.cos(el), -math.sin(el)])
    out = folder / "actions"
    out.mkdir(exist_ok=True)
    images = {}
    for state, frame in states:
        g = s.gallery(ch.set_spec(), units=[(utype, t) for t in tiers], camera=cam, lit={"normal_maps": True},
                      state=state, frame=frame)
        for (_, tier), im in g.images.items():
            p = out / f"{utype}_{state}_{frame}_{tier}.png"
            im.save(p)
            images[f"{state}_{frame}_{tier}"] = str(p)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    cell = 340
    sheet = Image.new("RGB", (len(states) * cell, len(tiers) * (cell + 28) + 4), (232, 232, 236))
    d = ImageDraw.Draw(sheet)
    for ci, (state, frame) in enumerate(states):
        for ti, tier in enumerate(tiers):
            p = images.get(f"{state}_{frame}_{tier}")
            if not p:
                continue
            im = Image.open(p).convert("RGB")
            w, h = im.size
            im = im.crop((int(w * 0.18), int(h * 0.05), int(w * 0.82), int(h * 0.95)))
            im.thumbnail((cell, cell))
            y = ti * (cell + 28) + 28
            sheet.paste(im, (ci * cell + (cell - im.width) // 2, y))
            d.text((ci * cell + 6, y - 24), f"{labels.get(tier, tier)}: {utype} {state}, frame {frame}", fill=(20, 20, 24),
                   font=font)
    sheet.save(out / f"{utype}_sheet.png")
    return {"sheet": str(out / f"{utype}_sheet.png"), "images": images}


TURNTABLE_FRAMES = 12
TURNTABLE_SCALE = 11.0  # the view's width in game units: a unit (about 8-10 tall, a spear more) fills the square


def turntable(folder: Path, name: str, tier: int, eu4: Path, utype: str = "cavalry", frames: int = TURNTABLE_FRAMES,
              size: int = 480) -> list[str]:
    """The staged unit at ``tier`` turned round once, as the game draws it (lit, its idle pose, the game's close
    camera's elevation): ``frames`` images under ``folder/turntable``, for the Library's rotating preview."""
    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.export.gallery import GAME_CLOSE
    from eu4unitgen.export.pipeline import Session, Settings
    folder = Path(folder)
    ch = CharacterSpec.load(folder / f"{name}.json")
    s = Session(Settings(eu4_path=Path(eu4), owned_dlc=[], prefix="hr_", stage_dir=folder / "turntable_stage"))
    v = GAME_CLOSE["view"]
    el = math.atan2(-v[2], math.hypot(v[0], v[1]))
    out = folder / "turntable"
    out.mkdir(exist_ok=True)
    paths = []
    for i in range(frames):
        a = math.radians(180 - 360 * i / frames)  # from the side (VIEWS) round past the three-quarter and the front
        cam = dict(GAME_CLOSE, width=size, height=size, ortho_scale=TURNTABLE_SCALE, view=[math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), -math.sin(el)])
        g = s.gallery(ch.set_spec(), units=[(utype, tier)], camera=cam, lit={"normal_maps": True})
        for im in g.images.values():
            p = out / f"frame_{i:02d}.png"
            im.save(p)
            paths.append(str(p))
    return paths
