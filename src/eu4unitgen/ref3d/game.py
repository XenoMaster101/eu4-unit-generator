"""Track M finish line, step 2: the EU4 game version of a master (SPEC M17).

From a master (``<subject>/master/vNNN``) into ``<subject>/game/vNNN`` (nothing overwritten):
  repose   Blender job game_master: the master moved onto the vanilla horse skeleton at rest, its albedo kept
           for the bake, the high surface and one closed solid;
  low      Blender job fresh_horse (test 15's pipeline), the solid as its source model: fitted to vanilla's box,
           skirts trimmed under ``hem_y`` except round the legs, a coarse voxel solid, an even remesh to about
           ``target_tris``, UVs, chain weights from the vanilla horse;
  colour   Blender job bake_master: the master's albedo, its classes and the specular map (gloss per class)
           baked onto the low mesh;
  normal   Blender job bake_horse: the high surface's normals (and ambient occlusion, multiplied into the colour);
  stage    a test-mod spec (one tag, tier 1 the new horse, tier 2 the vanilla horse for comparison), written
           to ``build/<test>/`` and staged with ``eu4unitgen.cli test-mod --spec``.

usage: python -m eu4unitgen.ref3d.game build <subject.json> --master vNNN [--test testNN (a new name)]
       python -m eu4unitgen.ref3d.game stage --test testNN <subject.json>:vNNN [<subject.json>:vNNN ...]
         (several game versions in one test: tier 1, 2, ... in order; the next tier is the vanilla horse)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from eu4unitgen.ref3d import s1

GAME_DEFAULTS = {
    "voxel_solid": 0.1,
    # fresh_horse (test 15's settings for its TripoSG horse, the rider cut and lump off: there is no rider)
    # (test 21: the tail is fresh_horse's own, along vanilla's tail bones; 1,650 body triangles leave room for the
    # slabs (~720), ears (~220) and plume (~320) under 3,000)
    # fit off: the master is already on the vanilla skeleton (test 21's first build: without the master's tail the
    # nose-to-tail box fit scaled the horse off vanilla's leg bones and the skirt trim cut its legs)
    "fresh": {"fit": False, "hem_y": 4.6, "leg_radius": 0.65, "tail_gap": 0.35, "voxel": 0.3, "relax": 4, "target_tris": 1550,
              "quads": 1550, "smooth": 12, "influences": 3, "chain_weights": True, "tail": {"build": True, "cut": 0.8},
              # M20: legs never thinner than vanilla's (cataphract_auto: legs hidden by armour and tassels fitted
              # 0.14-0.4 wide against horse 2's 0.7-1.3; the remesh cut the near foreleg off at its knee)
              "leg_allow": 1.0},
    "bake": {"size": 512, "cage": 0.4, "distance": 0.8},
    # the cloth as slabs (game_master): the body ~2,000 triangles and four slabs ~180 each stay under the
    # validator's 3,000 (test 20's first stage: 4,224 with 2,800 and 7 x 12 slabs)
    "pieces": {"piece_rows": 5, "piece_cols": 9, "piece_thick": 0.06, "hug_clear": 0.06},
    "normal": {"cage": 0.1, "distance": 0.35, "high_smooth": 0, "strength": 0.6, "samples": 32, "ao_strength": 0.35},
    # M20 (the user: the game head droops, long and thin): the fitted head pitch kept, half the neck's raise
    "repose": {"keep_head_pitch": True, "neck_keep": 0.5},
    # M20, the simpler pipeline (subject "pipeline": "cloak"): colour projected from the reference onto the low
    # mesh (ai/jobs/ref3d_project_texture) instead of baked from the master; armour and cloth as a cloak, a thin
    # continuous drape over the body (test 15's barding read well; test 16's separate slabs read as a tarp),
    # only where the parts show armour or a caparison
    "cloak": {"style": "drape", "offset": 0.06, "flare": 0.02, "fold": 0.05, "slits": [], "hem_length": 0.0,
              "readable": 0.7, "leg_lift": 1.2, "leg_clear": 0.4, "rider_cut": [7.6, 2.0]},
    "test": {"tag": "HDA", "gfx": "hellenicgfx", "set_name": "cataphract_test", "base_gfx": "westerngfx",
             "sprite_techs": [2, 3, 4],
             # test 15's unit colours for HDA (the engine's default otherwise)
             "unit_colours": [[40, 60, 160], [220, 190, 40], [230, 225, 210]]},
}


def eu4_path() -> Path:
    """The EU4 install (the base horse is read from it at run time, never copied: rule 1)."""
    p = os.environ.get("EU4_PATH")
    if not p:
        raise SystemExit("set EU4_PATH to the EU4 install")
    return Path(p)


def build(subject_path: Path, master_version: str, test: str | None, blender: str | None = None,
          ai_python: str | None = None) -> dict:
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays
    from eu4unitgen.core.project import default_blender_path

    if test and (Path("build") / test).exists():  # a test's folder is never written into twice (test 16's spec was lost so)
        raise SystemExit(f"build/{test} exists: pick a new test name")
    s = s1.load_subject(subject_path)
    cfg = {**GAME_DEFAULTS, **(s.get("game") or {})}
    root = s["_root"]
    mv = root / "master" / master_version
    mm = json.loads((mv / "manifest.json").read_text(encoding="utf-8"))
    ea = root / "spikes" / "ea" / "fit" / mm["ea"]["version"]
    rec = json.loads((ea / "result.json").read_text(encoding="utf-8"))["candidates"][mm["ea"]["candidate"]]
    out = s1.next_version(root / "game")
    br = BlenderRunner(blender or default_blender_path(), log_dir=out / "logs", timeout=7200)
    t0 = time.time()
    log = []

    def note(msg):
        log.append(msg)
        print(msg, flush=True)

    grids = {k: [p["rows"], p["cols"]] for k, p in mm["assemble"]["parts"].items()
             if k.startswith(("panel_", "chest_")) and isinstance(p, dict) and "rows" in p and not k.endswith("_tassels")}
    from eu4unitgen.ref3d import ed2d
    mvn = s1.main_view(s)  # the main view (M18)
    vL = ed2d.build_args(s, out)["views"][mvn]
    plab = sorted((mv / "cloth_left").glob("**/final/**/panel_labels.png")) or sorted((mv / "cloth_left").glob("**/panel_labels.png"))
    labels = {}
    if plab:
        labels.update({"panel_left": {"image": str(plab[-1]), "uv": "pattern"}, "panel_right": {"image": str(plab[-1]), "uv": "pattern"}})
    labels.update({"chest_left": {"image": vL["labels"], "uv": "projected_L"}, "chest_right": {"image": vL["labels"], "uv": "projected_L"}})
    rp = br.run_job("game_master", {"master_blend": str(mv / "assemble" / "master.blend"), "grids": grids, "labels": labels,
                                    "main_view": mvn, **cfg["repose"],
                                    "clean": {"image": vL["image"], "labels": vL["labels"], "classes": s["_template"]["classes"]},
                                    "skeleton": str(root / "spikes" / "ea" / "inputs" / "skeleton.json"),
                                    "body": {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")},
                                    "voxel": cfg["voxel_solid"], **cfg["pieces"], "out_dir": str(out / "repose")}).result
    note(f"repose: high {rp['high']['tris']} tris, solid {rp['solid']['tris']} tris")
    base = (eu4_path() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes()
    (out / "debug").mkdir(exist_ok=True)
    cloak_mode = s.get("pipeline") == "cloak"
    found = s.get("parts_found") or {}
    cloak = cfg["cloak"] if cloak_mode and (found.get("armour") or found.get("caparison")) else None
    if cloak_mode:
        note("cloak: " + ("armour/cloth shown: a drape over the body" if cloak else "none (no armour or caparison shown)"))
    fr = br.run_job("fresh_horse", {**cfg["fresh"], **({"barding": cloak} if cloak else {}),
                                    "source": {"kind": "model", "path": rp["solid"]["path"]},
                                    "pieces": rp["pieces"]["path"],
                                    "vanilla": base_arrays(base), "geometry_out": str(out / "low_geometry.json"),
                                    "preview_out": str(out / "low.obj"), "bake_source_out": str(out / "low_source.obj"),
                                    "debug_dir": str(out / "debug")}).result
    note(f"low: {fr['final']['tris']} tris, {fr['final']['vertices']} vertices, {fr['final']['pieces']} piece(s); "
         f"fit x{fr['fit']['scale']:.3f}")
    name = "horse"
    if cloak_mode:  # the colour projected from the reference (M20)
        bc = project_colour(s, mvn, vL, out, name, cfg, ai_python, cloak)
        note(f"colour: projected; {bc['seen_share'] * 100:.0f}% of the texels seen, {bc['covered_share'] * 100:.0f}% "
             f"under the cloak (coat); warp landmarks {len(bc['warp_landmarks'])}")
    else:
        bc = None
    bc = bc or br.run_job("bake_master", {"reposed_blend": str(out / "repose" / "reposed.blend"), "low": str(out / "low_geometry.json"),
                                    "fit": fr["fit"], **cfg["bake"], "classes": s["_template"]["classes"],
                                    "out_dir": str(out), "name": name}).result
    if not cloak_mode:
        note(f"colour: {bc['reached'] * 100:.0f}% of the texels reached; classes {bc['class_share']}")
    # the normal map from the high surface (the master's parts, fitted like the low mesh); AO into the colour
    hi = out / "high_fitted.obj"
    fit_obj(Path(rp["high"]["path"]), hi, fr["fit"])
    nb = br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(hi)], "diffuse": bc["diffuse"],
                                   "size": cfg["bake"]["size"], **cfg["normal"], "preview": True, "out_dir": str(out),
                                   "name": f"{name}b"}).result
    note(f"normal: {json.dumps({k: nb.get(k) for k in ('tilt', 'found', 'match')}, default=str)[:300]}")
    tex = {"diffuse": (nb.get("paths") or {}).get("diffuse") or nb.get("diffuse") or bc["diffuse"], "specular": bc["specular"],
           "normal": (nb.get("paths") or {}).get("normal") or str(out / f"{name}b_normal.png")}
    st = stage_test(test, [horse_entry(s, out / "low_geometry.json", tex)], cfg) if test else None
    if st:
        note(f"stage: exit {st['exit']}; {st['stage']}")
    manifest = {"stage": "game version", "subject": s["name"], "version": out.name, "master": master_version,
                "settings": cfg, "repose": rp, "low": fr, "colour": bc, "normal": nb, "textures": tex,
                "geometry": str(out / "low_geometry.json"), "test": st,
                "log": log, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


def project_colour(s: dict, mvn: str, vL: dict, out: Path, name: str, cfg: dict, ai_python: str | None, cloak) -> dict:
    """The low mesh's colour and gloss projected from the main view (ai/jobs/ref3d_project_texture): the
    landmarks' thin-plate warp, the body under the cloak coat-coloured, unseen texels from their own part."""
    from eu4unitgen.ai.env import AIJobRunner
    sv = s["views"][mvn]
    fg = Path(vL["image"]).with_name(Path(vL["image"]).stem + "_fg.png")
    r = AIJobRunner(ai_python).run_job("ref3d_project_texture", {
        "geometry": str((out / "low_geometry.json").resolve()), "image": vL["image"], "mask": str(fg),
        "labels": vL["labels"], "landmarks": sv.get("landmarks") or {}, "classes": s["_template"]["classes"],
        "gloss": GLOSS_BY_CLASS, "cloak_parts": CLOAK_PARTS if cloak else [], "size": cfg["bake"]["size"],
        "out_dir": str(out.resolve()), "name": name}).result
    return {**r, "diffuse": r["diffuse"], "specular": r["specular"]}


# fresh_horse's drape barding: its panel parts (_horse_barding: 2 cloth panel, 3 scale panel)
CLOAK_PARTS = [2, 3]
# gloss per label class (bake_master's: _horse_paint.GLOSS)
GLOSS_BY_CLASS = {"coat": 5, "leg": 5, "scale": 14, "cloth": 4, "trim": 12, "chanfron": 16, "leather": 6, "tail": 7,
                  "plume": 4}


def fit_obj(src: Path, dst: Path, fit: dict):
    """An OBJ (preview axes: file (x, y, -z) of PDX) with fresh_horse's fit applied (PDX: x s + offset)."""
    s, (ox, oy, oz) = fit["scale"], fit["offset"]
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if line.startswith("v "):
                x, y, z = map(float, line.split()[1:4])
                g.write(f"v {x * s + ox:.5f} {y * s + oy:.5f} {-((-z) * s + oz):.5f}\n")
            else:
                g.write(line)


def horse_entry(s: dict, geometry: Path, tex: dict) -> dict:
    """A loadout horse (core.loadout) for one game version."""
    views = ", ".join(s["views"])
    return {"base": "cavalry_horse", "geometry": str(Path(geometry).resolve()),
            "textures": {k: str(Path(v).resolve()) for k, v in tex.items()}, "rights": s.get("rights", "user"),
            "source": f"Track M master of {s['name']} (assets_in/{s['name']}; views {views})", "_name": s["name"]}


def stage_test(test: str, horses: list, cfg: dict) -> dict:
    """A test-mod spec with ``horses`` at tiers 1, 2, ... (the next tier: the vanilla horse), written to
    ``build/<test>/`` and staged there (refused if that folder exists)."""
    folder = Path("build") / test
    if folder.exists():
        raise SystemExit(f"build/{test} exists: pick a new test name")
    t = cfg["test"]
    names = ", ".join(f"tier {i + 1} {h['_name']}" for i, h in enumerate(horses))
    spec = {"schema": 1, "name": f"{test}: game horses ({names}); the vanilla horse at tier {len(horses) + 1} for comparison",
            "set": {"tags": [t["tag"]], "gfx": t["gfx"], "set_name": t["set_name"], "base_gfx": t["base_gfx"],
                    "unit_colours": {t["tag"]: t["unit_colours"]},
                    **({"unit_names": t["unit_names"]} if t.get("unit_names") else {})},
            "loadout": {"schema": 1, "horses": {str(i + 1): {k: v for k, v in h.items() if not k.startswith("_")}
                                                for i, h in enumerate(horses)}},
            "test": {"sprite_techs": t["sprite_techs"]}}
    folder.mkdir(parents=True)
    (folder / f"{test}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    cmd = [sys.executable, "-m", "eu4unitgen.cli", "test-mod", "--spec", str(folder / f"{test}.json"),
           "--stage-dir", str(folder / "stage")]
    st = subprocess.run(cmd, capture_output=True, text=True)
    (folder / "log.txt").write_text(st.stdout + st.stderr, encoding="utf-8")
    return {"spec": str(folder / f"{test}.json"), "stage": str(folder / "stage"), "exit": st.returncode}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("subject", type=Path)
    b.add_argument("--master", required=True)
    b.add_argument("--test", help="stage it alone in a new build/<test> folder (refused if it exists)")
    b.add_argument("--blender")
    c = sub.add_parser("stage")
    c.add_argument("--test", required=True)
    c.add_argument("horses", nargs="+", help="<subject.json>:vNNN (a game version), tier 1 first")
    a = p.parse_args(argv)
    if a.cmd == "stage":
        horses = []
        for h in a.horses:
            subj, ver = h.rsplit(":", 1)
            s = s1.load_subject(Path(subj))
            m = json.loads((s["_root"] / "game" / ver / "manifest.json").read_text(encoding="utf-8"))
            horses.append(horse_entry(s, Path(m.get("geometry") or s["_root"] / "game" / ver / "low_geometry.json"), m["textures"]))
        print(json.dumps(stage_test(a.test, horses, GAME_DEFAULTS), indent=1))
        return
    m = build(a.subject, a.master, a.test, a.blender)
    print(json.dumps({"version": m["version"], "test": m["test"], "seconds": m["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
