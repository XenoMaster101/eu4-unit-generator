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

usage: python -m eu4unitgen.ref3d.game build <subject.json> --master vNNN --test testNN (a new name)
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
    "fresh": {"hem_y": 4.6, "leg_radius": 0.65, "tail_gap": 0.35, "voxel": 0.3, "relax": 4, "target_tris": 2000,
              "quads": 2000, "smooth": 12, "influences": 3, "chain_weights": True, "tail": {"build": "auto", "cut": 0.8}},
    "bake": {"size": 512, "cage": 0.4, "distance": 0.8},
    # the cloth as slabs (game_master): the body ~2,000 triangles and four slabs ~180 each stay under the
    # validator's 3,000 (test 20's first stage: 4,224 with 2,800 and 7 x 12 slabs)
    "pieces": {"piece_rows": 5, "piece_cols": 9, "piece_thick": 0.06},
    "normal": {"cage": 0.1, "distance": 0.35, "high_smooth": 0, "strength": 0.6, "samples": 32, "ao_strength": 0.35},
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


def build(subject_path: Path, master_version: str, test: str, blender: str | None = None) -> dict:
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.fresh_mesh import base_arrays
    from eu4unitgen.core.project import default_blender_path

    if (Path("build") / test).exists():  # a test's folder is never written into twice (test 16's spec was lost so)
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
    rp = br.run_job("game_master", {"master_blend": str(mv / "assemble" / "master.blend"), "grids": grids,
                                    "skeleton": str(root / "spikes" / "ea" / "inputs" / "skeleton.json"),
                                    "body": {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")},
                                    "voxel": cfg["voxel_solid"], **cfg["pieces"], "out_dir": str(out / "repose")}).result
    note(f"repose: high {rp['high']['tris']} tris, solid {rp['solid']['tris']} tris")
    base = (eu4_path() / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes()
    (out / "debug").mkdir(exist_ok=True)
    fr = br.run_job("fresh_horse", {**cfg["fresh"], "source": {"kind": "model", "path": rp["solid"]["path"]},
                                    "pieces": rp["pieces"]["path"],
                                    "vanilla": base_arrays(base), "geometry_out": str(out / "low_geometry.json"),
                                    "preview_out": str(out / "low.obj"), "bake_source_out": str(out / "low_source.obj"),
                                    "debug_dir": str(out / "debug")}).result
    note(f"low: {fr['final']['tris']} tris, {fr['final']['vertices']} vertices, {fr['final']['pieces']} piece(s); "
         f"fit x{fr['fit']['scale']:.3f}")
    name = "horse"
    bc = br.run_job("bake_master", {"reposed_blend": str(out / "repose" / "reposed.blend"), "low": str(out / "low_geometry.json"),
                                    "fit": fr["fit"], **cfg["bake"], "classes": s["_template"]["classes"],
                                    "out_dir": str(out), "name": name}).result
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
    spec = test_spec(s, cfg, out / "low_geometry.json", tex, test)
    stage = Path("build") / test
    stage.mkdir(parents=True, exist_ok=True)
    (stage / f"{test}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    cmd = [sys.executable, "-m", "eu4unitgen.cli", "test-mod", "--spec", str(stage / f"{test}.json"),
           "--stage-dir", str(stage / "stage")]
    st = subprocess.run(cmd, capture_output=True, text=True)
    (stage / "log.txt").write_text(st.stdout + st.stderr, encoding="utf-8")
    note(f"stage: exit {st.returncode}; {stage / 'stage'}")
    manifest = {"stage": "game version", "subject": s["name"], "version": out.name, "master": master_version,
                "settings": cfg, "repose": rp, "low": fr, "colour": bc, "normal": nb, "textures": tex,
                "test": {"spec": str(stage / f"{test}.json"), "stage": str(stage / "stage"), "exit": st.returncode},
                "log": log, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


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


def test_spec(s: dict, cfg: dict, geometry: Path, tex: dict, test: str) -> dict:
    t = cfg["test"]
    return {"schema": 1, "name": f"{test}: the {s['name']} game horse (tier 1), the vanilla horse at tier 2 for comparison",
            "set": {"tags": [t["tag"]], "gfx": t["gfx"], "set_name": t["set_name"], "base_gfx": t["base_gfx"],
                    "unit_colours": {t["tag"]: t["unit_colours"]}},
            "loadout": {"schema": 1, "horses": {"1": {
                "base": "cavalry_horse", "geometry": str(geometry.resolve()),
                "textures": {k: str(Path(v).resolve()) for k, v in tex.items()},
                "rights": s.get("rights", "user"),
                "source": f"Track M master of {s['name']} (assets_in/{s['name']}; references L, R, S)"}}},
            "test": {"sprite_techs": t["sprite_techs"]}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("subject", type=Path)
    b.add_argument("--master", required=True)
    b.add_argument("--test", required=True, help="a new build/<test> folder (refused if it exists)")
    b.add_argument("--blender")
    a = p.parse_args(argv)
    m = build(a.subject, a.master, a.test, a.blender)
    print(json.dumps({"version": m["version"], "test": m["test"], "seconds": m["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
