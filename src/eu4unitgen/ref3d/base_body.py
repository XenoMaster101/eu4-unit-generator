"""Track W1: the base body (SPEC Track W). One plain human of our own, made once from an A-pose picture through
TRELLIS.2, on the game's skeleton with our own topology and weights, tuned until it moves like the game's body.

  raw     the picture cut out and TRELLIS.2's two candidates; the better outline is ``raw.glb`` (``raw``)
  build   ``blender/jobs/base_body.py``: the template of rings on the game's bones, shaped to the raw model
          (``build``), then its deformation and review against the game's body (``gate``)

The folder (``assets_in/base_body`` in a checkout) keeps ``rights.json``: the picture's rights (rule 8).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def raw(picture: Path, folder: Path, rights: str, cfg=None) -> dict:
    """The picture cut out, TRELLIS.2's candidates scored on their outline, the better one as ``raw.glb``."""
    import shutil

    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core import config
    from eu4unitgen.ref3d import trellis
    cfg = cfg or config.load()
    folder = Path(folder).resolve()
    tr = folder / "trellis"
    tr.mkdir(parents=True, exist_ok=True)
    (folder / "rights.json").write_text(json.dumps({"picture": str(Path(picture).resolve()), "rights": rights,
                                                    "made": time.strftime("%Y-%m-%d")}, indent=1), encoding="utf-8")
    inp = tr / "input"
    if not (inp / "input_rgba.png").is_file():
        trellis.cutout(Path(picture), inp, cfg.ai())
    if not (tr / "raw.glb").is_file():
        cands = trellis.generate_candidates((inp / "input_rgba.png").resolve(), tr, cfg.ai_root_path())
        br = BlenderRunner(cfg.blender(), log_dir=tr / "logs", timeout=1200)
        for c in cands:
            c.update(trellis.score_candidate(br, Path(c["glb"]), inp / "input_fg.png", tr / f"cand_s{c['seed']}"))
        best = max(range(len(cands)), key=lambda i: cands[i].get("iou") or 0.0)
        (tr / "candidates.json").write_text(json.dumps({"candidates": cands, "best": best}, indent=1), encoding="utf-8")
        shutil.copyfile(cands[best]["glb"], tr / "raw.glb")
    return json.loads((tr / "candidates.json").read_text(encoding="utf-8"))


# the gate (SPEC Track W, B): against the game's body in the same frames
GATE = {"flips_factor": 1.25, "flips_min": 3, "bends_over": 0.03, "joints_spine": 0.2, "joints_legs": 0.2}
DEFORM_SETS = ("polearm_twohanded", "spear_shield", "rifleman", "cavalry_rider")
REVIEW_SETS = ("polearm_twohanded", "spear_shield")


def build(folder: Path, cfg_over: dict | None = None, cfg=None) -> Path:
    """The base body from ``folder/trellis/raw.glb`` into ``folder/game/vNNN``: the job (geometry, the fitted raw
    model), its colours and normal map baked from the raw model, a manifest. -> the version folder."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core import config
    from eu4unitgen.core.fresh_mesh import base_arrays
    from eu4unitgen.ref3d import trellis
    cfg = cfg or config.load()
    folder = Path(folder).resolve()
    glb = folder / "trellis" / "raw.glb"
    out = trellis.next_version(folder / "game")
    br = BlenderRunner(cfg.blender(), log_dir=out / "logs", timeout=3600)
    base_file = Path(cfg.eu4()) / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
    res = br.run_job("base_body", {"vanilla": base_arrays(base_file.read_bytes()), "source": {"path": str(glb), "merge": 1e-5},
                                   "geometry_out": str(out / "low_geometry.json"), "preview_out": str(out / "low.obj"),
                                   "textured_out": str(out / "textured.obj"), "cfg": cfg_over or {}}).result
    basecolor = trellis.glb_basecolor(glb, out / "raw_basecolor.png")
    common = {"gloss": trellis.PROCESS["gloss"], **trellis.PROCESS["bake"], "image": str(basecolor)}
    bc = br.run_job("bake_colour", {**common, "size": 1024, "country_mask": False, "low": str(out / "low_geometry.json"),
                                    "high": str(out / "textured.obj"), "out_dir": str(out), "name": "body"}).result
    br.run_job("bake_horse", {"low": str(out / "low_geometry.json"), "high": [str(out / "textured.obj")],
                              "diffuse": bc["diffuse"], "size": 1024, **trellis.PROCESS["normal"], "preview": False,
                              "out_dir": str(out), "name": "bodyb", "paint": False})
    m = {"stage": "base body (Track W1)", "glb": str(glb), "version": out.name, "fresh": res,
         "geometry": str(out / "low_geometry.json"), "parts": {},
         "textures": {"diffuse": str(out / "bodyb_diffuse.png"), "specular": bc["specular"],
                      "normal": str(out / "bodyb_normal.png")}}
    (out / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return out


def gate(vdir: Path, cfg=None) -> dict:
    """The base body against the game's body (SPEC Track W, B): folds under each set's animations
    (``check_mesh_deform``), squash in the animation review under the soldiers' sets, the joints inside it.
    -> {"deform", "review", "joints", "pass", "fails"}, also ``gate.json``."""
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core import config
    from eu4unitgen.ref3d import review, trellis, unitstage
    cfg = cfg or config.load()
    vdir = Path(vdir).resolve()
    m = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    br = BlenderRunner(cfg.blender(), log_dir=vdir / "logs", timeout=3600)
    base_file = Path(cfg.eu4()) / "gfx" / "models" / "Units" / "western_generic_01_type1.mesh"
    base = base_file.read_bytes()
    fails, deform = [], {}
    for st in DEFORM_SETS:
        d = trellis.rider_deform(br, vdir, base, base_file, renders=False, prefix=f"{st}_")
        deform[st] = {a: {"ours": d["ours"][a].get("flips_visible", d["ours"][a]["flips"]),
                          "ours_with_hidden": d["ours"][a]["flips"], "game": d["vanilla"][a]["flips"]} for a in d["ours"]}
        for a, x in deform[st].items():
            lim = max(x["game"] * GATE["flips_factor"], x["game"] + GATE["flips_min"])
            if x["ours"] > lim:
                fails.append(f"{st} {a}: {x['ours']} folds (the game's {x['game']}, at most {lim:.0f})")
    subject = {"rights": json.loads((vdir.parents[1] / "rights.json").read_text(encoding="utf-8"))["rights"],
               "original": "base body"}
    rev = {}
    for st in REVIEW_SETS:
        out = vdir / f"review_{st}"
        sp = unitstage.spec(f"base body under {st}", 1, None, None, unitstage.unit_names("base_body"), unitstage.TEST,
                            soldier=unitstage.rider_entry(vdir, subject))
        sp["loadout"].setdefault("slots", {})["1/infantry"] = {"parts": [], "stance": st}
        stg = unitstage.stage(out, sp, "preview")
        if stg["exit"] != 0:
            raise RuntimeError(f"staging the base body failed; see {out / 'log.txt'}")
        r = review.review(out, 1, st, Path(cfg.eu4()))
        per = {}
        for state, x in r["states"].items():
            o = max((s.get("bends", 0) for s in x["ours"]), default=0.0)
            g = max((s.get("bends", 0) for s in x["game"]), default=0.0)
            per[state] = {"ours": round(o, 4), "game": round(g, 4)}
            if o > g + GATE["bends_over"]:
                fails.append(f"{st} {state}: {o:.0%} squashed (the game's {g:.0%})")
        rev[st] = {"states": per, "flags": r["summary"]["flags"], "sheet": r["sheet"]}
    j = (m.get("fresh") or {}).get("joints") or {}
    spine = max((j.get("spine_off") or {}).values(), default=None)
    legs = max((v for d in (j.get("legs_off") or {}).values() for v in d), default=None)
    if spine is not None and spine > GATE["joints_spine"]:
        fails.append(f"spine {spine} off its bones")
    if legs is not None and legs > GATE["joints_legs"]:
        fails.append(f"knees and ankles {legs} off their bones")
    res = {"deform": deform, "review": rev, "joints": {"spine": spine, "legs": legs}, "tris": (m["fresh"].get("final") or {}).get("tris"),
           "pass": not fails, "fails": fails}
    (vdir / "gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def current(folder: Path | None = None) -> Path | None:
    """The newest base body version that passed its gate (``gate.json``), or None: the soldiers' builds use it."""
    from eu4unitgen.ref3d import flow
    root = Path(folder) if folder else flow.folder_of("base_body")
    for v in sorted((root / "game").glob("v[0-9][0-9][0-9]"), reverse=True):
        g = v / "gate.json"
        if g.is_file() and json.loads(g.read_text(encoding="utf-8")).get("pass"):
            return v
    return None


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Track W1: the base body")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw")
    r.add_argument("picture", type=Path)
    r.add_argument("folder", type=Path)
    r.add_argument("--rights", required=True)
    b = sub.add_parser("build")
    b.add_argument("folder", type=Path)
    b.add_argument("--cfg", default="{}", help="SHAPE overrides as JSON")
    g = sub.add_parser("gate")
    g.add_argument("version", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "raw":
        print(json.dumps(raw(a.picture, a.folder, a.rights), indent=1, default=str))
    elif a.cmd == "build":
        v = build(a.folder, json.loads(a.cfg))
        print(v)
        print(json.dumps(gate(v), indent=1))
    else:
        print(json.dumps(gate(a.version), indent=1))


if __name__ == "__main__":
    main()
