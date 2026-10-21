"""Track M, the self-reading pipeline (SPEC M18), step 4: a new horse from one reference image with no
hand setup.

  new   the subject folder ``assets_in/<name>/`` from the image alone:
          1. the view: the whole image, the horse mask (BiRefNet); a horse facing right is mirrored to
             show a left side (``mirror``); the view is named ``side`` (``main_view``: no fixed names);
          2. SAM 2 segments and the parts job (ref3d/parts.py): labels, parts, zones and landmarks;
          3. the subject config written from them: zones (head, legs, tail), landmarks, the body length
             (chest to buttock), the head's pivot (poll), E-A's camera window for a side view (when the
             feet of both pairs show apart), the fit's workers from the machine's cores;
          4. the skeleton input (``spikes/ea/inputs/skeleton.json``) from the EU4 install's
             cavalry_horse.mesh: bone positions, the seat (the back's surface at Saddle_Node), the
             vanilla positions and triangles (for the vanilla-distance check), round 5's B parameters.
  run   the pipeline on it, untouched: TripoSG shells, the E-A fit, the refine, the master, the game
        version (and a staged test with ``--test``). One heavy job at a time; each step starts only with
        enough free RAM (memory.STEP_MIN_GB; else it stops and says so), and the fit runs as many processes
        as the free RAM holds (memory.fit_workers). A stopped run continues: finished steps are skipped,
        and an unfinished shells version keeps its finished seeds. The lowest free RAM per step is logged.

The only inputs are the image, a name, the rights declaration (rule 8: the user's statement, never
inferred) and the AI environment's python; the model folders follow from it
(``<AI root>/rembg_models``, ``weights/sam2.1-hiera-large``, ``weights/siglip2-so400m-patch14-384``,
``TripoSG``, ``weights/TripoSG``) unless the environment variables name others.

usage: python -m eu4unitgen.ref3d.auto new <image> --name <subject> --rights "<declaration>" --ai-python <env python>
       python -m eu4unitgen.ref3d.auto run <subject.json> --ai-python <env python> [--test testNN]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import ai_root

MAIN = "side"
# round 5's horse B (the bake-off's parametric horse; E-A's "b2" candidate): the app's own parameters, the
# same for every subject
B2_PARAMS = {"barrel_depth": 0.5283, "tuck_up": 0.06, "belly_sag": 0.015, "withers_rise": 0.0243, "croup_rise": -0.1,
             "rump_drop": 0.09, "breast_ahead": 0.13, "rump_behind": 0.11, "breast_width": 0.1569, "shoulder_width": 0.2139,
             "rib_width": 0.271, "hip_width": 0.2567, "buttock_width": 0.1854, "roundness": 2.4, "top_narrow": 0.25,
             "breast_round": 0.1, "rump_round": 0.14, "neck_base_depth": 0.48, "neck_throat_depth": 0.1771, "neck_width": 0.6,
             "crest": 0.75, "head_length": 0.28, "head_pitch": 40.0, "jowl_depth": 0.2403, "forehead_width": 0.1138,
             "muzzle_depth": 0.1265, "muzzle_width": 0.0822, "ear_length": 0.05, "upper_arm": 0.115, "forearm": 0.0812,
             "knee": 0.046, "stifle": 0.1082, "gaskin": 0.0676, "hock": 0.046, "cannon": 0.0325, "fetlock": 0.0419,
             "pastern": 0.0325, "hoof_top": 0.0379, "hoof_bottom": 0.05, "hoof_height": 0.04, "leg_flat": 0.85,
             "rider_clear": 2.1}
# a true side view's camera (the cataphract's S: a clean side reference)
SIDE_CAMERA = {"az": [-25, 25], "el": [-5, 15], "roll": [-3, 3], "persp": [0.005, 0.03]}


def ai_paths(ai_python: str) -> dict:
    """The model folders under the AI root (or the environment variables naming them)."""
    root = ai_root(ai_python)
    if root is None:
        raise SystemExit(f"{ai_python}: not inside an AI root (<root>/<env>/Scripts/python.exe)")
    return {"root": root, "u2net": os.environ.get("U2NET_HOME") or str(root / "rembg_models"),
            "sam2": str(root / "weights" / "sam2.1-hiera-large"),
            "siglip": str(root / "weights" / "siglip2-so400m-patch14-384"),
            "triposg_repo": os.environ.get("EU4UNITGEN_TRIPOSG_REPO") or str(root / "TripoSG"),
            "triposg_weights": os.environ.get("EU4UNITGEN_TRIPOSG_WEIGHTS") or str(root / "weights" / "TripoSG")}


def seat_height(p: list, tri: list, z: float, half: float = 0.5) -> float:
    """The back's surface over Saddle_Node: the highest triangle hit by vertical rays at x 0, z +- ``half``
    (fresh_horse.surface_top's rule, in plain Python)."""
    best = None
    for dz in (-half, -half / 2, 0.0, half / 2, half):
        qx, qz = 0.0, z + dz
        for t in range(0, len(tri), 3):
            a, b, c = (p[3 * tri[t + k]:3 * tri[t + k] + 3] for k in range(3))
            d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
            if abs(d) < 1e-12:
                continue
            w1 = ((b[2] - c[2]) * (qx - c[0]) + (c[0] - b[0]) * (qz - c[2])) / d
            w2 = ((c[2] - a[2]) * (qx - c[0]) + (a[0] - c[0]) * (qz - c[2])) / d
            w3 = 1 - w1 - w2
            if min(w1, w2, w3) < 0:
                continue
            y = w1 * a[1] + w2 * b[1] + w3 * c[1]
            best = y if best is None else max(best, y)
    return best


def skeleton_input(eu4: Path) -> dict:
    """E-A's skeleton input from the install (read at run time, never copied into the repo: rule 1)."""
    from eu4unitgen.core.fresh_mesh import base_arrays
    b = base_arrays((eu4 / "gfx" / "models" / "Units" / "cavalry_horse.mesh").read_bytes())
    p, tri = b["p"], b["tri"]
    return {"note": "derived at runtime from the user install (cavalry_horse.mesh): bone positions, seat; vanilla "
                    "p/tri only for the vanilla_distance check; git-ignored, never shipped (written by ref3d.auto)",
            "bones": b["bones"], "seat": seat_height(p, tri, b["bones"]["Saddle_Node"][2]),
            "vanilla_p": p, "vanilla_tri": tri, "b2_params": B2_PARAMS}


def facing_left(fg: Image.Image) -> bool:
    """The end whose outer fifth reaches higher holds the head (the ears over the croup)."""
    m = fg.convert("L").point(lambda x: 255 if x > 127 else 0)
    x0, y0, x1, y1 = m.getbbox()
    w = x1 - x0
    top = lambda a, b: m.crop((a, 0, b, m.height)).getbbox()[1]  # noqa: E731
    return top(x0, x0 + w // 5) <= top(x1 - w // 5, x1)


def new(image: Path, name: str, rights: str, ai_python: str, eu4: Path) -> dict:
    from eu4unitgen.ai.env import AIJobRunner
    from eu4unitgen.ref3d import parts

    image = Path(image).resolve()
    folder = Path("assets_in") / name
    if folder.exists():
        raise SystemExit(f"{folder} exists: pick a new name")
    paths = ai_paths(ai_python)
    t0 = time.time()
    folder.mkdir(parents=True)
    W, H = Image.open(image).size
    # 1. the view, its mask, which way the horse faces
    out = s1.next_version(folder / "refs" / "clean")
    img = Image.open(image).convert("RGB")
    raw = out / f"{MAIN}_raw.png"
    img.save(raw)
    ai = AIJobRunner(ai_python, env={"U2NET_HOME": paths["u2net"]})
    fg = ai.run_job("ref3d_mask", {"images": {f"{MAIN}_seed0": str(raw)}, "out_dir": str(out)}).result
    fgp = Path(fg[f"{MAIN}_seed0"]["path"])
    mirror = not facing_left(Image.open(fgp))
    if mirror:  # a right side: mirrored to show a left side (the template's near legs are the horse's left)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        Image.open(fgp).transpose(Image.FLIP_LEFT_RIGHT).save(fgp)
    img.save(out / f"{MAIN}_seed0.png")
    raw.unlink()
    Image.new("L", (W, H), 0).save(out / f"{MAIN}_mask_used.png")
    rel = os.path.relpath(image, folder.resolve())
    s = {"schema": 1, "name": name, "template": "horse", "rights": rights, "original": rel,
         "coords": "the view's pixels (x right, y down)" + ("; the image mirrored (the horse faced right)" if mirror else ""),
         "written_by": "ref3d.auto new (M18: read from the image, no hand setup)",
         "main_view": MAIN,
         "views": {MAIN: {"source": rel, "source_sha256": s1.sha256(image), "crop": [0, 0, W, H], "mirror": mirror,
                          "role": "the reference, read automatically (a side view)", "use_for": ["everything (the only view)"],
                          "landmarks": {}, "details": []}},
         "chosen": {MAIN: "0"}, "approved": {"clean": out.name, "labels": out.name, "note": "ref3d.auto: as read"},
         "ed2d": {"unbroken": 5, "seed": 100, "residual": 0.01, "warp_px": 2.5, "light": 0.12, "red_shift": 15},
         "ea": {"workers": max(1, min(6, (os.cpu_count() or 2) // 2)),
                "note": "workers: half the machine's cores (at most 6); the rest EA_DEFAULTS",
                "refine": {"trunk_views": [MAIN]}}}
    sp = folder / "subject.json"
    sp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    # 2. segments, parts, zones, landmarks
    m = parts.run(sp, ai_python, paths["siglip"], paths["sam2"])
    L = m["labels"][MAIN]
    zones = json.loads(Path(L["zones"]).read_text(encoding="utf-8"))
    lm = L["layout"]["landmarks"]
    # 3. the config from them
    v = s["views"][MAIN]
    v["zones"] = {z: zones[z] for z in ("head", "legs", "tail") if zones.get(z)}
    v["landmarks"] = lm
    v["body_length"] = {"from": "chest_front", "to": "buttock", "axis": "x"}
    v["head"] = {"zones": ["head"], "pivot": "poll"}
    feet = sum(1 for k in ("fore_near_hoof", "fore_far_hoof", "hind_near_hoof", "hind_far_hoof") if k in lm)
    apart = feet == 4 and lm["fore_near_hoof"] != lm["fore_far_hoof"] and lm["hind_near_hoof"] != lm["hind_far_hoof"]
    if apart:  # all four feet show apart: a side view (its camera window as the cataphract's S)
        s["ea"]["cameras"] = {MAIN: SIDE_CAMERA}
    s["approved"] = {"clean": out.name, "labels": m["version"], "note": "ref3d.auto: as read (no review)"}
    s["parts_found"] = {k: p["present"] for k, p in L["parts"].items()}
    sp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    # 4. the skeleton input
    inp = folder / "spikes" / "ea" / "inputs"
    inp.mkdir(parents=True)
    (inp / "skeleton.json").write_text(json.dumps(skeleton_input(eu4)), encoding="utf-8")
    rep = {"subject": str(sp), "mirrored": mirror, "labels": m["version"], "parts": s["parts_found"],
           "landmarks": sorted(lm), "side_view_window": apart, "sheet": m["sheets"]["parts"],
           "seconds": round(time.time() - t0, 1)}
    (folder / "auto_new.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def run(subject: Path, ai_python: str, test: str | None = None, blender: str | None = None) -> dict:
    """The whole pipeline on a subject ``new`` wrote, untouched: shells, fit, refine, master, game (test)."""
    from eu4unitgen.ref3d import ea, game, master
    paths = ai_paths(ai_python)
    os.environ.setdefault("EU4UNITGEN_TRIPOSG_PYTHON", ai_python)
    os.environ.setdefault("EU4UNITGEN_TRIPOSG_REPO", paths["triposg_repo"])
    os.environ.setdefault("EU4UNITGEN_TRIPOSG_WEIGHTS", paths["triposg_weights"])
    folder = Path(subject).resolve().parent
    log_path = folder / "auto_run.json"
    rep = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {"steps": {}}

    from eu4unitgen.ref3d import memory

    def step(key, fn):
        if key in rep["steps"]:  # done in an earlier (stopped) run
            print(f"{key}: done earlier ({rep['steps'][key]['version']})", flush=True)
            return rep["steps"][key]
        total, free = memory.ram_gb()
        need = memory.STEP_MIN_GB.get(key, 8.0)
        if free < need:
            raise SystemExit(f"{key}: {free:.1f} GB of {total:.1f} GB RAM free, {need:.0f} GB wanted: close programs "
                             f"and run again (finished steps are kept)")
        t = time.time()
        with memory.LowWatermark() as wm:
            m = fn()
        rep["steps"][key] = {"version": m["version"], "seconds": round(time.time() - t, 1),
                             "free_ram_gb": {"start": round(free, 1), "lowest": round(wm.low or free, 1)}}
        log_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"{key}: {m['version']} ({rep['steps'][key]['seconds']} s)", flush=True)
        return rep["steps"][key]

    sdir = folder / "spikes" / "ea" / "shells"
    open_ = sorted(p.name for p in sdir.glob("v*") if not (p / "manifest.json").is_file()) if sdir.is_dir() else []
    sh = step("shells", lambda: ea.shells(subject, [11 * (k + 1) for k in range(8)], ai_python,
                                          resume=open_[-1] if open_ else None))
    workers = memory.fit_workers(memory.ram_gb()[1], 6)
    rep["fit_workers"] = workers
    print(f"fit: {workers} process(es) for the free RAM", flush=True)
    ft = step("fit", lambda: ea.fit(subject, ai_python, sh["version"], workers=workers))
    rf = step("refine", lambda: ea.refine(subject, ai_python, ft["version"]))
    ms = step("master", lambda: master.build(subject, rf["version"], ai_python, blender))
    gm = step("game", lambda: game.build(subject, ms["version"], test, blender))
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new")
    a.add_argument("image", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--rights", required=True, help="the rights declaration (rule 8): who made the image and that you may use it")
    a.add_argument("--ai-python", required=True)
    b = sub.add_parser("run")
    b.add_argument("subject", type=Path)
    b.add_argument("--ai-python", required=True)
    b.add_argument("--test")
    b.add_argument("--blender")
    args = p.parse_args(argv)
    if args.cmd == "new":
        eu4 = os.environ.get("EU4_PATH")
        if not eu4:
            raise SystemExit("set EU4_PATH to the EU4 install (the vanilla horse's skeleton is read from it)")
        print(json.dumps(new(args.image, args.name, args.rights, args.ai_python, Path(eu4)), indent=1))
    else:
        print(json.dumps(run(args.subject, args.ai_python, args.test, args.blender), indent=1))


if __name__ == "__main__":
    main()
