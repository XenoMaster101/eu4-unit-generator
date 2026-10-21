"""Track M spike E-A (SPEC M9): which body wins in both reference cameras: TripoSG shells (N seeds
on the cleaned view L), the procedural anatomy fitted to the 2D references, a hybrid (the fit with
TripoSG-derived widths as a prior), or round 5's B?

Steps (each a new version folder under ``<subject>/spikes/ea/``; nothing overwritten):
  shells: TripoSG on the approved cleaned view L (background removed by the approved horse mask),
          one raw mesh per seed (``shells/vNNN``).
  fit:    the camera and pose per view and the anatomy shape, by least squares in the AI
          environment (job ref3d_ea_fit), for every candidate; all scored by the M5 gates in both
          cameras; three restarts (``fit/vNNN``). The job checkpoints as it goes; a run stopped
          (the AI drive dropped out, the PC restarted) continues with ``--resume vNNN``, on the
          arguments saved in that folder.

  refine: round 13 (the user's H3; no long searches): an earlier fit's candidate refitted in place with
          anatomy style 2 and every current view (a view the earlier fit didn't have gets its camera
          first); the head, the leg sections, the cameras and poses; the proportions stay
          (``fit/vNNN``, the candidate under the same name).

usage: python -m eu4unitgen.ref3d.ea shells <subject.json> --seeds 8
       python -m eu4unitgen.ref3d.ea fit <subject.json> --ai-python <env python> [--shells vNNN] [--resume vNNN]
       python -m eu4unitgen.ref3d.ea refine <subject.json> --ai-python <env python> --from vNNN [--resume vNNN]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes


def shell_input(s: dict, out: Path) -> Path:
    """The approved cleaned view L, its background removed by the approved horse mask (RGBA),
    cropped to the horse with a margin: TripoSG's input."""
    root, ap = s["_root"], s["approved"]
    k = s["chosen"]["L"]
    clean = root / "refs" / "clean" / ap["clean"]
    rgb = Image.open(clean / f"L_seed{k}.png").convert("RGB")
    alpha = Image.open(clean / f"L_seed{k}_fg.png").convert("L")
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    x0, y0, x1, y1 = alpha.point(lambda v: 255 if v > 127 else 0).getbbox()
    m = int(0.05 * max(x1 - x0, y1 - y0))
    rgba = rgba.crop((max(0, x0 - m), max(0, y0 - m), min(rgb.width, x1 + m), min(rgb.height, y1 + m)))
    p = out / "input_L.png"
    rgba.save(p)
    return p


def shells(subject_path: Path, seeds: list[int], python: str | None = None) -> dict:
    from eu4unitgen.ai.triposg import TripoSGRunner

    s = s1.load_subject(subject_path)
    r = TripoSGRunner(python)
    check_writes({"out_dir": str(s["_root"] / "spikes")}, allowed_roots(r.python), "E-A shells")
    out = s1.next_version(s["_root"] / "spikes" / "ea" / "shells")
    t0 = time.time()
    img = shell_input(s, out)
    res = {}
    for seed in seeds:
        res[str(seed)] = r.generate(img, out / f"seed{seed}.glb", seed=seed)
        print("seed", seed, {k: res[str(seed)][k] for k in ("faces", "components", "seconds_generate")}, flush=True)
    manifest = {"stage": "E-A shells", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "input": {"path": str(img), "sha256": s1.sha256(img)}, "seeds": res,
                "settings": {"steps": 50, "guidance": 7.0, "octree_depth": 9}, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


EA_DEFAULTS = {
    "candidates": ["b2", "procedural", "hybrid"], "restarts": 3, "seed": 7,
    # assumed: armour and cloth thickness over the barrel, as a fraction of the seat height
    "armour_allowance": 0.03,
    "body": {"legs": ["leg", "coat"], "head": ["chanfron", "coat", "leather"]},
    "occluders": {"legs": ["cloth", "trim", "leather", "scale", "tail", "plume", "chanfron"],
                  "head": ["plume", "cloth", "trim", "scale"]},
    # E-A pilot 7: the outlines outweighed the head's landmarks (worst 6.7%, the gate 2.5%)
    "landmark_weights": {k: 3.0 for k in ("poll", "eye", "muzzle_front", "nostril", "bit_ring", "chin", "throat",
                                          "ear_tip_a", "ear_tip_b")},
    # evaluations per stage (pilot 7 ran out in every outline stage at 1500 / 1500 / 2000)
    "evals": {"landmarks": 400, "outlines: camera+pose": 4000, "outlines: shape": 3000, "outlines: joint": 5000},
    # per view: {param: [lo, hi]} narrowing the camera's bounds (the subject config sets each view's
    # azimuth window: which side of the horse the view shows)
    "cameras": {},
    # per view: the neck's sideways bend and the head's turn (E-A pilot 8: the reference's head turns
    # to the viewer; with sagittal joints only, its landmarks stayed 3-5% off)
    "head_turn": True,
    # E-A round 10 (pilots 7-9: the head's landmarks stuck at 5-7%): the head and chanfron as their own
    # shape (a free profile at ``stations`` along the head) in a stage of their own against the head view
    "head_fit": {"view": "L", "stations": 8, "evals": 3000},
    # SPEC S3's fallback: each view fitted on its own; the master is L's body with R's rump (M10)
    "independent": True,
}


REFINE_DEFAULTS = {
    "candidate": "hybrid",
    # the user (round 13): S is the main evidence for the head and legs (unoccluded), L for the head too;
    # leg length and proportions are not taken from S (its legs are shorter: its own factor, NUIS)
    "head_views": ["L", "S"], "leg_views": ["L", "S"], "nuisance_legs": ["S"],
    # round 13c: the leg sections stay style 2's (fitted, they collapsed onto the feathering and occluders:
    # cannons half a horse's); the trunk landmarks steer S only; S's legs stand square
    "fit_leg_shape": False, "trunk_views": ["S"], "straight_legs": ["S"],
    # Powell evaluations per stage (each evaluation renders the stage's views)
    "stages": {"camera": 1500, "head": 2000, "legs": 2000, "polish": 1000},
}


def fit_args(s: dict, out: Path, shells_dir: Path | None, skeleton: Path) -> dict:
    """The ref3d_ea_fit job's arguments from the subject config and the approved H1 outputs."""
    from eu4unitgen.ref3d import ed2d
    base = ed2d.build_args(s, out)
    cfg = {**EA_DEFAULTS, **(s.get("ea") or {})}
    gates = dict(base["gates"])
    if cfg.get("head_landmark_max"):  # the user's head allowance (painted reference; SPEC M15)
        gates["landmark_max_head"] = cfg["head_landmark_max"]
    views = {}
    for n, v in base["views"].items():
        sv = s["views"][n]
        zones = s1.zones_in_crop(sv)
        z = {"legs": zones.get("legs", [])}
        if sv.get("head"):
            z["head"] = [q for name in sv["head"]["zones"] if name != "plume" for q in zones[name]]
        views[n] = {k: v[k] for k in ("image", "labels", "fg", "inpainted", "never_use", "landmarks", "body_length")}
        views[n]["zones"] = z
    shells = sorted(str(p) for p in shells_dir.glob("seed*.glb")) if shells_dir else []
    return {"skeleton": str(skeleton), "views": views, "classes": base["classes"], "gates": gates,
            "label_scale": gates["label_scale"], "shells": shells, "out_dir": str(out), **cfg}


def _view_key(r: dict) -> tuple:
    """A candidate's standing in one camera: gates passed, worst region IoU, worst landmark."""
    g = r["gates"]
    return (sum(1 for x in g.values() if x["pass"]), g["region_iou"]["worst"] or 0.0, -(g["landmarks"]["worst"] or 9.0))


def beats(cand: dict, ref: dict) -> bool:
    """SPEC M9 'beats round 5's B on the gates in both cameras': in every camera at least as many
    gates passed, worst region IoU and worst landmark no worse; strictly better in at least one."""
    better = False
    for vn, r in cand["views"].items():
        a, b = _view_key(r), _view_key(ref["views"][vn])
        if a[0] < b[0] or a[1] < b[1] - 1e-9 or a[2] < b[2] - 1e-9:
            return False
        better |= a != b
    return better


def verdict(res: dict, converge: float = 0.25, low_confidence: tuple = ()) -> dict:
    """E-A's pass: the best fitted candidate passes the region and landmark gates in both cameras,
    beats round 5's B, and its restarts end within ``converge`` (relative) of the best cost. Views in
    ``low_confidence`` (the user's rule for R, SPEC M15) are scored and reported but don't block."""
    c = res["candidates"]
    fitted = [n for n in ("procedural", "hybrid") if n in c]
    if not fitted:
        return {"pass": False, "why": "no fitted candidate"}
    best = max(fitted, key=lambda n: min(_view_key(r) for vn, r in c[n]["views"].items() if vn not in low_confidence))
    rec = c[best]
    gates_ok = all(g["pass"] for vn, r in rec["views"].items() if vn not in low_confidence for g in r["gates"].values())
    costs = [r["cost"] for r in rec.get("restarts", [])]
    if rec.get("independent") and low_confidence:  # judged on the deciding views' own fits (R doesn't decide)
        per = [c.get(f"{best}_{vn}", {}).get("restarts", []) for vn in rec["views"] if vn not in low_confidence]
        if per and all(per):
            costs = [sum(x[k]["cost"] for x in per) for k in range(min(len(x) for x in per))]
    conv = bool(costs) and (max(costs) - min(costs)) <= converge * max(min(costs), 1e-9)
    b = "b2" in c and beats({**rec, "views": {vn: r for vn, r in rec["views"].items() if vn not in low_confidence}}, c["b2"])
    return {"pass": gates_ok and b and conv, "best": best, "gates_pass": gates_ok, "beats_b2": b,
            "low_confidence": list(low_confidence),
            "restarts_converge": conv, "restart_costs": costs,
            "worst": {vn: {k: g["worst"] for k, g in r["gates"].items()} for vn, r in rec["views"].items()}}


def sheet(res: dict, s: dict, path: Path, version: str):
    """Every candidate in both cameras: the shaded body over the darkened reference (landmark
    errors in red), the outline over the reference, and its numbers."""
    rows = []
    for name, rec in res["candidates"].items():
        tiles = []
        for vn, r in rec["views"].items():
            tiles.append((f"{vn}: body over the reference", Image.open(r["render"]), False))
            tiles.append((f"{vn}: outline", Image.open(r["outline"]), False))
        txt = []
        for vn, r in rec["views"].items():
            reg = ", ".join(f"{z} {v:.2f}" for z, v in r["regions"].items() if v is not None)
            txt.append(f"{vn}: IoU {reg}; landmarks worst {r['gates']['landmarks']['worst'] * 100:.1f}% "
                       f"(rms {r['landmark_rms'] * 100:.1f}%); outside mask {r['outside_share'] * 100:.0f}%")
            txt.append("   gates: " + ", ".join(f"{k} {'ok' if g['pass'] else 'FAIL'}" for k, g in r["gates"].items()))
        vd = rec.get("vanilla_distance")
        if vd:
            txt.append(f"vanilla distance median {vd['median']:.2f}, within 0.15: {vd['within_0.15'] * 100:.0f}%")
        if rec.get("restarts"):
            txt.append("restart costs: " + ", ".join(f"{x['cost']:.3f}" for x in rec["restarts"]))
        tiles.append(("", text_tile("\n".join(s1._wrap(t, 46) for t in txt)), False))
        rows.append((name, tiles))
    for name, rec in (res.get("shells") or {}).items():
        tiles = [(f"{vn}: shell", Image.open(r["render"]), False) for vn, r in rec["views"].items()]
        txt = [f"{vn}: horse IoU {r['horse_iou']:.2f}; " + ", ".join(f"{z} {v:.2f}" for z, v in r["regions"].items() if v is not None)
               for vn, r in rec["views"].items()]
        txt.append(f"placement cost {rec['cost']:.3f}" + ("  (BEST: hybrid prior)" if name == res.get("best_shell") else ""))
        tiles.append(("", text_tile("\n".join(s1._wrap(t, 46) for t in txt)), False))
        rows.append((f"TripoSG\n{name}", tiles))
    return s1.grid_sheet(rows, f"E-A body candidates in both reference cameras: {s['name']} {version}", path,
                         tile=(300, 360))


def text_tile(text: str, size=(300, 360)):
    return s1.text_tile(text, size)


def fit(subject_path: Path, ai_python: str, shells_version: str | None, resume: str | None = None) -> dict:
    from eu4unitgen.ai.env import AIJobRunner

    s = s1.load_subject(subject_path)
    root = s["_root"] / "spikes" / "ea"
    check_writes({"out_dir": str(root)}, allowed_roots(ai_python), "E-A fit")
    skeleton = root / "inputs" / "skeleton.json"
    if not skeleton.is_file():
        raise FileNotFoundError(f"{skeleton}: write the vanilla skeleton first (bones, seat; read from the install)")
    t0 = time.time()
    if resume:
        out = root / "fit" / resume
        if (out / "manifest.json").is_file():
            raise FileExistsError(f"{out} is finished (manifest.json); nothing to resume")
        args = json.loads((out / "args.json").read_text(encoding="utf-8"))
        shells_version = args.get("shells_version")
    else:
        out = s1.next_version(root / "fit")
        args = fit_args(s, out, root / "shells" / shells_version if shells_version else None, skeleton)
        args["shells_version"] = shells_version
        (out / "args.json").write_text(json.dumps(args, indent=1), encoding="utf-8")
    res = AIJobRunner(ai_python, timeout=6 * 3600).run_job("ref3d_ea_fit", args).result
    sheet(res, s, out / "ea_sheet.png", out.name)
    manifest = {"stage": "E-A fit", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "shells_version": shells_version, "resumed": bool(resume),
                "settings": {k: v for k, v in args.items() if k != "views"},
                "verdict": verdict(res, low_confidence=tuple(args.get("low_confidence") or ())), "sheet": str(out / "ea_sheet.png"),
                "result": res, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def refine(subject_path: Path, ai_python: str, from_version: str, resume: str | None = None) -> dict:
    """Round 13's refit of an earlier fit's candidate (module doc) into a new fit version."""
    from eu4unitgen.ai.env import AIJobRunner

    s = s1.load_subject(subject_path)
    root = s["_root"] / "spikes" / "ea"
    check_writes({"out_dir": str(root)}, allowed_roots(ai_python), "E-A refine")
    t0 = time.time()
    if resume:
        out = root / "fit" / resume
        args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    else:
        out = s1.next_version(root / "fit")
        src = root / "fit" / from_version
        args = fit_args(s, out, None, root / "inputs" / "skeleton.json")
        rf = {**REFINE_DEFAULTS, **((s.get("ea") or {}).get("refine") or {})}
        rf["stages"] = {**REFINE_DEFAULTS["stages"], **rf.get("stages", {})}
        args["refine"] = {**rf, "from": str(src / "result.json"), "from_version": from_version, "anatomy_style": 2}
        args["candidates"] = [rf["candidate"]]
        (out / "args.json").write_text(json.dumps(args, indent=1), encoding="utf-8")
    res = AIJobRunner(ai_python, timeout=6 * 3600).run_job("ref3d_ea_fit", args).result
    sheet(res, s, out / "ea_sheet.png", out.name)
    rec = res["candidates"][args["refine"]["candidate"]]
    lc = tuple(args.get("low_confidence") or ())
    manifest = {"stage": "E-A refine", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "refined_from": args["refine"]["from_version"], "settings": {k: v for k, v in args.items() if k != "views"},
                "gates": {vn: {k: g["pass"] for k, g in r["gates"].items()} for vn, r in rec["views"].items()},
                "gates_pass": all(g["pass"] for vn, r in rec["views"].items() if vn not in lc for g in r["gates"].values()),
                "low_confidence": list(lc), "sheet": str(out / "ea_sheet.png"), "result": res,
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("shells")
    a.add_argument("subject", type=Path)
    a.add_argument("--seeds", type=int, default=8)
    a.add_argument("--python", default=os.environ.get("EU4UNITGEN_TRIPOSG_PYTHON"))
    b = sub.add_parser("fit")
    b.add_argument("subject", type=Path)
    b.add_argument("--ai-python", required=True)
    b.add_argument("--shells", help="shells version (vNNN) to align and use as the hybrid's prior")
    b.add_argument("--resume", help="fit version (vNNN) of a stopped run: continue it from its checkpoints")
    c = sub.add_parser("refine")
    c.add_argument("subject", type=Path)
    c.add_argument("--ai-python", required=True)
    c.add_argument("--from", dest="from_version", required=True, help="the fit version (vNNN) whose candidate is refitted")
    c.add_argument("--resume", help="refine version (vNNN) of a stopped run")
    args = p.parse_args(argv)
    if args.cmd == "refine":
        m = refine(args.subject, args.ai_python, args.from_version, args.resume)
        print(json.dumps({"version": m["version"], "gates": m["gates"], "seconds": m["seconds"]}, indent=1))
        return
    if args.cmd == "fit":
        from eu4unitgen.ai.env import AIDriveLost
        try:
            m = fit(args.subject, args.ai_python, args.shells, args.resume)
        except AIDriveLost as exc:
            last = sorted((s1.load_subject(args.subject)["_root"] / "spikes" / "ea" / "fit").glob("v*"))[-1].name
            print(f"{exc}\n\nProgress is saved. Once the drive is back, resume with:\n"
                  f"  python -m eu4unitgen.ref3d.ea fit {args.subject} --ai-python {args.ai_python} --resume "
                  f"{args.resume or last}", flush=True)
            raise SystemExit(75)
        print(json.dumps({"version": m["version"], "seconds": m["seconds"]}, indent=1))
    elif args.cmd == "shells":
        m = shells(args.subject, [11 * (k + 1) for k in range(args.seeds)], args.python)
        print(json.dumps({"version": m["version"], "seconds": m["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
