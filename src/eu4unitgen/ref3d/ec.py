"""Track M spike E-C (SPEC M9, M4 S5): one caparison side panel simulated as cloth from a flat
pattern on the fitted body. Does the hem land where the references put it, without cutting into the
body or the armour under it, identically on every rerun? And does a texture projected from the views
continue without a visible seam onto the parts no view shows (M9's texture-consistency spike)?

Steps (a new version folder under ``<subject>/spikes/ec/``; nothing overwritten):
  1. The body: E-A's best fitted candidate (``--ea vNNN``), in view L's pose.
  2. Rounds (Blender job ec_panel): simulate, then correct the pattern (SPEC M6: the hem by
     optimising the drop per control column) from the hem landmarks of L and R, until every hem
     landmark is within ``hem_target`` or ``rounds`` run out.
  3. The final build (texture, renders), then ``reruns`` more simulations in fresh Blender processes
     on the same arguments: the panel's vertices must match.
  4. The seam metric (AI environment, job ref3d_ec_measure); the proof sheet; manifest with verdict.

usage: python -m eu4unitgen.ref3d.ec run <subject.json> --ea vNNN --ai-python <env python>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes

EC_DEFAULTS = {
    "rounds": 12,
    "reruns": 5,
    "withers_m": 1.55,  # assumed master scale, as E-B
    "armour_m": 0.02,  # the scale armour under the cloth (M10: R shows it under the red cloth)
    "penetration_limit_m": 0.001,
    "hem_target": 0.015,  # stop correcting once every hem landmark is within 1.5% (the gate is 2.5%)
    "rerun_tol": 1e-5,  # units: identical reruns
    "seam_p95_max": 8.0,  # CIEDE2000 kL=2: the M5 colour gate, at the observed/unseen border
    # 95% of edges within 20% of the pattern (my threshold: SPEC S5 names stretch, no number; p99 and the max are
    # reported: on pilot 12's body ~5% of vertical edges stretch 25-40% where the cloth leaves the barrel's bulge)
    "stretch_max": 1.2,
    "low_confidence": ["R"],  # the user's rule (SPEC M15): R's hem landmarks reported, not gated or steering
    "panel_classes": ["cloth", "trim"],
    "texture_px": 512,
    "samples": 64,
    "panel": {"cols": 48, "rows": 24, "drop_m": [0.75] * 7, "start_clear_m": 0.01, "extend": 0.08},
    # a heavy wool caparison (Blender cloth, SI units)
    # the panel's own hem landmarks (hem_front is the chest panel's)
    "hem_landmarks": {"L": ["hem_side", "hem_belly", "hem_rear"], "R": ["hem_side", "hem_rear"]},
    # (E-C: stiffer, 120 / 80 at quality 15, made the hem jump 3-28% between rounds: the panel caught on
    # the legs one way or another)
    "cloth": {"quality": 10, "panel_mass_kg": 1.5, "air_damping": 1.0, "stiffness": 40.0, "shear": 40.0, "bending": 10.0,
              "damping": 5.0, "collision_quality": 4, "distance_m": 0.005, "friction": 0.5, "frames": 120},
}


def job_args(s: dict, cfg: dict, ea_dir: Path, cand: str, out: Path) -> dict:
    from eu4unitgen.ref3d import ed2d
    ea_args = json.loads((ea_dir / "args.json").read_text(encoding="utf-8"))
    rec = json.loads((ea_dir / "result.json").read_text(encoding="utf-8"))["candidates"][cand]
    views = {}
    for vn, v in ed2d.build_args(s, out)["views"].items():
        W, H = Image.open(v["image"]).size
        views[vn] = {"image": v["image"], "labels": v["labels"], "inpainted": v["inpainted"], "size": [W, H],
                     "camera": rec["cameras"][vn], "body_length": v["body_length"],
                     "hem_landmarks": {k: p for k, p in v["landmarks"].items() if k in cfg["hem_landmarks"].get(vn, [])}}
    return {"skeleton": ea_args["skeleton"], "body": {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")},
            "classes": ea_args["classes"], "views": views, "panel_classes": cfg["panel_classes"],
            "panel": cfg["panel"], "cloth": cfg["cloth"], "armour_m": cfg["armour_m"], "withers_m": cfg["withers_m"],
            "penetration_limit_m": cfg["penetration_limit_m"], "texture_px": cfg["texture_px"],
            "samples": cfg["samples"], "out_dir": str(out)}


def hem_worst(r: dict, skip=()) -> float:
    """The worst hem landmark (0 when none is observed: the far panel)."""
    return max([x["error"] for vn, v in r["hem"].items() if vn not in skip for x in v.values()], default=0.0)


def correct_drops(panel: dict, r: dict, a: dict, skip=()) -> list[str]:
    """SPEC M6 for the hem: each hem landmark's vertical miss (px) becomes a change of the drop
    (px / the view's scale at the target, in metres), spread over the control columns near that
    landmark's column (Gaussian, a quarter of the panel wide); the views' changes are averaged."""
    cols = panel["cols"]
    ctrl = [float(d) for d in panel["drop_m"]]
    n = len(ctrl)
    pos = [(cols - 1) * k / (n - 1) for k in range(n)]
    num, den = [0.0] * n, [0.0] * n
    mpu = r["metres_per_unit"]
    for vn, hem in r["hem"].items():
        if vn in skip:
            continue
        cam = a["views"][vn]["camera"]
        px_per_unit = cam["focal"] / cam["dist"]
        for x in hem.values():
            for k in range(n):
                w = math.exp(-(((pos[k] - x["column"]) / (cols / 4)) ** 2))
                num[k] += w * x["dv_px"] / px_per_unit * mpu
                den[k] += w
    step = [0.5 * num[k] / den[k] if den[k] > 0.2 else 0.0 for k in range(n)]  # damped: E-C's hem oscillated
    panel["drop_m"] = [max(0.2, d + s_) for d, s_ in zip(ctrl, step)]
    return [f"drops + [{', '.join(f'{x * 100:+.1f}' for x in step)}] cm"]


def gates(r: dict, cfg: dict, g: dict, reruns: dict | None, seam: dict | None) -> dict:
    lc = tuple(cfg.get("low_confidence", ()))
    out = {"hem_landmarks": {"pass": hem_worst(r, lc) <= g["landmark_max"], "value": hem_worst(r, lc), "limit": g["landmark_max"],
                             "all_views": hem_worst(r), "low_confidence": list(lc)},
           "intersections": {"pass": r["intersections"]["over_limit"] == 0, "value": r["intersections"]["over_limit"],
                             "max_mm": r["intersections"]["max_mm"]},
           "stretch": {"pass": r["stretch"]["p95"] <= cfg["stretch_max"], "value": r["stretch"]["p95"],
                       "p99": r["stretch"]["p99"], "max": r["stretch"]["max"], "limit": cfg["stretch_max"]}}
    if reruns is not None:
        out["reruns_identical"] = {"pass": reruns["max_diff"] <= cfg["rerun_tol"], "value": reruns["max_diff"],
                                   "limit": cfg["rerun_tol"], "runs": reruns["runs"]}
    if seam is not None:
        out["texture_seam"] = {"pass": seam["p95"] is not None and seam["p95"] <= cfg["seam_p95_max"],
                               "value": seam["p95"], "mean": seam["mean"], "limit": cfg["seam_p95_max"]}
    return out


def run(subject_path: Path, ea_version: str, ai_python: str, blender: str | None = None,
        candidate: str | None = None, out: Path | None = None, over: dict | None = None,
        extra: dict | None = None) -> dict:
    """E-C (or, from the master build: ``out`` the panel's folder, ``over`` settings, ``extra`` job args;
    with ``rounds`` 0 the pattern is built as given: the far panel, from the near one's)."""
    from eu4unitgen.ai.env import AIJobRunner
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.project import default_blender_path
    from eu4unitgen.ref3d.eb import best_candidate

    s = s1.load_subject(subject_path)
    root = s["_root"] / "spikes"
    check_writes({"out_dir": str(root / "ec")}, allowed_roots(ai_python), "E-C")
    ea_dir = root / "ea" / "fit" / ea_version
    cand = candidate or best_candidate(json.loads((ea_dir / "manifest.json").read_text(encoding="utf-8")))
    cfg = json.loads(json.dumps(EC_DEFAULTS))
    for src in (s.get("ec") or {}, over or {}):  # the subject's settings, then the caller's (the master)
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    g = json.loads((ea_dir / "args.json").read_text(encoding="utf-8"))["gates"]
    if out is None:
        out = s1.next_version(root / "ec")
    else:
        check_writes({"out_dir": str(out)}, allowed_roots(ai_python), "E-C")
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    br = BlenderRunner(blender or default_blender_path(), log_dir=out, timeout=3600)
    rounds = []
    for k in range(1, cfg["rounds"] + 1):
        d = out / f"round{k:02d}"
        d.mkdir()
        a = job_args(s, cfg, ea_dir, cand, d)
        a.update(extra or {})
        a["finish"] = False
        r = br.run_job("ec_panel", a).result
        used = json.loads(json.dumps(cfg["panel"]))
        lc = tuple(cfg.get("low_confidence", ()))
        did = [] if hem_worst(r, lc) <= cfg["hem_target"] else correct_drops(cfg["panel"], r, a, lc)
        rounds.append({"round": k, "panel": used, "result": r, "corrections": did})
        print(f"E-C round {k}: hem worst {hem_worst(r, lc) * 100:.2f}% (all views {hem_worst(r) * 100:.2f}%), "
              f"intersections {r['intersections']['over_limit']}, stretch p99 {r['stretch']['p99']:.3f}" + (f"; next: {'; '.join(did)}" if did else ""), flush=True)
        if not did:
            break
    if rounds and rounds[-1]["corrections"]:  # ran out of rounds: build the best pattern simulated
        lc = tuple(cfg.get("low_confidence", ()))
        cfg["panel"] = min(rounds, key=lambda q: hem_worst(q["result"], lc))["panel"]
    d = out / "final"
    d.mkdir()
    a = job_args(s, cfg, ea_dir, cand, d)
    a.update(extra or {})
    r = br.run_job("ec_panel", a).result
    base = json.loads((d / "panel_vertices.json").read_text(encoding="utf-8"))
    diffs = []
    for k in range(1, cfg["reruns"]):
        dd = out / f"rerun{k}"
        dd.mkdir()
        ar = job_args(s, cfg, ea_dir, cand, dd)
        ar.update(extra or {})
        ar["finish"] = False
        br.run_job("ec_panel", ar)
        other = json.loads((dd / "panel_vertices.json").read_text(encoding="utf-8"))
        diffs.append(max(abs(p - q) for u, v in zip(other, base) for p, q in zip(u, v)))
    reruns = {"runs": cfg["reruns"], "max_diff": max(diffs) if diffs else 0.0, "diffs": diffs}
    seam = AIJobRunner(ai_python, timeout=1800).run_job(
        "ref3d_ec_measure", {"texture": r["texture"]["path"], "evidence": r["texture"]["evidence"], "radius": 6}).result
    gt = gates(r, cfg, g, reruns, seam)
    verdict = {"pass": all(x["pass"] for x in gt.values()), "failed": [n for n, x in gt.items() if not x["pass"]]}
    sheet_path = out / "ec_sheet.png"
    sheet(s, a, r, gt, seam, sheet_path, out.name, cand, ea_version)
    manifest = {"stage": "E-C cloth panel", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "ea": {"version": ea_version, "candidate": cand}, "settings": cfg, "rounds": rounds,
                "final": {"result": r, "reruns": reruns, "seam": seam, "gates": gt}, "verdict": verdict,
                "sheet": str(sheet_path), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def _hem_overlay(view: dict, r: dict, vn: str, path_img: str) -> Image.Image:
    from PIL import ImageDraw
    im = Image.open(path_img).convert("RGB")
    dr = ImageDraw.Draw(im)
    for name, xy in view["hem_landmarks"].items():
        x, y = xy
        dr.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 0), width=2)
        e = r["hem"][vn][name]["error"]
        dr.text((x + 7, y - 7), f"{name} {e * 100:.1f}%", fill=(255, 255, 0))
    return im


def sheet(s, a, r, gt, seam, path, version, cand, ea_version):
    f = r["files"]
    rows = []
    for vn, v in a["views"].items():
        tiles = [(f"reference {vn} (cleaned); hem landmarks", _hem_overlay(v, r, vn, v["image"]), False),
                 (f"panel in {vn}'s solved camera (Cycles)", _hem_overlay(v, r, vn, f[f"{vn}_cycles"]), False)]
        rows.append((f"view {vn}", tiles))
    txt = [f"E-A {ea_version} candidate: {cand}",
           f"panel {r['panel']['width_m']:.2f} m wide, drops " + ", ".join(f"{d:.2f}" for d in r["panel"]["drops_m"][::8]) + " m",
           f"{r['panel']['verts']} verts, {r['panel']['tris']} tris"]
    for n, x in gt.items():
        txt.append(f"{n}: {'PASS' if x['pass'] else 'FAIL'} {x['value']:.4g}" if isinstance(x["value"], float)
                   else f"{n}: {'PASS' if x['pass'] else 'FAIL'} {x['value']}")
    for vn, h in r["hem"].items():
        txt.append(f"{vn} hem: " + ", ".join(f"{k[4:]} {x['error'] * 100:.1f}%" for k, x in h.items()))
    txt.append(f"texture: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in r["texture"]["share"].items()))
    txt.append(f"seam dE2000 mean {seam['mean']:.1f}, p95 {seam['p95']:.1f}" if seam.get("mean") is not None else "seam: no border")
    txt.append(f"stretch p95 {r['stretch']['p95']:.3f}, p99 {r['stretch']['p99']:.3f}, max {r['stretch']['max']:.3f}")
    tex = Image.open(r["texture"]["path"]).convert("RGB")
    evi = Image.open(r["texture"]["evidence"]).convert("RGB")
    rows.append(("panel", [("close-up (Cycles)", Image.open(f["closeup"]).convert("RGB"), False),
                           ("pattern texture", tex, False),
                           ("evidence: green L, blue R, red prior", evi, False),
                           ("", s1.text_tile("\n".join(s1._wrap(t, 44) for t in txt), (420, 460)), False)]))
    return s1.grid_sheet(rows, f"E-C caparison side panel (cloth): {s['name']} {version}", path, tile=(420, 460))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("subject", type=Path)
    a.add_argument("--ea", required=True)
    a.add_argument("--candidate")
    a.add_argument("--ai-python", required=True)
    a.add_argument("--blender")
    args = p.parse_args(argv)
    m = run(args.subject, args.ea, args.ai_python, args.blender, args.candidate)
    print(json.dumps({"version": m["version"], "verdict": m["verdict"], "seconds": m["seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
