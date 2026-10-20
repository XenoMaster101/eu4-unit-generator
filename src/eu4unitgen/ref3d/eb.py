"""Track M spike E-B (SPEC M9, M4 S4): scale plates on the fitted neck and chest. Do they conform to
the surface (no penetrations, no gaps), match the reference's plate period in view L, cover what the
labels say, and look right up close (M10 requirement 8: a close-up of overlap, curvature, thickness)?

Steps (a new version folder under ``<subject>/spikes/eb/``; nothing overwritten):
  1. The body: E-A's best fitted candidate (``--ea vNNN``), in view L's pose and solved camera.
  2. Plates (Blender job eb_plates): coverage from L's labels, rows round the neck and breast,
     penetration and gap checks, renders in L's camera; then the AI-environment job ref3d_eb_measure:
     coverage IoU and the plate period against the cleaned reference.
  3. Corrections (SPEC M6), up to ``rounds``: plate pitch and column spacing scaled by the measured
     period ratios (y: row pitch, x: plate width); tilt raised while plates cut into each other;
     standoff raised while plates cut into the body.
  4. The final round renders the close-ups (Cycles) and the evidence close-up; the proof sheet and
     manifest with the verdict.

usage: python -m eu4unitgen.ref3d.eb run <subject.json> --ea vNNN --ai-python <env python>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes

EB_DEFAULTS = {
    "view": "L",
    "rounds": 6,
    "withers_m": 1.55,  # assumed master scale: a 15-hand horse at the withers (the reference gives no size)
    "penetration_limit_m": 0.001,  # SPEC M5: zero penetrations over 1 mm
    "body_visible_max": 0.01,  # assumed: at most 1% of the covered surface shows the body between plates
    "period_target": 0.05,  # calibrate the plate size to within 5% (the gate allows 15%)
    "mirror_tol": 0.3,  # units: a triangle's mirror image must lie this close to a known triangle
    "visibility_tol": 0.05,  # units (~6 mm): an occluder nearer than this to a sample doesn't hide it
    "occluders": ["leather", "trim", "plume"],  # straps, trim and plume lie over the armour: unknown under them
    "samples": 96,
    "plate": {
        "pitch_m": 0.05, "spacing_m": 0.035,  # first guesses; calibrated against L's period
        "width_frac": 0.9,  # of the column spacing: dark slits between columns, as in the reference
        "height_frac": 1.35,  # of the row pitch: each row overlaps the next by a third of its pitch
        "thickness_m": 0.002, "standoff_m": 0.004, "backing_m": 0.002, "clearance_m": 0.001,
        "tilt_clear": 1.5, "tilt_extra_deg": 0.0, "curve_radius_m": 0.25, "stagger": 0.0,
    },
}


def best_candidate(ea_manifest: dict) -> str:
    v = ea_manifest.get("verdict") or {}
    return v.get("best") or "procedural"


def job_args(s: dict, cfg: dict, ea_dir: Path, cand: str, out: Path) -> tuple[dict, dict]:
    """(Blender job args, measure job args) for one round."""
    from eu4unitgen.ref3d import ed2d
    ea_args = json.loads((ea_dir / "args.json").read_text(encoding="utf-8"))
    res = json.loads((ea_dir / "result.json").read_text(encoding="utf-8"))
    rec = res["candidates"][cand]
    vn = cfg["view"]
    v = ed2d.build_args(s, out)["views"][vn]
    W, H = Image.open(v["image"]).size
    tol = max(1, round(ea_args["label_scale"] * v["body_length"]))
    body = {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")}
    b = {"skeleton": ea_args["skeleton"], "body": body, "view_name": vn, "classes": ea_args["classes"],
         "view": {"labels": v["labels"], "inpainted": v["inpainted"], "size": [W, H]},
         "occluders": cfg["occluders"], "plate": cfg["plate"], "withers_m": cfg["withers_m"],
         "penetration_limit_m": cfg["penetration_limit_m"], "label_tol_px": tol, "mirror_tol": cfg["mirror_tol"], "visibility_tol": cfg["visibility_tol"],
         "samples": cfg["samples"], "out_dir": str(out), "debug_dir": str(out) if cfg.get("debug") else None}
    m = {"image": v["image"], "labels": v["labels"], "fg": v["fg"], "inpainted": v["inpainted"],
         "classes": ea_args["classes"], "occluders": cfg["occluders"], "lit": str(out / f"{vn}_lit.png"),
         "id": str(out / f"{vn}_id.png"), "label_scale": ea_args["label_scale"], "body_length": v["body_length"]}
    return b, m


def gates(r: dict, m: dict, cfg: dict, g: dict) -> dict:
    pen = r["penetrations"]
    per = (m.get("plate_period") or {}).get("ratio")
    cov = m.get("coverage_iou")
    vis = r["gaps"]["body_visible"]
    return {
        "penetrations": {"pass": pen["plate_plate_pairs_over"] == 0 and pen["plate_body_over"] == 0,
                         "value": {"plate_plate": pen["plate_plate_pairs_over"], "plate_body": pen["plate_body_over"],
                                   "max_mm": max(pen["plate_plate_max_mm"], pen["plate_body_max_mm"])}},
        "gaps": {"pass": vis is not None and vis <= cfg["body_visible_max"], "value": vis, "limit": cfg["body_visible_max"]},
        "plate_period": {"pass": per is not None and abs(per - 1) <= g["plate_period_tol"], "value": per,
                         "limit": g["plate_period_tol"]},
        "coverage_iou": {"pass": cov is not None and cov >= g["coverage_iou_min"], "value": cov,
                         "limit": g["coverage_iou_min"]},
    }


def correct(plate: dict, r: dict, m: dict, cfg: dict) -> list[str]:
    """SPEC M6 for plates: the size by the period ratios, tilt for plate-plate cuts, standoff for
    plate-body cuts. Mutates ``plate``; returns what changed."""
    did = []
    ax = m.get("axes") or {}
    for axis, key in (("y", "pitch_m"), ("x", "spacing_m")):
        q = ax.get(axis)
        if q and abs(q["ratio"] - 1) > cfg["period_target"]:
            plate[key] = plate[key] / q["ratio"]
            did.append(f"{key} / {q['ratio']:.3f}")
    # crossing plates and plates in the body are fixed per plate in the job (local tilt, then
    # removal); globally, more tilt and standoff only while the job had to remove over 5% of them
    rm = r.get("removed") or {}
    n = max(rm.get("of", 1), 1)
    if rm.get("crossing", 0) / n > 0.05 and plate["tilt_clear"] < 4:
        plate["tilt_clear"] *= 1.5
        did.append(f"tilt_clear x1.5 ({rm['crossing']} plates removed for crossing)")
    if rm.get("in_body", 0) / n > 0.05 and plate["standoff_m"] < 0.012:  # padding over 12 mm isn't armour any more
        plate["standoff_m"] *= 1.5
        did.append(f"standoff x1.5 ({rm['in_body']} plates removed: in the body)")
    return did


def run(subject_path: Path, ea_version: str, ai_python: str, blender: str | None = None,
        candidate: str | None = None) -> dict:
    from eu4unitgen.ai.env import AIJobRunner
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.project import default_blender_path

    s = s1.load_subject(subject_path)
    root = s["_root"] / "spikes"
    check_writes({"out_dir": str(root / "eb")}, allowed_roots(ai_python), "E-B")
    ea_dir = root / "ea" / "fit" / ea_version
    ea_manifest = json.loads((ea_dir / "manifest.json").read_text(encoding="utf-8"))
    cand = candidate or best_candidate(ea_manifest)
    cfg = json.loads(json.dumps(EB_DEFAULTS))
    for k, v in (s.get("eb") or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    out = s1.next_version(root / "eb")
    t0 = time.time()
    br = BlenderRunner(blender or default_blender_path(), log_dir=out, timeout=3600)
    ai = AIJobRunner(ai_python, timeout=1800)
    g = json.loads((ea_dir / "args.json").read_text(encoding="utf-8"))["gates"]
    rounds = []
    final = None
    for k in range(1, cfg["rounds"] + 1):
        last = k == cfg["rounds"]
        d = out / f"round{k:02d}"
        d.mkdir()
        b, m = job_args(s, cfg, ea_dir, cand, d)
        b["closeups"] = False
        r = br.run_job("eb_plates", b).result
        mr = ai.run_job("ref3d_eb_measure", m).result
        gt = gates(r, mr, cfg, g)
        plate_used = dict(cfg["plate"])
        did = correct(cfg["plate"], r, mr, cfg)
        rounds.append({"round": k, "plate": plate_used, "blender": r, "measure": mr, "gates": gt, "corrections": did})
        print(f"E-B round {k}: " + json.dumps({n: x["value"] for n, x in gt.items()}, default=float)
              + (f"; next: {'; '.join(did)}" if did else ""), flush=True)
        if not did or last:
            cfg["plate"] = plate_used if not did else cfg["plate"]
            break
    # the final build at the settled settings, with the close-ups
    d = out / "final"
    d.mkdir()
    b, m = job_args(s, cfg, ea_dir, cand, d)
    b["closeups"] = True
    r = br.run_job("eb_plates", b).result
    mr = ai.run_job("ref3d_eb_measure", m).result
    gt = gates(r, mr, cfg, g)
    final = {"plate": dict(cfg["plate"]), "blender": r, "measure": mr, "gates": gt}
    verdict = {"pass": all(x["pass"] for x in gt.values()), "failed": [n for n, x in gt.items() if not x["pass"]],
               "rounds": len(rounds)}
    sheet_path = out / "eb_sheet.png"
    sheet(s, final, m, sheet_path, out.name, cand, ea_version)
    manifest = {"stage": "E-B plates", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "ea": {"version": ea_version, "candidate": cand}, "settings": cfg, "rounds": rounds, "final": final,
                "verdict": verdict, "sheet": str(sheet_path), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def _crop_box(id_path: Path, margin: int = 24):
    im = Image.open(id_path).convert("RGB")
    red = im.point(lambda v: 255 if v > 127 else 0).split()[0]
    g = im.split()[1].point(lambda v: 255 if v > 127 else 0)
    box = Image.composite(red, Image.new("L", im.size, 0), g.point(lambda v: 255 - v)).getbbox()
    if not box:
        return (0, 0, *im.size)
    x0, y0, x1, y1 = box
    return (max(0, x0 - margin), max(0, y0 - margin), min(im.width, x1 + margin), min(im.height, y1 + margin))


def sheet(s: dict, final: dict, m: dict, path: Path, version: str, cand: str, ea_version: str):
    """The reference's neck and chest beside the plates in the same camera, the coverage overlay, and
    the close-ups (M10 requirement 8), with the numbers."""
    r, mr, gt = final["blender"], final["measure"], final["gates"]
    f = r["files"]
    box = _crop_box(Path(m["id"]))
    ref = Image.open(m["image"]).convert("RGB").crop(box)
    tiles1 = [("reference L (cleaned), neck and chest", ref, False),
              ("plates in L's solved camera (Cycles)", Image.open(f["cycles_view"]).convert("RGB").crop(box), False),
              ("coverage: ref scale green, plates red, both yellow", Image.open(mr["overlay"]).convert("RGB").crop(box), False)]
    p = r["plates"]
    pen = r["penetrations"]
    per = mr.get("plate_period") or {}
    ax = mr.get("axes") or {}
    mpu = r["metres_per_unit"]
    txt = [f"E-A {ea_version} candidate: {cand}",
           f"{p['count']} plates, {p['rows']} rows; {p['tris']:,} tris",
           f"plate {p['width_units'] * mpu * 100:.1f} x {p['height_units'] * mpu * 100:.1f} cm, "
           f"{p['thickness_units'] * mpu * 1000:.1f} mm thick; pitch {p['pitch_units'] * mpu * 100:.1f} cm; "
           f"tilt {p['tilt_deg']:.1f} deg; standoff {p['standoff_units'] * mpu * 1000:.1f} mm",
           f"evidence: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in r["evidence_share"].items() if v)]
    for n, x in gt.items():
        txt.append(f"{n}: {'PASS' if x['pass'] else 'FAIL'} {json.dumps(x['value'], default=lambda q: round(q, 3)) if not isinstance(x['value'], float) else round(x['value'], 3)}")
    if per.get("ref_px"):
        txt.append(f"period ({per['axis']}): ref {per['ref_px']:.1f} px, plates {per['cand_px']:.1f} px")
    for a_ in ("x", "y"):
        if ax.get(a_):
            txt.append(f"  {a_}: ref {ax[a_]['ref_px']:.1f} px, plates {ax[a_]['cand_px']:.1f} px")
    txt.append(f"plate-plate max {pen['plate_plate_max_mm']:.2f} mm; plate-body max {pen['plate_body_max_mm']:.2f} mm")
    txt.append(f"not under a plate (slits, backing shows): {r['gaps']['not_under_a_plate'] * 100:.1f}%")
    tiles1.append(("", s1.text_tile("\n".join(s1._wrap(t, 44) for t in txt), (420, 460)), False))
    tiles2 = [("close-up: neck and chest (Cycles)", Image.open(f["closeup"]).convert("RGB"), False),
              ("detail: overlap, curvature, thickness", Image.open(f["closeup_detail"]).convert("RGB"), False),
              ("evidence: green observed, amber inferred,\nblue symmetry, red prior",
               Image.open(f["closeup_evidence"]).convert("RGB"), False)]
    return s1.grid_sheet([("view L", tiles1), ("close-ups", tiles2)],
                         f"E-B scale plates on the fitted neck and chest: {s['name']} {version}", path, tile=(420, 460))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("subject", type=Path)
    a.add_argument("--ea", required=True, help="E-A fit version (vNNN) whose best body carries the plates")
    a.add_argument("--candidate", help="E-A candidate (default: the E-A verdict's best)")
    a.add_argument("--ai-python", required=True)
    a.add_argument("--blender")
    args = p.parse_args(argv)
    m = run(args.subject, args.ea, args.ai_python, args.blender, args.candidate)
    print(json.dumps({"version": m["version"], "verdict": m["verdict"], "seconds": m["seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
