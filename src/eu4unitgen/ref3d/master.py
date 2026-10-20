"""Track M master build (SPEC M4 S2-S8, M10): the high-detail master of the subject, from E-A's body and
the generators E-B and E-C proved, evaluated with the M5 suite in both reference cameras, up to H3.

Steps (``<subject>/master/vNNN``, never overwritten; M10 requirement 7: the master stays separate):
  armour       E-B's generator over every scale area L or R shows (the neck, chest and barrel; R votes
               on the barrel only: the body stands in L's pose), its calibration loop and gates.
  cloth_left   E-C's near-side panel (L, R's hem as lower confidence).
  cloth_right  the far-side panel: the near panel's seam and pattern mirrored, its texture (symmetry).
  assemble     body, armour, panels, tail, plume, tassels, projected decals; renders in both cameras.
  breaks       E-D in 3D: the same five breaks made on the master and rendered through the cameras.
  evaluate     the M5 suite on the master and on every break (AI environment); then the H3 sheet: the
               master next to the reference at the same angle, E-B's plate close-up, and the plain list
               of matches, mismatches and guesses.

usage: python -m eu4unitgen.ref3d.master build <subject.json> --ea vNNN --ai-python <env python>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes

BREAKS = ["trim_removed", "caparison_short", "plates_bigger", "head_5deg", "wrong_red"]
DECAL_CLASS = {"eye": "chanfron", "bit_rosette": "trim", "breast_medallion": "trim", "crupper_rosette": "trim",
               "tail_tassel": "trim"}  # details on the body; the crosses, fringe and hem band are in the panels' textures
HEAD_LM = ["poll", "eye", "muzzle_front", "nostril", "bit_ring", "chin", "throat", "ear_tip_a", "ear_tip_b"]
MASTER_DEFAULTS = {"withers_m": 1.55, "tail_drop": 0.78, "tassel_step_m": 0.09, "tassel_len_m": 0.08, "samples": 128}


def view_info(s: dict, ea_args: dict, rec: dict, out: Path) -> dict:
    from eu4unitgen.ref3d import ed2d
    base = ed2d.build_args(s, out)["views"]
    views = {}
    for vn, v in base.items():
        W, H = Image.open(v["image"]).size
        views[vn] = {"image": v["image"], "labels": v["labels"], "fg": v["fg"], "inpainted": v["inpainted"],
                     "never_use": v["never_use"], "landmarks": v["landmarks"], "body_length": v["body_length"],
                     "details": v["details"], "size": [W, H], "camera": rec["cameras"][vn]}
    return views


def build(subject_path: Path, ea_version: str, ai_python: str, blender: str | None = None,
          candidate: str | None = None) -> dict:
    from eu4unitgen.ai.env import AIJobRunner
    from eu4unitgen.blender.runner import BlenderRunner
    from eu4unitgen.core.project import default_blender_path
    from eu4unitgen.ref3d import eb, ec

    s = s1.load_subject(subject_path)
    root = s["_root"]
    check_writes({"out_dir": str(root / "master")}, allowed_roots(ai_python), "master")
    ea_dir = root / "spikes" / "ea" / "fit" / ea_version
    ea_manifest = json.loads((ea_dir / "manifest.json").read_text(encoding="utf-8"))
    ea_args = json.loads((ea_dir / "args.json").read_text(encoding="utf-8"))
    cand = candidate or eb.best_candidate(ea_manifest)
    rec = json.loads((ea_dir / "result.json").read_text(encoding="utf-8"))["candidates"][cand]
    cfg = {**MASTER_DEFAULTS, **(s.get("master") or {})}
    out = s1.next_version(root / "master")
    t0 = time.time()
    log = []

    def note(msg):
        log.append(msg)
        print(msg, flush=True)

    views = view_info(s, ea_args, rec, out)
    tol = {vn: max(1, round(ea_args["label_scale"] * v["body_length"])) for vn, v in views.items()}
    # ---- armour: every scale area L or R shows
    cover = [{**{k: views["L"][k] for k in ("labels", "inpainted", "never_use", "camera")}, "parts": None},
             {**{k: views["R"][k] for k in ("labels", "inpainted", "never_use", "camera")}, "parts": [0],
              "label_tol_px": tol["R"], "close_px": max(2, round(0.02 * views["R"]["body_length"]))}]
    m_arm = eb.run(subject_path, ea_version, ai_python, blender, cand, out=out / "armour",
                   extra={"region": "all", "cover_views": cover})
    note(f"armour: {m_arm['verdict']}")
    arm_res = m_arm["final"]["blender"]
    # ---- cloth: the near panel, then the far one by symmetry
    m_cl = ec.run(subject_path, ea_version, ai_python, blender, cand, out=out / "cloth_left")
    note(f"cloth left: {m_cl['verdict']}")
    left_panel = m_cl["settings"]["panel"]
    m_cr = ec.run(subject_path, ea_version, ai_python, blender, cand, out=out / "cloth_right",
                  over={"rounds": 0, "reruns": 1, "panel": {**left_panel, "side": "right"}, "hem_landmarks": {"L": [], "R": []}},
                  extra={"texture_from": m_cl["final"]["result"]["texture"]["path"]})
    note("cloth right: built from the left panel (symmetry)")
    # the chest panel isn't built: the near foreleg steps through it and a one-piece cloth tangles round
    # the forelegs (master dev run, E-C front mode); it needs a pattern split at the legs (listed at H3)
    panels = []
    for name, m, ev in (("panel_left", m_cl, "observed L / R, prior"), ("panel_right", m_cr, "symmetry")):
        r = m["final"]["result"]
        d = Path(r["texture"]["path"]).parent
        panels.append({"name": name, "vertices": str(d / "panel_vertices.json"), "texture": r["texture"]["path"],
                       "rows": r["panel"]["rows"], "cols": r["panel"]["cols"], "evidence": ev})
    decals = [{"name": d["name"], "view": vn, "xy": d["xy"], "size": d["size"], "class": DECAL_CLASS[d["name"]]}
              for vn, v in views.items() for d in v["details"] if d["name"] in DECAL_CLASS]
    base_args = {"skeleton": ea_args["skeleton"], "body": {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")},
                 "classes": ea_args["classes"], "views": views, "armour_dir": str(Path(arm_res["files"]["lit"]).parent),
                 "verts_per_plate": arm_res["plates"]["verts_per_plate"], "panels": panels, "decals": decals,
                 "cloth_seam_y": min(m_cl["final"]["result"]["seam"]["y"]), **cfg}
    br = BlenderRunner(blender or default_blender_path(), log_dir=out / "logs", timeout=7200)
    sets = {}
    asm = br.run_job("master_assemble", {**base_args, "out_dir": str(out / "assemble")}).result
    sets["master"] = {"files": asm["files"], "landmarks": asm["landmarks"]}
    note(f"assembled in {asm['seconds']} s ({asm['cycles_device']})")
    for b in BREAKS:
        r = br.run_job("master_assemble", {**base_args, "break": b, "out_dir": str(out / "breaks" / b)}).result
        sets[b] = {"files": r["files"], "landmarks": r["landmarks"]}
        note(f"break {b}: rendered")
    template = s["_template"]
    gates = dict(ea_args["gates"])
    ev = AIJobRunner(ai_python, timeout=3600).run_job("ref3d_master_eval", {
        "classes": ea_args["classes"], "regions": template["regions"], "gates": gates, "views": views, "sets": sets,
        "low_confidence": ea_args.get("low_confidence") or [], "head_landmarks": HEAD_LM}).result
    master_ev = ev["sets"]["master"]
    e_d = {b: {"caught": bool(ev["sets"][b]["failed_any_view"]), "failed": ev["sets"][b]["failed_any_view"]}
           for b in BREAKS}
    verdict = {"master_gates_pass": master_ev["pass"],
               "failed": {vn: r["failed"] for vn, r in master_ev["views"].items()},
               "ed3d_every_break_caught": all(x["caught"] for x in e_d.values()), "ed3d": e_d,
               "armour": m_arm["verdict"], "cloth_left": m_cl["verdict"]}
    sheet_path = out / "h3_sheet.png"
    h3_sheet(s, views, asm, master_ev, verdict, m_arm, m_cl, sheet_path, out.name, ea_version, cand)
    manifest = {"stage": "master build", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "ea": {"version": ea_version, "candidate": cand}, "settings": cfg, "assemble": asm,
                "evaluation": ev, "verdict": verdict, "log": log, "sheet": str(sheet_path),
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def _fit(im: Image.Image, size) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail(size)
    return im


def h3_sheet(s, views, asm, mev, verdict, m_arm, m_cl, path, version, ea_version, cand):
    """H3 (SPEC M7): the master next to the reference at the same angle in both cameras, the plate
    close-ups (M10 requirement 8), the unseen side, and the plain lists."""
    f = asm["files"]
    rows = []
    for vn, v in views.items():
        r = mev["views"][vn]
        g = r["gates"]
        txt = [f"view {vn}" + (" (lower confidence: its camera drifted)" if r["low_confidence"] else ""),
               *(f"{k}: not measured" if x.get("not_measured") else
                 f"{k}: {'PASS' if x['pass'] else 'FAIL'} {x['value']:.3g} ({x.get('worst')})" if isinstance(x.get("value"), (int, float))
                 else f"{k}: {'PASS' if x['pass'] else 'FAIL'}" for k, x in g.items())]
        lm = g["landmarks"]
        if lm.get("head_value") is not None:
            txt.append(f"  head landmarks {lm['head_value'] * 100:.1f}% (allowance {lm['head_limit'] * 100:.0f}%)")
        rows.append((f"view {vn}", [(f"reference {vn}", Image.open(v["image"]).convert("RGB"), False),
                                    (f"master in {vn}'s solved camera", Image.open(f[f"{vn}_lit"]).convert("RGB"), False),
                                    (f"master classes in {vn}", Image.open(f[f"{vn}_id"]).convert("RGB"), False),
                                    ("", s1.text_tile("\n".join(s1._wrap(t, 44) for t in txt), (420, 460)), False)]))
    arm_detail = m_arm["final"]["blender"]["files"].get("closeup_detail")
    tiles = [("plates: neck and chest (master)", Image.open(f["closeup_plates"]), False)]
    if arm_detail:
        tiles.append(("plates: overlap, curvature, thickness", Image.open(arm_detail), False))
    tiles += [("unseen right side (symmetry and prior)", Image.open(f["unseen_right"]), False),
              ("front (the chest panel isn't built)", Image.open(f["front"]), False)]
    rows.append(("close-ups", tiles))
    return s1.grid_sheet(rows, f"H3: master {version} next to the reference ({s['name']}; body E-A {ea_version} {cand})",
                         path, tile=(420, 460))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build")
    a.add_argument("subject", type=Path)
    a.add_argument("--ea", required=True)
    a.add_argument("--candidate")
    a.add_argument("--ai-python", required=True)
    a.add_argument("--blender")
    args = p.parse_args(argv)
    m = build(args.subject, args.ea, args.ai_python, args.blender, args.candidate)
    print(json.dumps({"version": m["version"], "verdict": m["verdict"], "seconds": m["seconds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
