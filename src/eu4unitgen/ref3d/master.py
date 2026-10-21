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

Only what the labels show is built (M18): each view's ``<view>_parts.json`` (S1 parts, ref3d/parts.py), a
part present when any view shows it. No armour: no E-B run and no plates. No caparison: no E-C run, no
panels, no chest panel, no tassels. A saddle cloth: the cloth shell. No chanfron: the head is coat. No
plume: none. Labels without parts files (hand colour rules) build everything, as before.

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
# ornament comes from the original's views (the user, round 13: S confirms and fills gaps, the original has priority)
DECAL_VIEWS = ("L", "R")
HEAD_LM = ["poll", "eye", "muzzle_front", "nostril", "bit_ring", "chin", "throat", "ear_tip_a", "ear_tip_b"]
MASTER_DEFAULTS = {"withers_m": 1.55, "tail_drop": 0.78, "tassel_step_m": 0.09, "tassel_len_m": 0.08, "samples": 128,
                   # round 14 (the user at H3: "bright silver lamellae"): metals coloured by their brightest quarter
                   # (the painting's median scale pixel is a dark brown), fully metallic, a dark grey backing
                   "metal_quantile": 0.75, "plate_material": {"metallic": 1.0, "roughness": 0.3, "backing": [0.05, 0.05, 0.055]},
                   # the neck and breast's plates at 0.6 of the barrel's (S: neck rows ~14 px apart, rump ~23 px)
                   "front_scale": 0.6,
                   # the chest panel split at the legs (round 14): static drapes from the lower breast (top_frac of
                   # the breast's depth up from its bottom) to below_knee x H under the knees, round to side_reach x H
                   # behind the shoulder; L's image projected on them (my calls: S's front skirt hangs from ~0.3 up
                   # the breast to just under the knees)
                   "chest_panel": {"top_frac": 0.3, "below_knee": 0.04, "side_reach": 0.25, "gap_deg": 4.0,
                                   "clear_m": 0.03, "flare": 0.03, "rows": 16, "cols": 20}}


PARTS = ("armour", "chanfron", "caparison", "saddle_cloth", "saddle", "bridle", "plume", "bare_coat", "tail")


def subject_parts(s: dict, views) -> dict:
    """{part: present} merged over the views' parts files (present in any view); None when no view has one."""
    found = {}
    for vn in views:
        p = s1.view_dirs(s, vn)[1] / f"{vn}_parts.json"
        if p.is_file():
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                found[k] = found.get(k, False) or bool(v["present"])
    return found or None


def parts_box(s: dict, view: str, part: str):
    """A part's box (view crop px) in that view's parts file, or None."""
    p = s1.view_dirs(s, view)[1] / f"{view}_parts.json"
    if not p.is_file():
        return None
    q = json.loads(p.read_text(encoding="utf-8")).get(part) or {}
    return q.get("box") if q.get("present") else None


def seat_rise(bones: dict, k: dict) -> float:
    """How far the seat rises with the leg factors ``k`` (as _ref3d_body.skeleton proportions it)."""
    zf, zb = bones["HorseFrontLeftUp"][2], bones["HorseBackLeftUp"][2]
    w = min(max((bones["Saddle_Node"][2] - zf) / (zb - zf), 0.0), 1.0)
    return ((k.get("leg_f", 1.0) - 1) * bones["HorseFrontLeftUp"][1] * (1 - w)
            + (k.get("leg_b", 1.0) - 1) * bones["HorseBackLeftUp"][1] * w)


def view_info(s: dict, ea_args: dict, rec: dict, out: Path) -> dict:
    """The views with their solved cameras. A view fitted with its own (shorter) legs (round 13: S) sees the
    master's trunk higher by the seat's difference: its camera is moved up by that much, so the trunk, neck
    and head line up and the master's longer legs reach below that view's ground."""
    from eu4unitgen.ref3d import ed2d
    base = ed2d.build_args(s, out)["views"]
    bones = json.loads(Path(ea_args["skeleton"]).read_text(encoding="utf-8"))["bones"]
    views = {}
    for vn, v in base.items():
        W, H = Image.open(v["image"]).size
        cam = dict(rec["cameras"][vn])
        ls = (rec["poses"].get(vn) or {}).get("leg_scale")
        if ls:
            sk = rec["skeleton"]
            d = seat_rise(bones, sk) - seat_rise(bones, {"leg_f": sk.get("leg_f", 1.0) * ls[0], "leg_b": sk.get("leg_b", 1.0) * ls[1]})
            cam["cy"] += d * cam["focal"] / cam["dist"]
        views[vn] = {"image": v["image"], "labels": v["labels"], "fg": v["fg"], "inpainted": v["inpainted"],
                     "never_use": v["never_use"], "landmarks": v["landmarks"], "body_length": v["body_length"],
                     "details": v["details"], "size": [W, H], "camera": cam, "leg_scale": ls}
        ldir = s1.view_dirs(s, vn)[1]
        if (ldir / f"{vn}_texture.png").is_file():  # a self-read view (M18): the filled texture to project
            views[vn].update(texture=str(ldir / f"{vn}_texture.png"), texture_labels=str(ldir / f"{vn}_texture_labels.png"))
    return views


def tail_profile(s: dict, views: dict, view: str = "S", n: int = 12) -> list | None:
    """The tail from a side view (round 13: S is the main evidence for the tail): per station from the
    top of the visible hair to its tip, (drop below the top, centre behind the top, front-back half-width),
    in scene units (the view's px per unit at the target). None without that view or its tail landmarks."""
    v = views.get(view)
    if not v or not all(k in v["landmarks"] for k in ("tail_top_visible", "tail_tip")):
        return None
    tail = tuple(s["_template"]["classes"]["tail"])
    lab = Image.open(v["labels"]).convert("RGB")
    px = lab.load()
    ppu = v["camera"]["focal"] / v["camera"]["dist"]
    (x0, y0), (x1, y1) = v["landmarks"]["tail_top_visible"], v["landmarks"]["tail_tip"]
    out = []
    for k in range(n):
        y = int(round(y0 + (y1 - y0) * (k + 0.5) / n))
        xs = [x for x in range(max(0, int(x0) - 250), min(lab.size[0], int(x1) + 250)) if px[x, y] == tail]
        if len(xs) < 3:
            continue
        lo, hi = min(xs), max(xs)
        out.append([(y - y0) / ppu, (0.5 * (lo + hi) - x0) / ppu, 0.5 * (hi - lo) / ppu])
    return out or None


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
    found = subject_parts(s, views)
    has = (lambda k: True) if found is None else (lambda k: found.get(k, False))  # noqa: E731
    build_parts = {"armour": has("armour"), "chanfron": has("chanfron"), "plume": has("plume"), "tail": has("tail"),
                   "saddle_cloth": found is not None and has("saddle_cloth") and not has("caparison")}
    note(f"parts: {'all (no parts files: hand labels)' if found is None else ', '.join(k for k, v in found.items() if v)}")
    # ---- armour: every scale area L or R shows
    # (generality, finish line step 3: a subject may have only view L)
    m_arm = arm_res = None
    if build_parts["armour"]:
        cover = [{**{k: views["L"][k] for k in ("labels", "inpainted", "never_use", "camera")}, "parts": None}]
        if "R" in views:
            cover.append({**{k: views["R"][k] for k in ("labels", "inpainted", "never_use", "camera")}, "parts": [0],
                          "label_tol_px": tol["R"], "close_px": max(2, round(0.02 * views["R"]["body_length"]))})
        m_arm = eb.run(subject_path, ea_version, ai_python, blender, cand, out=out / "armour",
                       extra={"region": "all", "cover_views": cover, "front_scale": cfg["front_scale"]})
        note(f"armour: {m_arm['verdict']}")
        arm_res = m_arm["final"]["blender"]
    else:
        note("armour: none shown, not built")
    # ---- cloth: the near panel, then the far one by symmetry
    m_cl, panels, seam = None, [], None
    if has("caparison"):
        m_cl = ec.run(subject_path, ea_version, ai_python, blender, cand, out=out / "cloth_left")
        note(f"cloth left: {m_cl['verdict']}")
        # the far panel: the near one mirrored across the centre plane (master v007: simulated on its own, it
        # caught on the far foreleg L's pose lifts onto the rock and was dragged forward into a flat slab; its
        # texture was symmetry anyway)
        rl = m_cl["final"]["result"]
        dl = Path(rl["texture"]["path"]).parent
        (out / "cloth_right").mkdir(parents=True, exist_ok=True)
        vr = [[-x, y, z] for x, y, z in json.loads((dl / "panel_vertices.json").read_text(encoding="utf-8"))]
        (out / "cloth_right" / "panel_vertices.json").write_text(json.dumps(vr), encoding="utf-8")
        note("cloth right: the left panel mirrored (symmetry)")
        # the chest panel: not simulated (a one-piece cloth tangled round the stepping foreleg); master_assemble
        # drapes it, split at the legs (round 14)
        for name, d, ev in (("panel_left", dl, "observed L / R, prior"), ("panel_right", out / "cloth_right", "symmetry")):
            lab_p = Path(rl["texture"].get("labels") or (dl / "panel_labels.png"))
            panels.append({"name": name, "vertices": str(d / "panel_vertices.json"), "texture": rl["texture"]["path"],
                           "labels": str(lab_p) if lab_p.is_file() else None,
                           "rows": rl["panel"]["rows"], "cols": rl["panel"]["cols"], "evidence": ev})
        seam = min(rl["seam"]["y"])
    else:
        note("caparison: none shown, no panels" + (" (a saddle cloth instead)" if build_parts["saddle_cloth"] else ""))
    decals = [{"name": d["name"], "view": vn, "xy": d["xy"], "size": d["size"], "class": DECAL_CLASS[d["name"]]}
              for vn, v in views.items() if vn in DECAL_VIEWS for d in v["details"] if d["name"] in DECAL_CLASS]
    base_args = {"skeleton": ea_args["skeleton"], "body": {k: rec[k] for k in ("shape", "skeleton", "cameras", "poses")},
                 "classes": ea_args["classes"], "views": views,
                 "armour_dir": str(Path(arm_res["files"]["lit"]).parent) if arm_res else None,
                 "verts_per_plate": arm_res["plates"]["verts_per_plate"] if arm_res else None, "panels": panels,
                 "decals": decals, "cloth_seam_y": seam, "tail_profile": tail_profile(s, views, "S" if "S" in views else "L"),
                 **cfg, "build": build_parts,
                 # a self-read subject (M18): the main view projected on the body; the saddle cloth's box in it
                 "project_body": found is not None and "texture" in views["L"], "saddle_cloth_box": parts_box(s, "L", "saddle_cloth")}
    if not has("caparison"):
        base_args["chest_panel"] = None
    br = BlenderRunner(blender or default_blender_path(), log_dir=out / "logs", timeout=7200)
    sets = {}
    asm = br.run_job("master_assemble", {**base_args, "out_dir": str(out / "assemble")}).result
    sets["master"] = {"files": asm["files"], "landmarks": asm["landmarks"]}
    note(f"assembled in {asm['seconds']} s ({asm['cycles_device']})")
    # a break needs its part: no plates to enlarge without armour, no caparison to shorten without one
    breaks = [b for b in BREAKS if not (b == "plates_bigger" and not build_parts["armour"])
              and not (b == "caparison_short" and not panels)]
    for b in breaks:
        r = br.run_job("master_assemble", {**base_args, "break": b, "out_dir": str(out / "breaks" / b)}).result
        sets[b] = {"files": r["files"], "landmarks": r["landmarks"]}
        note(f"break {b}: rendered")
    template = s["_template"]
    gates = dict(ea_args["gates"])
    ev_args = {"classes": ea_args["classes"], "regions": template["regions"], "gates": gates, "views": views,
               "low_confidence": ea_args.get("low_confidence") or [], "head_landmarks": HEAD_LM,
               "trunk_views": (ea_args.get("refine") or {}).get("trunk_views") or []}
    ev = AIJobRunner(ai_python, timeout=3600).run_job("ref3d_master_eval", {**ev_args, "sets": sets}).result
    # step 4 (round 14): the cloth's hue. While L's cloth misses the hue gate, the panels' cloth texels turn back
    # by the measured (signed) difference and the master is assembled and scored again (twice at most)
    hue_log = []
    for k in range(2):
        mv = ev["sets"]["master"]["views"]["L"]
        g = mv["gates"]["hue_deg"]
        dh = (mv.get("hue_signed") or {}).get("cloth")
        if g["pass"] or dh is None or not panels:
            break
        for pn in panels:
            if pn["name"] == "panel_right":
                continue  # it shares the left panel's texture
            src = Path(pn["texture"])
            dst = out / "hue" / f"{pn['name']}_r{k + 1}.png"
            hue_rotate(src, Path(pn["labels"]) if pn.get("labels") else None, -dh, s["_template"]["classes"]["cloth"], dst)
            for q in panels:
                if q["texture"] == str(src):
                    q["texture"] = str(dst)
        hue_log.append({"round": k + 1, "measured_deg": dh, "turned_deg": -dh})
        note(f"hue: cloth {dh:+.1f} deg in L; the panels' cloth turned by {-dh:+.1f} deg")
        asm = br.run_job("master_assemble", {**base_args, "panels": panels, "out_dir": str(out / "assemble")}).result
        sets["master"] = {"files": asm["files"], "landmarks": asm["landmarks"]}
        ev = AIJobRunner(ai_python, timeout=3600).run_job("ref3d_master_eval", {**ev_args, "sets": sets}).result
    master_ev = ev["sets"]["master"]
    e_d = {b: {"caught": bool(ev["sets"][b]["failed_any_view"]), "failed": ev["sets"][b]["failed_any_view"]}
           for b in breaks}
    verdict = {"master_gates_pass": master_ev["pass"],
               "failed": {vn: r["failed"] for vn, r in master_ev["views"].items()},
               "ed3d_every_break_caught": all(x["caught"] for x in e_d.values()), "ed3d": e_d,
               "armour": m_arm["verdict"] if m_arm else "not built (none shown)",
               "cloth_left": m_cl["verdict"] if m_cl else "not built (no caparison shown)", "parts": found}
    sheet_path = out / "h3_sheet.png"
    h3_sheet(s, views, asm, master_ev, verdict, m_arm, m_cl, sheet_path, out.name, ea_version, cand)
    manifest = {"stage": "master build", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "ea": {"version": ea_version, "candidate": cand}, "settings": cfg, "assemble": asm,
                "evaluation": ev, "verdict": verdict, "hue_correction": hue_log, "log": log, "sheet": str(sheet_path),
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    return manifest


def hue_rotate(src: Path, labels: Path | None, deg: float, cloth_rgb, dst: Path):
    """The texture with its cloth texels' hue turned by ``deg`` (its labels say which are cloth; all when none)."""
    im = Image.open(src).convert("RGB")
    h, s_, v = im.convert("HSV").split()
    shift = int(round(deg / 360.0 * 256)) % 256
    h2 = h.point(lambda x: (x + shift) % 256)
    turned = Image.merge("HSV", (h2, s_, v)).convert("RGB")
    if labels is not None and Path(labels).is_file():
        lab = Image.open(labels).convert("RGB").resize(im.size, Image.NEAREST)
        r, g, b = lab.split()
        tr, tg, tb = cloth_rgb
        mask = Image.merge("RGB", (r.point(lambda x: 255 if x == tr else 0), g.point(lambda x: 255 if x == tg else 0),
                                   b.point(lambda x: 255 if x == tb else 0))).convert("L").point(lambda x: 255 if x == 255 else 0)
        turned = Image.composite(turned, im, mask)
    dst.parent.mkdir(parents=True, exist_ok=True)
    turned.save(dst)
    return dst


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
    arm_detail = m_arm["final"]["blender"]["files"].get("closeup_detail") if m_arm else None
    tiles = [("plates: neck and chest (master)" if m_arm else "neck and chest (no armour)", Image.open(f["closeup_plates"]), False)]
    if arm_detail:
        tiles.append(("plates: overlap, curvature, thickness", Image.open(arm_detail), False))
    tiles += [("unseen right side (symmetry and prior)", Image.open(f["unseen_right"]), False),
              ("front" + (" (the chest panel split at the legs)" if m_cl else ""), Image.open(f["front"]), False)]
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
