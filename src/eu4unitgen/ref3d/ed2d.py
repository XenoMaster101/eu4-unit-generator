"""Track M spike E-D in 2D: metric sensitivity on the approved H1 references (SPEC M9, M12).

The cleaned reference stands in for a perfect master's render: unbroken variants (render-like
differences only) must pass every M5 gate; each deliberate break must fail at least one. The AI job
``ref3d_ed2d`` makes and scores the variants; this side builds its arguments from the subject config
and the template, then the sheet, the calibration table and the manifest in
``<subject>/spikes/ed2d/vNNN`` (nothing overwritten). Reruns in 3D once E-A has a master.

usage: python -m eu4unitgen.ref3d.ed2d <subject.json> --ai-python <env python>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1
from eu4unitgen.ref3d.writeguard import allowed_roots, check_writes

GATES = Path(__file__).with_name("templates") / "gates.json"


def body_lengths(s: dict, fg_heights: dict) -> dict:
    """Per view body length in px: between two landmarks (along an axis), or another view's length
    scaled by the ratio of the horse's heights (a rear 3/4 view foreshortens length, not height)."""
    out = {}
    for n, v in s["views"].items():
        b = v.get("body_length")
        if b and "from" in b:
            (x0, y0), (x1, y1) = v["landmarks"][b["from"]], v["landmarks"][b["to"]]
            out[n] = abs(x1 - x0) if b.get("axis") == "x" else ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    for n, v in s["views"].items():
        b = v.get("body_length")
        if b and "like" in b:
            out[n] = out[b["like"]] * fg_heights[n] / fg_heights[b["like"]]
    return out


def fg_height(path: Path) -> int:
    m = Image.open(path).convert("L").point(lambda x: 255 if x > 127 else 0)
    box = m.getbbox()
    return box[3] - box[1] if box else 0


def build_args(s: dict, out: Path) -> dict:
    # the fitted views: those with key details (M18: or an empty list, a self-read view: the eye is in its texture)
    views = {n: k for n, k in s["chosen"].items() if s["views"][n].get("details") is not None}
    dirs = {n: s1.view_dirs(s, n) for n in views}
    fgs = {n: dirs[n][0] / f"{n}_seed{k}_fg.png" for n, k in views.items()}
    bl = body_lengths({**s, "views": {n: s["views"][n] for n in views}}, {n: fg_height(p) for n, p in fgs.items()})
    t = s["_template"]
    a = {"views": {}, "classes": {c: rgb for c, rgb in t["classes"].items() if c not in ("unclassified", "excluded")},
         "regions": t["regions"], "gates": json.loads(GATES.read_text(encoding="utf-8"))["gates"],
         "out_dir": str(out), **s["ed2d"]}
    for n, k in views.items():
        v = s["views"][n]
        clean, labels = dirs[n]
        crop = lambda p: s1.to_crop([p], v["crop"])[0]
        head = v.get("head")
        a["views"][n] = {
            "image": str(clean / f"{n}_seed{k}.png"), "labels": str(labels / f"{n}_labels.png"), "fg": str(fgs[n]),
            "inpainted": str(clean / f"{n}_mask_used.png"), "never_use": s1.never_use_in_crop(v),
            "landmarks": {name: crop(p) for name, p in v["landmarks"].items()}, "body_length": bl[n],
            "head": ([q for z in head["zones"] for q in s1.zones_in_crop(v)[z]] if head else None),
            "pivot": (crop(v["landmarks"][head["pivot"]]) if head else None),
            "details": [{**d, "xy": crop(d["xy"])} for d in v["details"]]}
    return a


def calibration(res: dict) -> dict:
    """Per gate: the worst value over the unbroken variants and each break's value, per view."""
    table = {}
    for n, vr in res["views"].items():
        for vn, r in vr.items():
            for g, x in r["gates"].items():
                row = table.setdefault(g, {"limit": x["limit"], "unbroken_worst": {}, "breaks": {}})
                if vn.startswith("unbroken"):
                    w = row["unbroken_worst"].get(n)
                    low_bad = g in ("region_iou", "coverage_iou", "key_details")
                    bad = (lambda q: abs(q - 1)) if g == "plate_period" else (lambda q: -q if low_bad else q)
                    if x["value"] is not None and (w is None or bad(x["value"]) > bad(w[0])):
                        row["unbroken_worst"][n] = (x["value"], x["worst"], vn)
                else:
                    row["breaks"].setdefault(vn, {})[n] = (x["value"], x["worst"], x["pass"])
    return table


def verdict(res: dict) -> dict:
    unbroken_fail = {f"{n}/{vn}": r["failed"] for n, vr in res["views"].items() for vn, r in vr.items()
                     if vn.startswith("unbroken") and r["failed"]}
    breaks = {}
    for n, vr in res["views"].items():
        for vn, r in vr.items():
            if not vn.startswith("unbroken"):
                breaks.setdefault(vn, {})[n] = r["failed"]
    caught = {b: any(f for f in per.values()) for b, per in breaks.items()}
    return {"pass": not unbroken_fail and all(caught.values()), "unbroken_failures": unbroken_fail,
            "breaks_caught": caught, "breaks_failed_gates": breaks}


def sheet(res: dict, s: dict, path: Path, version: str):
    rows = []
    for n, vr in res["views"].items():
        k = s["chosen"][n]
        clean = s1.view_dirs(s, n)[0]
        tiles = [("reference (cleaned)", Image.open(clean / f"{n}_seed{k}.png"), False)]
        for vn, r in vr.items():
            if vn.startswith("unbroken") and vn != "unbroken_0":
                continue
            failed = r["failed"]
            cap = (f"{vn}\n" + ("passes every gate" if not failed else "FAILS: " + ", ".join(failed)))
            tiles.append((s1._wrap(cap, 34), Image.open(r["image"]), vn.startswith("unbroken") and not failed))
        rows.append((f"view {n}", tiles))
    return s1.grid_sheet(rows, f"E-D (2D) metric sensitivity: {s['name']} {version} "
                               "(green = unbroken passes; every break must fail a gate)",
                         path, tile=(260, 330),
                         notes=["unbroken: camera residual, smooth warp, lighting field, blur, noise (seeded); "
                                "breaks get the same perturbation as unbroken_0."])


def run(subject_path: Path, ai_python: str) -> dict:
    from eu4unitgen.ai.env import AIJobRunner

    s = s1.load_subject(subject_path)
    root = s["_root"]
    check_writes({"out_dir": str(root / "spikes")}, allowed_roots(ai_python), "E-D 2D")
    out = s1.next_version(root / "spikes" / "ed2d")
    t0 = time.time()
    args = build_args(s, out)
    res = AIJobRunner(ai_python).run_job("ref3d_ed2d", args).result
    cal, ver = calibration(res), verdict(res)
    sheet(res, s, out / "ed2d_sheet.png", out.name)
    manifest = {"stage": "E-D 2D", "subject": s["name"], "version": out.name, "approved": s["approved"],
                "inputs": {"subject": {"path": str(subject_path), "sha256": s1.sha256(subject_path)}},
                "settings": {k: v for k, v in args.items() if k not in ("views",)},
                "body_length": {n: v["body_length"] for n, v in args["views"].items()},
                "verdict": ver, "calibration": cal, "results": res,
                "sheet": str(out / "ed2d_sheet.png"), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("subject", type=Path)
    p.add_argument("--ai-python", required=True)
    a = p.parse_args(argv)
    m = run(a.subject, a.ai_python)
    print(json.dumps({"version": m["version"], "verdict": m["verdict"], "sheet": m["sheet"], "seconds": m["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
