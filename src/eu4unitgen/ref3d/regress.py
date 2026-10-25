"""The regression set (Track P1.4): horse 2, the cataphract horse and the cataphract rider, rebuilt untouched from
their raw TRELLIS.2 models after every pipeline change; their quality numbers compared with the last good run.

  run       rebuild each (``assets_in/regression/<name>``: raw.glb, input/, subject.json; versions under ``runs/``),
            measure it (``metrics``; the quality report too), compare with ``regression/baseline.json`` and flag
            anything worse (``compare``); a report under ``build/regression/<time>/``
  baseline  store a run's (or given versions') numbers as the new baseline (the user approved them in game)

The baseline as first stored (2026-10-24): test 35's cataphract (horse ``cataphract_budget/horse/v001``, rider
``cataphract_budget/rider/v004``, approved in game) and horse 2's untouched end-to-end run (``horse2_t2`` v001).

usage: python -m eu4unitgen.ref3d.regress run [--only NAME ...]
       python -m eu4unitgen.ref3d.regress baseline --run build/regression/<time>    (or --version NAME=<dir> ...)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SET = {
    "horse2": {"kind": "horse"},
    "cataphract_horse": {"kind": "horse", "country_mask": True, "emblem": "emblem.png"},
    "cataphract_rider": {"kind": "rider", "palette_from": "cataphract_horse"},
}
# a metric worse than the baseline by more than this is flagged ("higher": bigger is better)
RULES = {"flips": ("lower", lambda b: max(6.0, 0.25 * b)), "max": ("lower", lambda b: 1.0),
         "iou": ("higher", lambda b: 0.02), "ref_iou": ("higher", lambda b: 0.02), "uv_used": ("higher", lambda b: 0.03),
         "part_iou": ("higher", lambda b: 0.03), "part_de": ("lower", lambda b: 3.0),
         "seat_gap": ("lower", lambda b: 0.1), "side_ratio": ("lower", lambda b: 0.05),
         "bake_reached": ("higher", lambda b: 0.02), "silver_cap": ("lower", lambda b: 0.03),
         "gloss_gold": ("higher", lambda b: max(0.01, 0.4 * b)), "gloss_silver": ("higher", lambda b: max(0.01, 0.4 * b)),
         "head_marks": ("higher", lambda b: 0.5), "emblem_texels": ("higher", lambda b: 0.25 * b),
         "locate_fit": ("higher", lambda b: 0.05), "group_flips": ("lower", lambda b: max(5.0, 0.3 * b))}
# colours: compared by distance from the baseline, either way (Lab for colours, plain for shares)
NEAR = {"cloth_primary": 8.0, "mask_share": 0.04}
# measures that are there or not (True in the baseline, False now: worse)
FLAGS = ("head_band", "reins", "mane", "tail")
# every measure ``metrics`` can return (the known-fixes registry names them as tests: "regress:<name>")
METRICS = ("tris", "flips_idle", "flips_moving", "flips_attack", "max_idle", "max_moving", "max_attack", "iou_side",
           "iou_front34", "iou_rear34", "uv_used", "ref_iou", "parts", "part_iou_shield", "part_de_shield",
           "part_method_shield", "seat_gap", "side_ratio", "bake_reached", "silver_cap", "gloss_gold", "gloss_silver",
           "head_band", "head_marks", "emblem_texels", "reins", "mane", "tail", "locate_fit", "group_flips_head",
           "group_flips_arms", "group_flips_legs", "group_flips_torso", "cloth_primary", "mask_share")
NOTE_ONLY = {"tris": 0.03}  # changes reported, never flagged


def checkout() -> Path:
    from eu4unitgen.core import config
    return config.checkout_root() or config.load().work()


def baseline_path() -> Path:
    return checkout() / "regression" / "baseline.json"


def source(name: str) -> Path:
    from eu4unitgen.core import config
    return config.load().work() / "assets_in" / "regression" / name


def _rule(metric: str):
    for prefix, r in RULES.items():
        if metric == prefix or metric.startswith(prefix + "_"):
            return r
    return None


def metrics(manifest: dict, kind: str, quality: dict | None = None) -> dict:
    """The numbers a model is judged by (module doc)."""
    m = {}
    final = (manifest.get("fresh") or {}).get("final") or {}
    if final.get("tris"):
        m["tris"] = final["tris"]
    for anim, x in ((manifest.get("deform") or {}).get("ours") or {}).items():
        m[f"flips_{anim}"] = x.get("flips")
        m[f"max_{anim}"] = x.get("max")
    col = manifest.get("colour") or {}
    if col.get("default_primary"):  # the country colour the cloth's mask carries (test 35: 255 77 70)
        m["cloth_primary"] = list(col["default_primary"])
    if col.get("country_mask_share") is not None:
        m["mask_share"] = col["country_mask_share"]
    if col.get("reached") is not None:
        m["bake_reached"] = col["reached"]
    for k, v in (col.get("gloss_shares") or {}).items():
        m[f"gloss_{k}"] = v
    if kind == "horse":
        fr = manifest.get("fresh") or {}
        for v, x in ((manifest.get("compare") or {}).get("iou") or {}).items():
            m[f"iou_{v}"] = x
        if col.get("covered") is not None:
            m["uv_used"] = col["covered"]
        seat = ((fr.get("back") or {}).get("seat") or {})
        if seat.get("ours") is not None and seat.get("vanilla") is not None:
            m["seat_gap"] = round(abs(seat["ours"] - seat["vanilla"]), 3)
        sl = ((fr.get("fit") or {}).get("slim") or {})
        if sl.get("after") is not None:
            m["side_ratio"] = sl["after"]
        md = col.get("metal_diffuse") or {}
        if md.get("after") and md.get("raw_silver_lum"):
            m["silver_cap"] = round(md["after"] / md["raw_silver_lum"], 3)
        m["head_band"] = bool((col.get("head_band") or {}).get("band"))
        m["head_marks"] = len(col.get("head_marks") or {})
        if col.get("emblem"):
            m["emblem_texels"] = sum(v.get("decal_texels", 0) for v in col["emblem"].values())
        m["reins"] = bool((fr.get("reins") or {}).get("built"))
        m["tail"] = bool((fr.get("hair") or {}).get("tube"))
    else:
        m["parts"] = sorted((manifest.get("parts") or {}).keys())
        for g, n in ((((manifest.get("deform") or {}).get("ours") or {}).get("idle") or {}).get("groups") or {}).items():
            m[f"group_flips_{g}"] = n
        loc = (manifest.get("part_checks") or {}).get("locate") or {}
        if loc.get("fit_iou") is not None:
            m["locate_fit"] = loc["fit_iou"]
        for key, rec in (manifest.get("part_checks") or {}).items():
            if not isinstance(rec, dict) or "tried" not in rec:
                continue
            used = next((t for t in rec["tried"] if rec.get("method", "").startswith(t.get("method", "?"))), None)
            if used and used.get("iou") is not None:
                m[f"part_iou_{key}"] = used["iou"]
                m[f"part_de_{key}"] = used["colour_de"]
            m[f"part_method_{key}"] = rec.get("method")
    sil = ((quality or {}).get("measures") or {}).get("silhouette") or {}
    if sil.get("iou") is not None:
        m["ref_iou"] = sil["iou"]
    return m


def compare(new: dict, base: dict) -> dict:
    """-> {"worse": [...], "better": [...], "notes": [...]}: each a sentence."""
    out = {"worse": [], "better": [], "notes": []}
    for k, b in base.items():
        if k not in new:
            out["notes"].append(f"{k}: in the baseline ({b}), not measured now")
            continue
        n = new[k]
        if k == "parts":
            lost = sorted(set(b) - set(n))
            if lost:
                out["worse"].append(f"parts lost: {', '.join(lost)}")
            gained = sorted(set(n) - set(b))
            if gained:
                out["notes"].append(f"parts added: {', '.join(gained)}")
            continue
        if k.startswith("part_method_"):
            if n != b:
                out["notes"].append(f"{k[12:]} now built by {n} (was {b})")
            continue
        if k in NEAR:
            from eu4unitgen.core.palette import delta_e
            d = delta_e(n, b) if isinstance(b, list) else abs(n - b)
            if d > NEAR[k]:
                out["worse"].append(f"{k}: {b} -> {n} (off by {d:.2f})")
            continue
        if k in FLAGS:
            if b and not n:
                out["worse"].append(f"{k}: was there, now missing")
            elif n and not b:
                out["better"].append(f"{k}: now there")
            continue
        if k in NOTE_ONLY:
            if b and abs(n - b) / b > NOTE_ONLY[k]:
                out["notes"].append(f"{k}: {b} -> {n}")
            continue
        r = _rule(k)
        if r is None or n is None or b is None:
            continue
        way, tol = r
        d = (n - b) if way == "higher" else (b - n)
        if d < -tol(abs(b)):
            out["worse"].append(f"{k}: {b} -> {n}")
        elif d > tol(abs(b)):
            out["better"].append(f"{k}: {b} -> {n}")
    for k in new:
        if k not in base and not k.startswith("part_method_"):
            out["notes"].append(f"{k}: {new[k]} (new measure, no baseline)")
    return out


def build(name: str, built: dict, blender: str | None = None) -> Path:
    """Rebuild one model from its raw model -> its version folder."""
    from eu4unitgen.ref3d import trellis
    s = SET[name]
    src = source(name)
    if s["kind"] == "horse":
        m = trellis.process(src / "raw.glb", src / "runs", emblem=(src / s["emblem"]) if s.get("emblem") else None,
                            country_mask=bool(s.get("country_mask")), blender=blender)
    else:
        pf = None
        if s.get("palette_from") and s["palette_from"] in built:
            hm = json.loads((built[s["palette_from"]] / "manifest.json").read_text(encoding="utf-8"))
            pf = {"diffuse": hm["textures"]["diffuse"], "specular": hm["textures"]["specular"],
                  "primary": (hm.get("colour") or {}).get("default_primary"), "geometry": hm["geometry"]}
        m = trellis.process_rider(src / "raw.glb", src / "runs", blender=blender, palette_from=pf,
                                  reference={"image": str(src / "input" / "input_rgb.png"), "fg": str(src / "input" / "input_fg.png")},
                                  part_cache=src / "parts_cache")
    return Path(m["geometry"]).parent


def measure(name: str, vdir: Path) -> tuple[dict, dict]:
    """(metrics, the quality report) of a built version."""
    from eu4unitgen.ref3d import quality as q
    kind = SET[name]["kind"]
    manifest = json.loads((Path(vdir) / "manifest.json").read_text(encoding="utf-8"))
    view = "side" if kind == "horse" else "front"
    game = ((manifest.get("compare") or {}).get("images") or {}).get("game", {}).get(view)
    if not game:  # versions built before the images were recorded
        hits = sorted((Path(vdir) / "compare").glob(f"game*{view}*.png"))
        game = str(hits[0]) if hits else None
    raw = game.replace("game", "raw") if game and Path(game.replace("game", "raw")).is_file() else None
    crops = Path(vdir) / "parts" / "crops"
    rep = q.report(manifest, kind, source(name) / "input" / "input_fg.png", Path(game) if game else None,
                   raw_view=Path(raw) if raw else None,
                   exclude=[crops / f"{k}_mask.png" for k in ("lance", "shield", "crest")] if kind == "rider" else None)
    q.save(rep, Path(vdir))
    return metrics(manifest, kind, rep), rep


def run(only: list[str] | None = None, blender: str | None = None) -> dict:
    from eu4unitgen.core import config
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = config.load().work() / "build" / "regression" / stamp
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(baseline_path().read_text(encoding="utf-8")) if baseline_path().is_file() else {"models": {}}
    built, res = {}, {"stamp": stamp, "models": {}}
    for name in SET:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            vdir = build(name, built, blender)
            built[name] = vdir
            m, rep = measure(name, vdir)
            cmp_ = compare(m, (base["models"].get(name) or {}).get("metrics", {}))
            res["models"][name] = {"version": str(vdir), "metrics": m, "compare": cmp_, "quality": rep["summary"],
                                   "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # noqa: BLE001 - a model that fails to build is the worst regression
            res["models"][name] = {"error": f"{type(exc).__name__}: {exc}", "compare": {"worse": ["failed to build"]}}
    res["worse"] = {n: r["compare"]["worse"] for n, r in res["models"].items() if r["compare"]["worse"]}
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report_md(res, base), encoding="utf-8")
    res["report"] = str(out / "report.md")
    return res


def report_md(res: dict, base: dict) -> str:
    lines = [f"# Regression run {res['stamp']}", ""]
    lines.append("**Nothing got worse.**" if not res.get("worse") else "**Worse than the baseline:** "
                 + "; ".join(f"{n}: {', '.join(w)}" for n, w in res["worse"].items()))
    for name, r in res["models"].items():
        lines += ["", f"## {name}", ""]
        if r.get("error"):
            lines.append(f"Failed: {r['error']}")
            continue
        b = (base["models"].get(name) or {})
        lines.append(f"Built {r['version']} in {r['seconds']} s; baseline: {b.get('from', 'none')}. Quality: {r['quality']}")
        for k in ("worse", "better", "notes"):
            if r["compare"][k]:
                lines.append(f"- {k}: " + "; ".join(r["compare"][k]))
    return "\n".join(lines) + "\n"


def store_baseline(entries: dict[str, tuple[Path, str]]) -> Path:
    """``entries``: {name: (version folder, where it comes from)} -> the baseline file (other models kept)."""
    p = baseline_path()
    base = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"models": {}}
    base["note"] = "the regression set's last good numbers (Track P1.4); derived numbers only, no assets"
    for name, (vdir, frm) in entries.items():
        m, _ = measure(name, vdir)
        base["models"][name] = {"from": frm, "stored": time.strftime("%Y-%m-%d"), "metrics": m}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(base, indent=1), encoding="utf-8")
    return p


def add_new_measures(run_dir: Path) -> Path:
    """A run's measures that the baseline doesn't have yet added to it (the stored ones kept: test 35's stay)."""
    rep = json.loads((Path(run_dir) / "report.json").read_text(encoding="utf-8"))
    p = baseline_path()
    base = json.loads(p.read_text(encoding="utf-8"))
    for name, r in rep["models"].items():
        if "metrics" not in r:
            continue
        b = base["models"].setdefault(name, {"from": f"regression run {rep['stamp']}", "metrics": {}})
        added = {k: v for k, v in r["metrics"].items() if k not in b["metrics"]}
        b["metrics"].update(added)
        if added:
            b.setdefault("added", {})[rep["stamp"]] = sorted(added)
    p.write_text(json.dumps(base, indent=1), encoding="utf-8")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    b = sub.add_parser("baseline")
    b.add_argument("--run", type=Path, help="a regression run's folder: its versions become the baseline")
    b.add_argument("--version", nargs="*", default=[], help="NAME=<version dir>")
    b.add_argument("--add-new", action="store_true", help="with --run: only add the measures the baseline lacks")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = run(a.only)
        print(Path(res["report"]).read_text(encoding="utf-8"))
        raise SystemExit(1 if res.get("worse") else 0)
    entries = {}
    if a.run and a.add_new:
        print(add_new_measures(a.run))
        return
    if a.run:
        rep = json.loads((a.run / "report.json").read_text(encoding="utf-8"))
        entries = {n: (Path(m["version"]), f"regression run {rep['stamp']}") for n, m in rep["models"].items() if "version" in m}
    for v in a.version:
        name, d = v.split("=", 1)
        entries[name] = (Path(d), d)
    print(store_baseline(entries))


if __name__ == "__main__":
    main()
