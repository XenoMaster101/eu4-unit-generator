"""Track M, the self-reading pipeline (SPEC M18), step 1: the region labels and the parts present in
each view, read from the image (AI job ref3d_parts: SAM 2 segments named by SigLIP 2, colours fitted
per image, the layout from the silhouette) instead of the subject config's colour rules and zones.

Writes a new labels version (``<subject>/refs/clean/vNNN``: labels, overlay, materials, zones,
parts per view; the H1 labels sheet) on the approved cleaned views. With ``--compare vNNN`` it also
scores the new labels against that version's labels (``approved``: each view's approved labels) (the cataphract's hand-approved ones): pixel
agreement per class.

usage: python -m eu4unitgen.ref3d.parts <subject.json> --ai-python <env python> --siglip <weights dir>
         [--sam2 <weights dir>] [--compare vNNN] [--views L,S] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from eu4unitgen.ref3d import s1

# The materials SigLIP tells apart (several prompts each; a material scores its best prompt). Plain
# words for what each looks like, no colours: the colours come from the image.
MATERIALS = {
    "metal": ["metal scale armour", "lamellar armour made of small metal plates", "polished steel armour",
              "a metal helmet plate", "chain mail", "armour plate covering a horse's head"],
    "gold": ["gold embroidery", "a golden ornament", "gold fringe and tassels", "a brass buckle", "gilded trim"],
    "cloth": ["a fabric horse blanket", "a woven cloth saddle pad", "an embroidered textile", "a cloth caparison",
              "folded fabric"],
    "leather": ["a leather saddle", "a leather strap", "a leather bridle", "reins", "a stirrup"],
    # not "a horse's head": SigLIP gave the armoured head to it (cataphract v008: chanfron recall 0.05)
    "coat": ["the body of a horse", "a horse's leg", "horse hide with short hair", "a hoof"],
    "hair": ["a horse's tail", "a horse's mane", "long black horse hair"],
    "plume": ["a feather plume", "a tuft of dyed feathers"],
}

PARTS_DEFAULTS = {
    "seed": 0,
    # segments: smaller than this share of the horse are not scored; larger than big_segment (the
    # whole horse) name nothing (horse 2 v002: its bay coat went to leather)
    "min_segment": 0.0004, "big_segment": 0.25,
    # a segment's crop: its box plus crop_margin, the rest dimmed to crop_dim (trials v012-v014: 0.3/0.35
    # let the armour around a leg name it metal; grey (0) put armour and a plume on horse 2)
    "crop_margin": 0.15, "crop_dim": 0.15, "crop_size": 384, "batch": 16,
    # the image's palette (cataphract v008: per-material colour centres let cloth absorb the gold
    # embroidery): palette_k colours, each named by the segments where it is at least ``minority``;
    # a colour under that share of its segment and named with at least ``confident`` keeps its own
    "palette_k": 12, "lightness_weight": 0.5, "minority": 0.3, "confident": 0.6,
    "majority": 0.003,  # the majority filter's radius, of the image's long side
    # layout (_ref3d_silhouette)
    "open_frac": 0.012, "trunk_run": 0.35, "tail_from": 0.7, "tail_min": 0.01, "head_len": 0.17,
    # parts: present when at least this much shows
    "chanfron_min": 0.15, "armour_min": 0.05, "cloth_min": 0.005, "hem_reach": 0.08, "capar_span": 0.6,
    "saddle_min": 0.004, "bridle_min": 0.01, "plume_min": 0.0008, "bare_min": 0.25,
    # landmarks (step 3): a foot's columns reach within foot_band x (ground - back) of the ground; the
    # fetlock fetlock_up x (ground - back) above the hoof (horse 2's hand landmarks: 65 px of 725)
    "foot_band": 0.03, "fetlock_up": 0.09,
    # the view as a texture: filled from the horse eroded by this share of the long side (v010: the mask's
    # anti-aliased edge spread grey over the far legs), leaving out pixels within backdrop_de (Lab) of the backdrop
    "texture_erode": 0.005, "backdrop_de": 15.0,
}


def run(subject_path: Path, ai_python: str, siglip: str, sam2: str | None = None, compare: str | None = None,
        views: list[str] | None = None, overrides: dict | None = None) -> dict:
    from eu4unitgen.ai.env import AIJobRunner

    s = s1.load_subject(subject_path)
    root = s["_root"]
    chosen = s.get("chosen") or {}
    names = views or [n for n in s["views"] if n in chosen]
    inputs = {}
    for n in names:
        cdir, ldir = s1.view_dirs(s, n)
        k = chosen[n]
        segs = ldir / f"{n}_segments.npy"
        inputs[n] = {"image": str(cdir / f"{n}_seed{k}.png"), "fg": str(cdir / f"{n}_seed{k}_fg.png"),
                     "inpainted": str(cdir / f"{n}_mask_used.png"), "segments": str(segs) if segs.is_file() else None}
    out = s1.next_version(root / "refs" / "clean")
    h1 = root / "h1" / out.name
    h1.mkdir(parents=True)
    t0 = time.time()
    ai = AIJobRunner(ai_python)
    missing = {n: v["image"] for n, v in inputs.items() if not v["segments"]}
    if missing:
        if not sam2:
            raise ValueError(f"views {sorted(missing)} have no SAM 2 segments: give --sam2")
        sg = ai.run_job("ref3d_segments", {"images": missing, "weights": sam2, "out_dir": str(out),
                                           **(s.get("segments") or {"points_per_crop": 64, "pred_iou_thresh": 0.5,
                                                                    "stability_score_thresh": 0.7})}).result
        for n in missing:
            inputs[n]["segments"] = sg[n]["npy"]
    if compare:
        for n in names:
            ref = s1.view_dirs(s, n)[1] if compare == "approved" else root / "refs" / "clean" / compare
            if (ref / f"{n}_labels.png").is_file():
                inputs[n]["compare"] = str(ref / f"{n}_labels.png")
    for n in names:  # hand landmarks, when the config has them: the silhouette's scored against them
        v = s["views"][n]
        if v.get("landmarks"):
            inputs[n]["landmarks_ref"] = {k: s1.to_crop([p], v["crop"])[0] for k, p in v["landmarks"].items()}
    settings = {**PARTS_DEFAULTS, **(s.get("parts") or {}), **(overrides or {})}
    lab = ai.run_job("ref3d_parts", {"views": inputs, "weights": siglip, "materials": MATERIALS,
                                     "colours": s["_template"]["classes"], "settings": settings, "out_dir": str(out)},
                     timeout=3600).result
    secs = lab.pop("_seconds", None)
    scores = {n: lab[n]["agreement"] for n in names if lab[n].get("agreement")}
    sheet(s, inputs, lab, scores, h1 / "h1_parts.png", out.name, compare)
    manifest = {"stage": "S1 parts (self-reading)", "subject": s["name"], "version": out.name,
                "inputs": inputs, "settings": settings, "materials": MATERIALS, "labels": lab,
                "compare": {"version": compare, "scores": scores} if compare else None,
                "sheets": {"parts": str(h1 / "h1_parts.png")}, "job_seconds": secs, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def sheet(s, inputs, lab, scores, path, version, compare):
    rows = []
    for n, v in inputs.items():
        L = lab[n]
        present = [k for k, p in L["parts"].items() if p["present"]]
        absent = [k for k, p in L["parts"].items() if not p["present"]]
        info = [f"facing {L['layout']['facing']}; {L['segments_scored']} segments scored; {L['seconds']} s",
                "PRESENT: " + ", ".join(present), "absent: " + ", ".join(absent), ""]
        for k in present:
            p = L["parts"][k]
            info.append(f"{k}: {p['share'] * 100:.1f}% " + " ".join(f"{c}{tuple(rgb)}" for c, rgb in list(p["colours"].items())[:2]))
        if L.get("landmark_error"):
            le = L["landmark_error"]
            info += ["", f"landmarks found {len(L['layout']['landmarks'])}; vs the hand ones (of the body length):",
                     s1._wrap(", ".join(f"{k} {x * 100:.1f}%" for k, x in sorted(le.items(), key=lambda kv: -kv[1])), 44)]
        if n in scores:
            sc = scores[n]
            info += ["", f"agreement with {compare}: {sc['overall'] * 100:.0f}%",
                     "recall " + ", ".join(f"{c} {x:.2f}" for c, x in sc["recall"].items()),
                     "precision " + ", ".join(f"{c} {x:.2f}" for c, x in sc["precision"].items())]
        legend = ", ".join(f"{m}" for m in L["material_legend"])
        tiles = [("the view", Image.open(v["image"]), False),
                 (f"materials (SigLIP prior + colours)\n{legend}", Image.open(L["materials"]), False),
                 ("labels (template classes) and zones", Image.open(L["overlay"]), False)]
        if compare and n in scores:
            tiles.append((f"{Path(v['compare']).parent.name} labels (reference)", Image.open(v["compare"]), False))
        tiles.append(("", s1.text_tile("\n".join(s1._wrap(t, 44) for t in info), (380, 460)), False))
        rows.append((f"view {n}", tiles))
    legend = "  ".join(f"{c}: {tuple(rgb)}" for c, rgb in s["_template"]["classes"].items())
    return s1.grid_sheet(rows, f"S1 parts, read from the image: {s['name']} {version}", path, tile=(380, 460),
                         notes=["classes (label colours): " + legend])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("subject", type=Path)
    p.add_argument("--ai-python", required=True)
    p.add_argument("--siglip", required=True)
    p.add_argument("--sam2")
    p.add_argument("--compare", metavar="vNNN|approved")
    p.add_argument("--views")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a setting for this run (a trial)")
    a = p.parse_args(argv)
    m = run(a.subject, a.ai_python, a.siglip, a.sam2, a.compare, a.views.split(",") if a.views else None,
            {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)})
    print(json.dumps({"version": m["version"], "sheets": m["sheets"], "seconds": m["seconds"],
                      "parts": {n: [k for k, p in L["parts"].items() if p["present"]] for n, L in m["labels"].items()},
                      "compare": m["compare"]}, indent=1))


if __name__ == "__main__":
    main()
