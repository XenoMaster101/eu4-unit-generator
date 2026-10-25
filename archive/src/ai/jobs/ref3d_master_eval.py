"""Track M master build, S8 evaluation: the M5 metric suite (the one E-D calibrated, _ref3d_metrics)
on the master's renders in each solved reference camera, for the master and for each deliberate break
(E-D in 3D). Runs ONLY in the AI environment.

Per render set and view: the candidate is the albedo render (colour), the class-ID render decoded to
classes (region, coverage, plate period), its alpha (silhouette) and the master's projected landmarks.
The user's allowances (SPEC M15) are applied on top of M5's gates: head landmarks within
``landmark_max_head``; views in ``low_confidence`` (R) are scored and reported but don't decide.

args: {"classes", "regions", "gates", "views": {name: {"image", "labels", "fg", "inpainted",
       "never_use", "landmarks", "body_length", "details"}}, "sets": {name: {"files": {view_kind: png},
       "landmarks": {view: {name: xy}}}}, "low_confidence": [view], "head_landmarks": [name], "result_path"}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


TRUNK = {"chest_front", "buttock", "croup"}
METALS = ("scale", "chanfron")  # the armour's metals: colour scored on the lit render (SPEC M17)  # E-A round 13: they steer (and are scored in) the trunk views only


def main(args_path):
    from PIL import Image

    import _ref3d_metrics as M
    import ref3d_ed2d as E
    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    classes = list(a["classes"])
    g = a["gates"]
    head = set(a.get("head_landmarks") or ())
    lc = set(a.get("low_confidence") or ())
    refs = {}
    for vn, v in a["views"].items():
        img = np.array(Image.open(v["image"]).convert("RGB"))
        cls = E.decode(np.array(Image.open(v["labels"]).convert("RGB")), a["classes"])
        fg = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127
        never = E.poly_mask(fg.shape, v.get("never_use"))
        refs[vn] = {"image": img, "cls": np.where(fg & ~never, cls, -1), "fg": fg & ~never, "valid": ~never,
                    "obs": fg & ~inp & ~never & (cls >= 0), "landmarks": {k: tuple(p) for k, p in v["landmarks"].items()}}
    res = {"sets": {}}
    for sn, st in a["sets"].items():
        out = {}
        for vn, v in a["views"].items():
            alb = np.array(Image.open(st["files"][f"{vn}_albedo"]).convert("RGBA"))
            idm = np.array(Image.open(st["files"][f"{vn}_id"]).convert("RGBA"))
            fgc = idm[..., 3] > 127
            ccls = np.where(fgc, E.decode(idm[..., :3], a["classes"]), -1)
            # composite the albedo over the reference's background (colour is compared inside regions only)
            img = np.where(alb[..., 3:4] > 127, alb[..., :3], refs[vn]["image"])
            clm = {k: tuple(p) for k, p in (st["landmarks"].get(vn) or {}).items()
                   if k not in TRUNK or vn in a.get("trunk_views", ())}  # the trunk's points only where they steer
            rlm = refs[vn]["landmarks"]
            for x_, y_ in (("ear_tip_a", "ear_tip_b"),):  # an unordered pair, as E-A matches it
                if all(k in clm and k in rlm for k in (x_, y_)):
                    same = np.hypot(*np.subtract(clm[x_], rlm[x_])) + np.hypot(*np.subtract(clm[y_], rlm[y_]))
                    swap = np.hypot(*np.subtract(clm[y_], rlm[x_])) + np.hypot(*np.subtract(clm[x_], rlm[y_]))
                    if swap < same:
                        clm[x_], clm[y_] = clm[y_], clm[x_]
            cand = {"image": img.astype(np.uint8), "cls": ccls, "fg": fgc, "landmarks": clm}
            cfg = {"classes": classes, "regions": a["regions"], "body_length": v["body_length"],
                   "details": v["details"], "gates": g}
            mets, gates = M.evaluate(refs[vn], cand, cfg)
            # metals on the lit render (the user, 2026-10-21): a metal's look is what it reflects, so the flat
            # albedo of silver plates can't match a painting's median pixel; the lit render, composited over the
            # reference's background like the albedo, is compared with the reference instead
            lit_p = st["files"].get(f"{vn}_lit")
            if lit_p and METALS:
                lit = np.array(Image.open(lit_p).convert("RGB"))
                lit_img = np.where(fgc[..., None], lit, refs[vn]["image"]).astype(np.uint8)
                rr = g["label_scale"] * v["body_length"]
                rc, cc = M.coarse(refs[vn]["cls"], len(classes), rr), M.coarse(ccls, len(classes), rr)
                de_l, dh_l, _ = M.colour(refs[vn]["image"], lit_img, rc, cc, refs[vn]["obs"], classes, g["region_min_px"])
                for c in METALS:
                    if c in de_l:
                        mets["colour"][c] = de_l[c]
                    if c in dh_l:
                        mets["hue"][c] = dh_l[c]
                    elif c in mets["hue"]:
                        del mets["hue"][c]
                mets["metals_on_lit"] = [c for c in METALS if c in de_l]
                gates = M.gates(mets, g)
            # the signed hue difference per chromatic class (candidate minus reference, degrees; round 14: the
            # master corrects its cloth's hue by it), on the classes' cores as M5's colour metric takes them
            from scipy import ndimage as ndi
            from skimage.color import rgb2lab
            la, lb = rgb2lab(refs[vn]["image"] / 255.0), rgb2lab(cand["image"] / 255.0)
            signed = {}
            for i, c in enumerate(classes):
                ra = ndi.binary_erosion((refs[vn]["cls"] == i) & refs[vn]["obs"], iterations=2)
                cb = ndi.binary_erosion((ccls == i) & refs[vn]["obs"], iterations=2)
                if ra.sum() < 200 or cb.sum() < 200:
                    continue
                ma, mb = la[ra].mean(axis=0), lb[cb].mean(axis=0)
                if np.hypot(ma[1], ma[2]) >= 15.0:
                    d = np.degrees(np.arctan2(mb[2], mb[1]) - np.arctan2(ma[2], ma[1]))
                    signed[c] = float((d + 180) % 360 - 180)
            # the head allowance: each landmark against its own limit
            lms = mets["landmarks"]
            lim = {k: g.get("landmark_max_head", g["landmark_max"]) if k in head else g["landmark_max"] for k in lms}
            bad = {k: x for k, x in lms.items() if x is not None and x > lim[k]}
            body = [x for k, x in lms.items() if k not in head and x is not None]
            hd = [x for k, x in lms.items() if k in head and x is not None]
            gates["landmarks"] = {"pass": not bad, "worst": max(lms, key=lambda k: lms[k] or 0) if lms else None,
                                  "value": max(body) if body else None, "head_value": max(hd) if hd else None,
                                  "limit": g["landmark_max"], "head_limit": g.get("landmark_max_head", g["landmark_max"]),
                                  "over": bad}
            for k, x in gates.items():  # M5's gates pass a metric that measured nothing; the master's don't
                if x.get("value") is None and k != "landmarks":
                    x["pass"], x["not_measured"] = False, True
            unmatched = sorted(set(refs[vn]["landmarks"]) - set(cand["landmarks"]))
            out[vn] = {"metrics": mets, "gates": gates, "failed": [k for k, x in gates.items() if not x["pass"]], "hue_signed": signed,
                       "landmarks_not_built": unmatched, "low_confidence": vn in lc}
            print(sn, vn, "failed:", out[vn]["failed"], flush=True)
        decisive = [vn for vn in out if vn not in lc]
        res["sets"][sn] = {"views": out, "pass": all(not out[vn]["failed"] for vn in decisive),
                           "failed_any_view": sorted({k for vn in out for k in out[vn]["failed"]})}
    Path(a["result_path"]).write_text(json.dumps(res, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
