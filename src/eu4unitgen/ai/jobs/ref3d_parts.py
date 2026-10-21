"""Track M S1 job (the self-reading pipeline, SPEC M18): region labels and the parts present in a
reference view, read from the image itself: no hand-written colour rules and no hand-drawn zones.
Runs ONLY in the AI environment (torch, transformers, OpenCV, SciPy).

args: {"views": {name: {"image", "fg" (horse mask), "segments" (SAM 2 ids, .npy), "inpainted" | null,
       "compare": a labels PNG to score against (optional), "landmarks_ref": {name: [x, y]} hand landmarks to
       score against (optional)}},
       "weights": a SigLIP 2 folder (google/siglip2-so400m-patch14-384, Apache-2.0),
       "materials": {material: [prompt, ...]}, "colours": {class: [r, g, b]} (the template's),
       "settings": {...} (PARTS_DEFAULTS in ref3d/parts.py), "out_dir", "result_path"}
Steps per view:
  1. every SAM 2 segment of the horse (one covering most of it split by the palette first) is cropped (its box with a margin, the rest of the image dimmed)
     and SigLIP scores it against each material's prompts: the segment's material prior;
  2. colours fitted per image: each material's colour centres (k-means in Lab) from the segments that
     SigLIP is sure of; every pixel then takes the material that best combines its segment's prior and
     its colour (gold embroidery inside a red cloth segment turns trim; a pixel in no segment goes by
     colour alone); a small majority filter;
  3. the horse's layout from its silhouette (_ref3d_silhouette): which way it faces, the head, the
     legs below the belly, the tail at the rear; materials become the template's classes there
     (metal on the head: chanfron; coat on the legs: leg; hair at the rear: tail, elsewhere: mane = coat;
     a plume only on top of the head);
  4. the parts present (armour, chanfron, caparison, saddle cloth, saddle, bridle, plume, bare coat,
     tail), each with its pixel share, box and fitted colours: ``<name>_parts.json``.
Writes ``<name>_labels.png``, ``<name>_labels_overlay.png``, ``<name>_materials.png`` (the SigLIP
prior per segment), ``<name>_landmarks.json`` (the silhouette's landmarks, _ref3d_silhouette.landmarks),
``<name>_texture.png`` and ``<name>_texture_labels.png`` (the view and its labels, the background filled
with the nearest horse pixel: what the master projects), ``<name>_segments_used.npy`` and ``<name>_segment_priors.json`` (the segments as
scored and their priors), ``<name>_zones.json`` (the zones in crop px). Result per view: as ref3d_labels'
(shares, palette, labels, overlay) plus "parts", "zones", "layout", "seconds", and with ``compare``
"agreement": the pixel agreement on the horse, overall, per reference class (recall) and per new
class (precision).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def lab_of(rgb):
    import cv2
    import numpy as np
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) * \
        np.array([100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)


def segment_crops(rgb, seg, ids, margin, dim, size):
    import numpy as np
    from PIL import Image
    H, W = seg.shape
    crops = []
    for sid in ids:
        ys, xs = np.nonzero(seg == sid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        b = [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(W, cx + side / 2)), int(min(H, cy + side / 2))]
        im = rgb.astype(np.float32).copy()
        out = seg != sid
        im[out] = im[out] * dim + 128 * (1 - dim)
        crops.append(Image.fromarray(im[b[1]:b[3], b[0]:b[2]].astype(np.uint8)).resize((size, size)))
    return crops


def siglip_priors(model, proc, crops, text_emb, batch):
    import torch
    out = []
    for i in range(0, len(crops), batch):
        x = proc(images=crops[i:i + batch], return_tensors="pt").to("cuda")
        with torch.no_grad():
            e = model.get_image_features(**x)
            e = getattr(e, "pooler_output", e)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append((e @ text_emb.T).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, text_emb.shape[0])


def agreement(cls, ref_rgb, fg, colours):
    import numpy as np
    names = [c for c in colours if c != "excluded"]
    b = np.full(fg.shape, "", dtype=object)
    for c in names:
        b[(np.abs(ref_rgb.astype(int) - colours[c].astype(int)).sum(-1) <= 6)] = c
    m = fg & (b != "")
    out = {"overall": round(float((cls[m] == b[m]).mean()), 4), "recall": {}, "precision": {}}
    for c in names:
        rm, nm = m & (b == c), m & (cls == c)
        if rm.sum() >= 200:
            out["recall"][c] = round(float((cls[rm] == c).mean()), 3)
        if nm.sum() >= 200:
            out["precision"][c] = round(float((b[nm] == c).mean()), 3)
    return out


def main(args_path):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    import _ref3d_silhouette as sil

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    st = a["settings"]
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    model = AutoModel.from_pretrained(a["weights"], torch_dtype=torch.float16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(a["weights"])
    mats = list(a["materials"])
    prompts = [p for m in mats for p in a["materials"][m]]
    owner = np.array([i for i, m in enumerate(mats) for _ in a["materials"][m]])
    with torch.no_grad():
        tx = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        te = model.get_text_features(**tx)
        te = getattr(te, "pooler_output", te)
        te = te / te.norm(dim=-1, keepdim=True)
    scale = float(model.logit_scale.exp())
    colours = {k: np.array(v, np.uint8) for k, v in a["colours"].items()}
    rng = np.random.default_rng(st.get("seed", 0))
    res = {}
    for name, v in a["views"].items():
        t0 = time.time()
        rgb = np.array(Image.open(v["image"]).convert("RGB"))
        fg = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127 if v.get("inpainted") else np.zeros_like(fg)
        seg = np.load(v["segments"]) if v.get("segments") else np.zeros(fg.shape, np.int32)
        seg = np.where(fg, seg, 0)
        nfg = int(fg.sum())
        from scipy.cluster.vq import kmeans2, vq
        # the image's own palette: k-means in Lab over the horse (lightness down-weighted: shade)
        lab = lab_of(rgb).reshape(fg.shape + (3,)) * np.array([st["lightness_weight"], 1, 1], np.float32)
        X = lab[fg]
        sample = X[rng.choice(len(X), min(len(X), 60000), replace=False)]
        centres, _ = kmeans2(sample, st["palette_k"], seed=rng, minit="++")
        clu = np.full(fg.shape, -1, np.int32)
        clu[fg] = vq(X, centres)[0]
        # 1. SigLIP per SAM 2 segment (its box with a margin, the rest dimmed); a segment covering most
        # of the horse names nothing as a whole: it is split by the palette, each colour a segment
        # (horse 2: the bay coat, one segment with the legs and head, voted for itself)
        nxt = int(seg.max()) + 1
        for sid, cnt in zip(*np.unique(seg[seg > 0], return_counts=True)):
            if cnt > st["big_segment"] * nfg:
                m = seg == sid
                seg[m] = nxt + clu[m]
                nxt += len(centres)
        ids, counts = np.unique(seg[seg > 0], return_counts=True)
        keep = (counts >= st["min_segment"] * nfg) & (counts <= st["big_segment"] * nfg)
        ids = ids[keep]
        prior = np.zeros((0, len(mats)))
        if len(ids):
            crops = segment_crops(rgb, seg, ids, st["crop_margin"], st["crop_dim"], st["crop_size"])
            sims = siglip_priors(model, proc, crops, te, st["batch"]).numpy() * scale
            pm = np.stack([sims[:, owner == i].max(axis=1) for i in range(len(mats))], axis=1)
            prior = np.exp(pm - pm.max(axis=1, keepdims=True))
            prior /= prior.sum(axis=1, keepdims=True)
        # 2. each palette colour's material: the votes of the segments where it is a main colour (at
        # least ``minority`` of the segment), weighted by their priors (a colour shown alone named
        # nothing: v009 put every colour of the armoured horse in metal)
        votes = np.zeros((len(centres), len(mats)))
        mat = np.full(fg.shape, -1, np.int32)
        minority = np.zeros(fg.shape, bool)
        for k_, s_ in enumerate(ids):
            m = seg == s_
            mat[m] = int(prior[k_].argmax())
            cc, cn = np.unique(clu[m], return_counts=True)
            main_ = cn >= st["minority"] * m.sum()
            for c, n_ in zip(cc[main_], cn[main_]):
                votes[c] += n_ * prior[k_]
            minority |= m & np.isin(clu, cc[~main_])
        cprior = np.where(votes.sum(1, keepdims=True) > 0, votes / np.maximum(votes.sum(1, keepdims=True), 1e-9), 1.0 / len(mats))
        cmat, cconf = cprior.argmax(1), cprior.max(1)
        # pixels in no scored segment go by their colour; a minority colour named with confidence keeps its own
        free = fg & (mat < 0)
        mat[free] = cmat[clu[free]]
        over = fg & minority & (cconf[np.maximum(clu, 0)] >= st["confident"])
        mat[over] = cmat[clu[over]]
        cm = {int(c): {"lab": centres[c].round(1).tolist(), "material": mats[int(cmat[c])], "confidence": round(float(cconf[c]), 3),
                       "share": round(float((clu == c).sum() / nfg), 4)} for c in range(len(centres))}
        # majority filter (one-hot box blur)
        r = max(1, int(round(st["majority"] * max(fg.shape))))
        votes = np.stack([cv2.blur((mat == i).astype(np.float32), (2 * r + 1, 2 * r + 1)) for i in range(len(mats))], -1)
        mat = np.where(fg, votes.argmax(-1), -1)
        # 3. the layout and the template's classes
        cls, zones, parts, lay = sil.analyse(fg, {m: mat == i for i, m in enumerate(mats)}, rgb, st)
        # outputs
        labimg = np.zeros_like(rgb)
        shares, palette = {}, {}
        for c, col in colours.items():
            m = fg & (cls == c)
            if not m.any() or c == "excluded":
                continue
            labimg[m] = col
            shares[c] = round(float(m.sum() / nfg), 4)
            ob, ip = m & ~inp, m & inp
            palette[c] = {"observed": np.median(rgb[ob], axis=0).astype(int).tolist() if ob.any() else None,
                          "inpainted": np.median(rgb[ip], axis=0).astype(int).tolist() if ip.any() else None,
                          "pixels": int(m.sum())}
        Image.fromarray(labimg).save(out / f"{name}_labels.png")
        over = rgb.copy()
        over[fg] = (0.45 * rgb[fg] + 0.55 * labimg[fg]).astype(np.uint8)
        over[~fg] = (0.35 * rgb[~fg]).astype(np.uint8)
        for z, polys in zones.items():
            for p in polys:
                cv2.polylines(over, [np.array(p, np.int32)], True, (255, 255, 255), 2)
        for k_, (x_, y_) in lay["landmarks"].items():
            cv2.circle(over, (int(x_), int(y_)), 6, (255, 255, 0), 2)
            cv2.putText(over, k_, (int(x_) + 8, int(y_) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        Image.fromarray(over).save(out / f"{name}_labels_overlay.png")
        (out / f"{name}_landmarks.json").write_text(json.dumps(lay["landmarks"], indent=1), encoding="utf-8")
        mpal = rng.integers(60, 255, (len(mats), 3)).astype(np.uint8)
        mimg = np.zeros_like(rgb)
        for i in range(len(mats)):
            mimg[mat == i] = mpal[i]
        Image.fromarray(mimg).save(out / f"{name}_materials.png")
        (out / f"{name}_zones.json").write_text(json.dumps(zones), encoding="utf-8")
        # the view as a texture (M18 step 2): the background filled with the nearest horse pixel, image and labels
        # (from the horse eroded by ``texture_erode`` of the long side: the mask's edge is anti-aliased backdrop)
        from scipy.ndimage import binary_erosion, distance_transform_edt
        src = binary_erosion(fg & (cls != "unclassified"), iterations=max(1, int(st["texture_erode"] * max(fg.shape))))
        # and not the backdrop seen through gaps (reins, tail hair): pixels near its colour (its median outside the mask)
        L0 = lab_of(rgb).reshape(fg.shape + (3,))
        if (~fg).any():
            bd = np.median(L0[~fg], axis=0)
            src &= np.linalg.norm(L0 - bd, axis=-1) > st["backdrop_de"]
        _, (iy, ix) = distance_transform_edt(~src, return_indices=True)
        Image.fromarray(rgb[iy, ix]).save(out / f"{name}_texture.png")
        Image.fromarray(labimg[iy, ix]).save(out / f"{name}_texture_labels.png")
        np.save(out / f"{name}_segments_used.npy", seg)
        (out / f"{name}_segment_priors.json").write_text(json.dumps(
            {int(s_): {m: round(float(x), 3) for m, x in zip(mats, prior[k_])} for k_, s_ in enumerate(ids)}), encoding="utf-8")
        (out / f"{name}_parts.json").write_text(json.dumps(parts, indent=1), encoding="utf-8")
        res[name] = {"shares": shares, "palette": palette, "inpainted_share": round(float((inp & fg).sum() / nfg), 4),
                     "unclassified_share": shares.get("unclassified", 0.0), "segment_share": 0.0, "neighbour_share": 0.0,
                     "excluded_share": 0.0, "labels": str(out / f"{name}_labels.png"),
                     "overlay": str(out / f"{name}_labels_overlay.png"), "materials": str(out / f"{name}_materials.png"),
                     "material_legend": {m: mpal[i].tolist() for i, m in enumerate(mats)},
                     "palette": cm,
                     "segments_scored": int(len(ids)), "zones": str(out / f"{name}_zones.json"),
                     "parts": parts, "layout": lay, "seconds": round(time.time() - t0, 1)}
        if v.get("landmarks_ref"):  # the hand landmarks (crop px): each found one's error, of the body length
            ref = v["landmarks_ref"]
            bl = abs(ref["chest_front"][0] - ref["buttock"][0]) if "chest_front" in ref and "buttock" in ref else max(fg.shape)
            err = lambda a_, b_: float(np.hypot(*(np.array(a_) - np.array(b_)))) / bl  # noqa: E731
            le = {k: round(err(p_, ref[k]), 4) for k, p_ in lay["landmarks"].items() if k in ref}
            ea_, eb_ = lay["landmarks"].get("ear_tip_a"), lay["landmarks"].get("ear_tip_b")
            if ea_ and eb_ and "ear_tip_a" in ref and "ear_tip_b" in ref:  # the ears: an unordered pair (as E-A matches them)
                if err(ea_, ref["ear_tip_b"]) + err(eb_, ref["ear_tip_a"]) < le["ear_tip_a"] + le["ear_tip_b"]:
                    le["ear_tip_a"], le["ear_tip_b"] = round(err(ea_, ref["ear_tip_b"]), 4), round(err(eb_, ref["ear_tip_a"]), 4)
            res[name]["landmark_error"] = le
            print(name, "landmark error", res[name]["landmark_error"], flush=True)
        if v.get("compare"):
            res[name]["agreement"] = agreement(cls, np.array(Image.open(v["compare"]).convert("RGB")), fg, colours)
            print(name, "agreement", res[name]["agreement"], flush=True)
        print(name, shares, {k: p["present"] for k, p in parts.items()}, flush=True)
    res["_seconds"] = round(time.time() - t_all, 1)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
