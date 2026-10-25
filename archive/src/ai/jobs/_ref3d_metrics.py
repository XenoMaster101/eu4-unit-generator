"""Track M fidelity metrics (SPEC M5), the 2D part: a candidate (a render of the master in a solved
reference camera, or an E-D test variant) against a cleaned reference view. Runs ONLY in the AI
environment (numpy, SciPy, OpenCV, scikit-image); imported by the ref3d_* jobs.

Everything works on arrays in the reference view's pixels:
  image (H, W, 3) uint8; cls (H, W) int class index, -1 = none; fg (H, W) bool;
  obs (H, W) bool: the reference's observed pixels (horse, not inpainted, not excluded);
  landmarks {name: (x, y)}; body_length in px.
Metrics are reported each on its own; ``gates`` turns them into pass/fail (hard, never combined).
"""
import numpy as np


def coarse(cls, n_classes, radius):
    """Majority class within a (2r+1) box: the scale at which coverage is judged (E-D 2D v001:
    pixel-exact labels of rivets and shaded gaps fail any render that is a few px off)."""
    import cv2
    if radius < 1:
        return cls
    k = 2 * int(round(radius)) + 1
    votes = np.stack([cv2.blur((cls == i).astype(np.float32), (k, k)) for i in range(n_classes)])
    out = votes.argmax(axis=0).astype(np.int32)
    out[votes.max(axis=0) < 1e-6] = -1
    out[cls < 0] = -1  # keep the silhouette exact
    return out


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else None


def tol_iou(a, b, radius):
    """IoU where a pixel counts as matched when the other mask is within ``radius`` px (E-D 2D v003:
    a leg is ~30 px wide, so the allowed camera residual alone cost exact IoU ~20%). Intersection =
    the mean of the two matched counts; union = |a| + |b| - intersection."""
    import cv2
    if radius < 1:
        return iou(a, b)
    if not a.any() and not b.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(radius)) + 1,) * 2)
    da, db = (cv2.dilate(m.astype(np.uint8), k) > 0 for m in (a, b))
    inter = 0.5 * ((a & db).sum() + (b & da).sum())
    return float(inter / (a.sum() + b.sum() - inter))


def _boundary(m):
    from scipy import ndimage as ndi
    return m & ~ndi.binary_erosion(m)


def silhouette(ref_fg, cand_fg, valid, body_length, radius=0):
    """Horse IoU and contour distance (mean, p95; fraction of body length) inside ``valid``."""
    from scipy import ndimage as ndi
    a, b = ref_fg & valid, cand_fg & valid
    ba, bb = _boundary(a), _boundary(b)
    if not ba.any() or not bb.any():
        return {"iou": tol_iou(a, b, radius), "contour_mean": None, "contour_p95": None}
    da, db = ndi.distance_transform_edt(~ba), ndi.distance_transform_edt(~bb)
    d = np.concatenate([db[ba], da[bb]])
    return {"iou": tol_iou(a, b, radius), "contour_mean": float(d.mean() / body_length),
            "contour_p95": float(np.percentile(d, 95) / body_length)}


def region_ious(ref_cls, cand_cls, obs, regions, classes, min_px, radius=0):
    """Per region (a group of classes): IoU on the observed pixels; None when the region has fewer
    than ``min_px`` observed reference pixels (not scored)."""
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for r, members in regions.items():
        ids = [idx[c] for c in members if c in idx]
        a, b = np.isin(ref_cls, ids) & obs, np.isin(cand_cls, ids) & obs
        out[r] = tol_iou(a, b, radius) if a.sum() >= min_px else None
    return out


def coverage(ref_cls, cand_cls, obs, classes, min_share, radius=0):
    """Per class IoU on the observed pixels (classes with at least ``min_share`` of them), and the
    confusion matrix (rows: reference class, share of its pixels per candidate class)."""
    n = max(int(obs.sum()), 1)
    ious, conf = {}, {}
    for i, c in enumerate(classes):
        a = (ref_cls == i) & obs
        if a.sum() / n >= min_share:
            ious[c] = tol_iou(a, (cand_cls == i) & obs, radius)
        if a.any():
            row = np.bincount(cand_cls[a] + 1, minlength=len(classes) + 1) / a.sum()
            conf[c] = {("none" if j == 0 else classes[j - 1]): round(float(v), 4) for j, v in enumerate(row) if v > 0}
    return ious, conf


def colour(ref_img, cand_img, ref_cls, cand_cls, obs, classes, min_px, core_px=2, min_chroma=15.0):
    """Region-mean colour per class, on each region's core (eroded ``core_px``, so misaligned
    borders don't mix neighbours in), exposure-compensated (the candidate's L* shifted so the
    horse's median L* matches: a render's light is not the painting's). Returns CIEDE2000 with the
    textile weighting kL = 2 (lightness counts half: light varies, dye doesn't), and the hue angle
    difference in degrees for chromatic classes (reference C* >= ``min_chroma``): light scales
    brightness, not hue, so a wrong dye shows here first (E-D 2D v001: a 15 deg red shift stayed
    under dE 5)."""
    from scipy import ndimage as ndi
    from skimage.color import deltaE_ciede2000, rgb2lab
    la, lb = rgb2lab(ref_img / 255.0), rgb2lab(cand_img / 255.0)
    shift = float(np.median(lb[..., 0][obs & (cand_cls >= 0)]) - np.median(la[..., 0][obs & (ref_cls >= 0)]))
    de, dh = {}, {}
    for i, c in enumerate(classes):
        a = ndi.binary_erosion((ref_cls == i) & obs, iterations=core_px)
        b = ndi.binary_erosion((cand_cls == i) & obs, iterations=core_px)
        if a.sum() < min_px or b.sum() < min_px:
            continue
        ma, mb = la[a].mean(axis=0), lb[b].mean(axis=0)
        mb[0] -= shift
        de[c] = float(deltaE_ciede2000(ma[None], mb[None], kL=2)[0])
        if np.hypot(ma[1], ma[2]) >= min_chroma:
            d = np.degrees(np.arctan2(mb[2], mb[1]) - np.arctan2(ma[2], ma[1]))
            dh[c] = float(abs((d + 180) % 360 - 180))
    return de, dh, shift


def landmarks(ref_lm, cand_lm, body_length):
    return {k: float(np.hypot(cand_lm[k][0] - x, cand_lm[k][1] - y) / body_length)
            for k, (x, y) in ref_lm.items() if k in cand_lm}


def _gray(img):
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _gradmag(img):
    import cv2
    g = _gray(img)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def edge_ssim(ref_img, cand_img, valid):
    """SSIM of the gradient magnitudes over the horse's bounding box (secondary: style-sensitive)."""
    from skimage.metrics import structural_similarity
    ys, xs = np.nonzero(valid)
    if not len(ys):
        return None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    a, b = _gradmag(ref_img)[sl], _gradmag(cand_img)[sl]
    rng = float(max(a.max(), b.max(), 1e-6))
    return float(structural_similarity(a, b, data_range=rng))


def _mean_acorr(gray, tiles, n):
    """Autocorrelation of the band-passed texture, averaged over the tiles (normalised per tile)."""
    import cv2
    acc = np.zeros((2 * n, 2 * n))
    w = np.outer(np.hanning(n), np.hanning(n))
    for y, x in tiles:
        t = gray[y:y + n, x:x + n]
        t = cv2.GaussianBlur(t, (0, 0), 1.0) - cv2.GaussianBlur(t, (0, 0), 8)
        f = np.fft.fft2((t - t.mean()) * w, s=(2 * n, 2 * n))
        ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(f) ** 2)))
        acc += ac / max(ac.max(), 1e-9)
    return acc / max(len(tiles), 1)


def _refine(prof, k):
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    den = a - 2 * b + c
    return k + (0.5 * (a - c) / den if den else 0.0)


def _first_peak(prof, lo, hi):
    """First local maximum in [lo, hi): (sub-pixel lag, prominence over the lowest value before it)."""
    for k in range(lo, hi):
        if prof[k] > prof[k - 1] and prof[k] >= prof[k + 1]:
            return _refine(prof, k), float(prof[k] - prof[1:k].min())
    return None, 0.0


def plate_period(ref_img, cand_img, ref_cls, cand_cls, obs, scale_id, n=64, stride=16, lo=5, hi=30,
                 min_scale=0.85, min_prominence=0.05):
    """Plate period from the autocorrelation averaged over every tile that is scale in both maps
    (E-D 2D v003: per-tile FFT peaks jumped between harmonics under blur). The reference's first
    peak along x (plate width) and y (row pitch); the axis with the more prominent peak is used; the
    candidate's peak is the highest within 0.6-1.7x the reference lag (no half or double harmonic).
    Returns {ref_px, cand_px, ratio, axis, tiles}."""
    ra, rb = (ref_cls == scale_id) & obs, cand_cls == scale_id
    H, W = ra.shape
    tiles = [(y, x) for y in range(0, H - n, stride) for x in range(0, W - n, stride)
             if ra[y:y + n, x:x + n].mean() >= min_scale and rb[y:y + n, x:x + n].mean() >= min_scale]
    none = {"ref_px": None, "cand_px": None, "ratio": None, "axis": None, "tiles": len(tiles)}
    if not tiles:
        return none
    aa, ab = _mean_acorr(_gray(ref_img), tiles, n), _mean_acorr(_gray(cand_img), tiles, n)
    profs = {"x": (aa[n, n:], ab[n, n:]), "y": (aa[n:, n], ab[n:, n])}
    (lag, prom), ax = max(((_first_peak(pa, lo, hi), ax) for ax, (pa, _) in profs.items()), key=lambda r: r[0][1])
    if lag is None or prom < min_prominence:
        return none
    pb = profs[ax][1]
    k0, k1 = max(2, int(np.floor(0.6 * lag))), min(len(pb) - 2, int(np.ceil(1.7 * lag)))
    k = k0 + int(np.argmax(pb[k0:k1 + 1]))
    cand = _refine(pb, k)
    return {"ref_px": float(lag), "cand_px": float(cand), "ratio": float(cand / lag), "axis": ax, "tiles": len(tiles)}


def details(ref_img, cand_img, items, body_length, tol_frac):
    """Key details: each reference patch (gradient magnitude) searched in the candidate within the
    landmark tolerance; normalised cross-correlation of the best match and its offset."""
    import cv2
    ga, gb = _gradmag(ref_img), _gradmag(cand_img)
    tol = tol_frac * body_length
    out = {}
    H, W = ga.shape
    for d in items:
        x, y, s = int(round(d["xy"][0])), int(round(d["xy"][1])), int(d["size"])
        h = s // 2
        if not (h <= x < W - h and h <= y < H - h):
            out[d["name"]] = {"ncc": None, "offset": None}
            continue
        patch = ga[y - h:y + h, x - h:x + h]
        r = int(np.ceil(tol))
        y0, y1, x0, x1 = max(0, y - h - r), min(H, y + h + r), max(0, x - h - r), min(W, x + h + r)
        res = cv2.matchTemplate(gb[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(res)
        off = float(np.hypot(x0 + loc[0] + h - x, y0 + loc[1] + h - y) / body_length)
        out[d["name"]] = {"ncc": float(best), "offset": off}
    return out


def evaluate(ref, cand, cfg):
    """ref / cand: dicts of arrays (image, cls, fg; ref also obs, valid). cfg: classes, regions,
    body_length, details, gates. Returns the metrics and the gate verdicts."""
    g = cfg["gates"]
    classes, bl = cfg["classes"], cfg["body_length"]
    r = g["label_scale"] * bl
    rc, cc = coarse(ref["cls"], len(classes), r), coarse(cand["cls"], len(classes), r)
    m = {"label_scale_px": r, "silhouette": silhouette(ref["fg"], cand["fg"], ref["valid"], bl, r),
         "regions": region_ious(rc, cc, ref["obs"], cfg["regions"], classes, g["region_min_px"], r)}
    m["coverage"], m["confusion"] = coverage(rc, cc, ref["obs"], classes, g["coverage_min_share"], r)
    m["colour"], m["hue"], m["exposure_shift_L"] = colour(ref["image"], cand["image"], rc, cc, ref["obs"],
                                                          classes, g["region_min_px"])
    m["landmarks"] = landmarks(ref["landmarks"], cand["landmarks"], bl)
    m["edge_ssim"] = edge_ssim(ref["image"], cand["image"], ref["valid"] & ref["fg"])
    m["plate_period"] = plate_period(ref["image"], cand["image"], rc, cc, ref["obs"], classes.index("scale"))
    m["details"] = details(ref["image"], cand["image"], cfg["details"], bl, g["landmark_max"])
    return m, gates(m, g)


def gates(m, g):
    """Hard gates (SPEC M5; values assumed until E-D calibrates them). Each: pass, the worst item."""
    def worst(d, low_is_bad):
        items = [(v, k) for k, v in d.items() if v is not None]
        return (min if low_is_bad else max)(items) if items else (None, None)

    out = {}
    v, k = worst({**m["regions"], "horse": m["silhouette"]["iou"]}, True)
    out["region_iou"] = {"pass": v is None or v >= g["region_iou_min"], "worst": k, "value": v, "limit": g["region_iou_min"]}
    v, k = worst(m["landmarks"], False)
    out["landmarks"] = {"pass": v is None or v <= g["landmark_max"], "worst": k, "value": v, "limit": g["landmark_max"]}
    v, k = worst(m["coverage"], True)
    out["coverage_iou"] = {"pass": v is None or v >= g["coverage_iou_min"], "worst": k, "value": v, "limit": g["coverage_iou_min"]}
    v, k = worst(m["colour"], False)
    out["colour_de2000"] = {"pass": v is None or v <= g["delta_e_max"], "worst": k, "value": v, "limit": g["delta_e_max"]}
    v, k = worst(m["hue"], False)
    out["hue_deg"] = {"pass": v is None or v <= g["hue_max_deg"], "worst": k, "value": v, "limit": g["hue_max_deg"]}
    dets = {n: (d["ncc"] if d["ncc"] is not None and d["offset"] <= g["landmark_max"] else -1.0)
            for n, d in m["details"].items()}
    v, k = worst(dets, True)
    out["key_details"] = {"pass": v is None or v >= g["detail_ncc_min"], "worst": k, "value": v, "limit": g["detail_ncc_min"]}
    r = m["plate_period"]["ratio"]
    out["plate_period"] = {"pass": r is None or abs(r - 1) <= g["plate_period_tol"], "worst": "scale", "value": r,
                           "limit": g["plate_period_tol"]}
    return out
