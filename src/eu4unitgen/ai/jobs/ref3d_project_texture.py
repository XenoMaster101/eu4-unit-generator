"""AI job (M19, the simpler pipeline): a game mesh's textures projected from a side-view reference.

The mesh is a geometry JSON in PDX space (``p``, ``n``, ``u0``, ``tri``: Y up, the horse facing -Z, its
left side at +X), as fresh_horse writes it. The reference is a side view of the horse facing left (its
left side: ref3d.auto mirrors a right-facing one), with its mask and, optionally, its labels.

  1. fit: the mesh's side silhouette (z, y) onto the reference's mask, by scale and offset per axis
     (bounding boxes first, then the overlap maximised);
  2. projection: every texel of the mesh's UV layout gets its surface point (barycentric in UV space)
     and takes the reference's colour (and label) at that point's (z, y). Orthographic along x, so the
     far side takes the near side's colours (the mirror);
  3. unseen texels (the surface turned away from the side: the top of the back and neck, the belly, the
     chest and the rump seen end-on, ``facing`` under ``min_facing``; or outside the eroded mask) take
     the colour of the nearest seen texel in 3D (the matching visible region, not a stretched one);
  4. islands bled outwards (mips and seams), the specular map from the labels (gloss per class).

args: {"geometry": .json (with "weights": the mesh's landmarks), "image": png, "mask": png (white = horse),
       "labels": png | null, "landmarks": {name: [u, v]} | null (the image's; with 4 or more shared with the
       mesh, a thin-plate spline through them replaces the plain fit: a raised head or other legs line up),
       "classes": {name: [r, g, b]} (the template's label colours), "gloss": {name: value},
       "size": texture size, "min_facing" (default 0.35), "erode" (mask pixels, default 3), "fill_k" (default 48),
       "out_dir", "name", "result_path"}
Result: the maps' paths, the fit (scale, offset per axis; silhouette IoU), the share of texels seen
directly, the debug images (the silhouette overlay, the seen mask).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from scipy.spatial import cKDTree


def load_geo(path):
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    p = np.array(g["p"], dtype=np.float64).reshape(-1, 3)
    n = np.array(g["n"], dtype=np.float64).reshape(-1, 3)
    uv = np.array(g["u0"], dtype=np.float64).reshape(-1, 2)
    tri = np.array(g["tri"], dtype=np.int64).reshape(-1, 3)
    return p, n, uv, tri


def raster(pts2, tri, W, H, attrs=None):
    """Triangles with 2D pixel-space corners -> (covered mask, triangle id, barycentrics) on a W x H grid."""
    tid = np.full((H, W), -1, dtype=np.int64)
    bary = np.zeros((H, W, 3))
    for t, (i, j, k) in enumerate(tri):
        a, b, c = pts2[i], pts2[j], pts2[k]
        x0, x1 = int(max(0, np.floor(min(a[0], b[0], c[0])))), int(min(W - 1, np.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = int(max(0, np.floor(min(a[1], b[1], c[1])))), int(min(H - 1, np.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        w1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        w2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        w3 = 1 - w1 - w2
        inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
        if not inside.any():
            continue
        yy, xx = np.nonzero(inside)
        tid[y0 + yy, x0 + xx] = t
        bary[y0 + yy, x0 + xx] = np.stack([w1[inside], w2[inside], w3[inside]], 1)
    return tid >= 0, tid, bary


def side_silhouette(p, tri, W, H, frame):
    """The mesh seen from its left side in ``frame`` (z0, y1, scale): z right, y up -> pixels."""
    z0, y1, s = frame
    pts = np.stack([(p[:, 2] - z0) * s, (y1 - p[:, 1]) * s], 1)
    return raster(pts, tri, W, H)[0]


def fit(p, tri, mask):
    """Scale and offset per axis taking the mesh's (z, y) onto the mask's pixels: (sz, oz, sy, oy) with
    u = sz z + oz, v = -sy y + oy; and the silhouette IoU."""
    H, W = mask.shape
    k = 384 / W  # work at 384 px wide
    m = ndimage.zoom(mask.astype(float), k, order=1) > 0.5
    h, w = m.shape
    # the mesh's silhouette once, in its own frame, at a fine scale
    lo, hi = p.min(0), p.max(0)
    S = 600 / (hi[2] - lo[2])
    sil = side_silhouette(p, tri, int((hi[2] - lo[2]) * S) + 2, int((hi[1] - lo[1]) * S) + 2, (lo[2], hi[1], S))
    ys, xs = np.nonzero(m)
    box = (xs.min(), xs.max(), ys.min(), ys.max())

    def params_from(x):  # x: the mesh box's left, right, top, bottom in mask pixels
        sz = (x[1] - x[0]) / (hi[2] - lo[2])
        sy = (x[3] - x[2]) / (hi[1] - lo[1])
        return sz, x[0] - sz * lo[2], sy, x[2] + sy * hi[1]

    def placed(x):
        sz, oz, sy, oy = params_from(x)
        # mask pixel (u, v) -> silhouette pixel ((u - oz)/sz - lo_z) S, (hi_y - (oy - v)/sy) S
        mat = np.diag([S / sy, S / sz])
        off = [(hi[1] - oy / sy) * S, (-oz / sz - lo[2]) * S]
        return ndimage.affine_transform(sil.astype(float), mat, off, output_shape=(h, w), order=1) > 0.5

    def cost(x):
        q = placed(x)
        return -(q & m).sum() / max((q | m).sum(), 1)

    x0 = np.array(box, dtype=float)
    r = optimize.minimize(cost, x0, method="Nelder-Mead", options={"xatol": 0.3, "fatol": 1e-4, "maxiter": 400,
                                                                   "initial_simplex": x0 + np.vstack([np.zeros(4), np.eye(4) * 6])})
    sz, oz, sy, oy = params_from(r.x)
    return (sz / k, oz / k, sy / k, oy / k), float(-r.fun), placed(r.x), m


# the warp's landmarks: the near hooves only (the far legs hide behind the near ones in a side view), no
# croup (the trunk's top is flat: its highest point wanders along the back)
LANDMARKS = ("muzzle_front", "poll", "chin", "throat", "chest_front", "buttock", "tail_top_visible",
             "fore_near_hoof", "hind_near_hoof")


def mesh_landmarks(p, weights):
    """The side-view landmarks on the mesh, (z, y) each, from its vanilla-skeleton weights: the head's
    front, top, bottom front and bottom back; the trunk's front, back and top; the tail's top; each
    hoof's lowest point (near = the left legs, +X)."""
    dom = [max(w, key=w.get) if w else "" for w in weights]
    dom = np.array(dom)
    def pts(*bones):
        return p[np.isin(dom, bones)]
    out = {}
    head = pts("HorseHead")
    if len(head):
        zc, yc = head[:, 2].mean(), head[:, 1].mean()
        f = head[head[:, 2] <= zc]
        r = head[head[:, 2] > zc]
        out["muzzle_front"] = head[head[:, 2].argmin()]
        out["poll"] = head[head[:, 1].argmax()]
        out["chin"] = f[f[:, 1].argmin()]
        lo = r[r[:, 1] < yc]
        if len(lo):
            out["throat"] = lo[lo[:, 2].argmax()]
    trunk = pts("HorseNeckBase", "HorseSpineTop", "HorseSpineMiddle", "HorseSpineEnd")
    if len(trunk):
        nb = pts("HorseNeckBase", "HorseSpineTop")
        y0 = np.percentile(nb[:, 1], 30)
        low = nb[nb[:, 1] <= y0]
        out["chest_front"] = low[low[:, 2].argmin()] if len(low) else nb[nb[:, 2].argmin()]
        back = pts("HorseSpineEnd")
        out["buttock"] = back[back[:, 2].argmax()]
        out["croup"] = back[back[:, 1].argmax()]
    tail = pts("HorseTailMiddle", "HorseTailTip")
    if len(tail):
        out["tail_top_visible"] = tail[tail[:, 1].argmax()]
    for key, leg in (("fore_near_hoof", "FrontLeft"), ("hind_near_hoof", "BackLeft"),
                     ("fore_far_hoof", "FrontRight"), ("hind_far_hoof", "BackRight")):
        q = pts(*(f"Horse{leg}{b}" for b in ("Down", "Heel", "Knee")))
        if len(q):
            out[key] = q[q[:, 1].argmin()]
    return {k: (float(v[2]), float(v[1])) for k, v in out.items()}


def warp(fitp, mesh_lm, image_lm):
    """(z, y) -> image (u, v): a thin-plate spline through the shared landmarks (plus the silhouette
    fit's box corners, so it stays sane away from them); the fit alone without landmarks."""
    sz, oz, sy, oy = fitp
    aff = lambda z, y: (sz * z + oz, -sy * y + oy)  # noqa: E731
    keys = [k for k in LANDMARKS if k in mesh_lm and k in image_lm]
    if len(keys) < 4:
        return lambda z, y: aff(z, y), keys
    from scipy.interpolate import RBFInterpolator
    src = np.array([mesh_lm[k] for k in keys])
    dst = np.array([image_lm[k] for k in keys], dtype=np.float64)
    # anchors: the box corners, mapped by the fit (they hold the far regions to the overall fit)
    zl, zh, yl, yh = src[:, 0].min(), src[:, 0].max(), src[:, 1].min(), src[:, 1].max()
    pad_z, pad_y = (zh - zl) * 0.4, (yh - yl) * 0.4
    corners = np.array([(z, y) for z in (zl - pad_z, zh + pad_z) for y in (yl - pad_y, yh + pad_y)])
    src = np.vstack([src, corners])
    dst = np.vstack([dst, np.stack(aff(corners[:, 0], corners[:, 1]), 1)])
    f = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=1.0)
    return (lambda z, y: tuple(f(np.stack([z, y], 1)).T)), keys


def bleed(img, filled, iters=None):
    """Fill the empty texels with their nearest filled one (islands bled outwards)."""
    idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


def main(args_path):
    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    name, size = a["name"], int(a.get("size", 1024))
    p, n, uv, tri = load_geo(a["geometry"])
    img = np.asarray(Image.open(a["image"]).convert("RGB")).astype(np.float64)
    mask = np.asarray(Image.open(a["mask"]).convert("L")) > 127
    H, W = mask.shape
    (sz, oz, sy, oy), iou, placed, m_small = fit(p, tri, mask)
    # 2. texels -> surface points
    uvp = uv * size  # the JSON keeps the image convention (v down): its row
    cov, tid, bary = raster(uvp, tri, size, size)
    T = tri[tid[cov]]
    B = bary[cov][..., None]
    P = (p[T] * B).sum(1)
    N = (n[T] * B).sum(1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    lm_img = a.get("landmarks") or {}
    lm_mesh = mesh_landmarks(p, json.loads(Path(a["geometry"]).read_text(encoding="utf-8"))["weights"]) if lm_img else {}
    to_img, used = warp((sz, oz, sy, oy), lm_mesh, lm_img)
    u, v = to_img(P[:, 2], P[:, 1])
    u, v = np.asarray(u), np.asarray(v)
    er = ndimage.binary_erosion(mask, iterations=int(a.get("erode", 3)))
    ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & er[vi, ui]
    facing = np.abs(N[:, 0])
    seen = inside & (facing >= float(a.get("min_facing", 0.35)))
    col = np.zeros((len(P), 3))
    for c in range(3):  # bilinear
        col[:, c] = ndimage.map_coordinates(img[..., c], [np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], order=1)
    lab = None
    if a.get("labels"):
        L = np.asarray(Image.open(a["labels"]).convert("RGB")).astype(np.int64)
        lab = L[vi, ui]
    # 3. the unseen take their nearest seen texel in 3D (mirrored: |x|, so either side's seen texels count)
    Q = P * np.array([0.0, 1.0, 1.0]) + np.abs(P[:, :1]) * np.array([1.0, 0.0, 0.0])
    # (the mean of the ``fill_k`` nearest, weighted by distance: a soft colour from the matching region, not
    # one texel's content dragged over the gap; the label is the nearest one's)
    tree = cKDTree(Q[seen])
    k = int(a.get("fill_k", 48))
    dist, nn = tree.query(Q[~seen], k=k)
    idx = np.nonzero(seen)[0]
    w = 1.0 / (dist + 0.05)
    col[~seen] = (col[idx[nn]] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    if lab is not None:
        lab[~seen] = lab[idx[nn[:, 0]]]
    # 4. the maps
    diff = np.zeros((size, size, 3))
    diff[cov] = col
    diff = bleed(diff, cov)
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(out / f"{name}_diffuse.png")
    gloss_img = np.full((size, size), 5.0)
    shares = {}
    if lab is not None:
        names = list(a["classes"])
        pal = np.array([a["classes"][k] for k in names])
        d = ((lab[:, None, :] - pal[None]) ** 2).sum(2)
        cls = d.argmin(1)
        g = np.array([a.get("gloss", {}).get(k, 5) for k in names], dtype=np.float64)[cls]
        gloss_img[cov] = g
        gloss_img = bleed(gloss_img[..., None], cov)[..., 0]
        shares = {names[k]: round(float((cls == k).mean()), 3) for k in np.unique(cls)}
    spec = np.zeros((size, size, 4), dtype=np.uint8)
    spec[..., 3] = np.clip(gloss_img, 0, 255).astype(np.uint8)
    Image.fromarray(spec, "RGBA").save(out / f"{name}_specular.png")
    seen_img = np.zeros((size, size, 3), dtype=np.uint8)
    seen_img[cov] = np.where(seen[:, None], [60, 200, 60], [220, 60, 60])
    Image.fromarray(seen_img).save(out / f"{name}_seen.png")
    over = np.zeros(placed.shape + (3,), dtype=np.uint8)
    over[m_small] = (0, 120, 255)
    over[placed] = over[placed] // 2 + np.array([127, 60, 0], dtype=np.uint8)
    Image.fromarray(over).save(out / f"{name}_fit.png")
    res = {"diffuse": str(out / f"{name}_diffuse.png"), "specular": str(out / f"{name}_specular.png"),
           "fit": {"sz": sz, "oz": oz, "sy": sy, "oy": oy, "iou": round(iou, 3)}, "warp_landmarks": used,
           "mesh_landmarks": lm_mesh,
           "texels": int(cov.sum()), "seen_share": round(float(seen.mean()), 3), "class_share": shares,
           "debug": {"seen": str(out / f"{name}_seen.png"), "fit": str(out / f"{name}_fit.png")}}
    Path(a["result_path"]).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
