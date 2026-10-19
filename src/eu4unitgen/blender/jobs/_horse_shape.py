"""Reshaping a generated horse onto the vanilla horse's rest pose (Track C, test 15; Blender jobs
only: numpy and mathutils). PDX mesh space: Y up, the horse facing -Z, its left side at +X.

The mesh is moved to fit the bones, never the bones to the mesh:
  * legs: each leg's centreline, tracked up from the hoof slice by slice inside its own cross
    section (skirts further out are ignored), maps by distance from the ground onto the vanilla
    leg chain (ground under the heel -> Heel -> Knee -> Down -> Up): bent or striding legs come
    out straight, at their own length;
  * head and neck: the same feature finder on both meshes (the head's principal axis, its back
    third's centre = the poll region, the neck centreline): our head is turned and moved onto
    vanilla's, the neck maps by fraction of its length;
  * torso: anchors that stay put.
The landmark pairs drive a 3D biharmonic spline (phi(r) = r plus an affine part), applied to every
vertex. Diagnostics (landmarks, per-leg straightening) come back for the job's result.
"""
import math

import numpy as np

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")


def verts(obj) -> np.ndarray:
    a = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def faces(obj) -> np.ndarray:
    """Triangles (m x 3) of ``obj``'s mesh (its faces fanned)."""
    out = []
    for poly in obj.data.polygons:
        v = poly.vertices
        out += [(v[0], v[i], v[i + 1]) for i in range(1, len(v) - 1)]
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def set_verts(obj, co: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
    obj.data.update()


def smooth(t):
    """Smoothstep of ``t`` clipped to 0..1 (arrays)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def steepness(before: np.ndarray, after: np.ndarray, obj) -> dict:
    """How abruptly a warp moved neighbouring vertices: per edge of ``obj``, |difference of the two
    ends' displacements| / edge length. Over 1 the edge's ends can cross (a fold or a crack on the
    fine surface, which the reduction hides and the normal bake shows). -> {"p999", "max", "over_1"}."""
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    d = after - before
    length = np.linalg.norm(before[e[:, 0]] - before[e[:, 1]], axis=1)
    e = e[length > 0.05 * np.median(length)]  # sliver edges of the voxel solid: any move looks steep
    k = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / np.linalg.norm(before[e[:, 0]] - before[e[:, 1]], axis=1)
    w = int(np.argmax(k))
    return {"p999": round(float(np.percentile(k, 99.9)), 3), "max": round(float(k[w]), 3),
            "over_1": int((k > 1.0).sum()), "worst_at": [round(float(x), 2) for x in before[e[w, 0]]]}


def chain(bones: dict, names: list[str]) -> np.ndarray:
    return np.array([bones[n] for n in names], dtype=np.float64)


def leg_chain(bones: dict, leg: str) -> np.ndarray:
    """Vanilla leg from the ground under the heel up to the leg's top joint."""
    heel = np.array(bones[f"Horse{leg}Heel"], dtype=np.float64)
    ground = np.array([heel[0], 0.0, heel[2]])
    return np.vstack([ground, chain(bones, [f"Horse{leg}{p}" for p in ("Heel", "Knee", "Down", "Up")])])


def along(poly: np.ndarray, s: float) -> np.ndarray:
    """The point at arc length ``s`` along ``poly`` (clamped; past the end it continues straight)."""
    seg = np.diff(poly, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    if s <= 0:
        return poly[0].copy()
    for p, d, l in zip(poly[:-1], seg, lens):
        if s <= l:
            return p + d * (s / l)
        s -= l
    return poly[-1] + seg[-1] / max(lens[-1], 1e-9) * s


def arc(poly: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def track_leg(co: np.ndarray, leg: str, mid_z: float, top_y: float, step: float = 0.35, radius: float = 0.8):
    """The leg's centreline from its hoof up to ``top_y``: per slice of height ``step``, the
    centre of the vertices within ``radius`` (in x, z) of the slice below's centre. Stops where
    the cross-section gets wider than a leg (a skirt or the body). -> (points, why it stopped)."""
    left, front = leg.endswith("Left"), leg.startswith("Front")
    side = co[((co[:, 0] > 0) == left) & ((co[:, 2] < mid_z) == front)]
    if not len(side):
        return np.zeros((0, 3)), "no vertices"
    floor = side[:, 1].min()
    low = side[side[:, 1] < floor + 0.5]
    centre = low[:, [0, 2]].mean(axis=0)
    pts = [np.array([centre[0], floor, centre[1]])]
    y = floor + step
    while y < top_y:
        sl = side[(side[:, 1] >= y - step / 2) & (side[:, 1] < y + step / 2)]
        near = sl[np.linalg.norm(sl[:, [0, 2]] - centre, axis=1) < radius]
        if len(near) < 4:
            return np.array(pts), f"lost at y {y:.2f}"
        spread = np.linalg.norm(near[:, [0, 2]] - near[:, [0, 2]].mean(axis=0), axis=1).max()
        if spread > radius * 0.98 and y > floor + 2.0:
            return np.array(pts), f"widens at y {y:.2f}"
        centre = near[:, [0, 2]].mean(axis=0)
        pts.append(np.array([centre[0], y, centre[1]]))
        y += step
    return np.array(pts), "reached the top"


def head_features(co: np.ndarray, neck_base: np.ndarray, depth: float = 3.5):
    """(head centre, head axis pointing to the muzzle, poll centre, neck centreline) of a horse
    mesh: the head = the front ``depth`` units above y 8; the axis = its principal direction,
    turned to point forward; the poll = the centre of the head's back third along that axis; the
    neck = per 0.5 along z from the neck base to the poll, the centre of the upper cross-section."""
    up = co[co[:, 1] > 8.0]
    front = up[:, 2].min()
    head = up[up[:, 2] < front + depth]
    centre = head.mean(axis=0)
    _, _, vt = np.linalg.svd(head - centre, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    if axis[2] > 0:
        axis = -axis
    t = (head - centre) @ axis
    poll = head[t < t.min() + (t.max() - t.min()) / 3].mean(axis=0)
    neck = []
    for z in np.arange(neck_base[2], poll[2], -0.5):
        sl = co[(np.abs(co[:, 2] - z) < 0.25) & (co[:, 1] > neck_base[1] - 2.5) & (np.abs(co[:, 0]) < 1.6)]
        if len(sl):
            top = sl[sl[:, 1] > sl[:, 1].max() - 2.0]
            neck.append(top.mean(axis=0))
    return centre, axis, poll, np.array(neck)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 3x3 rotation turning unit vector ``a`` onto unit vector ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def spline(src: np.ndarray, dst: np.ndarray):
    """3D biharmonic spline (phi(r) = r) with an affine part, mapping ``src`` onto ``dst``."""
    n = len(src)
    k = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    p = np.hstack([np.ones((n, 1)), src])
    a = np.zeros((n + 4, n + 4))
    a[:n, :n], a[:n, n:], a[n:, :n] = k + np.eye(n) * 1e-6, p, p.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = dst
    sol = np.linalg.solve(a, rhs)
    w, aff = sol[:n], sol[n:]

    def apply(x: np.ndarray, chunk: int = 40000) -> np.ndarray:
        out = np.empty_like(x)
        for i in range(0, len(x), chunk):
            c = x[i:i + chunk]
            out[i:i + chunk] = np.linalg.norm(c[:, None, :] - src[None, :, :], axis=2) @ w + \
                np.hstack([np.ones((len(c), 1)), c]) @ aff
        return out
    return apply


def reshape(obj, vanilla, bones: dict, leg_top: float = 6.0, anchors: int = 60, seed: int = 0) -> dict:
    """Warp ``obj`` onto the vanilla rest pose (module doc). -> diagnostics."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    co, vco = verts(obj), verts(vanilla)
    mid_z = (B["HorseFrontLeftUp"][2] + B["HorseBackLeftUp"][2]) / 2
    src, dst, log = [], [], {"legs": {}}
    for leg in LEGS:
        pts, why = track_leg(co, leg, mid_z, leg_top)
        target = leg_chain(B, leg)
        if len(pts) < 3:
            log["legs"][leg] = {"tracked": int(len(pts)), "stopped": why}
            continue
        s = arc(pts)
        for i in range(0, len(pts), 2):
            src.append(pts[i])
            dst.append(along(target, s[i]))
        tip = pts[0]
        log["legs"][leg] = {"tracked": int(len(pts)), "stopped": why, "hoof": tip.round(2).tolist(),
                            "hoof_to": along(target, 0).round(2).tolist(), "top_y": float(pts[-1][1]),
                            "max_move": float(max(np.linalg.norm(along(target, s[i]) - pts[i]) for i in range(len(pts))))}
    oc, oaxis, opoll, oneck = head_features(co, B["HorseNeckBase"])
    vc, vaxis, vpoll, vneck = head_features(vco, B["HorseNeckBase"])
    r = rotation_between(oaxis, vaxis)
    head = co[(co[:, 1] > 8.0) & (co[:, 2] < co[co[:, 1] > 8.0][:, 2].min() + 3.5)]
    rng = np.random.default_rng(seed)
    for p in head[rng.choice(len(head), size=min(12, len(head)), replace=False)]:
        src.append(p)
        dst.append(vpoll + r @ (p - opoll))
    log["head"] = {"turn_deg": float(math.degrees(math.acos(max(-1, min(1, float(oaxis @ vaxis)))))),
                   "poll": opoll.round(2).tolist(), "poll_to": vpoll.round(2).tolist()}
    if len(oneck) >= 2 and len(vneck) >= 2:
        os_, vs_ = arc(oneck), arc(vneck)
        for f in (0.25, 0.5, 0.75):
            src.append(along(oneck, f * os_[-1]))
            dst.append(along(vneck, f * vs_[-1]))
    body = co[(co[:, 2] > B["HorseNeckBase"][2] + 0.5) & (co[:, 2] < B["HorseSpineEnd"][2] + 2.0) & (co[:, 1] > leg_top + 1.5)]
    for p in body[rng.choice(len(body), size=min(anchors, len(body)), replace=False)]:
        src.append(p)
        dst.append(p)
    warp = spline(np.array(src), np.array(dst))
    new = warp(co)
    set_verts(obj, new)
    log["landmarks"] = len(src)
    log["max_vertex_move"] = float(np.linalg.norm(new - co, axis=1).max())
    log["head_after"] = {"poll": head_features(new, B["HorseNeckBase"])[2].round(2).tolist()}
    return log


def _span(co, tris, point, direction, reach=10.0):
    """(start, end) along ``direction`` (as offsets from ``point``) of the solid nearest ``point``
    on a ray through it, or None."""
    import _horse_measure as hm
    hits = hm.ray_hits(co, tris, point - direction * reach, direction) - reach
    pairs = [(hits[i], hits[i + 1]) for i in range(0, len(hits) - 1, 2)]
    if not pairs:
        return None
    return min(pairs, key=lambda p: 0.0 if p[0] <= 0 <= p[1] else min(abs(p[0]), abs(p[1])))


def neck_warp(obj, vanilla, bones: dict, allow: float = 1.15, samples=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85)) -> dict:
    """The neck thinned towards vanilla's: at fractions of NeckBase -> HorseHead, the thickness
    perpendicular to the neck (in the vertical plane) is measured by a ray on both meshes; where
    ours is thicker than vanilla's x ``allow``, that cross-section is scaled about its own centre
    (never thickened). The ratio fades to 1 a quarter of the neck behind the base and at the head,
    and the shift fades out from 3 to 4.5 off the neck's axis. -> {fraction: [ours, vanilla,
    ratio]}, "steepness"."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    base, head = B["HorseNeckBase"], B["HorseHead"]
    axis = head - base
    length = float(np.linalg.norm(axis))
    axis /= length
    up = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    up = up if up[1] > 0 else -up
    co, tris = verts(obj), faces(obj)
    vco, vtri = verts(vanilla), faces(vanilla)
    ts, ratios, centres, log = [-0.25], [1.0], [0.0], {}
    for t in samples:
        point = base + axis * t * length
        o, v = _span(co, tris, point, up), _span(vco, vtri, point, up)
        if o is None or v is None:
            continue
        ratio = min(1.0, allow * (v[1] - v[0]) / (o[1] - o[0]))
        ts.append(t)
        ratios.append(ratio)
        centres.append((o[0] + o[1]) / 2)
        log[t] = [round(float(o[1] - o[0]), 3), round(float(v[1] - v[0]), 3), round(ratio, 3)]
    ts.append(1.05)
    ratios.append(1.0)
    centres.append(centres[-1])
    rel = co - base
    tv = (rel @ axis) / length
    radial = rel - np.outer(rel @ axis, axis)
    u = radial @ up
    near = (tv > -0.25) & (tv < 1.05)
    fade = 1.0 - smooth((np.linalg.norm(radial, axis=1) - 3.0) / 1.5)  # no hard edge round the neck
    r = np.interp(tv, ts, ratios)
    c = np.interp(tv, ts, centres)
    shift = np.where(near, (c + (u - c) * r) - u, 0.0) * fade
    set_verts(obj, co + shift[:, None] * up)
    log["steepness"] = steepness(co, verts(obj), obj)
    return log


def _leg_profile(points, chain_pts, ys, reach, slab):
    """Per height in ``ys``: (centre (x, z), mean radius) of the leg's slice in ``points``, found
    from the vanilla chain's point at that height by recentring; NaN where no leg is found."""
    out = np.full((len(ys), 3), np.nan)
    for i, y in enumerate(ys):
        sl = points[np.abs(points[:, 1] - y) < slab]
        k = int(np.clip(np.searchsorted(chain_pts[:, 1], y), 1, len(chain_pts) - 1))
        p0, p1 = chain_pts[k - 1], chain_pts[k]
        t = (y - p0[1]) / max(p1[1] - p0[1], 1e-9)
        centre = (p0 + (p1 - p0) * np.clip(t, 0, 1))[[0, 2]]
        for _ in range(4):
            near = sl[np.linalg.norm(sl[:, [0, 2]] - centre, axis=1) < reach]
            if len(near) < 8:
                break
            centre = (near[:, [0, 2]].min(axis=0) + near[:, [0, 2]].max(axis=0)) / 2
        else:
            out[i] = (centre[0], centre[1], np.linalg.norm(near[:, [0, 2]] - centre, axis=1).mean())
    return out


def leg_inflate(obj, vanilla, bones: dict, top: float, allow: float = 0.9, step: float = 0.25,
                reach: float = 0.85, most: float = 2.5, band: float = 2.0) -> dict:
    """Thin legs thickened towards vanilla's: per leg and height (ground to ``top``, where the
    legs meet the body), the mean radius of our slice and of vanilla's (surface samples, so
    vanilla's sparse vertices don't matter); where ours is under vanilla's x ``allow``, the slice's
    slice moves out from its centre (x, z): a vertex at distance d inside our radius r0 by
    (ratio - 1) x d, outside it by (ratio - 1) x r0 fading to nothing over ``band`` x that push
    (wider than the push, so neighbours never cross: no fold where the leg meets the panels or
    the body). Ratio at most ``most``, never thinning; it fades to 1 over the top 0.8 so the legs
    still meet the body. -> per leg: the thinnest ratio before; how many vertices moved; steepness."""
    import _horse_measure as hm
    co, vco = verts(obj), verts(vanilla)
    vs = hm.surface_samples(vco, faces(vanilla))
    os_ = hm.surface_samples(co, faces(obj), n=200000)
    ys = np.arange(0.2, top, step)
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    before = co.copy()
    best = np.full(len(co), np.inf)
    push = np.zeros(len(co))
    dirs = np.zeros((len(co), 2))
    log = {}
    for leg in LEGS:
        chain_pts = leg_chain(B, leg)[::-1][::-1]
        chain_pts = chain_pts[np.argsort(chain_pts[:, 1])]
        ours, van = _leg_profile(os_, chain_pts, ys, reach, 0.1), _leg_profile(vs, chain_pts, ys, reach, 0.1)
        ok = ~np.isnan(ours[:, 2]) & ~np.isnan(van[:, 2])
        if ok.sum() < 3:
            log[leg] = {"found": int(ok.sum())}
            continue
        ratio = np.clip(allow * van[ok, 2] / ours[ok, 2], 1.0, most)
        ratio = np.convolve(np.pad(ratio, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # smooth along the leg
        fade = np.clip((top - ys[ok]) / 0.8, 0, 1)
        ratio = 1 + (ratio - 1) * fade
        cx, cz = np.interp(co[:, 1], ys[ok], ours[ok, 0]), np.interp(co[:, 1], ys[ok], ours[ok, 1])
        r0 = np.interp(co[:, 1], ys[ok], ours[ok, 2])
        k = np.interp(co[:, 1], ys[ok], ratio) - 1.0
        d = np.hypot(co[:, 0] - cx, co[:, 2] - cz)
        out = k * r0  # the slice surface's push
        far = r0 + band * out
        amount = np.where(d < r0, k * d, out * (1.0 - smooth((d - r0) / np.maximum(far - r0, 1e-6))))
        mine = (d < far) & (co[:, 1] < top) & (d < best) & (amount > 0)
        best[mine] = d[mine]
        push[mine] = amount[mine]
        dirs[mine] = np.stack([co[mine, 0] - cx[mine], co[mine, 2] - cz[mine]], axis=1) / np.maximum(d[mine], 1e-6)[:, None]
        log[leg] = {"found": int(ok.sum()), "thinnest_ratio": round(float((ours[ok, 2] / van[ok, 2]).min()), 3),
                    "most_scale": round(float(ratio.max()), 3)}
    move = push > 0
    co[move, 0] += dirs[move, 0] * push[move]
    co[move, 2] += dirs[move, 1] * push[move]
    set_verts(obj, co)
    log["moved"] = int(move.sum())
    log["steepness"] = steepness(before, co, obj)
    return log


def _half_widths(points, ys, zs, cell):
    """Max |x| per (y, z) cell and side (+x, -x) of ``points``: arrays (2, len(ys), len(zs)), NaN where empty."""
    out = np.full((2, len(ys), len(zs)), np.nan)
    iy = np.round((points[:, 1] - ys[0]) / cell).astype(int)
    iz = np.round((points[:, 2] - zs[0]) / cell).astype(int)
    ok = (iy >= 0) & (iy < len(ys)) & (iz >= 0) & (iz < len(zs))
    for side, sel in ((0, points[:, 0] > 0), (1, points[:, 0] < 0)):
        m = ok & sel
        flat = np.full(len(ys) * len(zs), -np.inf)
        np.maximum.at(flat, iy[m] * len(zs) + iz[m], np.abs(points[m, 0]))
        flat[np.isinf(flat)] = np.nan
        out[side] = flat.reshape(len(ys), len(zs))
    return out


def _fill(grid, default):
    """NaN cells of a (2, ny, nz) grid take their filled neighbours' mean, growing inwards;
    ``default`` where nothing reaches."""
    ny, nz = grid.shape[1:]
    for _ in range(max(ny, nz)):
        empty = np.isnan(grid)
        if not empty.any():
            break
        pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        nb = np.stack([pad[:, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz] for dy, dz in ((0, 1), (0, -1), (1, 0), (-1, 0))])
        with np.errstate(all="ignore"):
            grid = np.where(empty, np.nanmean(nb, axis=0), grid)
    return np.nan_to_num(grid, nan=default)


def _blur(grid, passes):
    """``passes`` 3 x 3 box blurs of a (2, ny, nz) grid (edges repeated)."""
    ny, nz = grid.shape[1:]
    for _ in range(passes):
        pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)), mode="edge")
        grid = sum(pad[:, i:i + ny, j:j + nz] for i in range(3) for j in range(3)) / 9.0
    return grid


def _sample(grid, co, ys, zs, cell, side=None):
    """Bilinear lookup of a (2, ny, nz) grid at each vertex's (y, z), on its side (0: +x, 1: -x;
    ``side`` overrides), clamped to the grid."""
    fy = np.clip((co[:, 1] - ys[0]) / cell, 0, len(ys) - 1)
    fz = np.clip((co[:, 2] - zs[0]) / cell, 0, len(zs) - 1)
    y0i, z0i = np.floor(fy).astype(int), np.floor(fz).astype(int)
    y1i, z1i = np.minimum(y0i + 1, len(ys) - 1), np.minimum(z0i + 1, len(zs) - 1)
    ty, tz = fy - y0i, fz - z0i
    s = (co[:, 0] < 0).astype(int) if side is None else side
    return (grid[s, y0i, z0i] * (1 - ty) * (1 - tz) + grid[s, y1i, z0i] * ty * (1 - tz) +
            grid[s, y0i, z1i] * (1 - ty) * tz + grid[s, y1i, z1i] * ty * tz)


def flank_warp(obj, vanilla, bones: dict, allow: float = 1.05, y0: float = 6.0, y1: float = 13.0,
               cell: float = 0.5, least: float = 0.6) -> dict:
    """Each flank narrowed towards vanilla's half-width, side by side: the generated torso can be
    wider on one side (a 3/4-view reference: the near side's caparison flares) and wider than
    vanilla's where the rider's legs hang. Per (y, z) cell of the torso (neck base to behind the
    rump, ``y0``-``y1``) and side, our max |x| against vanilla's (both sides' mean; its thin
    stirrups left out); where ours is over vanilla's x ``allow`` that side's x is scaled down (to
    ``least`` at most; never widened). Cells without both surfaces take their neighbours' ratio and
    the grid is smoothed; vertices near the leg chains below y 9 keep their x (the legs are on
    their chains), fading out over 0.8 and over y 9-10. -> the narrowest ratio per side, steepness."""
    import _horse_measure as hm
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    co = verts(obj)
    ours = hm.surface_samples(co, faces(obj), n=250000)
    van = hm.surface_samples(verts(vanilla), faces(vanilla), n=120000)
    stir = (np.abs(van[:, 0]) > 2.9) & (van[:, 1] > 7.5) & (van[:, 1] < 10.5) & (van[:, 2] > -4.0) & (van[:, 2] < 0.5)
    van = van[~stir]
    ys = np.arange(y0, y1 + cell / 2, cell)
    zs = np.arange(B["HorseNeckBase"][2], B["HorseSpineEnd"][2] + 2.5 + cell / 2, cell)
    o = _half_widths(ours, ys, zs, cell)
    v = np.nanmean(_half_widths(van, ys, zs, cell), axis=0)
    with np.errstate(all="ignore"):
        ratio = np.clip(allow * v[None] / o, least, 1.0)
    # cells without both surfaces take their neighbours' ratio (a 1.0 there would make a step
    # next to a narrowed cell)
    ratio = _blur(_fill(ratio, 1.0), 3)
    r = _sample(ratio, co, ys, zs, cell)
    # faded in below y0 and beyond both ends of the grid along z (a hard edge there is a step)
    inside = smooth((co[:, 1] - (y0 - 1.0)) / 1.0) * smooth((co[:, 2] - (zs[0] - 1.5)) / 1.5) * \
        smooth(((zs[-1] + 1.5) - co[:, 2]) / 1.5)
    r = 1 + (r - 1) * inside
    keep = (1.0 - smooth((hm.leg_distance(co, bones) - 0.8) / 0.8)) * (1.0 - smooth((co[:, 1] - 9.0) / 1.0))
    r = 1 + (r - 1) * (1.0 - keep)  # the legs keep their x, fading out (no step at the leg's edge)
    before = co.copy()
    co[:, 0] *= r
    set_verts(obj, co)
    return {"narrowest": {"left": round(float(ratio[0].min()), 3), "right": round(float(ratio[1].min()), 3)},
            "moved": int((r < 0.999).sum()), "steepness": steepness(before, co, obj)}


def flatten_lump(obj, z0: float, z1: float, y0: float, y1: float, margin: float = 0.3,
                 cell: float = 0.5, fade: float = 1.5) -> dict:
    """Vertices in the box on either flank sticking out past the other flank's width at the same
    (y, z) (a rider's leg on one side) are pushed in to within ``margin`` of it (soft: the excess
    is compressed, not cut, so there is no crease). The widths are a grid (max |x| per ``cell`` and side, gaps filled, blurred) read bilinearly,
    and the limit rises away over ``fade`` at the box's edges: a smooth wall, not cell-sized
    terraces (which the normal bake shows). No vertex is deleted, so no hole opens.
    -> how many moved, steepness."""
    co = verts(obj)
    ys = np.arange(y0 - fade, y1 + fade + cell / 2, cell)
    zs = np.arange(z0 - fade, z1 + fade + cell / 2, cell)
    width = _blur(_fill(_half_widths(co, ys, zs, cell), 0.0), 2)
    other = _sample(width, co, ys, zs, cell, side=(co[:, 0] >= 0).astype(int))  # the opposite side's
    inside = smooth((co[:, 1] - y0) / fade) * smooth((y1 - co[:, 1]) / fade) * \
        smooth((co[:, 2] - z0) / fade) * smooth((z1 - co[:, 2]) / fade)
    before = co.copy()
    over = (np.abs(co[:, 0]) > other) & (inside > 0)
    # a soft clamp: the excess e past the other side's width becomes margin x (1 - exp(-e / margin)),
    # tangent to no change at the limit (a hard clamp leaves a crease where the wall begins); the
    # move itself fades out at the box's edges
    excess = np.abs(co[over, 0]) - other[over]
    to = np.sign(co[over, 0]) * (other[over] + margin * (1.0 - np.exp(-excess / margin)))
    co[over, 0] += (to - co[over, 0]) * inside[over]
    set_verts(obj, co)
    return {"vertices": int(over.sum()), "steepness": steepness(before, co, obj)}


def nearest_on_chains(co: np.ndarray, chains: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(distance, nearest point) of each vertex to the polylines."""
    best = np.full(len(co), np.inf)
    point = np.zeros_like(co)
    for poly in chains:
        for a, b in zip(poly[:-1], poly[1:]):
            d = b - a
            t = np.clip(((co - a) @ d) / max(d @ d, 1e-12), 0, 1)
            q = a + t[:, None] * d
            dist = np.linalg.norm(co - q, axis=1)
            closer = dist < best
            best[closer], point[closer] = dist[closer], q[closer]
    return best, point


def near_chains(co: np.ndarray, chains: list[np.ndarray], radius: float) -> np.ndarray:
    """Mask: vertices within ``radius`` of any of the polylines."""
    return nearest_on_chains(co, chains)[0] < radius


def bone_segments(bones: dict) -> dict:
    """Each deforming bone's segment (from the bone to its child; the chain ends extended: the
    hoof to the ground, the head to the muzzle, the tail end past its tip)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = {}
    for leg in LEGS:
        names = [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")]
        for a, b in zip(names[:-1], names[1:]):
            seg[a] = (B[a], B[b])
        heel = B[names[-1]]
        seg[names[-1]] = (heel, np.array([heel[0], 0.0, heel[2]]))
    neck = ["HorseNeckBase", "HorseNeckMiddle", "HorseHead"]
    for a, b in zip(neck[:-1], neck[1:]):
        seg[a] = (B[a], B[b])
    head = B["HorseHead"]
    seg["HorseHead"] = (head, head + np.array([0.0, -1.5, -3.3]))  # towards vanilla's muzzle
    tail = ["HorseSpineEnd", "HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd"]
    for a, b in zip(tail[1:-1], tail[2:]):
        seg[a] = (B[a], B[b])
    end, tip = B["HorseTailEnd"], B["HorseTailTip"]
    seg["HorseTailEnd"] = (end, end + (end - tip) * 0.5)
    return seg


def segment_distance(co: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((co - a) @ d) / max(d @ d, 1e-12), 0, 1)
    return np.linalg.norm(co - (a + t[:, None] * d), axis=1)


def chain_weights(co: np.ndarray, bones: dict, power: float = 4.0) -> tuple[list, dict]:
    """Per vertex: weights by distance to the bone chains (head and neck, each leg, the tail) in
    their regions, else None (the torso keeps nearest-surface weights). -> (rows, region counts).
    Each region includes its spine parent along the segment to the chain's first joint, so the
    weights blend into the torso instead of stepping at the region's border. Regions: head/neck
    in front of 0.8 behind the neck base, above y 9; a leg below y 9.3 within 0.9 of its chain
    (1.4 above y 7.5);
    the tail on the centre line (|x| < 0.7) behind the rump within 1.0 of its chain."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = bone_segments(bones)
    regions = {"head": ("HorseSpineTop", "HorseNeckBase", ["HorseNeckBase", "HorseNeckMiddle", "HorseHead"]),
               "tail": ("HorseSpineEnd", "HorseTailTop", ["HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd"])}
    for leg in LEGS:
        parent = "HorseSpineTop" if leg.startswith("Front") else "HorseSpineEnd"
        regions[leg] = (parent, f"Horse{leg}Up", [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")])
    rows, counts = [None] * len(co), {}
    for region, (parent, first, names) in regions.items():
        dist = {b: segment_distance(co, *seg[b]) for b in names}
        dist[parent] = segment_distance(co, B[parent], B[first])
        chain_min = np.min([dist[b] for b in names], axis=0)
        if region == "head":
            mask = (co[:, 2] < B["HorseNeckBase"][2] + 0.8) & (co[:, 1] > 9.0)
        elif region == "tail":
            mask = (np.abs(co[:, 0]) < 0.7) & (co[:, 2] > B["HorseSpineEnd"][2] + 1.5) & (chain_min < 1.0)
        else:  # the leg itself below the elbow or stifle, the shoulder or hip above it
            mask = (co[:, 1] < 9.3) & (chain_min < np.where(co[:, 1] < 7.5, 0.9, 1.4))
        idx = np.nonzero(mask)[0]
        counts[region] = int(len(idx))
        if not len(idx):
            continue
        order = [parent] + names
        d = np.stack([dist[b][idx] for b in order], axis=1)
        w = 1.0 / np.maximum(d, 0.05) ** power
        for row, i in zip(w, idx):
            if rows[i] is None:  # first region wins
                top = np.argsort(-row)[:3]
                total = row[top].sum()
                rows[i] = {order[k]: float(row[k] / total) for k in top}
    return rows, counts


TAIL = ("HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd")


def clean_surface_rows(co: np.ndarray, rows: list, chain_rows: list, bones: dict, far: float = 1.0) -> int:
    """Nearest-surface weights of the torso (vertices without chain weights) cleaned of bones
    that only fit vanilla's own shape: the tail (our rump sits where vanilla's tail is: the
    caparison follows the hindquarters), the reins (``Rem``) and a leg's lower bones on vertices
    farther than ``far`` from that leg's chain (a skirt doesn't swing with the leg). Emptied
    rows fall back to the nearest spine bone. -> how many rows changed. In place."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = bone_segments(bones)
    lower = {}
    for leg in LEGS:
        d = np.min([segment_distance(co, *seg[f"Horse{leg}{p}"]) for p in ("Down", "Knee", "Heel")], axis=0)
        for p in ("Down", "Knee", "Heel"):
            lower[f"Horse{leg}{p}"] = d > far
    spine = ("HorseSpineTop", "HorseSpineEnd")
    changed = 0
    for i, row in enumerate(rows):
        if chain_rows[i] is not None or not row:
            continue
        keep = {b: w for b, w in row.items()
                if b not in TAIL and b != "Rem" and not (b in lower and lower[b][i])}
        if len(keep) != len(row):
            changed += 1
            if not keep:
                near = min(spine, key=lambda b: float(np.linalg.norm(co[i] - B[b])))
                keep = {near: 1.0}
            total = sum(keep.values())
            rows[i] = {b: w / total for b, w in keep.items()}
    return changed
