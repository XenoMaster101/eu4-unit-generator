"""A procedural horse body (Track C, test 16 bake-off horse B; Blender jobs only: numpy). PDX mesh
space: Y up, the horse facing -Z, its left side at +X, on the vanilla horse's skeleton.

Our own geometry: smooth lofted tubes along the skeleton (barrel, neck, head, each leg), joined
and voxel-remeshed by fresh_horse. Each tube's cross-section at a station is a 16-direction
outline (the outermost surface in each direction) measured on a template surface - the vanilla
horse, which the skeleton and animations belong to, so the joints bend where the game expects -
then fitted to the reference:
  * ``ratios`` (``fit_ratios``: ours / vanilla horse_proportions of the reference model, clipped
    to ``CLIP``): depth moves the belly (about the back line, so the seat stays at vanilla's),
    width the flanks, neck and leg their tubes' thickness;
  * ``topline``: the reference's back height minus vanilla's per station along the body
    (clipped to +-0.5, smoothed, faded to nothing at the saddle): withers and croup.
Thin template parts that would read as the template's own (reins, stirrups, saddle) are left
out of the samples. The tail isn't built here (fresh_horse's tail rule builds it).
"""
import numpy as np

import _horse_measure as hm

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")
# reference / vanilla ratios are clipped: the generated model's legs are sticks and its fused
# skirts widen the flanks, so those cues are only trusted near 1 (assumed, test 16)
CLIP = {"depth": (0.9, 1.1), "width": (0.85, 1.1), "neck": (0.85, 1.0), "leg": (0.95, 1.1)}


def fit_ratios(ref_co, ref_tris, van_co, van_tris, bones) -> dict:
    """{depth, width, neck, leg}: the reference's horse_proportions over vanilla's, clipped
    (CLIP); 1.0 where a measurement is missing. Also the raw ratios, for the log."""
    ours, van = hm.proportions(ref_co, ref_tris, bones), hm.proportions(van_co, van_tris, bones)
    raw, out = {}, {}
    for k, (lo, hi) in CLIP.items():
        o, v = hm.summary(ours).get(k), hm.summary(van).get(k)
        raw[k] = None if not o or not v else round(float(o / v), 3)
        out[k] = 1.0 if raw[k] is None else float(np.clip(raw[k], lo, hi))
    return {"ratios": out, "raw": raw}


def _frame(t):
    t = t / np.linalg.norm(t)
    side = np.array([1.0, 0.0, 0.0])
    if abs(t @ side) > 0.9:
        side = np.array([0.0, 0.0, 1.0])
    u = side - t * (t @ side)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    if v[1] < 0 and abs(t[1]) < 0.9:
        u, v = -u, -v
    return t, u, v


def _smooth(a, axis, passes=1):
    for _ in range(passes):
        a = (np.roll(a, 1, axis) + 2 * a + np.roll(a, -1, axis)) / 4 if axis == 1 else \
            np.concatenate([a[:1], (a[:-2] + 2 * a[1:-1] + a[2:]) / 4, a[-1:]], axis=0)
    return a


def rings(samples, centres, tangents, nang=16, slab=0.2, reach=3.0, recentre=True):
    """Per station (``centres`` with ``tangents``): the outline of ``samples`` in the plane across
    the axis, as (centre, u, v, radii[nang]) - the outermost sample in each angular bin within
    ``reach``, the centre moved to the slice's middle, empty bins interpolated round the ring.
    None where the slice is empty."""
    out = []
    ang = 2 * np.pi * np.arange(nang) / nang
    for c, t in zip(centres, tangents):
        t, u, v = _frame(t)
        c = np.array(c, dtype=np.float64)
        sl = samples[np.abs((samples - c) @ t) < slab]
        res = None
        for _ in range(3 if recentre else 1):
            d = sl - c
            a, b = d @ u, d @ v
            near = np.hypot(a, b) < reach
            if near.sum() < 8:
                break
            if recentre:
                c = c + u * (a[near].max() + a[near].min()) / 2 + v * (b[near].max() + b[near].min()) / 2
            res = True
        if res is None:
            out.append(None)
            continue
        d = sl - c
        a, b = d @ u, d @ v
        r = np.hypot(a, b)
        near = r < reach
        th = np.mod(np.arctan2(b[near], a[near]), 2 * np.pi)
        k = np.round(th / (2 * np.pi / nang)).astype(int) % nang
        rad = np.full(nang, np.nan)
        np.fmax.at(rad, k, r[near])
        ok = ~np.isnan(rad)
        if ok.sum() < 3:
            out.append(None)
            continue
        idx = np.arange(nang)
        rad = np.interp(idx, np.concatenate([idx[ok] - nang, idx[ok], idx[ok] + nang]), np.tile(rad[ok], 3))
        out.append((c, u, v, rad))
    return out, ang


def tube(stations, ang, caps=True):
    """A closed tube through ``stations`` [(centre, u, v, radii)] -> (vertices, triangles)."""
    nang = len(ang)
    verts, tris = [], []
    for c, u, v, rad in stations:
        for th, r in zip(ang, rad):
            verts.append(c + u * np.cos(th) * r + v * np.sin(th) * r)
    m = len(stations)
    for i in range(m - 1):
        for j in range(nang):
            a, b = i * nang + j, i * nang + (j + 1) % nang
            tris += [(a, b + nang, b), (a, a + nang, b + nang)]
    if caps:
        for i, flip in ((0, True), (m - 1, False)):
            centre = len(verts)
            verts.append(stations[i][0])
            for j in range(nang):
                a, b = i * nang + j, i * nang + (j + 1) % nang
                tris.append((centre, a, b) if flip else (centre, b, a))
    return np.array(verts), np.array(tris)


def _chain_points(pts, step):
    """Points every ``step`` along a polyline (with its ends) and their tangents."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], max(2, int(np.ceil(cum[-1] / step)) + 1))
    p = np.stack([np.interp(s, cum, pts[:, k]) for k in range(3)], axis=1)
    t = np.gradient(p, axis=0)
    return p, t


def build(template_co, template_tris, bones, ratios=None, topline=None, nang=16, step=0.4, tri_keep=None) -> dict:
    """The body's tubes (module doc). ``topline``: [[z, dy]] (the reference's back height minus
    the template's); ``tri_keep``: the template triangles to sample (fresh_horse leaves out the
    reins and the tail by their bones). -> {"tubes": [(vertices, triangles)], "log"}."""
    r = {"depth": 1.0, "width": 1.0, "neck": 1.0, "leg": 1.0, **(ratios or {})}
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    pts = hm.surface_samples(template_co, template_tris if tri_keep is None else template_tris[tri_keep], n=200000)
    # leave out the template's thin tack: reins (above the neck, in front of the withers), the
    # saddle and stirrups (round the seat and hanging off the sides)
    seat = B["Saddle_Node"]
    tack = ((np.abs(pts[:, 2] - seat[2]) < 2.2) & ((pts[:, 1] > seat[1] - 0.6) | (np.abs(pts[:, 0]) > 2.6)))
    pts = pts[~tack]
    leg_d = hm.leg_distance(pts, bones)
    tubes, log = [], {}
    # barrel: along z from the neck base to the rump, sections in the (x, y) plane
    # from the breast (in front of the neck base, where only the chest under the neck counts) to the rump
    zf, zb = B["HorseNeckBase"][2] - 1.6, B["HorseSpineEnd"][2] + 2.9  # empty end slices drop out; further catches the dock
    zs = np.arange(zf, zb + 1e-6, step)
    centres = [np.array([0.0, (B["HorseSpineTop"][1] + B["HorseFrontLeftDown"][1]) / 2, z]) for z in zs]
    body_pts = pts[((leg_d > 0.9) | (pts[:, 1] > 8.6)) &
                   ~((pts[:, 2] < B["HorseNeckBase"][2] + 0.3) & (pts[:, 1] > B["HorseNeckBase"][1] - 1.5))]
    st, ang = rings(body_pts, centres, [np.array([0.0, 0.0, 1.0])] * len(zs), nang, reach=4.0)
    keep = [(z, s) for z, s in zip(zs, st) if s is not None]
    rad = _smooth(np.array([s[3] for _, s in keep]), 0, 1)
    dy = np.zeros(len(keep))
    if topline:
        tl = np.array(topline, dtype=np.float64)
        dy = np.clip(np.interp([z for z, _ in keep], tl[:, 0], tl[:, 1]), -0.5, 0.5)
        dy = _smooth(dy, 0, 2) * (1 - np.exp(-((np.array([z for z, _ in keep]) - seat[2]) / 1.5) ** 2))
    stations = []
    for (z, (c, u, v, _)), rr, d in zip(keep, rad, dy):
        a, b = np.cos(ang) * rr, np.sin(ang) * rr
        top = b.max()
        a = a * r["width"]
        b = np.where(b < 0, top + (b - top) * r["depth"], b)  # the belly moves; the back stays
        b = b + d * np.clip(np.sin(ang), 0, 1)  # the topline: the upper half only
        # back to radii in (u, v): the tube writer takes radii per fixed angle, so keep (a, b) as
        # an ellipse-free outline by passing the points through u, v directly
        stations.append((c, u, v, a, b))
    tubes.append(_outline_tube(stations))
    log["barrel"] = {"stations": len(stations), "z": [round(float(zs[0]), 2), round(float(zs[-1]), 2)]}
    # neck: NeckBase (behind and below, into the barrel) -> NeckMiddle -> Head
    # it stops short of the poll: past it the slices across the neck would cut the jaw
    neck_axis = np.array([B["HorseNeckBase"] + np.array([0, -0.8, 0.8]), B["HorseNeckMiddle"],
                          B["HorseNeckMiddle"] + (B["HorseHead"] - B["HorseNeckMiddle"]) * 0.7])
    p, t = _chain_points(neck_axis, step)
    neck_pts = pts[(pts[:, 2] < B["HorseNeckBase"][2] + 1.5) & (pts[:, 1] > 8.5)]
    st, ang = rings(neck_pts, p, t, nang, reach=2.2)
    st = [s for s in st if s is not None]
    tubes.append(tube([(c, u, v, rad * r["neck"]) for c, u, v, rad in st], ang))
    log["neck"] = {"stations": len(st)}
    # head: HorseHead -> the muzzle (the template's surface point furthest along the head)
    axis0 = np.array([0.0, -1.5, -3.3]) / np.linalg.norm([0.0, -1.5, -3.3])
    head_pts = pts[(pts[:, 2] < B["HorseHead"][2] + 1.0) & (pts[:, 1] > 9.0)]
    tip = head_pts[np.argmax((head_pts - B["HorseHead"]) @ axis0)]
    # from a little behind the poll (overlapping the neck's end) to the muzzle
    head_axis = np.array([B["HorseHead"] - axis0 * 0.5, tip - axis0 * 0.15])
    p, t = _chain_points(head_axis, step * 0.75)
    st, ang = rings(head_pts, p, t, nang, reach=1.5)
    st = [s for s in st if s is not None]
    tubes.append(tube(st, ang))
    log["head"] = {"stations": len(st), "muzzle": [round(float(x), 2) for x in tip]}
    # legs: Up -> Down -> Knee -> Heel -> the ground, each from the samples nearest its own chain
    nearest = _nearest_leg(pts, B)
    for leg in LEGS:
        chain = np.array([B[f"Horse{leg}{j}"] for j in ("Up", "Down", "Knee", "Heel")])
        chain = np.vstack([chain, [chain[-1][0], 0.05, chain[-1][2]]])
        p, t = _chain_points(chain, step * 0.75)
        mine = pts[(nearest == LEGS.index(leg)) & (pts[:, 1] < B[f"Horse{leg}Up"][1] - 0.3)]
        st, ang = rings(mine, p, t, nang, reach=1.0)
        st = [(c, u, v, rad * r["leg"]) for c, u, v, rad in (s for s in st if s is not None)]
        tubes.append(tube(st, ang))
        log[leg] = {"stations": len(st)}
    log["ratios"] = r
    return {"tubes": tubes, "log": log}


def _nearest_leg(pts, B) -> np.ndarray:
    """Per point: the index (LEGS) of the leg chain it is nearest."""
    dist = []
    for leg in LEGS:
        chain = [B[f"Horse{leg}{j}"] for j in ("Up", "Down", "Knee", "Heel")]
        chain.append(np.array([chain[-1][0], 0.0, chain[-1][2]]))
        best = np.full(len(pts), np.inf)
        for a, b in zip(chain[:-1], chain[1:]):
            d = b - a
            t = np.clip(((pts - a) @ d) / max(d @ d, 1e-12), 0, 1)
            best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * d), axis=1))
        dist.append(best)
    return np.argmin(np.stack(dist), axis=0)


def _outline_tube(stations):
    """A closed tube through stations given as (centre, u, v, a[nang], b[nang]): ring points
    centre + u a + v b (outlines that aren't star-shaped radii after the fit)."""
    verts, tris = [], []
    nang = len(stations[0][3])
    for c, u, v, a, b in stations:
        for x, y in zip(a, b):
            verts.append(c + u * x + v * y)
    m = len(stations)
    for i in range(m - 1):
        for j in range(nang):
            p, q = i * nang + j, i * nang + (j + 1) % nang
            tris += [(p, q + nang, q), (p, p + nang, q + nang)]
    for i, flip in ((0, True), (m - 1, False)):
        centre = len(verts)
        verts.append(np.mean(verts[i * nang:(i + 1) * nang], axis=0))
        for j in range(nang):
            p, q = i * nang + j, i * nang + (j + 1) % nang
            tris.append((centre, p, q) if flip else (centre, q, p))
    return np.array(verts), np.array(tris)


def topline(ref_co, ref_tris, van_co, van_tris, bones, step=0.5) -> list:
    """[[z, dy]]: the reference's back height minus the template's on the centre line, from the
    withers to the croup (rays down; None stations skipped)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    out = []
    for z in np.arange(B["HorseNeckBase"][2] + 0.5, B["HorseSpineEnd"][2] + 1.5, step):
        o, v = hm.top_bottom(ref_co, ref_tris, z), hm.top_bottom(van_co, van_tris, z)
        if o and v and o[0] is not None and v[0] is not None:
            out.append([round(float(z), 2), round(float(o[0] - v[0]), 3)])
    return out
