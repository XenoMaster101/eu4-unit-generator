"""Blender job (M23): a TRELLIS.2 rider to the game: a body on the vanilla infantry skeleton (the rider wears the
infantry pdxmesh and plays its horse_* animations, SPEC §7.2) and its lance and shield as rigid parts on the hand
nodes. PDX mesh space (Y up, facing -Z, the figure's right hand at -X), as fresh_horse; most steps are
fresh_horse's ``direct`` path.

  1. import, the seams welded (all pieces kept: the weapons are joined to the hands);
  2. the split, in the model's own units:
       lance: the column standing far to one side over the top of the body; its axis fitted to it, its shaft
       radius measured below the hand; the lance is what lies within 1.6 shaft radii of the axis (1.15 at the
       grip, so the fingers stay; 0.035 of the height round the blade and its tassel);
       shield: on the other side, the outermost points (sure shield) give its plane; the shield is what is
       connected to them within the plate's thickness of that plane;
  3. scale and place: the helmet's top (the highest point near the centre line that isn't the crest's red) at
     16.4, the feet on the ground, centred;
  4. each arm turned about its shoulder so the lance's grip points at Right_hand_node and the shield's at
     Left_hand_node (the rest of the gap slid), blended in over the arm's first quarter; each weapon carried with
     its hand;
  5. the crest (red, over the helmet) smoothed on the closed solid into one plume, like the horse's mane;
  6. the body: a closed solid (flood fill), quadric collapse to ``target_tris``, packed UVs, weights transferred
     from vanilla's body (nearest surface, smoothed), the cloth that hangs behind the legs on the hips (see
     ``cloth_weights``);
  7. each weapon: its own closed solid and collapse, packed UVs, exported twice: in the model's frame (its bake
     target) and in its node's frame (the part: a weapon along -Z from the grip, the rider's "down", levelled
     forward in game; the shield's face outwards along +X, its top along +Y).

args: {"vanilla": base arrays of the infantry body (p, tri, weights, bones), "source": {"path", "merge"},
       "target_tris", "part_tris": {"lance", "shield"}, "close_voxel", "smooth", "influences",
       "geometry_out", "preview_out", "fitted_out", "textured_out", "parts_dir", "result_path"}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _colour as cc  # noqa: E402
import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
import fresh_horse as fh  # noqa: E402

NECK = 14.1  # vanilla's neck: between top_spine (13.82) and head (14.5); the crest is looked for above it


def pca_axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt


SHIELD_FLAT, SHIELD_BROAD = 0.3, 0.035  # a shield: its thinnest spread under 0.3 of its width, its width over 0.035 H


def split(obj):
    """The lance and shield masks (module doc, 2), in the model's own units. -> (lance, shield, info)."""
    co = hs.verts(obj)
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    y0, y1 = co[:, 1].min(), co[:, 1].max()
    H = y1 - y0
    x0 = float(np.median(co[:, 0]))
    info = {"height_raw": round(float(H), 4)}
    lance = np.zeros(len(co), dtype=bool)
    top = co[co[:, 1] > y0 + 0.93 * H]
    far = top[np.abs(top[:, 0] - x0) > 0.15 * H]
    side = 0.0
    if len(far) > 20:
        xl, zl = np.median(far[:, 0]), np.median(far[:, 2])
        col = co[np.hypot(co[:, 0] - xl, co[:, 2] - zl) < 0.04 * H]
        c, vt = pca_axis(col)
        a = vt[0] if vt[0][1] > 0 else -vt[0]
        rel = co - c
        t = rel @ a
        r = np.linalg.norm(rel - t[:, None] * a, axis=1)
        tc = col @ a - c @ a
        tmin, tmax = float(tc.min()), float(tc.max())
        span = tmax - tmin
        low = (r < 0.04 * H) & (t > tmin) & (t < tmin + 0.35 * span)
        rs = float(np.median(r[low])) if low.any() else 0.01 * H
        hand = (r > 2.5 * rs) & (r < 0.06 * H) & (t > tmin + 0.2 * span) & (t < tmin + 0.8 * span)
        tg = float(np.median(t[hand])) if hand.any() else tmin + 0.5 * span
        lim = np.where(np.abs(t - tg) < 0.05 * H, 1.15 * rs, 1.6 * rs)
        lim = np.where(t > tg + 0.12 * H, np.maximum(lim, 0.035 * H), lim)
        lance = (r < lim) & (t >= tmin - 0.01 * H) & (t <= tmax + 0.01 * H)
        side = float(np.sign(xl - x0))
        # a broad head above the hand (a poleaxe's or a bill's blade, wider than a lance's 0.035): within 0.09 of the
        # height round the shaft, from 0.08 H over the grip (the hand stays below), on the weapon's side of the body
        # (the helmet is nearer the centre line). Measured on the English soldier: the head at 0.75-0.95 H, the hand
        # under 0.7; often a separate piece, so taken by place, not by connection
        head = (r < 0.09 * H) & (t > tg + 0.08 * H) & ((co[:, 0] - x0) * side > 0.14 * H)
        info["blade_added"] = int((head & ~lance).sum())
        lance = lance | head
        info["lance"] = {"axis": a.tolist(), "centre": c.tolist(), "shaft_radius": rs, "grip_t": tg,
                         "grip": (c + tg * a).tolist(), "t": [tmin, tmax], "vertices": int(lance.sum())}
    shield = np.zeros(len(co), dtype=bool)
    sgn = -side if side else 1.0
    out = (co[:, 0] - x0) * sgn
    sure = out > 0.75 * out.max()
    if sure.sum() > 50:
        sc, vt = pca_axis(co[sure])
        n = vt[2]
        if (sc - co.mean(0)) @ n < 0:
            n = -n
        d = (co - sc) @ n
        thick = float(np.percentile(np.abs(d[sure]), 95)) * 1.5 + 0.004 * H
        plane = co - d[:, None] * n
        ok = (np.abs(d) < thick) & (np.linalg.norm(plane - sc, axis=1) < 0.35 * H) & ~lance
        shield = fh.flood(ok, sure & ok, e)
        # a shield is a broad flat plate; the outermost points of an arm held out (a soldier without one) are a
        # forearm: long and round. Measured (the user's references, 2026-10-28): the cataphract's kite 0.089 flat,
        # 0.062 of the height broad; the English soldier's left forearm 0.571 and 0.022
        pts = co[shield]
        sv = np.sqrt(np.maximum(np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1], 0))             if len(pts) > 50 else np.zeros(3)
        flat, broad = float(sv[2] / max(sv[1], 1e-9)), float(sv[1] / H)
        info["shield_shape"] = {"flat": round(flat, 3), "broad": round(broad, 4)}
        if not (flat < SHIELD_FLAT and broad > SHIELD_BROAD):
            info["shield_rejected"] = "not a plate (an arm or a hand): no shield"
            return lance, np.zeros(len(co), dtype=bool), info
        # the hand behind it: body points just behind the plate near its middle
        behind = ~shield & ~lance & (d < 0) & (d > -0.08 * H) & (np.linalg.norm(plane - sc, axis=1) < 0.12 * H)
        grip = co[behind].mean(0) if behind.any() else sc - 0.03 * H * n
        info["shield"] = {"normal": n.tolist(), "centre": sc.tolist(), "thickness": thick, "grip": grip.tolist(),
                          "vertices": int(shield.sum())}
    return lance, shield, info


def feet_span(co, under=1.2):
    """How wide each foot is (its x extent under ``under``, left and right of the centre line): a boot kept or lost
    (the English soldier's tapered to spikes, 2026-10-28). -> [left, right]."""
    low = co[co[:, 1] < under]
    out = []
    for side in (1.0, -1.0):
        s = low[low[:, 0] * side > 0.1]
        # the 5th to 95th percentile: a thin stray near the ground (a polearm's butt beside the foot: the English
        # soldier's raw "foot" measured 3.1 for 1.4) doesn't count
        out.append(round(float(np.percentile(s[:, 0], 95) - np.percentile(s[:, 0], 5)), 3) if len(s) > 5 else 0.0)
    return out


def skirt_hem(co, crotch, gap, zc, step=0.1, floor=1.5):
    """How far the skirt or cloak hangs between the legs: the first height under the crotch with nothing on the centre
    line (within ``gap`` across, 1.2 of the torso's middle in depth). The crotch itself when the legs are bare there;
    ``floor`` when it hangs to the feet (split there, the feet stay apart)."""
    y = crotch
    while y > floor:
        sl = co[(np.abs(co[:, 1] - y) < step) & (np.abs(co[:, 0]) < gap) & (np.abs(co[:, 2] - zc) < 1.2)]
        if len(sl) < 3:
            return float(y)
        y -= step
    return float(floor)


def place(obj, bones, keep, top=16.4):
    """Scale (the helmet's top at ``top``: vanilla's body is 16.65 with its hat), feet on the ground, centred (x on
    the body's median, z on its torso's). The helmet's top: the highest point near the centre line that isn't the
    crest's red (the narrowest-slice neck took the crest for the head: the figure came out 15.7 tall).
    -> (scale, offset, helmet top before)."""
    co = hs.verts(obj)
    rgb, _ = fh.vertex_colours(obj)
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (rgb[:, 1] - rgb[:, 2]), 2 * rgb[:, 0] - rgb[:, 1] - rgb[:, 2])) + 360) % 360
    chroma = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
    body = co[keep]
    y0 = body[:, 1].min()
    H = body[:, 1].max() - y0
    is_crest, _ = cc.crest_colour(co, rgb, keep, y0 + 0.85 * H)
    red = is_crest(hue, chroma)
    x0 = float(np.median(body[:, 0]))
    head = keep & ~red & (np.abs(co[:, 0] - x0) < 0.06 * H) & (co[:, 1] > y0 + 0.75 * H)
    helmet = float(np.percentile(co[head, 1], 99.5)) if head.sum() > 20 else body[:, 1].max()
    s = top / (helmet - y0)
    torso = body[(body[:, 1] > y0 + 0.5 * H) & (body[:, 1] < y0 + 0.8 * (helmet - y0)) & (np.abs(body[:, 0] - x0) < 0.12 * H)]
    off = np.array([-x0 * s, -y0 * s, -float(np.median(torso[:, 2])) * s])
    hs.set_verts(obj, co * s + off)
    return s, off, helmet


def torso_half(co, bands=((8.6, 9.6), (9.6, 10.6), (10.6, 11.6)), reach=3.0):
    """The body's half-width (x) and half-depth (z) at the waist and chest: the 95th percentile of |x| and of
    |z - median z| over points within ``reach`` of the centre line (the arms are farther out at those heights)."""
    xs, zs = [], []
    for y0, y1 in bands:
        sl = co[(co[:, 1] >= y0) & (co[:, 1] < y1) & (np.abs(co[:, 0]) < reach)]
        if len(sl) > 20:
            xs.append(np.percentile(np.abs(sl[:, 0]), 95))
            zs.append(np.percentile(np.abs(sl[:, 2] - np.median(sl[:, 2])), 95))
    return (float(np.median(xs)) if xs else None), (float(np.median(zs)) if zs else None)


def slim_body(co, keep, vanilla_co, low=0.7, edge=2.0, legs=None):
    """M23 (the user, test 29: bulkier than vanilla's rider): x and z scaled about the centre line towards vanilla's
    rider body's waist and chest (half-width and half-depth: ``torso_half``), at least ``low`` (the armour, cloak and
    skirt need room); before the arms are turned, so the hands still reach their nodes. -> (kx, kz, measures)."""
    vx, vz = torso_half(vanilla_co)
    ox, oz = torso_half(co[keep])
    kx = float(np.clip(vx / ox, low, 1.0)) if vx and ox else 1.0
    kz = float(np.clip(vz / oz, low, 1.0)) if vz and oz else 1.0
    zc = float(np.median(co[keep & (co[:, 1] > 8.6) & (co[:, 1] < 11.6), 2]))
    SLIM_ZC["zc"] = zc
    SLIM_ZC["edge"] = edge
    # across the torso (|x| under ``edge``) x scales; beyond, the arms shift in whole (their shape and length kept:
    # scaled, the hands fell short of their nodes and the slide stretched the arms)
    ax = np.abs(co[:, 0])
    f = np.ones(len(co))
    if legs is not None:  # standing: the legs keep their stance (slimmed at 0.6 the cataphract's knees met): the
        # slimming whole above the hips, faded out to the knees, none below
        knee, hip = legs
        f = np.clip((co[:, 1] - knee) / max(hip - knee, 1e-6), 0.0, 1.0)
    k = 1.0 - (1.0 - kx) * f
    co[:, 0] = np.sign(co[:, 0]) * np.where(ax < edge, ax * k, ax - edge * (1 - k))
    co[:, 2] = zc + (co[:, 2] - zc) * (1.0 - (1.0 - kz) * f)
    return kx, kz, {"vanilla": [vx, vz], "ours": [ox, oz]}


# the joint fit (the user, T38: the English soldier squashed in game): one scale and a centring left the skeleton's
# joints outside the body (the cataphract's spine at 0.70-0.77 of his depth from the front against vanilla's 0.47-0.50,
# his neck bone at his back's surface; the English soldier's knees and feet 0.9-1.7 off his legs' centres): turning a
# joint outside a limb swings it on a lever and folds it. Each part moves whole (its volume and shape kept) so its
# centre is where vanilla's body has it, measured the same way on both.
SPINE_FIT = ("hip", "lower_spine", "mid_spine", "top_spine", "head")
LEG_BANDS = (1.0, 1.8, 2.6, 3.4, 4.2, 5.0)  # heights the legs are measured at (the ankle bone is at 0.92)
LEG_WIDE = 2.2  # a band wider than this on one side is a skirt or cloak, not a leg


def torso_depth(co, y, half=0.4, reach=1.2, strip=0.2):
    """The torso's front and back (z) near the centre line at height ``y``: across strips ``strip`` wide in x, the
    median of each strip's nearest and farthest point. Not percentiles of the vertices: those followed where a mesh is
    dense (the raw cataphract's thousands of plume vertices behind him moved his "front" 0.5 inwards)."""
    s = co[(np.abs(co[:, 1] - y) < half) & (np.abs(co[:, 0]) < reach)]
    if len(s) < 10:
        return None
    k = np.floor(s[:, 0] / strip).astype(int)
    fr, bk = [], []
    for i in np.unique(k):
        z = s[k == i, 2]
        if len(z) >= 3:
            fr.append(z.min())
            bk.append(z.max())
    return (float(np.median(fr)), float(np.median(bk))) if len(fr) >= 3 else None


def surface_points(obj, per=12, seed=0):
    """Points spread over a mesh's triangles (a low mesh has too few vertices to measure)."""
    co = hs.verts(obj)
    tri = np.array([(p.vertices[0], p.vertices[k], p.vertices[k + 1]) for p in obj.data.polygons
                    for k in range(1, len(p.vertices) - 1)])
    rng = np.random.default_rng(seed)
    u = rng.random((len(tri), per, 2))
    u = np.where(u.sum(2, keepdims=True) > 1, 1 - u, u)
    a, b, c = co[tri[:, 0]][:, None], co[tri[:, 1]][:, None], co[tri[:, 2]][:, None]
    return np.concatenate([co, (a + u[..., :1] * (b - a) + u[..., 1:] * (c - a)).reshape(-1, 3)])


def spine_fractions(co, bones):
    """Each spine bone's place in the body's depth at its height: 0 the front, 1 the back."""
    out = {}
    for b in SPINE_FIT:
        d = torso_depth(co, bones[b][1])
        if d and d[1] - d[0] > 1e-3:
            out[b] = round((bones[b][2] - d[0]) / (d[1] - d[0]), 3)
    return out


DEPTH_CLAMP = (0.8, 1.2)  # our depth counted at most this far from vanilla's: a cloak or plume behind isn't the body


def spine_targets(co, vanilla_co, bones):
    """Where each spine bone should sit in our body: from the front surface (nothing hangs in front of a chest), at
    vanilla's fraction of our depth clamped to ``DEPTH_CLAMP`` of vanilla's (the cataphract's cloak and plume behind
    him put his neck bone behind his back when the middle of the whole depth was used). -> {bone: (target z, bone z)}."""
    out = {}
    for b in SPINE_FIT:
        v, d = torso_depth(vanilla_co, bones[b][1]), torso_depth(co, bones[b][1])
        if v is None or d is None or v[1] - v[0] < 1e-3:
            continue
        vd = v[1] - v[0]
        frac = (bones[b][2] - v[0]) / vd
        depth = float(np.clip(d[1] - d[0], DEPTH_CLAMP[0] * vd, DEPTH_CLAMP[1] * vd))
        out[b] = (d[0] + frac * depth, bones[b][2])
    return out


def leg_line(co, side, top):
    """One leg's centre line (x and z as straight lines in y) through its bands under ``top``; bands where that side is
    wider than ``LEG_WIDE`` (a skirt) are skipped. -> (fx, fz) callables or None."""
    ys, xs, zs = [], [], []
    for y in LEG_BANDS:
        if y > top:
            break
        s = co[(np.abs(co[:, 1] - y) < 0.3) & (co[:, 0] * side > 0.25)]
        if len(s) < 8:
            continue
        lo, hi = np.percentile(s[:, 0], 5), np.percentile(s[:, 0], 95)
        if hi - lo > LEG_WIDE:
            continue
        ys.append(y)
        xs.append(float(np.median(s[:, 0])))
        zs.append(float(np.median(s[:, 2])))
    if len(ys) < 2:
        return None
    px, pz = np.polyfit(ys, xs, 1), np.polyfit(ys, zs, 1)
    return (lambda y: float(np.polyval(px, y))), (lambda y: float(np.polyval(pz, y)))


def leg_offsets(co, vanilla_co, bones, top):
    """How far each leg's centre is from vanilla's at the knee and the ankle: {side: (d_knee, d_foot)} (x, z)."""
    out = {}
    for side, s in (("L", 1.0), ("R", -1.0)):
        ours, van = leg_line(co, s, top), leg_line(vanilla_co, s, 9.0)
        if ours is None or van is None:
            continue
        out[side] = tuple(np.array([van[0](y) - ours[0](y), van[1](y) - ours[1](y)])
                          for y in (bones[f"{side}_knee"][1], bones[f"{side}_foot"][1]))
    return out


def joint_fit(co, keep, vanilla_co, bones, legs=True):
    """The joint fit (see ``SPINE_FIT``), in place on ``co``: the torso shifted in depth, by height, so each spine
    bone sits at vanilla's fraction of the depth (everything with it: arms, head, weapons by their height); then with
    ``legs`` each leg sheared under the thigh (none at the thigh, the knee's offset at the knee, the ankle's at the
    ankle and below), so the legs stand at vanilla's stance; the cloth between them takes each side's share.
    -> (shift(p): where a point measured before the fit went (the torso's shift only: for grips), report)."""
    tg = spine_targets(co[keep], vanilla_co, bones)
    ys = [bones[b][1] for b in tg]
    dz = [bz - t for t, bz in tg.values()]
    rep = {"spine_dz": {b: round(float(v), 3) for b, v in zip(tg, dz)}}
    if not ys:
        return (lambda p: np.asarray(p, dtype=np.float64)), rep

    def tz(y):
        return np.interp(y, ys, dz)
    co[:, 2] += tz(co[:, 1])
    if legs:
        thigh = (bones["L_thigh"][1] + bones["R_thigh"][1]) / 2
        offs = leg_offsets(co[keep], vanilla_co, bones, thigh - 1.0)
        rep["legs"] = {}
        for side, s in (("L", 1.0), ("R", -1.0)):
            if side not in offs:
                continue
            dk, df = offs[side]
            ky, fy = bones[f"{side}_knee"][1], bones[f"{side}_foot"][1]
            y = co[:, 1]
            sx = np.interp(y, [fy, ky, thigh], [df[0], dk[0], 0.0])
            sz = np.interp(y, [fy, ky, thigh], [df[1], dk[1], 0.0])
            w = keep * fh.smoothstep(co[:, 0] * s, -0.4, 0.4)
            co[:, 0] += w * sx
            co[:, 2] += w * sz
            rep["legs"][side] = {"knee": np.round(dk, 3).tolist(), "foot": np.round(df, 3).tolist()}
    return (lambda p: np.asarray(p, dtype=np.float64) + np.array([0.0, 0.0, float(tz(p[1]))])), rep


# the base game's proportions are a little stylised for reading at a distance (big head, hands and shoulders): ours
# are brought up to at least vanilla's, never down, and by at most these factors
STYLISE_MAX = {"head": 1.25, "hand": 1.3, "shoulders": 1.15}
HAND_R = 1.0  # a hand: what lies within this of its hand bone


def head_height(co):
    """How far the head rises over the neck (vanilla's body: 2.75 with its hat)."""
    return float(np.percentile(co[co[:, 1] > NECK - 0.2][:, 1], 99.5) - NECK) if (co[:, 1] > NECK).sum() > 10 else 0.0


def hand_span(co, bones, side):
    q = co[np.linalg.norm(co - np.array(bones[f"{side}_hand"]), axis=1) < HAND_R]
    return float(np.ptp(q, axis=0).max()) if len(q) > 5 else 0.0


def shoulder_width(co):
    s = co[(co[:, 1] > 12.0) & (co[:, 1] < 13.2) & (np.abs(co[:, 0]) < 3.0)]
    return float(np.percentile(s[:, 0], 98) - np.percentile(s[:, 0], 2)) if len(s) > 10 else 0.0


def stylise(co, keep, vanilla_co, bones):
    """In place: the head (above the neck, about the neck's centre), each hand (about its hand bone) and the shoulders
    (x above the chest, faded in from 11 to 12) scaled up to at least vanilla's size (``STYLISE_MAX`` at most).
    -> the factors and measures."""
    out = {}
    body = co[keep]
    vh, oh = head_height(vanilla_co), head_height(body)
    k = float(np.clip(vh / oh, 1.0, STYLISE_MAX["head"])) if oh > 0.3 else 1.0
    if k > 1.0:
        sel = keep & (co[:, 1] > NECK)
        centre = np.array([0.0, NECK, float(np.median(co[sel, 2]))])
        f = np.clip((co[sel, 1] - NECK) / 0.4, 0.0, 1.0)[:, None]  # faded in over the collar
        co[sel] = co[sel] + f * (k - 1.0) * (co[sel] - centre)
    out["head"] = {"vanilla": round(vh, 3), "ours": round(oh, 3), "k": round(k, 3)}
    for side in ("L", "R"):
        vs, os_ = hand_span(vanilla_co, bones, side), hand_span(co[keep], bones, side)
        k = float(np.clip(vs / os_, 1.0, STYLISE_MAX["hand"])) if os_ > 0.2 else 1.0
        if k > 1.0:
            h = np.array(bones[f"{side}_hand"])
            d = np.linalg.norm(co - h, axis=1)
            sel = keep & (d < HAND_R * 1.4)
            f = np.clip((HAND_R * 1.4 - d[sel]) / (HAND_R * 0.4), 0.0, 1.0)[:, None]  # full inside HAND_R, faded out
            co[sel] = co[sel] + f * (k - 1.0) * (co[sel] - h)
        out[f"{side}_hand"] = {"vanilla": round(vs, 3), "ours": round(os_, 3), "k": round(k, 3)}
    vw, ow = shoulder_width(vanilla_co), shoulder_width(co[keep])
    k = float(np.clip(vw / ow, 1.0, STYLISE_MAX["shoulders"])) if ow > 0.5 else 1.0
    if k > 1.0:
        f = np.clip(co[:, 1] - 11.0, 0.0, 1.0) * keep
        co[:, 0] *= 1.0 + f * (k - 1.0)
    out["shoulders"] = {"vanilla": round(vw, 3), "ours": round(ow, 3), "k": round(k, 3)}
    return out


# borders at 1.0 (or spread over neighbours) protected so much the weighted collapse stalled and the plain fallback
# thinned the limbs (57% of the English soldier's triangles); at 0.4 the limbs keep 74% (67% unweighted)
IMPORTANCE = {"border": 0.4, "limb": 0.7, "border_de": 25.0, "materials": 6, "limb_r": 1.2, "smooth": 0}


def importance(obj, colour_src, bones, cfg=None):
    """A vertex group ``importance`` (the user, after T38: the simplification rounded off the edges between materials
    and spent its triangles on flat areas): 1 where two distinct materials meet (the colours of the raw model, before
    the voxel remesh, carried over by nearest point; k-means on Lab, borders at least ``border_de`` apart), ``limb``
    on the arms and legs (within ``limb_r`` of their bone chains, the legs below the hips), 0 elsewhere; smoothed over
    the neighbours. -> shares."""
    from mathutils.kdtree import KDTree
    c = {**IMPORTANCE, **(cfg or {})}
    src_co, src_rgb = colour_src
    tree = KDTree(len(src_co))
    for i, q in enumerate(src_co):
        tree.insert(q, i)
    tree.balance()
    co = hs.verts(obj)
    rgb = np.array([src_rgb[tree.find(q)[1]] for q in co])
    import _paint
    lab = _paint._lab(np.clip(rgb / 255.0, 0, 1)[:, None, :])[:, 0, :]  # vertex_colours: 0-255
    rng = np.random.default_rng(0)
    cent = lab[rng.choice(len(lab), int(c["materials"]), replace=False)]
    for _ in range(12):
        lab_ = np.argmin(((lab[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.array([lab[lab_ == i].mean(0) if (lab_ == i).any() else cent[i] for i in range(len(cent))])
    apart = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1)) >= c["border_de"]
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    w = np.zeros(len(co))
    hit = apart[lab_[edges[:, 0]], lab_[edges[:, 1]]]
    w[edges[hit].ravel()] = c["border"]
    limb = np.zeros(len(co), dtype=bool)
    for chain in list(ARMS.values()) + list(LEGS.values()):
        _, d = chain_rows(co, bones, chain)
        limb |= d < c["limb_r"]
    limb &= ~((co[:, 1] > bones["hip"][1]) & (np.abs(co[:, 0]) < abs(bones["L_shoulder"][0])))  # not the torso
    w = np.maximum(w, limb * c["limb"])
    for _ in range(int(c["smooth"])):
        acc, n = w.copy(), np.ones(len(w))
        np.add.at(acc, edges[:, 0], w[edges[:, 1]])
        np.add.at(acc, edges[:, 1], w[edges[:, 0]])
        np.add.at(n, edges.ravel(), 1)
        w = np.maximum(w, acc / n)
    g = obj.vertex_groups.new(name="importance")
    for i, x in enumerate(w):
        g.add([i], float(x), "REPLACE")
    return {"border": round(float(hit.mean()), 4), "limb": round(float(limb.mean()), 4), "mean": round(float(w.mean()), 3)}


def gaps(co, bones):
    """The outline's gaps (the user, after T38: the base game's units read because arms, torso and legs are apart):
    between the legs at the knees, and between each forearm and the torso at the wrists' height (the widest empty
    run in x between them, 0 when joined; at the elbows the upper arm crosses the slice: vanilla's read 0.04)."""
    def run(sl, lo, hi):
        xs = np.sort(sl[(sl[:, 0] > lo) & (sl[:, 0] < hi), 0])
        return float(np.diff(np.concatenate([[lo], xs, [hi]])).max()) if len(xs) else hi - lo
    knee = co[np.abs(co[:, 1] - bones["L_knee"][1]) < 0.3]
    out = {"legs": round(run(knee, -0.9, 0.9), 3)}
    for side, s in (("L", 1.0), ("R", -1.0)):
        y = bones[f"{side}_wrist"][1]
        sl = co[np.abs(co[:, 1] - y) < 0.3]
        sl = sl[sl[:, 0] * s > 0] * np.array([s, 1.0, 1.0])
        out[f"{side}_arm"] = round(run(sl, 1.2, abs(bones[f"{side}_wrist"][0])), 3)
    return out


def joints_report(co, vanilla_co, bones):
    """The check (rule ``joints_inside``): each spine bone's distance from where ``spine_targets`` puts it, the
    fractions (for reading), and each leg's distance from vanilla's at the knee and the ankle."""
    thigh = (bones["L_thigh"][1] + bones["R_thigh"][1]) / 2
    offs = leg_offsets(co, vanilla_co, bones, thigh - 1.0)
    return {"gaps": gaps(co, bones), "gaps_vanilla": gaps(vanilla_co, bones),
            "spine_off": {b: round(abs(bz - t), 3) for b, (t, bz) in spine_targets(co, vanilla_co, bones).items()},
            "spine": spine_fractions(co, bones), "spine_vanilla": spine_fractions(vanilla_co, bones),
            "legs_off": {s: [round(float(np.linalg.norm(v)), 3) for v in d] for s, d in offs.items()}}


def free_hand(co, keep, bones, side):
    """An empty hand: the arm's far end along vanilla's shoulder-to-hand direction (within 45 degrees of it, the
    farthest 3% of the arm's points), or None."""
    sh = np.array(bones[f"{side}_shoulder"])
    d = np.array(bones[f"{side}_hand"]) - sh
    d /= np.linalg.norm(d)
    rel = co - sh
    dist = np.linalg.norm(rel, axis=1)
    cos = (rel @ d) / np.maximum(dist, 1e-9)
    arm = keep & (cos > np.cos(np.radians(45))) & (dist > 1.5) & (dist < 9.0)
    if arm.sum() < 30:
        return None
    t = rel[arm] @ d
    far = co[arm][t >= np.percentile(t, 97)]
    return far.mean(0)


# a large swing moves only what joins the hand (the English soldier's arm, 41.5 degrees: the jack came along as a
# sheet); a small one keeps its neighbours with it, which smooths the shoulder (the cataphract's 31 and 34 degrees:
# moving the arm alone raised its worst stretch from 2.4-2.8 to 4.7-9.8, regression 2026-10-28)
ARM_CONNECTED_DEG = 38.0


def swing_arm(co, movable, grip, target, shoulder, side, edges=None):
    """``movable`` points on one arm turned about the shoulder so the grip points where the target is (arm length
    kept: the hands were 4.6 and 5.9 from vanilla's, holding the lance in front and the shield at the hip; sliding
    them stretched the arms), blended in over the arm's first quarter; then the rest of the gap (length) slid,
    faded in the same way. -> (angle in degrees, the grip's end position)."""
    sh = np.asarray(shoulder, dtype=np.float64)
    g, tg = np.asarray(grip) - sh, np.asarray(target) - sh
    t = np.clip(((co - sh) @ g) / max(g @ g, 1e-9), 0, 1.2)
    near = np.linalg.norm(co - (sh + np.clip(t, 0, 1)[:, None] * g), axis=1) < 1.3
    m = movable & (co[:, 0] * side > 0) & near & (t > 0.0)
    k0 = np.cross(g, tg)
    big = math.degrees(math.atan2(np.linalg.norm(k0), g @ tg)) > ARM_CONNECTED_DEG
    if edges is not None and big:  # the arm: what joins the hand along the surface past the shoulder seam (t > 0.1), not all
        # that lies near the line: a hand held at the chest brought the jack's side along (the English soldier,
        # 2026-10-28: a sheet from the shoulder to the wrist)
        seed = m & (t > 0.85)
        if seed.any():
            arm = fh.flood(m & (t > 0.1), seed, edges)
            m = arm | (m & (t <= 0.1))
    k = np.cross(g, tg)
    ang = math.atan2(np.linalg.norm(k), g @ tg)
    k = k / max(np.linalg.norm(k), 1e-9)
    w = fh.smoothstep(t, 0.0, 0.25)
    rel = co[m] - sh
    th = (ang * w[m])[:, None]
    kx = np.cross(k[None], rel)
    rot = rel * np.cos(th) + kx * np.sin(th) + k[None] * (rel @ k)[:, None] * (1 - np.cos(th))
    end = sh + (g * np.cos(ang) + np.cross(k, g) * np.sin(ang) + k * (g @ k) * (1 - np.cos(ang)))
    slide = np.asarray(target) - end
    co[m] = sh + rot + w[m, None] * slide
    return math.degrees(ang), slide


SLIM_ZC = {}


def slimmed(p, kx, kz, co):
    """A point measured before ``slim_body``, moved as it moved the body."""
    zc, edge = SLIM_ZC.get("zc", 0.0), SLIM_ZC.get("edge", 2.0)
    ax = abs(p[0])
    x = np.sign(p[0]) * (ax * kx if ax < edge else ax - edge * (1 - kx))
    return np.array([x, p[1], zc + (p[2] - zc) * kz])


def carry(pts, grip, target, shoulder):
    """A weapon carried with its hand: the arm's full turn about the shoulder, then onto the target."""
    sh = np.asarray(shoulder, dtype=np.float64)
    g, tg = np.asarray(grip) - sh, np.asarray(target) - sh
    k = np.cross(g, tg)
    ang = math.atan2(np.linalg.norm(k), g @ tg)
    k = k / max(np.linalg.norm(k), 1e-9)
    rel = pts - sh
    rot = rel * math.cos(ang) + np.cross(k[None], rel) * math.sin(ang) + k[None] * (rel @ k)[:, None] * (1 - math.cos(ang))
    end = sh + (g * math.cos(ang) + np.cross(k, g) * math.sin(ang) + k * (g @ k) * (1 - math.cos(ang)))
    return sh + rot + (np.asarray(target) - end)


def frame_rotation(a_from, b_from, a_to, b_to):
    """The rotation taking the orthonormal pair (a_from, b_from) onto (a_to, b_to)."""
    def basis(a, b):
        a = a / np.linalg.norm(a)
        b = b - (b @ a) * a
        b /= np.linalg.norm(b)
        return np.stack([a, b, np.cross(a, b)], 1)
    return basis(np.asarray(a_to, float), np.asarray(b_to, float)) @ basis(np.asarray(a_from, float), np.asarray(b_from, float)).T


def part_object(src, mask, name):
    """A copy of ``src`` with only ``mask``'s vertices."""
    o = src.copy()
    o.data = src.data.copy()
    o.name = name
    bpy.context.scene.collection.objects.link(o)
    fh.delete_vertices(o, np.flatnonzero(~mask).tolist())
    return o


def plume_line(crest_pts, helmet_pts, helmet_top, head, fin, n=12):
    """M23: the raw plume's centre line, from the finial on the helmet's top: the crest's points in shells by their
    distance from the finial's top, each shell's centroid and side-to-side spread (its half width). Stops where the
    line would come down to the helmet (within its radius + 0.35, under its top + 0.3) or near the cloak (under the
    neck + 1.5): the gap the user asked for. In the head bone's frame (the mesh's axes, the bone at the origin)."""
    hc = np.median(helmet_pts[helmet_pts[:, 1] > helmet_top - 0.3][:, [0, 2]], axis=0)
    hr = float(np.percentile(np.hypot(helmet_pts[:, 0] - hc[0], helmet_pts[:, 2] - hc[1]), 90))
    base = np.array([hc[0], helmet_top - 0.1, hc[1]])
    tip = base + np.array([0.0, fin, 0.0])
    d = np.linalg.norm(crest_pts - tip, axis=1)
    edges = np.linspace(0.0, float(np.percentile(d, 97)), n + 1)
    line, width = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 20:
            continue
        c = crest_pts[m].mean(0)
        near_helmet = np.hypot(c[0] - hc[0], c[2] - hc[1]) < hr + 0.35 and c[1] < helmet_top + 0.3
        if near_helmet or c[1] < NECK + 1.5:
            break
        line.append(c)
        width.append(float(np.percentile(np.abs(crest_pts[m, 0] - c[0]), 80)))
    return {"finial_base": (base - head).round(4).tolist(), "finial_height": fin, "helmet_radius": round(hr, 3),
            "line": (np.array(line) - head).round(4).tolist() if line else [], "half_width": [round(w, 3) for w in width]}


def export_part(o, path_model, path_node, R, origin):
    """The part as bake_colour's low geometry (model frame) and as the mesh part's geometry (node frame)."""
    fh.export(o, path_model, None)
    g = json.loads(open(path_model, encoding="utf-8").read())
    p = (np.array(g["p"]).reshape(-1, 3) - origin) @ R.T
    with open(path_node, "w", encoding="utf-8") as f:
        json.dump({"p": p.ravel().round(5).tolist(), "tri": g["tri"], "uv": g["u0"]}, f)
    return {"tris": len(g["tri"]) // 3, "bounds": [p.min(0).round(3).tolist(), p.max(0).round(3).tolist()]}


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    vp = van["p"]
    vanilla = fh.mesh_object("vanilla", [vp[i:i + 3] for i in range(0, len(vp), 3)],
                             [van["tri"][i:i + 3] for i in range(0, len(van["tri"]), 3)])
    for vi, row in enumerate(van["weights"]):
        for bone, w in row.items():
            g = vanilla.vertex_groups.get(bone) or vanilla.vertex_groups.new(name=bone)
            g.add([vi], w, "REPLACE")
    bones = van["bones"]
    res = {}
    obj = fh.import_model(a["source"]["path"])
    fh.merge_seams(obj, float(a["source"].get("merge", 1e-5)))
    res["imported"] = {"tris": fh.tri_count(obj), "pieces": fh.count_pieces(obj)}
    lance, shield, res["split"] = split(obj)
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)  # the arms are found along them (swing_arm)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    s, off, _ = place(obj, bones, ~lance & ~shield)
    res["place"] = {"scale": float(s), "offset": off.tolist()}
    if a.get("fitted_out"):
        fh.write_obj(obj, a["fitted_out"], uv=True)
    co = hs.verts(obj)
    fitted = co.copy()  # placed, not yet slimmed or posed: as the reference shows it (Track P1.2's part crops)
    feet_raw = feet_span(fitted[~lance & ~shield])
    van_co = surface_points(vanilla)
    res["joints_before"] = joints_report(co[~lance & ~shield], van_co, bones)
    shift = lambda p: np.asarray(p, dtype=np.float64)  # noqa: E731 - the identity without the fit
    if a.get("joint_fit", True):
        held = co.copy()
        shift, res["joint_fit"] = joint_fit(co, ~lance & ~shield, van_co, bones, legs=True)
        for key, mask in (("lance", lance), ("shield", shield)):  # each weapon whole, as its grip moved
            info = res["split"].get(key)
            if info and mask.any():
                g = np.array(info["grip"]) * s + off
                co[mask] = held[mask] + (shift(g) - g)
    legs =((bones["L_knee"][1] + bones["R_knee"][1]) / 2, (bones["L_thigh"][1] + bones["R_thigh"][1]) / 2 + 0.8)         if a.get("fit") == "standing" else None
    kx, kz, measured = slim_body(co, ~lance & ~shield, hs.verts(vanilla), float(a.get("slim_low", 0.7)), legs=legs)
    res["slim"] = {"x": round(kx, 3), "z": round(kz, 3), **measured}
    moves, dirs = {}, {}
    li, si = res["split"].get("lance"), res["split"].get("shield")
    if li:
        grip = slimmed(shift(np.array(li["grip"]) * s + off), kx, kz, co)
        side = -1.0 if grip[0] < 0 else 1.0
        node = "Right_hand_node" if side < 0 else "Left_hand_node"
        sh = bones["R_shoulder" if side < 0 else "L_shoulder"]
        before = co.copy()
        ang, slide = swing_arm(co, ~lance & ~shield, grip, bones[node], sh, side, edges)
        co[lance] = carry(before[lance], grip, bones[node], sh)
        ax = carry(np.array([grip, grip + np.array(li["axis"])]), grip, bones[node], sh)
        dirs["lance"] = (ax[1] - ax[0], None)
        moves["lance"] = {"node": node, "grip": grip.round(3).tolist(), "turned_deg": round(ang, 1),
                          "slid": round(float(np.linalg.norm(slide)), 3)}
    if si:
        grip = slimmed(shift(np.array(si["grip"]) * s + off), kx, kz, co)
        side = -1.0 if grip[0] < 0 else 1.0
        node = "Right_hand_node" if side < 0 else "Left_hand_node"
        sh = bones["R_shoulder" if side < 0 else "L_shoulder"]
        before = co.copy()
        ang, slide = swing_arm(co, ~lance & ~shield, grip, bones[node], sh, side, edges)
        co[shield] = carry(before[shield], grip, bones[node], sh)
        ax = carry(np.array([grip, grip + np.array(si["normal"]), grip + np.array([0.0, 1.0, 0.0])]), grip, bones[node], sh)
        dirs["shield"] = (ax[1] - ax[0], ax[2] - ax[0])
        moves["shield"] = {"node": node, "grip": grip.round(3).tolist(), "turned_deg": round(ang, 1),
                           "slid": round(float(np.linalg.norm(slide)), 3)}
    held = {m["node"] for m in moves.values()}
    for side, s_, node in (("L", 1.0, "Left_hand_node"), ("R", -1.0, "Right_hand_node")):
        if node in held or not a.get("free_arm", True):
            continue
        # an empty hand to vanilla's hand too (the user, after T38: the English soldier's free arm hung about 20 degrees
        # under vanilla's, his forearm 0.19 from his side at the wrist against vanilla's 1.07)
        grip = free_hand(co, ~lance & ~shield, bones, side)
        if grip is not None:
            ang, slide = swing_arm(co, ~lance & ~shield, grip, bones[f"{side}_hand"], bones[f"{side}_shoulder"], s_, edges)
            moves[f"free_{side}"] = {"node": f"{side}_hand", "grip": grip.round(3).tolist(), "turned_deg": round(ang, 1),
                                     "slid": round(float(np.linalg.norm(slide)), 3)}
    if a.get("stylise", True):
        res["stylise"] = stylise(co, ~lance & ~shield, van_co, bones)
    hs.set_verts(obj, co)
    res["arms"] = moves
    # the crest: the crest's own colour over the helmet (the cataphract's is red)
    rgb, lum = fh.vertex_colours(obj)
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (rgb[:, 1] - rgb[:, 2]), 2 * rgb[:, 0] - rgb[:, 1] - rgb[:, 2])) + 360) % 360
    chroma = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
    is_crest, crest_h = cc.crest_colour(co, rgb, ~lance & ~shield, NECK + 1.2)
    crest = ~lance & ~shield & (co[:, 1] > NECK + 1.2) & is_crest(hue, chroma)
    res["crest_vertices"] = int(crest.sum())
    res["crest_hue"] = None if crest_h is None else round(crest_h, 1)
    # the helmet's steel: not the crest's colour, above the neck, near the centre line (its axis and radius for the
    # collar cut)
    steel = ~lance & ~shield & ~is_crest(hue, chroma) & (co[:, 1] > NECK) & (np.abs(co[:, 0]) < 1.5) \
        & (co[:, 1] < NECK + 2.6)
    helmet_pts = co[steel] if steel.any() else None
    crest_pts = co[crest]
    # Track P1.2: where each part lies in the placed model, to find it in the reference (ref3d.part_gen.locate)
    def sample(mask, n):
        idx = np.flatnonzero(mask)
        return fitted[idx[:: max(1, len(idx) // n)]].round(4).tolist() if len(idx) else []
    with open(os.path.join(a["parts_dir"], "fitted_points.json"), "w", encoding="utf-8") as f:
        json.dump({"all": sample(np.ones(len(fitted), dtype=bool), 30000), "lance": sample(lance, 3000),
                   "shield": sample(shield, 3000), "crest": sample(crest, 3000),
                   "helmet": sample(steel, 3000)}, f)
    # the parts, with their own raw copies (bake sources)
    parts = {}
    helmet_top = float(np.percentile(helmet_pts[:, 1], 99)) if helmet_pts is not None and len(helmet_pts) > 20 else NECK + 2.0
    if a.get("plume") == "part" and len(crest_pts) > 200 and helmet_pts is not None:
        # the plume as a rigid part on the head (the user, test 31: test 29's arc was right, but smoothed into the body
        # it merged into the helmet, mail and cloak; cut out as it was, its rise went with the gap): its centre line
        # for a clean tube (trellis.plume_part) out of a finial on the helmet's top
        res["plume_line"] = plume_line(crest_pts, helmet_pts, helmet_top, np.array(bones["head"]),
                                       float(a.get("finial_height", 0.6)))
    for key, mask in (("lance", lance), ("shield", shield)):
        if mask.sum() > 50:
            p = part_object(obj, mask, key)
            fh.write_obj(p, os.path.join(a["parts_dir"], f"{key}_textured.obj"), uv=True)
            parts[key] = p
    # the body: the weapons off; the bake source as it is
    colour_src = (co[~lance & ~shield].copy(), rgb[~lance & ~shield].copy())  # for the importance after the remesh
    fh.delete_vertices(obj, np.flatnonzero(lance | shield).tolist())
    if a.get("textured_out"):
        fh.write_obj(obj, a["textured_out"], uv=True)
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    fh.SOLID.clear()
    # the slot between the legs, from just under the crotch down: closed round both legs, the skirt was one block
    # that tore when the seated legs spread and swung forward; split, each half moves with its leg
    crotch = min(bones["L_thigh"][1], bones["R_thigh"][1]) - 0.5
    gap = float(a.get("leg_gap", 0.3))
    # and the cloth hanging under each arm, outside the torso: draped over the shield arm, it hung from it in one
    # sheet beside the leg and tore when the arm moved (the arms' flips: 64 against vanilla's 2)
    side_x = float(a.get("side_cut", 2.1))
    slit_x, slit = float(a.get("slit_x", 1.85)), float(a.get("slit_half", 0.0))  # slits off: no gain (54 against 57 flips)
    under = min(bones["R_hand"][1], bones["L_hand"][1]) - 1.0
    # the standing fit (infantry and the gun crew; the user, T37: the rider build stood hunched and bulky): the skirt
    # hangs whole (the legs split only below its hem) and the side cut stops at the crotch (from the ground up it took
    # the English soldier's boots, his feet 2.5 out: legs that tapered to spikes)
    standing = a.get("fit", "seated") == "standing"
    slot_top, side_lo = crotch, -1.0
    if standing:
        slot_top = min(crotch, skirt_hem(hs.verts(obj), crotch, gap, SLIM_ZC.get("zc", 0.0)))
        side_lo = crotch
        res["fit"] = {"kind": "standing", "hem": round(slot_top, 2)}
    else:
        res["fit"] = {"kind": "seated"}
    fh.SOLID.update({"closing": 0.2, "opening": 0.0, "relax": 3,
                     "carve": [[[-gap, -1.0, -20.0], [gap, slot_top, 20.0]],
                               # side slits: the skirt's front panels follow the thighs, its back ones the hips
                               *([[[slit_x, -1.0, -slit], [20.0, crotch, slit]], [[-20.0, -1.0, -slit], [-slit_x, crotch, slit]]]
                                 if slit > 0 else []),
                               [[side_x, side_lo, -20.0], [20.0, under, 20.0]], [[-20.0, side_lo, -20.0], [-side_x, under, 20.0]]]})
    res["leg_slot"] = {"half_width": gap, "top": round(slot_top, 2)}
    # the collar round the helmet (the user, test 30: the head read as a hooded blob): above the neck, anything farther
    # out than the helmet (its steel near the centre line: the median axis, the 90th percentile radius, + 0.1) is cut
    if a.get("collar_cut", True) and helmet_pts is not None and len(helmet_pts) > 50:
        hc = np.median(helmet_pts[:, [0, 2]], axis=0)
        hr = float(np.percentile(np.hypot(helmet_pts[:, 0] - hc[0], helmet_pts[:, 2] - hc[1]), 90)) + 0.1
        fh.SOLID["carve_out"] = [{"centre": hc.tolist(), "radius": hr, "y0": NECK - 0.2}]
        res["collar_cut"] = {"centre": hc.round(3).tolist(), "radius": round(hr, 3), "above": NECK - 0.2}
    res["side_cut"] = {"beyond_x": side_x, "below_y": round(under, 2)}
    vs = float(a.get("close_voxel", 0.1))
    fh.collapse_to(obj, 10 ** 9, vs)
    fh.keep_largest(obj)
    if len(crest_pts) and a.get("plume", "cut") in ("cut", "part"):  # (a part: its own tube replaces it)
        # the plume cut off the closed solid (smoothed, it was a heavy red arc over the head: the user, test 29); a
        # small crest part (a tuft on the head bone) takes its place in the loadout
        from mathutils.kdtree import KDTree
        tree = KDTree(len(crest_pts))
        for i, q in enumerate(crest_pts):
            tree.insert(q, i)
        tree.balance()
        steel_tree = None
        if helmet_pts is not None and len(helmet_pts):  # what is nearer the helmet's steel stays: the cut took the
            # cataphract's dome with the plume on it (his head 1.71 over the neck against vanilla's 2.75)
            steel_tree = KDTree(len(helmet_pts))
            for i, q in enumerate(helmet_pts):
                steel_tree.insert(q, i)
            steel_tree.balance()
        cut = [i for i, q in enumerate(hs.verts(obj)) if q[1] > NECK + 1.2 and (d := tree.find(q)[2]) < 0.4
               and (steel_tree is None or d < steel_tree.find(q)[2])]
        fh.delete_vertices(obj, cut)
        fh.keep_largest(obj)
        res["plume_voxels_cut"] = len(cut)
        res["plume_holes_filled"] = fh.fill_open(obj)
        crest_pts = crest_pts[:0]  # nothing left to keep rigid
    keep_group = None
    if a.get("importance", True):  # the triangle budget on material borders and the limbs, not flat areas
        res["importance"] = importance(obj, colour_src, bones)
        # inverted: the Decimate modifier's group marks what may go (as it is, the collapse stalled at 33,300 triangles;
        # inverted, the English soldier's limbs took 78% of his triangles against 67%)
        keep_group = ("importance", float(a.get("importance_factor", 5.0)), bool(a.get("importance_invert", True)))
    res["body_tris"] = fh.collapse_to(obj, int(a["target_tris"]), None, keep=keep_group)
    if obj.vertex_groups.get("importance"):
        obj.vertex_groups.remove(obj.vertex_groups["importance"])
    if res.get("plume_line"):
        # the finial on the low body's crown (the raw helmet's top points lean forward: it stood on the dome's front
        # edge); the line moved with it, keeping its shape
        v = hs.verts(obj)
        top = v[v[:, 1] > v[:, 1].max() - 0.2]
        crown = np.array([top[:, 0].mean(), v[:, 1].max() - 0.08, top[:, 2].mean()]) - np.array(bones["head"])
        pl = res["plume_line"]
        shift = crown - np.array(pl["finial_base"])
        pl["finial_base"] = crown.round(4).tolist()
        pl["line"] = (np.array(pl["line"]) + shift).round(4).tolist() if pl["line"] else []
        pl["crown_shift"] = shift.round(3).tolist()
    if a.get("even", 0):  # the collapse's slivers evened out (a few smoothing passes on the low mesh; their flips)
        fh.activate(obj)
        mod = obj.modifiers.new("even", "SMOOTH")
        mod.factor, mod.iterations = 0.4, int(a["even"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.data.uv_layers.new(name="UV")
    fh.packed_uv(obj, float(a.get("uv_margin", 0.002)))
    res["uv_area"] = round(fh.uv_area(obj), 4)
    wstats = {}
    fh.transfer_weights(obj, vanilla, a.get("smooth", 12), a.get("influences", 4), None, wstats)
    res["regions"] = rider_weights(obj, bones, a.get("influences", 4), int(a.get("weight_smooth", 12)), plume=crest_pts,
                                   standing=a.get("fit") == "standing")
    feet_game = feet_span(hs.verts(obj))
    res["joints"] = joints_report(surface_points(obj), van_co, bones)  # rule joints_inside
    res["feet"] = {"raw": feet_raw, "game": feet_game,
                   "ratio": round(min(g / r for g, r in zip(feet_game, feet_raw) if r) if any(feet_raw) else 1.0, 3)}
    verts, tris, unweighted = fh.export(obj, a["geometry_out"], a.get("preview_out"))
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": fh.count_pieces(obj)}
    # the parts in their nodes' frames
    res["parts"] = {}
    for key, p in parts.items():
        while p.data.uv_layers:
            p.data.uv_layers.remove(p.data.uv_layers[0])
        fh.SOLID.clear()
        fh.SOLID.update({"closing": 0.1, "opening": 0.0, "relax": 1})
        fh.collapse_to(p, int(a.get("part_tris", {}).get(key, 300)), 0.06)
        fh.keep_largest(p)
        p.data.uv_layers.new(name="UV")
        fh.packed_uv(p, 0.004)
        node = moves[key]["node"]
        origin = np.array(bones[node])
        a_dir, b_dir = dirs[key]  # after the arm's turn
        place_at = None
        if key == "lance":  # the axis (tip) to -Z (the rider's "down"), the figure's forward (-Z) to +Y
            R = frame_rotation(a_dir, np.array([0.0, 0.0, -1.0]), [0, 0, -1], [0, 1, 0])
        elif a.get("shield_frame"):
            # strapped to the forearm (the user, test 29: it sat on the rein hand over the horse's neck): its face
            # along the rider's outward side and its top up, in the hand node's frame as the seated pose has it
            # (rider_frames), centred half way to the elbow and out from the forearm
            f = a["shield_frame"]
            R = frame_rotation(a_dir, b_dir, f["out"], f["up"])
            # and lowered (centred on the raised forearm, the kite's top stood over the helmet: its rim showed as a
            # ribbon over the head from behind), a little smaller
            place_at = 0.5 * np.array(f["elbow"]) + float(a.get("shield_out", 0.6)) * np.array(f["out"])                 - float(a.get("shield_down", 1.5)) * np.array(f["up"])
            k = float(a.get("shield_scale", 0.85))
            R = R * k
        else:  # the face outwards to +X, the shield's top to +Y (the hoplon's convention: on the rein hand)
            R = frame_rotation(a_dir, b_dir, [1, 0, 0], [0, 1, 0])
        if place_at is not None:
            centre = hs.verts(p).mean(0)
            origin = centre - np.linalg.solve(R, place_at)  # (centre - origin) @ R.T == place_at (R may carry a scale)
        res["parts"][key] = {"node": node, "R": np.asarray(R).round(6).tolist(), "origin": np.asarray(origin).round(6).tolist(),
                             **({"strapped_at": place_at.round(3).tolist()} if place_at is not None else {}),
                             **export_part(p, os.path.join(a["parts_dir"], f"{key}_low.json"),
                                           os.path.join(a["parts_dir"], f"{key}_part.json"), R, origin)}
    _pdx.write_result(a, res)


ARMS = {"R": ["R_shoulder", "R_arm", "R_elbow", "R_wrist", "R_hand", "Right_hand_node"],
        "L": ["L_shoulder", "L_arm", "L_elbow", "L_wrist", "L_hand", "Left_hand_node"]}
LEGS = {"R": ["R_thigh", "R_knee", "R_foot", "R_toe"], "L": ["L_thigh", "L_knee", "L_foot", "L_toe"]}
SPINE = ["hip", "lower_spine", "mid_spine", "top_spine"]


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def chain_rows(co, bones, chain):
    """Weights along a bone chain: each segment owned by its first bone, inverse distance^4, two nearest
    segments blended (vanilla's joints are about half and half)."""
    pts = [np.array(bones[b]) for b in chain]
    d = np.stack([_seg_dist(co, pts[i], pts[i + 1])[0] for i in range(len(pts) - 1)], 1)
    w = 1.0 / np.maximum(d, 0.05) ** 4
    idx = np.argsort(-w, axis=1)[:, :2]
    rows = []
    for i in range(len(co)):
        a_, b_ = idx[i]
        tot = w[i, a_] + w[i, b_]
        rows.append({chain[a_]: w[i, a_] / tot, chain[b_]: w[i, b_] / tot})
    return rows, d.min(1)


def rider_weights(obj, bones, influences=4, smooth=12, back=0.8, plume=None, standing=False):
    """M23: region rules over the nearest-surface weights (which only hold where the new mesh hugs vanilla's body;
    the rider's plume, cloak and armour don't: the plume stretched on the shoulders' weights, the cloak mixed the
    spine, hips and thighs, ~100 flips seated against vanilla's 12):
      head: above the neck all on ``head`` (the helmet and plume rigid), faded over 13.9-14.4;
      arms: beyond the shoulders and near an arm's chain, that chain (shoulder to hand), faded in over 1 unit;
      the cloak: behind the back (z over ``back``) and under the shoulders, the spine by height (hip, lower,
      mid and top spine), never a leg or arm;
      legs and the skirt's front halves: under the hips, in front of the back plane, near a leg's chain, that
      chain, faded in under the hips;
    then ``smooth`` passes of neighbour averaging, at most ``influences`` bones. -> counts per region."""
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    co = hs.verts(obj)
    rows = [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in me.vertices]
    counts = {"head": 0, "arms": 0, "cloak": 0, "legs": 0}
    neck0, neck1 = 13.9, 14.4
    hip = np.array(bones["hip"])
    arm_rows = {s_: chain_rows(co, bones, ch) for s_, ch in ARMS.items()}
    leg_rows = {s_: chain_rows(co, bones, ch) for s_, ch in LEGS.items()}
    spine_y = [bones[b][1] for b in SPINE]

    def mix(row, new, f):
        out = {k: w * (1 - f) for k, w in row.items()}
        for k, w in new.items():
            out[k] = out.get(k, 0.0) + w * f
        return out
    if plume is not None and len(plume):  # the plume: all of it rigid with the head (it hangs below the neck)
        from mathutils.kdtree import KDTree
        tree = KDTree(len(plume))
        for i, q in enumerate(plume):
            tree.insert(q, i)
        tree.balance()
        on_plume = np.array([tree.find(q)[2] < 0.6 for q in co])
    else:
        on_plume = np.zeros(len(co), dtype=bool)
    counts["plume"] = int(on_plume.sum())
    for i, p in enumerate(co):
        x, y, z = p
        if on_plume[i]:
            rows[i] = {"head": 1.0}
            continue
        if y > neck0:
            rows[i] = mix(rows[i], {"head": 1.0}, float(fh.smoothstep(np.array(y), neck0, neck1)))
            counts["head"] += 1
            continue
        side = "L" if x > 0 else "R"
        sh = abs(bones[f"{side}_shoulder"][0])
        arow, adist = arm_rows[side]
        if abs(x) > sh and adist[i] < 1.2 and y > hip[1] and z <= back:
            rows[i] = mix(rows[i], arow[i], float(fh.smoothstep(np.array(abs(x)), sh, sh + 1.0)))
            counts["arms"] += 1
            continue
        # standing, only above the hips (the user, T38: the English soldier's calves and seat, 14% of him, took the
        # spine's weights and stayed behind while his legs walked; below the hips a coat moves with the legs)
        if z > back and y < bones["top_spine"][1] and not (standing and y < hip[1]):  # the cloak behind: the spine by height
            k = int(np.clip(np.searchsorted(spine_y, y) - 1, 0, len(SPINE) - 2))
            f = float(np.clip((y - spine_y[k]) / (spine_y[k + 1] - spine_y[k]), 0, 1))
            rows[i] = {SPINE[k]: 1 - f, SPINE[k + 1]: f}
            counts["cloak"] += 1
            continue
        lrow, ldist = leg_rows[side]
        if y < hip[1] and ldist[i] < 2.2:
            rows[i] = mix(rows[i], lrow[i], float(fh.smoothstep(np.array(hip[1] - y), 0.0, 1.0)))
            counts["legs"] += 1
    nb = [[] for _ in rows]
    for e in me.edges:
        a_, b_ = e.vertices
        nb[a_].append(b_)
        nb[b_].append(a_)
    for _ in range(smooth):
        new = []
        for i, row in enumerate(rows):
            if not nb[i]:
                new.append(row)
                continue
            m = {}
            for j in nb[i]:
                for k, w in rows[j].items():
                    m[k] = m.get(k, 0.0) + w / len(nb[i])
            new.append({k: 0.5 * row.get(k, 0.0) + 0.5 * m.get(k, 0.0) for k in set(row) | set(m)})
        rows = new
    for v, row in zip(me.vertices, rows):
        top = sorted(((k, w) for k, w in row.items() if w >= 0.01), key=lambda kv: -kv[1])[:influences]
        total = sum(w for _, w in top) or 1.0
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        for k, w in top:
            (obj.vertex_groups.get(k) or obj.vertex_groups.new(name=k)).add([v.index], w / total, "REPLACE")
    return counts


def cloth_weights(obj, bones, behind=0.6, influences=4):
    """The cloth hanging below the hips and behind the legs (the cloak's back and the skirt's back: farther than
    ``behind`` behind the nearest leg bone) takes the hips' weights, not a leg's: seated, the legs swing forward
    and the cloak dragged on them tore (the leg weights also pulled it into the horse). -> vertices changed."""
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    legs = {"L_thigh", "R_thigh", "L_knee", "R_knee", "L_foot", "R_foot", "L_toe", "R_toe"}
    hip = np.array(bones["hip"])
    n = 0
    for v in me.vertices:
        row = {names[g.group]: g.weight for g in v.groups if g.weight > 0}
        leg_w = sum(w for k, w in row.items() if k in legs)
        if leg_w <= 0 or v.co.y > hip[1]:
            continue
        side = "L" if v.co.x > 0 else "R"
        knee = np.array(bones[f"{side}_knee"])
        if v.co.z - knee[2] < behind:  # in front of or round the leg: it moves with it
            continue
        keep = {k: w for k, w in row.items() if k not in legs}
        f = float(np.clip((v.co.z - knee[2] - behind) / 0.8, 0.0, 1.0))  # blended in over 0.8 behind
        new = {k: w for k, w in row.items() if k in legs}
        new = {k: w * (1 - f) for k, w in new.items()}
        new.update({k: keep.get(k, 0.0) for k in keep})
        new["hip"] = new.get("hip", 0.0) + leg_w * f
        new = dict(sorted(new.items(), key=lambda kv: -kv[1])[:influences])  # the format's limit
        total = sum(new.values()) or 1.0
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        for k, w in new.items():
            if w > 1e-3:
                (obj.vertex_groups.get(k) or obj.vertex_groups.new(name=k)).add([v.index], w / total, "REPLACE")
        n += 1
    return {"behind_legs_on_hips": n}


main()
