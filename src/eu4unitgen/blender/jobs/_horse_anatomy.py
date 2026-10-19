"""The app's own parametric horse anatomy (Track C, round 5: bake-off horse B; Blender jobs only:
numpy). PDX mesh space: Y up, the horse facing -Z, its left side at +X.

No base-game geometry: the only things taken from the vanilla horse are its skeleton (bone
positions: where each part sits and bends), the seat height (``seat``: the back's surface under
the saddle, so the vanilla rider sits on it) and, in fresh_horse, the sanity limits. Every shape
comes from named parameters (``DEFAULTS``; lengths as fractions of the seat height ``H``, so they
scale with the skeleton) and the profile curves below:
  * barrel: along z from the breast (``breast_ahead`` in front of the front legs' Up bones) to
    the buttock (``rump_behind`` behind the hind ones). Top line: withers (``withers_rise``
    above the seat), the seat at the saddle, croup (``croup_rise``), buttock (``rump_drop`` below
    the croup). Belly line: ``barrel_depth`` under the top at the girth, sagging slightly to the
    saddle, ``tuck_up`` at the flank. Half-widths: breast, ``shoulder_width``, ``rib_width``,
    ``hip_width``, ``buttock_width``. Sections: superellipses (``roundness``), the upper half
    ``top_narrow`` narrower (a horse's ribcage is widest below the middle). Both ends close as
    ellipsoids (``breast_round``, ``rump_round``): no flat caps, no knobs;
  * neck: along the bones NeckBase -> NeckMiddle -> HorseHead (the crest line: the bones sit
    ``crest`` of the way up each section), from inside the withers; depth ``neck_base_depth`` to
    ``neck_throat_depth``, width ``neck_width`` x depth;
  * head: from the poll (HorseHead) ``head_length`` long, pitched ``head_pitch`` degrees below
    the horizontal; depth ``jowl_depth`` at the cheek to ``muzzle_depth``, half-width
    ``forehead_width`` to ``muzzle_width``; two ears ``ear_length`` long;
  * legs: along each leg chain (Up, Down, Knee, Heel, the ground), radii at the joints: front
    ``upper_arm``, ``forearm``, ``knee``, ``cannon``; hind ``stifle``, ``gaskin``, ``hock``,
    ``cannon``; both ``fetlock``, ``pastern``, and a hoof (``hoof_top`` to ``hoof_bottom`` over
    ``hoof_height``, flat under). Sections flattened side to side (``leg_flat``).
``fit`` drives the parameters from the reference model: it builds the default horse, measures
it and the reference the same way (_horse_measure.proportions; topline by rays), and scales
each group by reference / default within ``SANITY``. Next to the saddle the barrel is held under
``rider_clear`` (the vanilla rider's thighs hang there).
"""
import numpy as np

import _horse_measure as hm

DEFAULTS = {
    # barrel
    "barrel_depth": 0.42, "tuck_up": 0.06, "belly_sag": 0.015, "withers_rise": 0.035, "croup_rise": 0.0,
    "rump_drop": 0.09, "breast_ahead": 0.13, "rump_behind": 0.11, "breast_width": 0.11, "shoulder_width": 0.15,
    "rib_width": 0.19, "hip_width": 0.18, "buttock_width": 0.13, "roundness": 2.4, "top_narrow": 0.25,
    "breast_round": 0.1, "rump_round": 0.14,
    # neck
    "neck_base_depth": 0.30, "neck_throat_depth": 0.14, "neck_width": 0.6, "crest": 0.75,
    # head
    "head_length": 0.28, "head_pitch": 40.0, "jowl_depth": 0.19, "forehead_width": 0.09, "muzzle_depth": 0.1,
    "muzzle_width": 0.065, "ear_length": 0.05,
    # legs (radii)
    "upper_arm": 0.085, "forearm": 0.06, "knee": 0.034, "stifle": 0.08, "gaskin": 0.05, "hock": 0.034,
    "cannon": 0.024, "fetlock": 0.031, "pastern": 0.024, "hoof_top": 0.028, "hoof_bottom": 0.037,
    "hoof_height": 0.04, "leg_flat": 0.85,
    # the rider's thighs: the barrel's half-width beside the saddle stays under this (scene units;
    # assumed from round 3's rider_leg measure: the vanilla rider's thighs hang outside x ~2.5)
    "rider_clear": 2.1,
}
GROUPS = {  # which parameters each fitted measurement scales
    "depth": ("barrel_depth",),
    "width": ("breast_width", "shoulder_width", "rib_width", "hip_width", "buttock_width"),
    "neck": ("neck_base_depth",),
    # the reference's neck measures its armour too: the throat and the head take the square root of
    # that factor, so the neck thickens without dwarfing the head (assumed, round 5)
    "neck_sqrt": ("neck_throat_depth", "jowl_depth", "forehead_width", "muzzle_depth", "muzzle_width"),
    "leg": ("upper_arm", "forearm", "knee", "stifle", "gaskin", "hock", "cannon", "fetlock", "pastern",
            "hoof_top", "hoof_bottom"),
}
SANITY = (0.7, 1.6)  # only stops broken results (a missed measurement), not a pull towards any shape


def _smooth(a, passes=3):
    for _ in range(passes):
        a = np.concatenate([a[:1], (a[:-2] + 2 * a[1:-1] + a[2:]) / 4, a[-1:]])
    return a


def _frame(t):
    t = t / np.linalg.norm(t)
    side = np.array([1.0, 0.0, 0.0])
    u = side - t * (t @ side)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    if v[1] < 0:
        v = -v
    return t, u, v


def _ring(c, u, v, a, b, n, ang, narrow=0.0):
    """Superellipse ring: half-axes ``a`` (along u) and ``b`` (along v), exponent ``n``, the half
    towards +v ``narrow`` narrower."""
    cs, sn = np.cos(ang), np.sin(ang)
    x = a * np.sign(cs) * np.abs(cs) ** (2 / n) * (1 - narrow * np.clip(sn, 0, 1))
    y = b * np.sign(sn) * np.abs(sn) ** (2 / n)
    return c[None, :] + np.outer(x, u) + np.outer(y, v)


def _loft(rings, caps=True):
    """A closed tube through ``rings`` (each nang x 3) -> (vertices, triangles); end caps fan to
    each end ring's centre."""
    nang = len(rings[0])
    verts = np.vstack(rings)
    tris = []
    for i in range(len(rings) - 1):
        for j in range(nang):
            a, b = i * nang + j, i * nang + (j + 1) % nang
            tris += [(a, b + nang, b), (a, a + nang, b + nang)]
    if caps:
        extra = []
        for i, flip in ((0, True), (len(rings) - 1, False)):
            centre = len(verts) + len(extra)
            extra.append(rings[i].mean(axis=0))
            for j in range(nang):
                a, b = i * nang + j, i * nang + (j + 1) % nang
                tris.append((centre, a, b) if flip else (centre, b, a))
        verts = np.vstack([verts, extra])
    return verts, np.array(tris)


def _spline(points, step):
    """Catmull-Rom through ``points``, resampled every ``step`` -> (points, tangents)."""
    p = np.asarray(points, dtype=np.float64)
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        for t in np.linspace(0, 1, 24, endpoint=False):
            t2, t3 = t * t, t * t * t
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    dense.append(p[-1])
    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], max(3, int(np.ceil(cum[-1] / step)) + 1))
    out = np.stack([np.interp(s, cum, dense[:, k]) for k in range(3)], axis=1)
    return out, np.gradient(out, axis=0), s / cum[-1]


def build(bones: dict, seat: float, params: dict | None = None, nang: int = 20, step: float = 0.35) -> dict:
    """The horse's parts as closed tubes (module doc). -> {"tubes": [(v, f)], "params", "log"}."""
    P = {**DEFAULTS, **(params or {})}
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    H = float(seat)
    ang = 2 * np.pi * np.arange(nang) / nang
    tubes, log = [], {}
    # ---- barrel: key stations along z, profiles interpolated and smoothed
    fz, bz = B["HorseFrontLeftUp"][2], B["HorseBackLeftUp"][2]
    sz = B["Saddle_Node"][2]
    zF, zB = fz - P["breast_ahead"] * H, bz + P["rump_behind"] * H
    keys_z = np.array([zF, fz + 0.4, fz + 1.5, sz, bz - 1.8, bz, zB])
    top = H + np.array([P["withers_rise"] - 0.15, P["withers_rise"], 0.01, 0.0, 0.0, P["croup_rise"],
                        P["croup_rise"] - P["rump_drop"]]) * H
    girth_bottom = top[2] - P["barrel_depth"] * H
    bottom = np.array([girth_bottom + 0.02 * H, girth_bottom + 0.01 * H, girth_bottom,  # the breast as deep as the girth
                       girth_bottom - P["belly_sag"] * H, girth_bottom + P["tuck_up"] * H,
                       girth_bottom + (P["tuck_up"] + 0.05) * H, B["HorseBackLeftDown"][1] + 0.02 * H])  # the thigh down to the gaskin
    width = np.array([P["breast_width"], P["shoulder_width"], P["rib_width"], P["rib_width"], 0.95 * P["rib_width"],
                      P["hip_width"], P["buttock_width"]]) * H
    zs = np.arange(zF, zB + 1e-6, step)
    t_y, b_y, w = (_smooth(np.interp(zs, keys_z, k)) for k in (top, bottom, width))
    # beside the saddle the vanilla rider's thighs hang: the barrel stays inside them
    near_seat = np.exp(-((zs - sz) / 2.0) ** 2)
    w = w + (np.minimum(w, P["rider_clear"]) - w) * np.clip(near_seat * 1.6, 0, 1)
    rings = []
    for z, ty, by, ww in zip(zs, t_y, b_y, w):
        f = 1.0  # the ends close as ellipsoids
        lf, lb = P["breast_round"] * H, P["rump_round"] * H
        if z < zF + lf:
            f = np.sqrt(max(1 - ((zF + lf - z) / lf) ** 2, 0.04))
        elif z > zB - lb:
            f = np.sqrt(max(1 - ((z - (zB - lb)) / lb) ** 2, 0.04))
        c = np.array([0.0, (ty + by) / 2 + (1 - f) * (ty - by) * 0.15, z])
        rings.append(_ring(c, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), ww * f, (ty - by) / 2 * f,
                           P["roundness"], ang, P["top_narrow"]))
    tubes.append(_loft(rings))
    log["barrel"] = {"z": [round(float(zF), 2), round(float(zB), 2)], "depth": round(float(t_y[len(zs) // 3] - b_y[len(zs) // 3]), 2),
                     "max_half_width": round(float(w.max()), 2), "withers": round(float(top[1]), 2)}
    # ---- neck: the crest along the bones, from inside the withers to the poll
    start = np.array([0.0, top[1] - 0.1 * H, fz + 0.6])
    pts, tan, s = _spline([start, B["HorseNeckBase"], B["HorseNeckMiddle"], B["HorseHead"]], step)
    rings = []
    for p, t, q in zip(pts, tan, s):
        _, u, v = _frame(t)
        d = (P["neck_base_depth"] + (P["neck_throat_depth"] - P["neck_base_depth"]) * q ** 0.8) * H
        c = p - v * d * (P["crest"] - 0.5)  # the bones run near the top of the neck
        rings.append(_ring(c, u, v, d * P["neck_width"] / 2, d / 2, 2.2, ang))
    tubes.append(_loft(rings))
    log["neck"] = {"stations": len(rings), "base_depth": round(P["neck_base_depth"] * H, 2)}
    # ---- head: from the poll, pitched down
    pitch = np.radians(P["head_pitch"])
    hd = np.array([0.0, -np.sin(pitch), -np.cos(pitch)])
    L = P["head_length"] * H
    poll = B["HorseHead"]
    rings = []
    ts = np.linspace(-0.08, 1.0, max(6, int(L / step)))
    for t in ts:
        p = poll + hd * (t * L)
        _, u, v = _frame(hd)
        depth = np.interp(t, [-0.08, 0.2, 0.45, 1.0], [0.8 * P["jowl_depth"], P["jowl_depth"],
                                                      0.5 * (P["jowl_depth"] + P["muzzle_depth"]), P["muzzle_depth"]]) * H
        half = np.interp(t, [-0.08, 0.3, 1.0], [0.85 * P["forehead_width"], P["forehead_width"], P["muzzle_width"]]) * H
        f = np.sqrt(max(1 - ((t - 0.9) / 0.1) ** 2, 0.15)) if t > 0.9 else 1.0
        c = p - v * depth * 0.35  # the head's top line runs along the axis; the jaw hangs below
        rings.append(_ring(c, u, v, half * f, depth / 2 * f, 2.3, ang))
    tubes.append(_loft(rings))
    _, u, v = _frame(hd)
    for side in (1.0, -1.0):  # ears: thin cones up and a little back from the poll
        base = poll + u * side * 0.5 * P["forehead_width"] * H + v * 0.05 * H
        tip = base + (v * 0.9 - hd * 0.45) * P["ear_length"] * H
        e = [_ring(base + (tip - base) * k, u, np.cross(hd, u), 0.16 * (1 - k) + 0.02, 0.1 * (1 - k) + 0.02, 2.0, ang)
             for k in np.linspace(0, 1, 4)]
        tubes.append(_loft(e))
    log["head"] = {"length": round(L, 2), "pitch": P["head_pitch"], "muzzle": [round(float(x), 2) for x in poll + hd * L]}
    # ---- legs
    for leg in ("FrontLeft", "FrontRight", "BackLeft", "BackRight"):
        front = leg.startswith("Front")
        chain = [B[f"Horse{leg}{j}"] for j in ("Up", "Down", "Knee", "Heel")]
        hoof_h = P["hoof_height"] * H
        ground = np.array([chain[-1][0], 0.0, chain[-1][2]])
        nodes = chain + [np.array([chain[-1][0], hoof_h, chain[-1][2]]), ground]
        r_nodes = ([P["upper_arm"], P["forearm"], P["knee"], P["fetlock"], P["hoof_top"], P["hoof_bottom"]] if front else
                   [P["stifle"], P["gaskin"], P["hock"], P["fetlock"], P["hoof_top"], P["hoof_bottom"]])
        seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        # extra stations: the cannon thinner between the knee/hock and the fetlock, the pastern above the hoof
        k_s = list(cum) + [cum[2] + 0.3 * seg[2], cum[2] + 0.7 * seg[2], cum[3] + 0.5 * seg[3]]
        k_r = r_nodes + [P["cannon"] * 1.05, P["cannon"], P["pastern"]]
        order = np.argsort(k_s)
        k_s, k_r = np.array(k_s)[order], np.array(k_r)[order] * H
        ss = np.linspace(0, cum[-1], max(8, int(cum[-1] / (step * 0.7))))
        ss = np.unique(np.concatenate([ss, [cum[-2], cum[-2] + 0.02]]))  # a station at the hoof's top edge
        rad = np.interp(ss, k_s, k_r)
        for _ in range(3):  # smooth joints, not pipe segments (the hoof's edge stays: it is the last stations)
            rad[1:-3] = (rad[:-4] + 2 * rad[1:-3] + rad[2:-2]) / 4
        idx = [min(int(np.searchsorted(cum, s_, side="right") - 1), len(seg) - 1) for s_ in ss]
        pos = np.array([nodes[i] + (nodes[i + 1] - nodes[i]) * (s_ - cum[i]) / seg[i] for i, s_ in zip(idx, ss)])
        # ring planes turn gradually through the joints (rings square to each straight segment
        # overlap at a bend and leave a notch at the knee and hock)
        tans = np.gradient(pos, axis=0)
        tans /= np.linalg.norm(tans, axis=1, keepdims=True)
        for _ in range(6):
            tans[1:-1] = (tans[:-2] + 2 * tans[1:-1] + tans[2:]) / 4
        rings = []
        for s_, r, p, t in zip(ss, rad, pos, tans):
            if s_ >= cum[-2]:
                t = np.array([0.0, -1.0, 0.0])  # the hoof stands upright
            _, u, v = _frame(t)
            flat = P["leg_flat"] if s_ < cum[-2] else 1.0
            rings.append(_ring(p, u, v, r * flat, r * (2 - flat), 2.0, ang))
        tubes.append(_loft(rings))
        log[leg] = {"stations": len(rings)}
    return {"tubes": tubes, "params": P, "log": log}


def _joined(tubes):
    vs, fs, off = [], [], 0
    for v, f in tubes:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def measure(tubes, bones) -> dict:
    """The fitted measurements of a built horse (as fit compares them)."""
    co, tris = _joined(tubes)
    return hm.summary(hm.proportions(co, tris, bones))


def fit(bones: dict, seat: float, ref_co, ref_tris, params: dict | None = None) -> dict:
    """Parameters driven by the reference (module doc). -> {"params", "factors", "reference",
    "default", "topline"}."""
    base = {**DEFAULTS, **(params or {})}
    default = measure(build(bones, seat, base)["tubes"], bones)
    ref = hm.summary(hm.proportions(ref_co, ref_tris, bones))
    P, factors = dict(base), {}
    for key, names in GROUPS.items():
        root = key.endswith("_sqrt")
        k = key.removesuffix("_sqrt")
        r, d = ref.get(k), default.get(k)
        f = 1.0 if not r or not d else float(np.clip(r / d, *SANITY))
        f = f ** 0.5 if root else f
        factors[key] = round(f, 3)
        for n in names:
            P[n] = base[n] * f
    # the topline: the reference's withers and croup heights over the seat (rays down the centre line)
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    tl = {}
    for name, z in (("withers_rise", B["HorseFrontLeftUp"][2] + 0.4), ("croup_rise", B["HorseBackLeftUp"][2])):
        top, _ = hm.top_bottom(ref_co, ref_tris, z)
        if top is not None:
            P[name] = float(np.clip((top - seat) / seat, -0.1, 0.12))
            tl[name] = round(P[name], 3)
    keys = [k for k in GROUPS if not k.endswith("_sqrt")]
    return {"params": P, "factors": factors, "reference": {k: ref.get(k) for k in keys},
            "default": {k: default.get(k) for k in keys}, "topline": tl}
