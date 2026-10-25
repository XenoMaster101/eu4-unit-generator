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
Track M (round 13, the user's H3: "the head reads like a dog/donkey; the legs are smooth tubes")
adds a second style, ``anatomy`` 2 (``ANATOMY2``; the game's horses stay on 1): the head from a horse's
side and top profiles (flat face, deep round jaw, the jaw line rising to a narrow muzzle, the chin,
nostrils, eye bulges, the facial crest, the cheek and the mouth line) and cupped ears; the legs from
shaped sections (front, back and side extents per station: the elbow, the knee's block, the cannon's
tendons, the fetlock's bulge, the point of the hock), a pastern sloping forward (``pastern_ahead``)
and a hoof (toe forward of the coronet, heels upright, a flat sole).
``fit`` drives the parameters from the reference model: it builds the default horse, measures
it and the reference the same way (_horse_measure.proportions; topline by rays), and scales
each group by reference / default within ``SANITY``. Next to the saddle the barrel is held under
``rider_clear`` (the vanilla rider's thighs hang there), and above ``rider_knee`` inside a line
narrowing to ``rider_clear_top`` at ``rider_hip``.
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
    # above rider_knee x H the line narrows to rider_clear_top at rider_hip x H (round 6: test 18's B,
    # whose rider's leg showed 0.97 of vanilla's, beside the saddle: 2.1 at y 9.1, 1.58 at 11.0)
    "rider_knee": 0.735, "rider_hip": 0.89, "rider_clear_top": 1.58,
    # anatomy style (1: round 5's tubes; 2: Track M round 13) and the pastern's forward slope (style 2)
    "anatomy": 1, "pastern_ahead": 0.0,
}
# style 2's defaults (fractions of H; a 1.55 m horse): head ~0.36 H long, its depth at the cheek and at
# the muzzle, the forehead's and muzzle's half-widths; leg sections' half-extents (module doc)
ANATOMY2 = {
    "anatomy": 2, "head_length": 0.36, "jowl_depth": 0.21, "muzzle_depth": 0.108, "forehead_width": 0.061,
    "muzzle_width": 0.032, "ear_length": 0.095, "upper_arm": 0.055, "forearm": 0.055, "knee": 0.04,
    "stifle": 0.065, "gaskin": 0.05, "hock": 0.036, "cannon": 0.026, "fetlock": 0.036, "pastern": 0.024,
    "hoof_top": 0.03, "hoof_bottom": 0.04, "hoof_height": 0.045, "pastern_ahead": 0.035,
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


def head_direction(P) -> np.ndarray:
    """The head's axis from the poll: pitched down by head_pitch, turned by head_yaw (deg, + = +X)."""
    pitch, yaw = np.radians(P["head_pitch"]), np.radians(P.get("head_yaw", 0.0))
    return np.array([np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)])


HEAD_T = (-0.08, 1.0)  # the head's stations run from just behind the poll (t < 0) to the muzzle (1)


def head_profile_at(P, t):
    """Track M's free head shape (E-A round 10: the reference's head is under a chanfron that no
    parametric head matches): ``P["head_profile"]`` holds K depth multipliers, K width multipliers
    and K vertical offsets (fractions of the seat height, along the head's up), at K stations evenly
    over HEAD_T. -> (depth x, width x, offset) at ``t``; (1, 1, 0) without a profile (the game's head)."""
    hp = P.get("head_profile")
    if not hp:
        return 1.0, 1.0, 0.0
    k = len(hp) // 3
    ts = np.linspace(HEAD_T[0], HEAD_T[1], k)
    return (float(np.interp(t, ts, hp[:k])), float(np.interp(t, ts, hp[k:2 * k])), float(np.interp(t, ts, hp[2 * k:])))


# ---- style 2: the head. Profiles along the head (t: 0 the poll, 1 the muzzle's front) in head
# lengths: the face (top) and the jaw line (bottom) from the poll-muzzle axis, along the head's up;
# half-widths at the top, the widest (``HEAD2_QM`` of the way up) and the bottom of each section.
HEAD2_T = np.array([-0.06, 0.0, 0.08, 0.18, 0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95, 0.985, 1.0])
HEAD2_TOP = np.array([0.03, 0.07, 0.11, 0.13, 0.135, 0.13, 0.12, 0.115, 0.11, 0.10, 0.08, 0.045, 0.0])
HEAD2_BOT = np.array([-0.12, -0.22, -0.36, -0.44, -0.45, -0.38, -0.28, -0.21, -0.19, -0.22, -0.22, -0.15, -0.06])
HEAD2_WT = np.array([0.10, 0.12, 0.15, 0.17, 0.17, 0.15, 0.115, 0.095, 0.09, 0.10, 0.11, 0.09, 0.05])
HEAD2_WM = np.array([0.12, 0.14, 0.16, 0.17, 0.165, 0.14, 0.11, 0.09, 0.085, 0.10, 0.11, 0.09, 0.05])
HEAD2_WB = np.array([0.10, 0.12, 0.14, 0.14, 0.13, 0.10, 0.08, 0.07, 0.07, 0.08, 0.08, 0.07, 0.04])
HEAD2_QM = 0.2
# reference depth (top - bottom) at the cheek (t 0.3) and the muzzle (t 0.82), half-widths at the
# forehead (t 0.3, top) and the muzzle (t 0.82, top): jowl_depth, muzzle_depth, forehead_width and
# muzzle_width scale the profiles through them
HEAD2_REF = {"cheek": 0.585, "muzzle": 0.30, "forehead": 0.17, "muzzle_w": 0.09}
# relief (head lengths; + outward): (t0, q0, st, sq, amplitude); q is the section's height, -1 jaw to 1 face
HEAD2_BUMPS = [(0.30, 0.50, 0.05, 0.18, 0.022),    # the eye's orbit
               (0.20, 0.80, 0.04, 0.12, -0.012),   # the hollow above the eye
               (0.24, -0.40, 0.10, 0.35, 0.012),   # the cheek (masseter)
               (0.935, 0.30, 0.025, 0.16, -0.030)]  # the nostril's opening
HEAD2_CREST = ((0.30, 0.20), (0.56, 0.05), 0.010)  # the facial crest: a ridge from below the eye forward


def _smoothstep(a, b, x):
    k = np.clip((x - a) / (b - a), 0.0, 1.0)
    return k * k * (3 - 2 * k)


def head2_section(P, H, t):
    """Style 2's head section at ``t`` (scaled by the parameters and Track M's free profile): ->
    (top, bottom, half-widths (top, widest, bottom), superellipse exponent), lengths in scene units."""
    L = P["head_length"] * H
    k = _smoothstep(0.35, 0.8, t)
    sd = (1 - k) * P["jowl_depth"] * H / (HEAD2_REF["cheek"] * L) + k * P["muzzle_depth"] * H / (HEAD2_REF["muzzle"] * L)
    sw = ((1 - k) * P["forehead_width"] * H / (HEAD2_REF["forehead"] * L)
          + k * P["muzzle_width"] * H / (HEAD2_REF["muzzle_w"] * L))
    dm, wm, off = head_profile_at(P, t)
    top, bot = (np.interp(t, HEAD2_T, a) * L * sd * dm + off * H for a in (HEAD2_TOP, HEAD2_BOT))
    w = [np.interp(t, HEAD2_T, a) * L * sw * wm for a in (HEAD2_WT, HEAD2_WM, HEAD2_WB)]
    return top, bot, w, 2.7 - 0.5 * _smoothstep(0.3, 0.9, t)


def _head2_ring_xy(top, bot, w, n, ang):
    """A section's outline in the head's (side, up) coordinates, its normalised height q and |side|."""
    cs, sn = np.cos(ang), np.sin(ang)
    xn = np.sign(cs) * np.abs(cs) ** (2 / n)
    q = np.sign(sn) * np.abs(sn) ** (2 / n)
    wt, wmid, wb = w
    width = np.where(q > HEAD2_QM, wmid + (wt - wmid) * (q - HEAD2_QM) / (1 - HEAD2_QM),
                     wmid + (wb - wmid) * (HEAD2_QM - q) / (1 + HEAD2_QM))
    yc, hh = 0.5 * (top + bot), 0.5 * (top - bot)
    return xn * width, yc + q * hh, q, np.abs(xn)


def head2_relief(t, q, side_w, L):
    """Outward displacement (scene units) of the head's surface: eye, cheek, nostrils, crest, mouth."""
    d = 0.0
    for t0, q0, st, sq, amp in HEAD2_BUMPS:
        d = d + amp * np.exp(-((t - t0) / st) ** 2 - ((q - q0) / sq) ** 2)
    t0, q0 = 0.935, 0.30  # the nostril's rim round its opening
    r = np.sqrt(((t - t0) / 0.025) ** 2 + ((q - q0) / 0.16) ** 2)
    d = d + 0.012 * np.exp(-((r - 1.3) / 0.4) ** 2)
    (ta, qa), (tb, qb), amp = HEAD2_CREST
    k = np.clip(((t - ta) * (tb - ta) + (q - qa) * (qb - qa) / 16) / ((tb - ta) ** 2 + (qb - qa) ** 2 / 16), 0, 1)
    dist = np.sqrt(((t - (ta + k * (tb - ta))) / 0.02) ** 2 + ((q - (qa + k * (qb - qa))) / 0.08) ** 2)
    d = d + amp * np.exp(-dist ** 2)
    d = d - 0.010 * _smoothstep(0.82, 0.88, t) * np.exp(-((q + 0.40) / 0.06) ** 2)  # the mouth's line
    return d * side_w * L  # on the sides only


def head2_point(P, poll, H, t, side, q):
    """A point on style 2's head: ``t`` along it, ``side`` across (+1 the left surface, +X), ``q`` the
    section's height (-1 the jaw line, 1 the face)."""
    hd = head_direction(P)
    _, u, v = _frame(hd)
    top, bot, w, n = head2_section(P, H, t)
    wt, wmid, wb = w
    width = (wmid + (wt - wmid) * (q - HEAD2_QM) / (1 - HEAD2_QM) if q > HEAD2_QM
             else wmid + (wb - wmid) * (HEAD2_QM - q) / (1 + HEAD2_QM))
    xn = side * max(1 - abs(q) ** n, 0.0) ** (1 / n)  # on the superellipse at height q
    x = xn * width
    if side:
        x += np.sign(side) * head2_relief(t, q, abs(xn), P["head_length"] * H)
    return poll + hd * (t * P["head_length"] * H) + u * x + v * (0.5 * (top + bot) + q * 0.5 * (top - bot))


def ear2_tips(P, poll, H):
    """Style 2's ears (left, right): (base, axis, tip); up from the poll, spread, tips tilted forward."""
    hd = head_direction(P)
    _, u, v = _frame(hd)
    L = P["head_length"] * H
    top, _, w, _ = head2_section(P, H, 0.02)
    out = []
    for side in (1.0, -1.0):
        base = poll + hd * 0.02 * L + v * (top - 0.015 * L) + u * side * 0.55 * w[0]
        e = -hd * 0.80 + v * 0.35 + u * side * 0.30
        e /= np.linalg.norm(e)
        E = P["ear_length"] * H
        out.append((base, e, base + e * E + v * 0.08 * E))
    return out


def _head2(P, poll, H, ang_n=36):
    """Style 2's head and ears as closed tubes."""
    hd = head_direction(P)
    _, u, v = _frame(hd)
    L = P["head_length"] * H
    ang = 2 * np.pi * np.arange(ang_n) / ang_n
    rings = []
    for t in np.concatenate([np.linspace(-0.06, 0.9, 49), np.linspace(0.915, 1.0, 9)]):
        top, bot, w, n = head2_section(P, H, t)
        x, y, q, xa = _head2_ring_xy(top, bot, w, n, ang)
        x = x + np.sign(x) * head2_relief(t, q, xa, L)
        if t > 0.97:  # the muzzle's front rounds off
            f = np.sqrt(max(1 - ((t - 0.97) / 0.03) ** 2, 0.2))
            yc = 0.5 * (top + bot)
            x, y = x * f, yc + (y - yc) * f
        rings.append(poll + hd * (t * L) + np.outer(x, u) + np.outer(y, v))
    tubes = [_loft(rings)]
    for base, e, tip in ear2_tips(P, poll, H):
        side = np.sign((base - poll) @ u) or 1.0
        a1 = u * side - e * (e @ (u * side))  # across the ear, towards its open side
        a1 /= np.linalg.norm(a1)
        a2 = np.cross(e, a1)
        ea = ang[::3]
        er = []
        for k in np.linspace(0, 1, 8):
            c = base + (tip - base) * k
            wid = 0.075 * L * (1 - k) ** 0.7 + 0.004 * L
            th = 0.035 * L * (1 - k ** 1.5) + 0.004 * L
            cs, sn = np.cos(ea), np.sin(ea)
            xa = np.where(cs > 0, -0.4 * th * cs, th * cs)  # a crescent: the cup opens to the side
            er.append(c[None, :] + np.outer(xa, a1) + np.outer(wid * sn, a2))
        tubes.append(_loft(er))
    return tubes


# ---- style 2: the legs. Stations (segment of the chain, fraction along it, the parameter the extents
# scale, side, front and back half-extents as multiples of it); segments: 0 Up-Down, 1 Down-Knee,
# 2 Knee-Heel (the cannon), 3 Heel-hoof top (the pastern)
LEG2_FRONT = [(0, 0.0, "upper_arm", 1.0, 1.15, 1.0), (0, 0.6, "upper_arm", 0.9, 1.05, 1.05),
              (1, 0.0, "forearm", 1.25, 1.3, 1.5),     # the elbow
              (1, 0.25, "forearm", 1.1, 1.3, 1.05), (1, 0.7, "forearm", 0.78, 0.88, 0.72),
              (1, 0.9, "knee", 0.95, 0.95, 1.05),      # above the knee (the accessory carpal bone)
              (2, 0.0, "knee", 1.15, 1.1, 1.0),         # the knee
              (2, 0.1, "knee", 0.88, 0.82, 0.78)]
LEG2_HIND = [(0, 0.0, "stifle", 1.2, 1.3, 1.4), (0, 0.6, "stifle", 1.1, 1.2, 1.4),
             (1, 0.0, "stifle", 1.0, 1.15, 1.3),        # the stifle
             (1, 0.35, "gaskin", 1.0, 0.9, 1.25), (1, 0.75, "gaskin", 0.75, 0.65, 0.95),
             (1, 0.92, "hock", 0.95, 0.85, 1.55),       # the point of the hock
             (2, 0.0, "hock", 1.05, 1.0, 1.15),         # the hock
             (2, 0.12, "hock", 0.85, 0.8, 0.85)]
LEG2_LOWER = [(2, 0.3, "cannon", 1.0, 0.9, 1.1), (2, 0.75, "cannon", 1.0, 0.9, 1.2),
              (2, 0.95, "fetlock", 0.92, 0.85, 1.15),
              (3, 0.0, "fetlock", 1.05, 0.9, 1.5),      # the fetlock (the ergot behind)
              (3, 0.35, "pastern", 1.1, 1.05, 1.2), (3, 0.75, "pastern", 1.0, 1.0, 1.0),
              (3, 0.97, "hoof_top", 1.08, 1.0, 1.12)]   # the coronet's band


def _leg_frame(t):
    """(side, back) square to the tangent ``t``: side towards +X, back towards +Z (the horse's rear)."""
    bk = np.array([0.0, 0.0, 1.0]) - t * t[2]
    bk /= np.linalg.norm(bk)
    return np.cross(bk, t), bk


def _ring2(c, side, back, w, f, b, ang, n=2.2):
    cs, sn = np.cos(ang), np.sin(ang)
    x = w * np.sign(cs) * np.abs(cs) ** (2 / n)
    y = np.sign(sn) * np.abs(sn) ** (2 / n) * np.where(sn > 0, b, f)
    return c[None, :] + np.outer(x, side) + np.outer(y, back)


def _leg2(P, H, nodes, front, ang, step=0.15):
    """Style 2's leg: shaped sections along the chain (Up, Down, Knee, Heel, hoof top), then the hoof
    from the hoof top to the ground (``nodes``' last two)."""
    nodes = [np.asarray(q, dtype=np.float64) for q in nodes]
    seg = np.linalg.norm(np.diff(nodes[:5], axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    st = [(cum[i] + f * seg[i], P[p] * ws, P[p] * fs, P[p] * bs) for i, f, p, ws, fs, bs in
          (LEG2_FRONT if front else LEG2_HIND) + LEG2_LOWER]
    st.append((cum[4], P["hoof_top"] * 1.0, P["hoof_top"], P["hoof_top"] * 1.05))
    st = np.array(sorted(st))
    ss = np.unique(np.concatenate([np.linspace(0, cum[4], max(8, int(cum[4] / step))), st[:, 0]]))
    ext = np.stack([np.interp(ss, st[:, 0], st[:, k]) for k in (1, 2, 3)], axis=1) * H
    for _ in range(2):  # soften the station corners, keep the ends
        ext[1:-2] = (ext[:-3] + 2 * ext[1:-2] + ext[2:-1]) / 4
    idx = [min(int(np.searchsorted(cum, s_, side="right") - 1), 3) for s_ in ss]
    pos = np.array([nodes[i] + (nodes[i + 1] - nodes[i]) * (s_ - cum[i]) / seg[i] for i, s_ in zip(idx, ss)])
    tans = np.gradient(pos, axis=0)
    tans /= np.linalg.norm(tans, axis=1, keepdims=True)
    for _ in range(6):
        tans[1:-1] = (tans[:-2] + 2 * tans[1:-1] + tans[2:]) / 4
    rings = []
    for p, t, (w, f, b) in zip(pos, tans, ext):
        side, back = _leg_frame(t / np.linalg.norm(t))
        rings.append(_ring2(p, side, back, w, f, b, ang))
    # the hoof: toe forward of the coronet (the wall ~66 deg), heels nearly upright, a flat sole
    top, ground = nodes[4], nodes[5]
    hh = np.linalg.norm(ground - top)
    down = (ground - top) / max(hh, 1e-9)
    side, back = _leg_frame(down)
    ht, hb = P["hoof_top"] * H, P["hoof_bottom"] * H
    for k in (0.12, 0.4, 0.7, 0.92, 1.0):
        w = ht + (hb - ht) * k
        f = ht + (hb - ht) * k + 0.45 * hh * k
        b = 1.05 * ht + (0.95 * hb - 1.05 * ht) * k
        rings.append(_ring2(top + down * hh * k, side, back, w, f, b, ang))
    return _loft(rings)


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
    clear_y = H * np.array([P["rider_knee"], P["rider_hip"]])
    clear_x = np.array([P["rider_clear"], P["rider_clear_top"]])
    rings = []
    for z, ty, by, ww in zip(zs, t_y, b_y, w):
        f = 1.0  # the ends close as ellipsoids
        lf, lb = P["breast_round"] * H, P["rump_round"] * H
        if z < zF + lf:
            f = np.sqrt(max(1 - ((zF + lf - z) / lf) ** 2, 0.04))
        elif z > zB - lb:
            f = np.sqrt(max(1 - ((z - (zB - lb)) / lb) ** 2, 0.04))
        c = np.array([0.0, (ty + by) / 2 + (1 - f) * (ty - by) * 0.15, z])
        ring = _ring(c, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), ww * f, (ty - by) / 2 * f,
                     P["roundness"], ang, P["top_narrow"])
        # and above the rider's knees inside the line the rider's legs slope in along (round 6: a shallower
        # barrel is widest higher up, where only the cap on the widest kept it in before)
        hi = ring[:, 1] > clear_y[0]
        if hi.any():
            room = np.interp(ring[hi, 1], clear_y, clear_x) / np.maximum(np.abs(ring[hi, 0]), 1e-6)
            ring[:, 0] *= 1 + (min(1.0, float(room.min())) - 1) * min(1.0, 1.6 * float(np.exp(-((z - sz) / 2.0) ** 2)))
        rings.append(ring)
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
    if P.get("anatomy", 1) == 2:
        tubes += _head2(P, B["HorseHead"], H)
        log["head"] = {"length": round(P["head_length"] * H, 2), "pitch": P["head_pitch"], "style": 2}
        for leg in ("FrontLeft", "FrontRight", "BackLeft", "BackRight"):
            chain = [B[f"Horse{leg}{j}"] for j in ("Up", "Down", "Knee", "Heel")]
            heel = chain[-1]
            ahead = P["pastern_ahead"] * H
            hoof_top = B.get(f"Horse{leg}HoofTop", np.array([heel[0], P["hoof_height"] * H, heel[2] - ahead]))
            ground = B.get(f"Horse{leg}Sole", np.array([hoof_top[0], 0.0, hoof_top[2]]))
            tubes.append(_leg2(P, H, chain + [hoof_top, ground], leg.startswith("Front"), ang))
            log[leg] = {"style": 2}
        return {"tubes": tubes, "params": P, "log": log}
    # ---- head: from the poll, pitched down (and turned by head_yaw, positive to the horse's left:
    # Track M fits a reference whose head turns to the viewer; 0, the default, is the game's straight head)
    hd = head_direction(P)
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
        dm, wm, off = head_profile_at(P, t)
        depth, half = depth * dm, half * wm
        c = p - v * depth * 0.35 + v * off * H  # the head's top line runs along the axis; the jaw hangs below
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
        # a posed leg (Track M: a lifted hoof) gives its hoof's top and sole; else straight down to the ground
        ground = B.get(f"Horse{leg}Sole", np.array([chain[-1][0], 0.0, chain[-1][2]]))
        hoof_top = B.get(f"Horse{leg}HoofTop", np.array([chain[-1][0], hoof_h, chain[-1][2]]))
        down = (ground - hoof_top) / max(np.linalg.norm(ground - hoof_top), 1e-9)
        nodes = chain + [hoof_top, ground]
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
                t = down  # the hoof stands upright (along its own axis when posed)
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
