"""Barding panels and stirrups for a fresh horse (Track C, test 15 round 3, step 4; Blender jobs
only: numpy). PDX mesh space: Y up, the horse facing -Z, its left side at +X.

Panels are a thin shell hanging free round the body, not part of its surface: the generated
horse's fused skirts are cut off at the belly, and these hang in their place.
  * outline: the convex hull (x, z) of the body's slice at ``top`` (the legs crossing it are
    inside, so the panels hang outside them), pushed out by ``offset``, resampled every
    ``column``;
  * each column hangs straight down from the outline to its hem, flaring out by ``flare`` per
    unit of drop; its top row tucks ``tuck`` up into the body so there's no gap at the seam, and
    its last row is the hem band (``band_h``: painted gold, which is per face);
  * hems by position along the body (``hems``: [[z, y], ...], interpolated), never lower than
    the vanilla knee or hock + ``leg_clear`` (the legs show under them);
  * slits (no faces between two columns) at the ``slits`` z positions on both sides and at the
    front and rear centre, so the panels hang (and later swing) apart;
  * two-sided: every face again, reversed, ``back`` inside.
Stirrups: a trapezoid iron loop (a square bar) across the horse under the rider's boot sole
(``tread``, the left one; the right mirrored) and a strap up to ``strap_top``.
Tail (when the generated one is missing, fused or a stub: _horse_measure.tail_state): a tapered
tube along the vanilla tail chain from a root buried in the rump, its own piece, weighted along
the chain like vanilla's (two neighbouring bones per vertex).
Each part comes back as arrays (vertices, triangles, per-face part and hem, per-vertex anchor:
the point whose body weights it takes).
"""
import numpy as np

CLOTH_PANEL, SCALE_PANEL, IRON, LEATHER, TAIL = 2, 3, 4, 5, 6
TAIL_NODES = ("HorseSpineEnd", "HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd")


def hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(map(tuple, np.round(points, 5))))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def resample(loop: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` (about) along a closed polygon."""
    closed = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = seg.sum()
    n = max(8, int(round(total / step)))
    at = np.linspace(0, total, n, endpoint=False)
    cum = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for s in at:
        i = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-9)
        out.append(closed[i] + (closed[i + 1] - closed[i]) * t)
    return np.array(out)


def outline(body: np.ndarray, top: float, z0: float, z1: float, offset: float, column: float, slab: float = 0.4):
    """The panels' outline (x, z): hull of the body slice at ``top`` between ``z0`` and ``z1``,
    pushed out by ``offset`` from its centroid, every ``column``."""
    sl = body[(np.abs(body[:, 1] - top) < slab) & (body[:, 2] > z0) & (body[:, 2] < z1)][:, [0, 2]]
    h = hull(sl)
    h = np.vstack([h, h * [-1, 1]])  # mirrored: the outline is symmetric even if the body isn't quite
    h = hull(h)
    c = h.mean(axis=0)
    d = h - c
    h = h + d / np.linalg.norm(d, axis=1, keepdims=True) * offset
    return resample(h, column), c


def panels(body: np.ndarray, bones: dict, top: float = 8.0, offset: float = 0.08, column: float = 0.6,
           flare: float = 0.12, tuck: float = 0.5, hems=None, slits=None, leg_clear: float = 0.4,
           back: float = 0.02, seam_z: float | None = None, band_h: float = 0.35) -> dict:
    """The hanging panels (module doc). ``hems``: [[z, y], ...] the hem height along the body
    (front to back); ``slits``: z positions where the side panels part; ``seam_z``: scale panels
    in front of it, cloth behind. -> {"v", "f", "part", "hem", "anchor", "log"}."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    z0, z1 = B["HorseNeckBase"][2] - 2.0, B["HorseSpineEnd"][2] + 3.0
    loop, centre = outline(body, top, z0, z1, offset, column)
    knee = min(B[f"HorseFront{s}Knee"][1] for s in ("Left", "Right"))
    hock = min(B[f"HorseBack{s}Knee"][1] for s in ("Left", "Right"))
    mid_z = (B["HorseFrontLeftUp"][2] + B["HorseBackLeftUp"][2]) / 2
    hems = np.array(hems or [[z0, 4.7], [z1, 4.7]], dtype=np.float64)
    hem_ref = np.interp(loop[:, 1], hems[:, 0], hems[:, 1])
    floor = np.where(loop[:, 1] < mid_z, knee, hock) + leg_clear
    hem = np.maximum(hem_ref, floor)
    seam_z = B["HorseSpineTop"][2] + 0.5 if seam_z is None else seam_z
    # slits: a column pair is left open where the outline crosses a slit z on either side, and at
    # the front and rear centre (x changes sign)
    n = len(loop)
    nxt = np.roll(np.arange(n), -1)
    open_ = np.zeros(n, dtype=bool)
    for zs in (slits or []):
        open_ |= (loop[:, 1] - zs) * (loop[nxt, 1] - zs) < 0
    open_ |= (loop[:, 0] * loop[nxt, 0] < 0)
    v, f, part, fhem, anchor = [], [], [], [], []
    cols = []
    for i, (x, z) in enumerate(loop):
        d = np.array([x, z]) - centre
        d /= np.linalg.norm(d)
        drop = top - hem[i]
        bottom = np.array([x, z]) + d * flare * drop
        band = np.array([x, z]) + d * flare * (drop - band_h)
        inner = centre + (np.array([x, z]) - centre) * 0.88
        k = len(v)
        v += [[inner[0], top + tuck, inner[1]], [x, top, z], [band[0], hem[i] + band_h, band[1]],
              [bottom[0], hem[i], bottom[1]]]
        anchor += [[x, top, z]] * 4
        cols.append(k)
    for i in range(n):
        if open_[i]:
            continue
        a, b = cols[i], cols[nxt[i]]
        kind = SCALE_PANEL if (loop[i, 1] + loop[nxt[i], 1]) / 2 < seam_z else CLOTH_PANEL
        h = (hem[i] + hem[nxt[i]]) / 2
        for r in range(3):  # tuck, panel, hem band; outward-facing (the loop runs counter-clockwise seen from above)
            quad = [(a + r, b + r, b + r + 1), (a + r, b + r + 1, a + r + 1)]
            for t in quad:
                f.append(t)
                part.append(kind)
                fhem.append(h)
    v, f = np.array(v), np.array(f)
    # the back faces: the same columns pushed ``back`` towards the centre, reversed
    vb = v.copy()
    d = vb[:, [0, 2]] - centre
    vb[:, [0, 2]] -= d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * back
    fb = f[:, ::-1] + len(v)
    out_f = np.vstack([f, fb])
    return {"v": np.vstack([v, vb]), "f": out_f, "part": part * 2, "hem": fhem * 2,
            "anchor": np.vstack([anchor, anchor]),
            "log": {"columns": n, "open": int(open_.sum()), "tris": int(len(out_f)),
                    "hem_reference": [round(float(hem_ref.min()), 2), round(float(hem_ref.max()), 2)],
                    "hem_used": [round(float(hem.min()), 2), round(float(hem.max()), 2)],
                    "limited_by_legs": int((hem > hem_ref + 1e-6).sum()), "knee": round(float(knee), 2),
                    "hock": round(float(hock), 2)}}


def _bar(a: np.ndarray, b: np.ndarray, t: float, up=(0.0, 0.0, 1.0)):
    """A square bar from ``a`` to ``b`` of side ``t``: (8 vertices, 8 side triangles)."""
    d = b - a
    d = d / np.linalg.norm(d)
    s1 = np.cross(d, up)
    if np.linalg.norm(s1) < 1e-6:
        s1 = np.cross(d, (1.0, 0.0, 0.0))
    s1 = s1 / np.linalg.norm(s1) * t / 2
    s2 = np.cross(d, s1)
    s2 = s2 / np.linalg.norm(s2) * t / 2
    ring = [s1 + s2, -s1 + s2, -s1 - s2, s1 - s2]
    v = [a + r for r in ring] + [b + r for r in ring]
    f = []
    for k in range(4):
        k2 = (k + 1) % 4
        f += [(k, k2, 4 + k2), (k, 4 + k2, 4 + k)]
    return np.array(v), np.array(f)


def stirrups(tread, strap_top, width: float = 0.9, height: float = 0.9, bar: float = 0.08,
             strap: tuple = (0.18, 0.04)) -> dict:
    """Both stirrups (module doc): ``tread`` [x, y, z] of the left boot sole, ``strap_top`` where
    the left strap meets the saddle. -> {"v", "f", "part", "hem", "anchor", "log"}."""
    v, f, part = [], [], []

    def add(pv, pf, kind):
        f.extend((pf + len(v)).tolist())
        v.extend(pv.tolist())
        part.extend([kind] * len(pf))
    for s in (1.0, -1.0):
        t = np.array(tread, dtype=np.float64) * [s, 1, 1]
        top = np.array(strap_top, dtype=np.float64) * [s, 1, 1]
        corners = [t + [-width / 2, 0, 0], t + [width / 2, 0, 0], t + [width / 6, height, 0], t + [-width / 6, height, 0]]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            add(*_bar(np.array(a), np.array(b), bar), IRON)
        sv, sf = _bar(t + [0, height, 0], top, 1.0)
        # a flat strap: the bar's cross-section squashed to strap (width along z, thickness along x)
        mid = (sv[:4].mean(axis=0), sv[4:].mean(axis=0))
        flat = []
        for i, p in enumerate(sv):
            c = mid[0] if i < 4 else mid[1]
            q = p - c
            flat.append(c + [np.sign(q[0]) * strap[1] / 2, 0, np.sign(q[2]) * strap[0] / 2])
        add(np.array(flat), sf, LEATHER)
    v = np.array(v)
    return {"v": v, "f": np.array(f), "part": part, "hem": [0.0] * len(part), "anchor": v.copy(),
            "log": {"tris": len(part), "tread": [round(float(x), 2) for x in tread]}}


def tail(bones: dict, root, sides: int = 6, rings: int = 8, radii=((0.0, 0.3), (0.25, 0.36), (0.55, 0.62),
         (0.85, 0.5), (1.0, 0.18)), flat: float = 0.8, offset=(0.0, 0.0, 0.0)) -> dict:
    """The tail (module doc): ``root`` [x, y, z] inside the rump, then the vanilla tail bones
    TailTop -> TailEnd moved by ``offset`` (onto our rump; the weights stay on the bones); ``rings`` cross-sections (``sides``-gons, x radius ``flat`` x the z
    radius) evenly along it, radius by the fraction along (``radii``: [[fraction, radius]]: the
    dock, the hair widening, the taper), both ends capped. Per vertex ``chain``: its position as
    t = node index + fraction along TAIL_NODES (root = node 0), for ``tail_weights``.
    -> {"v", "f", "part", "hem", "anchor", "chain", "log"}."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    nodes = np.array([root] + [B[n] + np.asarray(offset, dtype=np.float64) for n in TAIL_NODES[1:]], dtype=np.float64)
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    rad = np.array(radii, dtype=np.float64)
    v, chain, f = [], [], []
    for k in range(rings):
        s = cum[-1] * k / (rings - 1)
        i = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        frac = (s - cum[i]) / seg[i]
        c = nodes[i] + (nodes[i + 1] - nodes[i]) * frac
        d = nodes[i + 1] - nodes[i]  # the ring's plane: perpendicular to its segment
        d /= np.linalg.norm(d)
        side = np.array([1.0, 0.0, 0.0])
        other = np.cross(d, side)
        other /= np.linalg.norm(other)
        r = np.interp(k / (rings - 1), rad[:, 0], rad[:, 1])
        for j in range(sides):
            a = 2 * np.pi * j / sides
            v.append(c + side * np.cos(a) * r * flat + other * np.sin(a) * r)
            chain.append(i + frac)
    for k in range(rings - 1):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            f += [(a, b, b + sides), (a, b + sides, a + sides)]
    for k, t, flip in ((0, 0.0, True), (rings - 1, float(len(seg)), False)):  # caps: a fan round an end point
        centre = len(v)
        v.append(np.mean(v[k * sides:(k + 1) * sides], axis=0))
        chain.append(t)
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            f.append((centre, b, a) if flip else (centre, a, b))
    v = np.array(v)
    # outward: the side faces' winding checked against the tube's axis at one face
    a, b, c = v[f[0][0]], v[f[0][1]], v[f[0][2]]
    if np.dot(np.cross(b - a, c - a), a - nodes[0]) < 0:
        f = [t[::-1] for t in f]
    n = len(f)
    return {"v": v, "f": np.array(f), "part": [TAIL] * n, "hem": [0.0] * n, "anchor": v.copy(),
            "chain": np.array(chain), "log": {"tris": n, "length": round(float(cum[-1]), 2),
                                              "root": [round(float(x), 2) for x in root]}}


def tail_weights(t: float) -> dict:
    """Bone weights at chain position ``t`` (``tail``): each bone (TAIL_NODES) is whole at the
    middle of its own segment and blends with the next round each joint (vanilla's tail: about
    half and half at a joint); the root's segment is HorseSpineEnd's."""
    u = float(np.clip(t - 0.5, 0.0, len(TAIL_NODES) - 1))
    i = min(int(np.floor(u)), len(TAIL_NODES) - 2)
    frac = u - i
    out = {TAIL_NODES[i]: 1.0 - frac, TAIL_NODES[i + 1]: frac}
    return {k: w for k, w in out.items() if w > 1e-3}


def _panel_runs(open_: np.ndarray) -> list[list[int]]:
    """The panels: runs of column indices between slits round the closed loop (column i is in a
    run with i + 1 when the face between them is kept)."""
    n = len(open_)
    if not open_.any():
        return [list(range(n))]
    start = int(np.argmax(open_))  # an open gap: the next column starts a panel
    runs, cur = [], []
    for k in range(1, n + 1):
        i = (start + k) % n
        cur.append(i)
        if open_[i]:  # no face from i onwards: the panel ends at i
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def drape(body: np.ndarray, bones: dict, offset: float = 0.08, column: float = 0.7, flare: float = 0.1,
          tuck: float = 0.5, hems=None, slits=None, hem_length: float = 0.25, leg_clear: float = 0.4,
          readable: float = 0.55, back: float = 0.02, seam_z: float | None = None, band_h: float = 0.3,
          fold: float = 0.14, fold_len: float = 2.1, sag: float = 0.25, corner: float = 0.4,
          tail_open: float = 1.5, top: float | None = None, leg_lift: float = 0.9, leg_cut: float = 1.1,
          rider_cut=(8.0, 1.6), above: float = 1.2, min_drop: float = 0.8) -> dict:
    """Barding panels as draped cloth (test 16: the straight hanging slabs read as a box). They
    start tucked into the barrel at its widest level (``top``; when None, the level between y 7.5
    and 10.5 whose body slice is widest) round the outline there, then fall with ``flare`` and
    folds: out and in by up to ``fold``, a wave every ``fold_len`` along the outline, growing from
    nothing at the top to full at the hem. Each panel's hem sags ``sag`` in its middle and its
    corners round up by ``corner``.
    Hems: the reference's (``hems``), never above the body's belly - 0.3, nor below a floor set by
    ``hem_length``: 0 = game-readable (``readable`` of the way from knee or hock up to the belly,
    so the legs show), 1 = reference-accurate (knee or hock + ``leg_clear`` only); below 0 shorter
    still (round 6, the readability rule's last step): -1 = ``above`` over the belly, the barrel's
    underside showing, -2 twice that (a band round the barrel). A hem stays ``min_drop`` under the
    top edge.
    Slits as panels(), at the front centre, and an opening ``tail_open`` wide at the rear centre
    for the tail. Without ``slits`` (round 5: one continuous caparison, not a row of aprons) the hem is
    one curve cut up by ``leg_lift`` over each leg (a Gaussian ``leg_cut`` wide along z), so the
    legs swing free under it; with slits each panel's hem sags (as round 4). ``rider_cut`` (y, width):
    beside the saddle the top edge dips to y, a Gaussian ``width`` wide along z, under the seated
    rider's leg (hung from the widest level all along, the cloth hid the rider's thighs); None: a
    straight top. Two-sided below the top.
    -> {"v", "f", "part", "hem", "anchor", "log"}."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    z0, z1 = B["HorseNeckBase"][2] - 2.0, B["HorseSpineEnd"][2] + 3.0
    if top is None:
        best, top = -1.0, 8.0
        for y in np.arange(7.5, 10.51, 0.25):
            sl = body[(np.abs(body[:, 1] - y) < 0.2) & (body[:, 2] > z0) & (body[:, 2] < z1)]
            if len(sl) > 8 and np.abs(sl[:, 0]).max() > best:
                best, top = float(np.abs(sl[:, 0]).max()), float(y)
    loop, centre = outline(body, top, z0, z1, offset, column)
    n = len(loop)
    nxt = np.roll(np.arange(n), -1)
    mid_z = (B["HorseFrontLeftUp"][2] + B["HorseBackLeftUp"][2]) / 2
    knee = min(B[f"HorseFront{s}Knee"][1] for s in ("Left", "Right"))
    hock = min(B[f"HorseBack{s}Knee"][1] for s in ("Left", "Right"))
    under = body[(np.abs(body[:, 0]) < 0.5) & (body[:, 2] > B["HorseFrontLeftUp"][2] + 1.0) &
                 (body[:, 2] < B["HorseBackLeftUp"][2] - 1.0) & (body[:, 1] > max(knee, hock) + 1.0)]  # the centre line, not the legs
    belly = float(np.percentile(under[:, 1], 2)) if len(under) else top - 2.0
    joint = np.where(loop[:, 1] < mid_z, knee, hock)
    floor_read = joint + (belly - joint) * readable
    floor_ref = joint + leg_clear
    hl = float(np.clip(hem_length, -2.0, 1.0))
    if hl >= 0:
        floor, cap = floor_read + (floor_ref - floor_read) * hl, belly - 0.3
    else:  # shorter than game-readable (round 6): up past the belly, ``above`` over it at -1, on at -2
        floor = floor_read + (belly + above - floor_read) * -hl
        cap = belly - 0.3 + (above + 0.3) * -hl
    hems = np.array(hems or [[z0, 4.7], [z1, 4.7]], dtype=np.float64)
    hem_ref = np.interp(loop[:, 1], hems[:, 0], hems[:, 1])
    hem = np.minimum(np.maximum(hem_ref, floor), cap)
    open_ = np.zeros(n, dtype=bool)
    for zs in (slits or []):
        open_ |= (loop[:, 1] - zs) * (loop[nxt, 1] - zs) < 0
    crosses = loop[:, 0] * loop[nxt, 0] < 0
    open_ |= crosses & (loop[:, 1] < mid_z)  # the front centre
    open_ |= (loop[:, 1] > mid_z) & (crosses | (np.abs(loop[:, 0]) < tail_open / 2) |
                                     (np.abs(loop[nxt, 0]) < tail_open / 2))  # round the tail
    runs = _panel_runs(open_)
    if slits:
        for run in runs:  # each panel's hem: sagging in the middle, the corners rounded up
            q = np.linspace(0.0, 1.0, len(run))
            hem[run] = hem[run] - sag * (1 - (2 * q - 1) ** 2) + corner * np.clip(1 - np.minimum(q, 1 - q) / 0.2, 0, 1) ** 2
    else:  # one caparison: the hem rises over each leg, sags between them
        leg_z = [B["HorseFrontLeftUp"][2], B["HorseBackLeftUp"][2]]
        lift = sum(np.exp(-((loop[:, 1] - z) / leg_cut) ** 2) for z in leg_z)
        hem = hem - sag + leg_lift * np.clip(lift, 0, 1)
        for run in runs:  # the edges at the chest and tail openings round up
            q = np.linspace(0.0, 1.0, len(run))
            hem[run] = hem[run] + corner * np.clip(1 - np.minimum(q, 1 - q) / 0.06, 0, 1) ** 2
    seg = np.linalg.norm(loop[nxt] - loop, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)[:-1]])
    wave = np.sin(2 * np.pi * arc / fold_len)
    seam_z = B["HorseSpineTop"][2] + 0.5 if seam_z is None else seam_z
    fracs = (0.0, 0.4, 0.75)  # of the drop: the top and two fold rows; then the band and the hem
    v, anchor, cols = [], [], []
    tops = np.full(n, top)
    if rider_cut is not None and rider_cut[0] < top:
        dip = np.exp(-((loop[:, 1] - B["Saddle_Node"][2] + 1.0) / rider_cut[1]) ** 2)  # round the rider's lower leg
        tops = top - (top - rider_cut[0]) * dip
    hem = np.minimum(hem, tops - min_drop)  # a short hem never meets the top (beside the saddle it dips)
    for i, (x, z) in enumerate(loop):
        d = np.array([x, z]) - centre
        d /= np.linalg.norm(d)
        ti = tops[i]
        drop = ti - hem[i]
        cols.append(len(v))
        inner = centre + (np.array([x, z]) - centre) * (0.88 - 0.08 * (top - ti) / max(top - 7.0, 1e-6))
        v.append([inner[0], ti + tuck, inner[1]])
        rows = [(ti - f * drop, f) for f in fracs] + [(hem[i] + band_h, (drop - band_h) / drop), (hem[i], 1.0)]
        for y, t in rows:
            p = np.array([x, z]) + d * (flare * (ti - y) + fold * t * wave[i])
            v.append([p[0], y, p[1]])
        anchor += [[x, ti, z]] * (len(rows) + 1)
    per = len(fracs) + 3  # points per column: tuck, top, two fold rows, band, hem
    f, part, fhem, hang = [], [], [], []
    for i in range(n):
        if open_[i]:
            continue
        a, b = cols[i], cols[nxt[i]]
        kind = SCALE_PANEL if (loop[i, 1] + loop[nxt[i], 1]) / 2 < seam_z else CLOTH_PANEL
        h = (hem[i] + hem[nxt[i]]) / 2
        for r in range(per - 1):  # outward-facing (the loop runs counter-clockwise seen from above)
            for t in ((a + r, b + r, b + r + 1), (a + r, b + r + 1, a + r + 1)):
                f.append(t)
                part.append(kind)
                fhem.append(h)
                hang.append(r >= 1)  # below the top: seen from inside too
    v, f, hang = np.array(v), np.array(f), np.array(hang)
    vb = v.copy()
    dd = vb[:, [0, 2]] - centre
    vb[:, [0, 2]] -= dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-9) * back
    fb = f[hang][:, ::-1] + len(v)
    out_f = np.vstack([f, fb])
    allv, alla = np.vstack([v, vb]), np.vstack([anchor, anchor])
    used = np.unique(out_f)  # the back copies of the tuck row (and open columns) have no faces
    remap = np.full(len(allv), -1)
    remap[used] = np.arange(len(used))
    out_f = remap[out_f]
    return {"v": allv[used], "f": out_f,
            "part": part + [p for p, k in zip(part, hang) if k], "hem": fhem + [h for h, k in zip(fhem, hang) if k],
            "anchor": alla[used],
            "log": {"style": "drape", "top": round(top, 2), "belly": round(belly, 2), "columns": n,
                    "panels": len(runs), "open": int(open_.sum()), "tris": int(len(out_f)),
                    "hem_reference": [round(float(hem_ref.min()), 2), round(float(hem_ref.max()), 2)],
                    "hem_used": [round(float(hem.min()), 2), round(float(hem.max()), 2)],
                    "hem_length": hem_length}}
