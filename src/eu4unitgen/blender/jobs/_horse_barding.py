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
Each part comes back as arrays (vertices, triangles, per-face part and hem, per-vertex anchor:
the point whose body weights it takes).
"""
import numpy as np

CLOTH_PANEL, SCALE_PANEL, IRON, LEATHER = 2, 3, 4, 5


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
