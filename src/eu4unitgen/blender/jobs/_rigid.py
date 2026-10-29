"""Rigid parts (the user, after the T39 review sheets: "weapons, shields and plumes must be rigid"). Pure Python, so
the Blender jobs (``fresh_rider``: the weapon straightened when it is split off) and the app (``ref3d.review`` and
the regression set: measured) share one measure.

  bow          how far a long weapon's shaft strays from the straight line between its ends, as a share of its
               length. The shaft is the thin part: its length cut into ``BINS`` slices, each slice's middle (its
               points' mean across the axis) and spread (the median distance from that middle); slices spread more than
               ``WIDE`` times the median slice are the head (a bill's blade, a tassel) and left out
  straighten   each point moved by the shaft's bend at its place along the length: the slices' middles (interpolated
               between slices, across the head's) less the straight chord between the end slices; past the end
               slices, none. The ends stay where they were and the head rides on the shaft. TRELLIS.2's cataphract
               lance waves about one and a half times along its length, 2.3% of it off the line (0.41 over 18.2:
               plainly bent near the tip in the review sheets, 2026-10-29); a fitted curve couldn't follow it
  distortion   how much a part's shape changes from its rest pose to a posed frame: the largest change of distance
               between sampled points, as a share of the part's size (0 for a part on one bone)
"""
import math

BINS = 24
WIDE = 1.5
BOW_MAX = 0.004  # a straight weapon: under 0.4% of its length off the line (0.07 on an 18-long lance)
RIGID_MAX = 0.002


def _unit(v):
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _median(v):
    s = sorted(v)
    n = len(s)
    return 0.5 * (s[(n - 1) // 2] + s[n // 2]) if n else 0.0


def axis(pts):
    """The points' mean and principal axis (power iteration on the covariance)."""
    n = len(pts)
    m = [sum(p[k] for p in pts) / n for k in range(3)]
    c = [[sum((p[i] - m[i]) * (p[j] - m[j]) for p in pts) for j in range(3)] for i in range(3)]
    v = [1.0, 0.9, 1.1]
    for _ in range(60):
        v = _unit([sum(c[i][j] * v[j] for j in range(3)) for i in range(3)])
    return m, v


def shaft(pts, bins=BINS, wide=WIDE):
    """The weapon's frame and its shaft's slices (each slice's middle: its points' mean, a ring's centre): {"centre", "axis", "e1", "e2", "lo", "hi", "t", "uv", "slices":
    [(t, u, v)]} (``t`` along the axis, ``u``, ``v`` across it, per point; the slices' middles)."""
    c, ax = axis(pts)
    other = [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = _unit(_cross(ax, other))
    e2 = _cross(ax, e1)
    rel = [[p[k] - c[k] for k in range(3)] for p in pts]
    t = [sum(r[k] * ax[k] for k in range(3)) for r in rel]
    uv = [(sum(r[k] * e1[k] for k in range(3)), sum(r[k] * e2[k] for k in range(3))) for r in rel]
    lo, hi = min(t), max(t)
    span = (hi - lo) or 1.0
    groups = [[] for _ in range(bins)]
    for i, ti in enumerate(t):
        groups[min(bins - 1, int((ti - lo) / span * bins))].append(i)
    rows = []
    for b, g in enumerate(groups):
        if len(g) < 3:
            continue
        mu, mv = sum(uv[i][0] for i in g) / len(g), sum(uv[i][1] for i in g) / len(g)
        r = _median([math.hypot(uv[i][0] - mu, uv[i][1] - mv) for i in g])
        rows.append((lo + (b + 0.5) * span / bins, mu, mv, r))
    rmed = _median([r[3] for r in rows]) if rows else 0.0
    slices = [(tc, mu, mv) for tc, mu, mv, r in rows if r <= wide * rmed]
    return {"centre": c, "axis": ax, "e1": e1, "e2": e2, "lo": lo, "hi": hi, "t": t, "uv": uv, "slices": slices}


def bow(pts) -> float:
    """The shaft's largest distance off the chord between its end slices, as a share of the weapon's length."""
    s = shaft(pts)
    sl = s["slices"]
    if len(sl) < 4:
        return 0.0
    (t0, u0, v0), (t1, u1, v1) = sl[0], sl[-1]
    dev = 0.0
    for tc, mu, mv in sl:
        f = (tc - t0) / ((t1 - t0) or 1.0)
        dev = max(dev, math.hypot(mu - (u0 + f * (u1 - u0)), mv - (v0 + f * (v1 - v0))))
    return dev / ((s["hi"] - s["lo"]) or 1.0)


def straighten(pts):
    """The weapon's points with its shaft straightened (module doc). -> (points, {"bow_before", "bow_after"})."""
    s = shaft(pts)
    sl = s["slices"]
    before = bow(pts)
    if len(sl) < 3:
        return [list(p) for p in pts], {"bow_before": round(before, 5), "bow_after": round(before, 5), "skipped": True}
    (t0, u0, v0), (t1, u1, v1) = sl[0], sl[-1]

    def bend(t):  # the slices' middles, interpolated, less the chord; none past the end slices
        if t <= t0 or t >= t1:
            return 0.0, 0.0
        k = next(i for i in range(1, len(sl)) if sl[i][0] >= t)
        (ta, ua, va), (tb, ub, vb) = sl[k - 1], sl[k]
        g = (t - ta) / ((tb - ta) or 1.0)
        f = (t - t0) / ((t1 - t0) or 1.0)
        return (ua + g * (ub - ua) - (u0 + f * (u1 - u0)), va + g * (vb - va) - (v0 + f * (v1 - v0)))
    out = []
    for p, ti in zip(pts, s["t"]):
        du, dv = bend(ti)
        out.append([p[k] - du * s["e1"][k] - dv * s["e2"][k] for k in range(3)])
    return out, {"bow_before": round(before, 5), "bow_after": round(bow(out), 5)}


def surface(p, tri, per=0.02):
    """Points spread over a mesh's surface (flat ``p`` and ``tri``): each triangle's corners and points on it about
    ``per`` apart (a low-poly part's few vertices bias a slice's middle)."""
    pts = [p[i:i + 3] for i in range(0, len(p), 3)]
    out = list(pts)
    for k in range(0, len(tri), 3):
        a, b, c = (pts[tri[k + j]] for j in range(3))
        n = max(1, min(12, int(max(math.dist(a, b), math.dist(b, c), math.dist(a, c)) / per)))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                w1, w2 = i / n, j / n
                out.append([a[q] + w1 * (b[q] - a[q]) + w2 * (c[q] - a[q]) for q in range(3)])
    return out


def distortion(rest, posed, n=40) -> float:
    """The part's largest change of shape between ``rest`` and ``posed`` (the same points, in order), as a share of
    its size (module doc)."""
    if len(rest) != len(posed) or len(rest) < 2:
        return 0.0
    idx = list(range(0, len(rest), max(1, len(rest) // n)))
    size = 0.0
    worst = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            d0 = math.dist(rest[idx[a]], rest[idx[b]])
            size = max(size, d0)
            worst = max(worst, abs(math.dist(posed[idx[a]], posed[idx[b]]) - d0))
    return worst / (size or 1.0)
