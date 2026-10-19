"""Horse proportions (Track C, test 15 round 3, step 4; Blender jobs only: numpy). PDX mesh space:
Y up, the horse facing -Z, its left side at +X, fitted to the vanilla horse's skeleton.

What reads at map zoom is the silhouette, so these are silhouette numbers, measured the same way
on our horse and on vanilla's. Thin parts (straps, stirrups, reins, free-hanging barding panels:
a ray's entry and exit hits closer than 0.5) are skipped, so the numbers describe the body:
  * depth: back to belly on the centre line at the girth, the saddle and the flank;
  * width: rays in from either side at a quarter, half and three quarters of the way from belly
    to back (the widest);
  * belly: the belly's height above the ground (how much leg shows under the body);
  * leg: each leg's thickness at the forearm/gaskin (Down-Knee) and the cannon (Knee-Heel)
    midpoints: the side and front extents of a thin slice of surface samples round the leg's own
    centre; thin legs under a deep body read as "sticks";
  * neck: the thickness at HorseNeckMiddle perpendicular to the neck (NeckBase -> Head), by a ray;
  * volume of the closed surface (open meshes: approximate);
  * hem (reported, not judged): the lowest surface that isn't a leg, i.e. a skirt's hem;
  * rider_leg (with a posed rider): the share of its lower legs and feet visible from the side.
``compare`` divides ours by vanilla's and flags what is past its limit.
"""
import numpy as np

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")

# ours / vanilla limits: past these the horse reads bloated or short-legged at map zoom. Assumed
# (test 15 round 3): a barded horse may be somewhat bulkier than vanilla's bare one, not 1.5x.
# Tightened after the in-game test 15 (depth 1.22x and the hem at 0.79x read as a sack on sticks).
LIMITS = {"depth": 1.15, "width": 1.25, "belly": 0.85, "leg": 0.75, "neck": 1.3, "volume": 1.3}
# the seated rider's lower legs seen from the side, ours / vanilla's: the reference shows the boot
# and stirrup outside the caparison (measured only when a posed rider is given)
LIMITS_RIDER = {"rider_leg": 0.9}
LOW_IS_BAD = ("belly", "leg", "rider_leg")


def ray_hits(co, tris, origin, direction):
    """Distances along the ray to every triangle it crosses (Moller-Trumbore, both sides)."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = (e1 * p).sum(axis=1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    t_ = origin - a
    u = (t_ * p).sum(axis=1) * inv
    q = np.cross(t_, e1)
    v = (q @ direction) * inv
    t = (q * e2).sum(axis=1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.sort(t[hit])


def solid_entry(hits, thick=0.5):
    """The first hit that enters something thicker than ``thick`` (its next hit is farther):
    straps, stirrups and reins are skipped. Hits pair up (in, out) on closed surfaces."""
    for i in range(0, len(hits) - 1, 2):
        if hits[i + 1] - hits[i] > thick:
            return hits[i]
    return hits[0] if len(hits) else None


def top_bottom(co, tris, z, x=0.0):
    """(back, belly) heights on the vertical line at (x, z): the first surface going down from
    above, the first going up from below the ground. None where the line misses."""
    down = solid_entry(ray_hits(co, tris, np.array([x, 60.0, z]), np.array([0.0, -1.0, 0.0])))
    up = solid_entry(ray_hits(co, tris, np.array([x, -5.0, z]), np.array([0.0, 1.0, 0.0])))
    return (None if down is None else 60.0 - down), (None if up is None else -5.0 + up)


def span_at(co, tris, point, direction, reach=10.0):
    """Thickness of the solid around ``point`` along ``direction``: the entry and exit of the hit
    pair that contains it, on a ray cast from ``reach`` away (any topology: no vertex sampling).
    None if the point isn't inside anything."""
    d = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    hits = ray_hits(co, tris, np.asarray(point) - d * reach, d)
    for i in range(0, len(hits) - 1, 2):
        if hits[i] <= reach <= hits[i + 1]:
            return float(hits[i + 1] - hits[i])
    return None


def surface_samples(co, tris, n=80000, seed=0):
    """``n`` points spread over the surface by area (slices of them work on any topology)."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]


def leg_thickness(samples, bones, leg, upper, lower, slab=0.1, reach=0.9):
    """Thickness of ``leg`` halfway between joints ``upper`` and ``lower``: the surface samples in
    a horizontal slab there, within ``reach`` (x, z) of the leg's own centre (found from the
    bone's point by recentring: a leg needn't sit exactly on the vanilla bone); the mean of its
    side (z) and front (x) extents. None where no leg is found."""
    a, b = (np.array(bones[f"Horse{leg}{j}"], dtype=np.float64) for j in (upper, lower))
    mid = (a + b) / 2
    sl = samples[np.abs(samples[:, 1] - mid[1]) < slab]
    centre = mid[[0, 2]]
    for _ in range(4):
        near = sl[np.linalg.norm(sl[:, [0, 2]] - centre, axis=1) < reach]
        if len(near) < 8:
            return None
        centre = (near[:, [0, 2]].min(axis=0) + near[:, [0, 2]].max(axis=0)) / 2
    return float((np.ptp(near[:, 0]) + np.ptp(near[:, 2])) / 2)


def leg_distance(co, bones):
    """Each vertex's distance to the nearest vanilla leg chain (ground under the heel to Up)."""
    best = np.full(len(co), np.inf)
    for leg in LEGS:
        names = [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")]
        pts = [np.array(bones[n], dtype=np.float64) for n in names]
        pts.append(np.array([pts[-1][0], 0.0, pts[-1][2]]))
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            t = np.clip(((co - a) @ d) / max(d @ d, 1e-12), 0, 1)
            best = np.minimum(best, np.linalg.norm(co - (a + t[:, None] * d), axis=1))
    return best


def volume(co, tris):
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    return float(abs((a * np.cross(b, c)).sum()) / 6.0)


def stations(bones):
    """The torso slices measured: girth (behind the front legs), saddle, flank (before the hind legs)."""
    front = (bones["HorseFrontLeftUp"][2] + bones["HorseFrontRightUp"][2]) / 2
    back = (bones["HorseBackLeftUp"][2] + bones["HorseBackRightUp"][2]) / 2
    return {"girth": front + 0.25 * (back - front), "saddle": bones["Saddle_Node"][2],
            "flank": front + 0.8 * (back - front)}


def proportions(co, tris, bones, leg_radius=0.8, slab=0.3):
    """Silhouette numbers of one mesh (module doc)."""
    co, tris = np.asarray(co, dtype=np.float64), np.asarray(tris, dtype=np.int64)
    out = {"slices": {}}
    legd = leg_distance(co, bones)
    for name, z in stations(bones).items():
        back, belly = top_bottom(co, tris, z)
        width = None
        if back is not None and belly is not None:
            spans = []
            for f in (0.25, 0.5, 0.75):
                y = belly + f * (back - belly)
                lt = solid_entry(ray_hits(co, tris, np.array([30.0, y, z]), np.array([-1.0, 0.0, 0.0])))
                rt = solid_entry(ray_hits(co, tris, np.array([-30.0, y, z]), np.array([1.0, 0.0, 0.0])))
                if lt is not None and rt is not None:
                    spans.append(60.0 - lt - rt)
            width = max(spans) if spans else None
        radius = leg_radius + np.clip(2.0 - co[:, 1], 0, None) * 0.5  # the hooves stick out
        not_leg = co[(np.abs(co[:, 2] - z) < slab) & (legd > radius)]
        out["slices"][name] = {
            "z": round(float(z), 3), "back": back, "belly": belly,
            "depth": None if back is None or belly is None else back - belly, "width": width,
            "hem": float(not_leg[:, 1].min()) if len(not_leg) else belly}
    samples = surface_samples(co, tris)
    out["legs"] = {leg: {"upper": leg_thickness(samples, bones, leg, "Down", "Knee"),
                         "cannon": leg_thickness(samples, bones, leg, "Knee", "Heel")} for leg in LEGS}
    base, mid, head = (np.array(bones[n], dtype=np.float64) for n in ("HorseNeckBase", "HorseNeckMiddle", "HorseHead"))
    axis = (head - base) / np.linalg.norm(head - base)
    side = np.array([1.0, 0.0, 0.0])
    up = np.cross(axis, side)
    up = up if up[1] > 0 else -up
    out["neck"] = {"thickness": span_at(co, tris, mid, up), "width": span_at(co, tris, mid, side)}
    out["volume"] = volume(co, tris)
    return out


def rider_legs(rider, seat_y, below=1.0, inner=1.0):
    """The seated rider's lower legs and feet: vertices more than ``below`` under the seat and
    more than ``inner`` off the centre line."""
    r = np.asarray(rider, dtype=np.float64)
    return r[(r[:, 1] < seat_y - below) & (np.abs(r[:, 0]) > inner)]


def visible_from_side(co, tris, points, eps=0.02):
    """Share of ``points`` not covered by the mesh seen from their own side (a ray from x = +-30
    in towards the centre line; any surface counts, thin panels too)."""
    if not len(points):
        return None
    seen = 0
    for p in points:
        s = 1.0 if p[0] > 0 else -1.0
        hits = ray_hits(co, tris, np.array([30.0 * s, p[1], p[2]]), np.array([-s, 0.0, 0.0]))
        seen += int(not len(hits) or 30.0 - hits[0] <= abs(p[0]) + eps)
    return seen / len(points)


def summary(p):
    """The numbers ``compare`` judges: mean depth, max width, mean belly height, mean leg
    thickness (both points, all legs), neck thickness, volume; and the lowest hem (reported)."""
    sl = list(p["slices"].values())

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None
    widths = [s["width"] for s in sl if s["width"] is not None]
    hems = [s["hem"] for s in sl if s["hem"] is not None]
    return {"depth": mean(s["depth"] for s in sl), "width": max(widths) if widths else None,
            "belly": mean(s["belly"] for s in sl),
            "leg": mean(v for leg in p["legs"].values() for v in leg.values()),
            "neck": p["neck"]["thickness"], "volume": p["volume"], "hem": min(hems) if hems else None,
            "rider_leg": p.get("rider_leg")}


def compare(ours, vanilla, limits=None):
    """{metric: {ours, vanilla, ratio, limit, flag}}: flag when the ratio is past its limit
    (above it; below it for belly and leg), or the metric couldn't be measured on ours."""
    o, v = summary(ours), summary(vanilla)
    limits = {**LIMITS, **(LIMITS_RIDER if v.get("rider_leg") else {}), **(limits or {})}
    out = {}
    for k, lim in limits.items():
        if o[k] is None or not v[k]:
            out[k] = {"ours": o[k], "vanilla": v[k], "ratio": None, "limit": lim, "flag": True}
            continue
        r = o[k] / v[k]
        out[k] = {"ours": round(o[k], 3), "vanilla": round(v[k], 3), "ratio": round(r, 3), "limit": lim,
                  "flag": bool(r < lim if k in LOW_IS_BAD else r > lim)}
    out["hem"] = {"ours": o["hem"], "vanilla": v["hem"], "ratio": None, "limit": None, "flag": False}
    return out
