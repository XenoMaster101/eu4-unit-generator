"""Map-scale readability (Track C, round 6; Blender jobs only: numpy). PDX mesh space: Y up, the
horse facing -Z, its left side at +X.

The user's rule (round 6): at the game camera (58 degrees down) a horse must show at least
``LEG_SHARE`` of vanilla's visible leg length, measured the same way on both:
  * each leg's axis (Down -> Knee -> Heel -> the ground under the heel, and Up -> Down above it)
    sampled every ``step`` in height from the ground to the Up bone;
  * a sample is visible when a ray from the camera to it (parallel rays: at map scale the camera
    is far) crosses no occluder before it;
  * occluders: every triangle that isn't the lower leg itself. A body triangle (part 0) counts as
    lower leg when its nearest skeleton segment is a leg's below the Up bone and it lies within
    ``leg_reach`` of it (the barrel's underside near the elbow is farther); every other part
    (caparison, stirrups, chanfron, tail) always occludes. The legs don't hide each other: a leg
    behind a leg still reads as a leg;
  * visible length = visible samples x ``step``, per leg, averaged over the four legs and the
    camera's azimuths (``AZIMUTHS``, from the front round the left side to the back; the right
    side mirrors the left).
``outline`` does the same for the head and neck (the axis HorseNeckBase -> HorseHead -> the
muzzle): how much of it the body and caparison leave in view.
``colours``: area-weighted mean colour of each region (legs below the Up bones, head, neck, the
rest: body and caparison) from the diffuse texture; ``contrast``: CIE76 colour difference
between the legs and the rest, and the head and the rest (a dark leg against a red caparison
reads; a grey leg against grey mail doesn't).
"""
import numpy as np

import _horse_measure as hm

LEG_SHARE = 0.8  # the user's rule (round 6): visible leg length >= 80% of vanilla's
ELEVATION = 58.0  # GAME_CLOSE's camera, degrees below the horizontal
AZIMUTHS = (0.0, 45.0, 90.0, 135.0, 180.0)  # degrees from the front, round the horse's left side
LEGS = hm.LEGS
CONTRAST_MIN = 12.0  # assumed (round 6): CIE76 delta E under which two regions merge at map scale


def view_dir(azimuth, elevation=ELEVATION):
    """Camera -> scene direction for a camera ``azimuth`` degrees round from the horse's front
    (-Z) towards its left (+X), ``elevation`` degrees above the ground, looking down at it."""
    a, e = np.radians(azimuth), np.radians(elevation)
    cam = np.array([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)])
    return -cam


def _chain(B, leg):
    names = [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")]
    pts = [B[n] for n in names]
    pts.append(np.array([pts[-1][0], 0.0, pts[-1][2]]))
    return np.array(pts)


def segments(bones):
    """Skeleton segments -> (a: s x 3, b: s x 3, lower: s booleans, the legs below the Up bone)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    a, b, lower = [], [], []
    for leg in LEGS:
        c = _chain(B, leg)
        for i in range(len(c) - 1):
            a.append(c[i])
            b.append(c[i + 1])
            lower.append(i > 0)
    trunk = ["HorseNeckBase", "HorseSpineTop", "HorseSpineMiddle", "HorseSpineEnd", "HorseTailTop", "HorseTailMiddle",
             "HorseTailTip", "HorseTailEnd"]
    for chain in (trunk, ["HorseNeckBase", "HorseNeckMiddle", "HorseHead"]):
        for p, q in zip(chain, chain[1:]):
            if p in B and q in B:
                a.append(B[p])
                b.append(B[q])
                lower.append(False)
    for leg in LEGS:  # the shoulders and hips: from the spine out to each Up bone
        up = B[f"Horse{leg}Up"]
        a.append(np.array([0.0, up[1], up[2]]))
        b.append(up)
        lower.append(False)
    return np.array(a), np.array(b), np.array(lower)


def _seg_dist(p, a, b):
    """Distances from points p (n x 3) to segments (s x 3 each) -> n x s."""
    ab = b - a
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    return np.linalg.norm(p[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)


def leg_triangles(co, tris, bones, part=None, leg_reach=1.0):
    """Which triangles are lower leg (module doc) -> booleans."""
    a, b, lower = segments(bones)
    cen = co[tris].mean(axis=1)
    d = _seg_dist(cen, a, b)
    near = d.argmin(axis=1)
    leg = lower[near] & (d[np.arange(len(cen)), near] < leg_reach)
    if part is not None:
        leg &= np.asarray(part) == 0
    return leg


def _visible(co, occ, p, d, far=60.0):
    hits = hm.ray_hits(co, occ, p - d * far, d)
    return not len(hits) or hits[0] >= far - 1e-4


def leg_samples(bones, step=0.25):
    """Points along each leg's axis every ``step`` in height, ground to the Up bone ->
    {leg: m x 3}."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    out = {}
    for leg in LEGS:
        c = _chain(B, leg)[::-1]  # the ground first: heights increase
        ys = np.arange(0.05, c[-1][1], step)
        out[leg] = np.stack([np.interp(ys, c[:, 1], c[:, k]) if k != 1 else ys for k in range(3)], axis=1)
    return out


def leg_visible(co, tris, bones, part=None, step=0.25, azimuths=AZIMUTHS, elevation=ELEVATION, leg_reach=1.0):
    """Visible leg length (module doc) -> {"length": mean per leg and view, "by_view": {azimuth:
    mean over the legs}, "by_leg": {leg: mean over the views}}."""
    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    occ = tris[~leg_triangles(co, tris, bones, part, leg_reach)]
    samples = leg_samples(bones, step)
    grid = np.zeros((len(azimuths), len(LEGS)))
    for i, az in enumerate(azimuths):
        d = view_dir(az, elevation)
        for j, leg in enumerate(LEGS):
            grid[i, j] = step * sum(_visible(co, occ, p, d) for p in samples[leg])
    return {"length": round(float(grid.mean()), 3),
            "by_view": {str(int(az)): round(float(grid[i].mean()), 3) for i, az in enumerate(azimuths)},
            "by_leg": {leg: round(float(grid[:, j].mean()), 3) for j, leg in enumerate(LEGS)}}


def outline(co, tris, bones, part=None, step=0.25, azimuths=AZIMUTHS, elevation=ELEVATION, reach=1.6):
    """Share of the head-and-neck axis (NeckBase -> NeckMiddle -> Head -> the muzzle, 2.6 down and 3
    forward of the poll, as the control horse) in view: occluders are the triangles farther than
    ``reach`` from that axis (the body behind it, the caparison over it), and any part but the
    body or the chanfron (1)."""
    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    axis = np.array([B["HorseNeckBase"], B["HorseNeckMiddle"], B["HorseHead"], B["HorseHead"] + [0, -2.6, -3.0]])
    cen = co[tris].mean(axis=1)
    near = _seg_dist(cen, axis[:-1], axis[1:]).min(axis=1) < reach
    if part is not None:
        near &= np.isin(np.asarray(part), (0, 1))
    occ = tris[~near]
    seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    ts = np.arange(0.0, seg.sum(), step)
    pts = np.stack([np.interp(ts, np.concatenate([[0], np.cumsum(seg)]), axis[:, k]) for k in range(3)], axis=1)
    seen = [np.mean([_visible(co, occ, p, view_dir(az, elevation)) for p in pts]) for az in azimuths]
    return round(float(np.mean(seen)), 3)


def _lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def regions(co, tris, bones, part=None):
    """Each triangle's region: "leg" (lower leg, as leg_visible's), "head", "neck" (nearest the
    head or neck axis and within 1.6 of it, body or chanfron), "rest"."""
    leg = leg_triangles(co, tris, bones, part)
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    cen = co[tris].mean(axis=1)
    head = _seg_dist(cen, B["HorseHead"][None], (B["HorseHead"] + [0, -2.6, -3.0])[None])[:, 0]
    neck = _seg_dist(cen, np.array([B["HorseNeckBase"], B["HorseNeckMiddle"]]),
                     np.array([B["HorseNeckMiddle"], B["HorseHead"]])).min(axis=1)
    body = np.ones(len(tris), dtype=bool) if part is None else np.isin(np.asarray(part), (0, 1))
    out = np.full(len(tris), "rest", dtype=object)
    out[body & (neck < 1.6) & (neck <= head)] = "neck"
    out[body & (head < 1.6) & (head < neck)] = "head"
    out[leg] = "leg"
    return out


def colours(co, tris, uv, image, bones, part=None):
    """Area-weighted mean sRGB colour of each region (``regions``), the diffuse ``image`` (h x w
    x 3, row 0 at the top) sampled at four points of each triangle (PDX UVs: v down from the top)
    -> {region: [r, g, b]}."""
    co = np.asarray(co, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    img = np.asarray(image, dtype=np.float64)[..., :3]
    h, w = img.shape[:2]
    reg = regions(co, tris, bones, part)
    area = 0.5 * np.linalg.norm(np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]), axis=1)
    out = {}
    for bary in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
        c = np.einsum("k,tkj->tj", np.array(bary), uv[tris])
        px = np.clip((c[:, 0] % 1.0) * w, 0, w - 1).astype(int)
        py = np.clip((c[:, 1] % 1.0) * h, 0, h - 1).astype(int)
        col = img[py, px]
        for r in ("leg", "head", "neck", "rest"):
            m = reg == r
            acc = out.setdefault(r, [np.zeros(3), 0.0])
            acc[0] += (col[m] * area[m, None]).sum(axis=0)
            acc[1] += area[m].sum()
    return {r: [round(float(x), 1) for x in s / max(n, 1e-12)] for r, (s, n) in out.items()}


def contrast(cols):
    """{"leg_rest", "head_rest"}: CIE76 delta E between the regions' mean colours."""
    lab = {r: _lab(c) for r, c in cols.items()}
    return {f"{r}_rest": round(float(np.linalg.norm(lab[r] - lab["rest"])), 1) for r in ("leg", "head")}
