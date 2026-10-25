"""Painting a fresh horse (Track C, test 15, step 8; Blender jobs only: numpy). PDX mesh space.

Every texel takes its colour from the 3D point it maps to (rasterised from the UVs), so patterns
run straight across UV islands. Regions by position on the rest-pose horse (``bones``: vanilla
bind positions) and a per-face ``part`` attribute (1 = the chanfron):

  chanfron  silver plate with a gold ridge down the middle;
  coat      the head (outside the chanfron) and the legs below the hem;
  trim      gold bands: just above the hem, and at the seam between scale and cloth;
  scale     the neck and the front of the body: silver scales in rows (projected on the plane
            the face looks along least), each darker at its lower edge;
  cloth     red: the rear caparison and the saddle, with a gold cross on each rear flank;
  panels    (parts 2 and 3, round 3) cloth or scale, with a gold band along each panel's hem;
  iron, leather  stirrups and straps (parts 4 and 5);
  hair      the built tail (part 6): the coat darkened, in streaks along it.

Round 5 (test 17: ours read "grey front, red back" at map scale, vanilla "brown horse, dark
legs"): the horse cues painted in as well -
  coat      a bay (palette "coat"), going to dark points (palette "points") below the knees and
            hocks;
  mane      a dark line along the crest of the neck (palette "mane");
  shading   painted light, as vanilla's diffuse has it: light from above (the smooth normal's up
            component) and darker towards the belly; ``shade`` sets its strength (0: none).

Maps: diffuse RGB, specular RGBA (RGB 0: no unit colours; alpha = gloss, small values as the base
game's), normal flat. Islands are bled outwards so mips and seams don't pick up the background.
"""
import numpy as np

COAT, SCALE, CLOTH, TRIM, CHANFRON, IRON, LEATHER, HAIR, MANE = range(9)
GLOSS = {COAT: 5, SCALE: 14, CLOTH: 4, TRIM: 12, CHANFRON: 16, IRON: 14, LEATHER: 6, HAIR: 7, MANE: 6}
# parts (_horse_barding): 2 cloth panel, 3 scale panel, 4 stirrup iron, 5 strap, 6 tail
DEFAULTS = {"iron": (70, 70, 74), "leather": (60, 38, 24), "points": (36, 26, 21), "mane": (30, 22, 18)}


def regions(centres: np.ndarray, normals: np.ndarray, part: np.ndarray, bones: dict, hem_y: float,
            face_hem: np.ndarray | None = None, face_head: np.ndarray | None = None) -> np.ndarray:
    """Per face: its region (module doc). ``face_hem``: each panel face's hem height;
    ``face_head``: each face's HorseHead weight (the head is where it is over 0.5; without it,
    near the poll-to-muzzle segment)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    y, z = centres[:, 1], centres[:, 2]
    seam = B["HorseSpineTop"][2] + 0.5
    out = np.full(len(centres), CLOTH)
    out[z < seam] = SCALE
    out[(z >= seam) & (z < seam + 0.4) & (y > hem_y)] = TRIM
    out[(y > hem_y) & (y < hem_y + 0.5)] = TRIM
    # the head: in front of the poll along the head's own direction (a z plane cut the front
    # of the neck into coat blotches, test 16b)
    # near the poll-to-muzzle segment of this mesh (its muzzle: the face furthest along vanilla's
    # poll-to-muzzle direction, as add_chanfron finds it)
    rel = centres - B["HorseHead"]
    tip = rel[np.argmax(rel @ np.array([0.0, -1.5, -3.3]))]
    length = float(np.linalg.norm(tip))
    head_dir = tip / max(length, 1e-6)
    t = rel @ head_dir
    off = np.linalg.norm(rel - t[:, None] * head_dir, axis=1)
    if face_head is not None:  # by the weights: wherever a generated head ended up
        out[face_head > 0.5] = COAT
    else:
        out[(t > -0.3) & (t < length + 0.5) & (off < 1.7)] = COAT
    out[y < hem_y] = COAT
    out[part == 1] = CHANFRON
    out[part == 2] = CLOTH
    out[part == 3] = SCALE
    if face_hem is not None:
        out[(part >= 2) & (part <= 3) & (y < face_hem + 0.35)] = TRIM
    out[part == 4] = IRON
    out[part == 5] = LEATHER
    out[part == 6] = HAIR
    return out


SCALE_SIZE = 0.25  # scene units; test 16: 0.32 read as big tiles and, soft-shaded, as wool; 0.2 with a hard rim aliased into speckle


def _scale_shade(a: np.ndarray, b: np.ndarray, size: float = SCALE_SIZE) -> np.ndarray:
    """Scales in rows on the plane (a, b), rows offset by half a scale: each scale's face flat
    and bright with a slight sheen towards its top, a thin dark rim at its rounded lower edge
    (crisp, so the pattern reads as metal rather than a soft texture)."""
    row = np.floor(b / size)
    fa = (a / size + 0.5 * (row % 2)) % 1.0 - 0.5
    fb = (b / size) % 1.0
    r = np.sqrt(fa ** 2 + (1.0 - fb) ** 2 * 0.7)  # distance from the scale's top centre
    face = 0.8 + 0.08 * fb  # brighter towards the top
    # the rim as a smooth dip (a hard one thinner than a texel aliases into white speckle)
    rim = np.exp(-((r - 0.55) / 0.08) ** 2)
    return face * (1.0 - 0.28 * rim)


def colour(points: np.ndarray, normals: np.ndarray, region: int, palette: dict, bones: dict) -> np.ndarray:
    """RGB (0-255 floats) of texels at ``points`` (with the face's ``normals``) in ``region``."""
    n = len(points)
    rng = np.random.default_rng(int(abs(points[0].sum()) * 1000) % 2 ** 32 if n else 0)
    noise = 1.0 + (rng.random(n) - 0.5) * 0.08
    if region == SCALE:
        # one projection for the whole front: rows along the height, wrapped round a vertical
        # axis in front of the chest (a plane per face, as before, broke the rows at every face
        # edge into a patchwork, test 16b)
        B = {k: np.array(v) for k, v in bones.items()}
        zc = B["HorseNeckBase"][2] + 1.5
        a = 2.5 * np.arctan2(points[:, 0], zc - points[:, 2])
        return np.array(palette["silver"])[None, :] * (_scale_shade(a, points[:, 1]) * noise)[:, None]
    if region == CLOTH:
        rgb = np.tile(np.array(palette["red"], dtype=np.float64), (n, 1)) * noise[:, None]
        B = {k: np.array(v) for k, v in bones.items()}
        cy, cz = 8.6, B["HorseSpineEnd"][2] - 1.2  # a cross on each rear flank
        arm, bar = 0.85, 0.17
        cross = ((np.abs(points[:, 1] - cy) < bar) & (np.abs(points[:, 2] - cz) < arm)) | \
                ((np.abs(points[:, 2] - cz) < bar) & (np.abs(points[:, 1] - cy) < arm))
        cross &= np.abs(points[:, 0]) > 1.2
        rgb[cross] = palette["gold"]
        return rgb
    if region == TRIM:
        return np.array(palette["gold"])[None, :] * noise[:, None]
    if region in (IRON, LEATHER):
        key = "iron" if region == IRON else "leather"
        return np.array(palette.get(key, DEFAULTS[key]), dtype=np.float64)[None, :] * noise[:, None]
    if region == HAIR:  # streaks across x (the tail hangs along y)
        streak = 0.8 + 0.12 * np.sin(points[:, 0] * 23.0 + np.sin(points[:, 1] * 3.0))
        return np.array(palette["coat"])[None, :] * (streak * noise)[:, None]
    if region == MANE:
        return np.array(palette.get("mane", DEFAULTS["mane"]), dtype=np.float64)[None, :] * noise[:, None]
    if region == CHANFRON:
        rgb = np.array(palette["silver"])[None, :] * 0.92 * noise[:, None]
        rgb[np.abs(points[:, 0]) < 0.12] = palette["gold"]
        return rgb
    # the coat (head, lower legs): quiet, the muzzle a little darker (HorseHead to the nose tip
    # is -z), no blotches
    B = {k: np.array(v) for k, v in bones.items()}
    muzzle = np.clip((B["HorseHead"][2] - 1.5 - points[:, 2]) / 1.5, 0, 1)
    rgb = np.array(palette["coat"])[None, :] * ((1.0 - 0.25 * muzzle) * (1.0 + (noise - 1.0) * 0.4))[:, None]
    # dark points: below the knees and hocks (+ 0.6), blending over 1.2
    knee = min(B["HorseFrontLeftKnee"][1], B["HorseBackLeftKnee"][1]) + 0.6
    k = np.clip((knee - points[:, 1]) / 1.2, 0, 1)[:, None]
    return rgb * (1 - k) + np.array(palette.get("points", DEFAULTS["points"]), dtype=np.float64)[None, :] * k


def mane_mask(points, normals, bones, half=0.3):
    """Texels of the mane: within ``half`` of the centre line along the crest of the neck (near the
    bones NeckBase -> HorseHead), on surfaces facing up."""
    B = {k: np.array(v) for k, v in bones.items()}
    a, b = B["HorseNeckBase"], B["HorseHead"]
    d = b - a
    t = np.clip(((points - a) @ d) / (d @ d), 0, 1)
    crest = np.linalg.norm(points - (a + t[:, None] * d), axis=1)
    return (np.abs(points[:, 0]) < half) & (normals[:, 1] > 0.3) & (crest < 1.3) & (t > 0.05) & (t < 0.97)


def paint(uv: np.ndarray, tris: np.ndarray, co: np.ndarray, face_n: np.ndarray, region: np.ndarray,
          palette: dict, bones: dict, size: int = 512, bleed: int = 6, shade: float = 1.0, mane: bool = False):
    """-> (diffuse RGB uint8, specular RGBA uint8, normal RGBA uint8, coverage share).
    ``uv``: per triangle corner (T, 3, 2) in Blender convention (v up); ``tris``: (T, 3) vertex
    indices into ``co``."""
    # smooth (vertex) normals for the painted light: face normals would paint every facet flat
    vn = np.zeros_like(co)
    for k in range(3):
        np.add.at(vn, tris[:, k], face_n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    diff = np.zeros((size, size, 3), dtype=np.float64)
    gloss = np.zeros((size, size), dtype=np.float64)
    filled = np.zeros((size, size), dtype=bool)
    for t in range(len(tris)):
        p = uv[t] * (size - 1)
        p[:, 1] = (size - 1) - p[:, 1]  # image rows run down
        lo = np.floor(p.min(axis=0)).astype(int)
        hi = np.ceil(p.max(axis=0)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(lo[0], 0), min(hi[0], size - 1) + 1),
                             np.arange(max(lo[1], 0), min(hi[1], size - 1) + 1))
        q = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = p
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        lam = np.linalg.solve(m, (q - a).T).T
        w0 = 1 - lam[:, 0] - lam[:, 1]
        inside = (lam[:, 0] >= -0.02) & (lam[:, 1] >= -0.02) & (w0 >= -0.02)
        if not inside.any():
            continue
        bary = np.stack([w0, lam[:, 0], lam[:, 1]], axis=1)[inside]
        pts = bary @ co[tris[t]]
        rgb = colour(pts, np.repeat(face_n[t][None, :], len(pts), axis=0), int(region[t]), palette, bones)
        if mane and int(region[t]) in (SCALE, COAT):  # the mane, per texel (off by default: on the mail it read as holes)
            m = mane_mask(pts, bary @ vn[tris[t]], bones)
            if m.any():
                rgb[m] = np.array(palette.get("mane", DEFAULTS["mane"]), dtype=np.float64)
        if shade:  # light from above, the belly darker
            ny = (bary @ vn[tris[t]])[:, 1]
            light = (0.72 + 0.4 * (0.5 + 0.5 * ny)) * (0.86 + 0.14 * np.clip((pts[:, 1] - 5.0) / 7.0, 0, 1))
            rgb = rgb * (1 + (light - 1) * shade)[:, None]
        px = q[inside].astype(int)
        diff[px[:, 1], px[:, 0]] = rgb
        gloss[px[:, 1], px[:, 0]] = GLOSS[int(region[t])]
        filled[px[:, 1], px[:, 0]] = True
    coverage = float(filled.mean())
    for _ in range(bleed):  # each pass copies a filled neighbour into empty texels
        grow = ~filled
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            take = grow & src
            diff[take] = np.roll(np.roll(diff, dy, axis=0), dx, axis=1)[take]
            gloss[take] = np.roll(np.roll(gloss, dy, axis=0), dx, axis=1)[take]
            filled |= take
            grow &= ~take
    diff[~filled] = palette["coat"]
    gloss[~filled] = GLOSS[COAT]
    spec = np.zeros((size, size, 4), dtype=np.uint8)
    spec[..., 3] = np.clip(gloss, 0, 255).astype(np.uint8)
    normal = np.zeros((size, size, 4), dtype=np.uint8)
    normal[...] = (128, 128, 255, 255)
    return np.clip(diff, 0, 255).astype(np.uint8), spec, normal, coverage
