"""Painting a fresh horse (Track C, test 15, step 8; Blender jobs only: numpy). PDX mesh space.

Every texel takes its colour from the 3D point it maps to (rasterised from the UVs), so patterns
run straight across UV islands. Regions by position on the rest-pose horse (``bones``: vanilla
bind positions) and a per-face ``part`` attribute (1 = the chanfron):

  chanfron  silver plate with a gold ridge down the middle;
  coat      the head (outside the chanfron) and the legs below the hem;
  trim      gold bands: just above the hem, and at the seam between scale and cloth;
  scale     the neck and the front of the body: silver scales in rows (projected on the plane
            the face looks along least), each darker at its lower edge;
  cloth     red: the rear caparison and the saddle, with a gold cross on each rear flank.

Maps: diffuse RGB, specular RGBA (RGB 0: no unit colours; alpha = gloss, small values as the base
game's), normal flat. Islands are bled outwards so mips and seams don't pick up the background.
"""
import numpy as np

COAT, SCALE, CLOTH, TRIM, CHANFRON = range(5)
GLOSS = {COAT: 5, SCALE: 14, CLOTH: 4, TRIM: 12, CHANFRON: 16}


def regions(centres: np.ndarray, normals: np.ndarray, part: np.ndarray, bones: dict, hem_y: float) -> np.ndarray:
    """Per face: its region (module doc)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    y, z = centres[:, 1], centres[:, 2]
    seam = B["HorseSpineTop"][2] + 0.5
    out = np.full(len(centres), CLOTH)
    out[z < seam] = SCALE
    out[(z >= seam) & (z < seam + 0.4) & (y > hem_y)] = TRIM
    out[(y > hem_y) & (y < hem_y + 0.5)] = TRIM
    out[z < B["HorseHead"][2] + 0.4] = COAT
    out[y < hem_y] = COAT
    out[part == 1] = CHANFRON
    return out


def _scale_shade(a: np.ndarray, b: np.ndarray, size: float = 0.32) -> np.ndarray:
    """Scales in rows on the plane (a, b): 1 on a scale's face, darker towards its lower edge
    and in the gaps (rows offset by half a scale)."""
    row = np.floor(b / size)
    fa = (a / size + 0.5 * (row % 2)) % 1.0 - 0.5
    fb = (b / size) % 1.0
    r = np.sqrt(fa ** 2 + (1.0 - fb) ** 2 * 0.8)
    return np.clip(1.15 - 0.55 * r, 0.55, 1.0) * (0.92 + 0.08 * (fb > 0.15))


def colour(points: np.ndarray, normals: np.ndarray, region: int, palette: dict, bones: dict) -> np.ndarray:
    """RGB (0-255 floats) of texels at ``points`` (with the face's ``normals``) in ``region``."""
    n = len(points)
    rng = np.random.default_rng(int(abs(points[0].sum()) * 1000) % 2 ** 32 if n else 0)
    noise = 1.0 + (rng.random(n) - 0.5) * 0.08
    if region == SCALE:
        ax = np.abs(normals[0]) if len(normals) else np.array([1, 0, 0])
        if ax[0] >= ax[1] and ax[0] >= ax[2]:
            a, b = points[:, 2], points[:, 1]
        elif ax[2] >= ax[1]:
            a, b = points[:, 0], points[:, 1]
        else:
            a, b = points[:, 0], points[:, 2]
        return np.array(palette["silver"])[None, :] * (_scale_shade(a, b) * noise)[:, None]
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
    if region == CHANFRON:
        rgb = np.array(palette["silver"])[None, :] * 1.08 * noise[:, None]
        rgb[np.abs(points[:, 0]) < 0.12] = palette["gold"]
        return rgb
    return np.array(palette["coat"])[None, :] * noise[:, None]


def paint(uv: np.ndarray, tris: np.ndarray, co: np.ndarray, face_n: np.ndarray, region: np.ndarray,
          palette: dict, bones: dict, size: int = 512, bleed: int = 6):
    """-> (diffuse RGB uint8, specular RGBA uint8, normal RGBA uint8, coverage share).
    ``uv``: per triangle corner (T, 3, 2) in Blender convention (v up); ``tris``: (T, 3) vertex
    indices into ``co``."""
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
