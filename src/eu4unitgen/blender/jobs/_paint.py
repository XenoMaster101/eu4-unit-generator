"""The game-painted pass (the user, after T38: the base game's units read at a distance because their textures are
hand-painted: dark creases, light tops, crisp material borders; ours were soft and low in contrast). Run after the
bake on the diffuse in texture space (numpy, inside Blender's bake jobs):

  materials  the texels clustered by colour (Lab, k-means, ``materials`` clusters) on a blurred copy, then a majority
             filter (``clean`` passes): each texel's material, in regions, not flickering texel to texel
  flatten    each material's own detail smoothed (``flat_passes`` blurs among texels of the same material), mixed in
             by ``flatten``: the bake's noise out, the colours of a painted area
  contrast   each material's lightness spread about its own middle by ``contrast`` (its colour kept)
  cavity     the normal map's curvature (the divergence of its tilt): creases darker by up to ``cavity``, ridges
             lighter by half that
  borders    a dark line (``border``, about two texels) where one material meets another on the same island, when
             their colours are at least ``border_de`` apart (CIE76): the lit and shaded halves of one white jack are
             two clusters, but no border
  top        light from above: lighter where the surface faces up, darker facing down, by ``top`` (the object-space
             normal's up component)
  sharpen    an unsharp mask (radius 1 texel) of strength ``sharpen``

Texels outside the islands are left as they are (the bake's margin).
"""
import numpy as np

PAINT = {"materials": 6, "clean": 3, "flatten": 0.6, "flat_passes": 4, "contrast": 1.3, "cavity": 0.35, "border": 0.3,
         "border_de": 25.0, "top": 0.18, "sharpen": 0.3}


def _lab(rgb):
    def lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = (lin(rgb[..., k]) for k in range(3))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def materials(rgb, mask, k, seed=0, iters=12):
    """Each texel's material (k-means on Lab over the island texels; -1 outside) and the materials' Lab centres."""
    lab = _lab(np.clip(rgb, 0, 1))
    pts = lab[mask]
    rng = np.random.default_rng(seed)
    sample = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    cent = sample[rng.choice(len(sample), k, replace=False)]
    for _ in range(iters):
        lab_ = np.argmin(((sample[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.array([sample[lab_ == i].mean(0) if (lab_ == i).any() else cent[i] for i in range(k)])
    out = np.full(mask.shape, -1, dtype=np.int32)
    lab_all = np.empty(len(pts), dtype=np.int32)
    for s in range(0, len(pts), 200000):  # in chunks: k x n distances
        lab_all[s:s + 200000] = np.argmin(((pts[s:s + 200000, None] - cent[None]) ** 2).sum(-1), 1)
    out[mask] = lab_all
    return out, cent


def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def _blur(a, mask):
    """A 3x3 box blur over the island texels only (neighbours outside count as the texel itself)."""
    acc = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb = _shift(a, dy, dx, 0)
            ok = _shift(mask, dy, dx, False)
            acc += np.where(ok[..., None] if a.ndim == 3 else ok, nb, a)
    return acc / 9.0


def majority(mat, mask, passes):
    """Each texel's material replaced by the most common among its 3x3 neighbours on the island (``passes`` times)."""
    k = int(mat.max()) + 1
    for _ in range(passes):
        votes = np.zeros(mat.shape + (k,), dtype=np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb = _shift(mat, dy, dx, -1)
                for i in range(k):
                    votes[..., i] += nb == i
        mat = np.where(mask, np.argmax(votes, -1), -1).astype(np.int32)
    return mat


def flat_within(rgb, mat, mask, passes):
    """Blurred among neighbours of the same material only."""
    out = rgb.copy()
    for _ in range(passes):
        acc = out.copy()
        n = np.ones(mask.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                same = (_shift(mat, dy, dx, -1) == mat) & mask
                acc += np.where(same[..., None], _shift(out, dy, dx, 0), 0)
                n += same
        out = acc / n[..., None]
    return out


def game_paint(diff, mask, tilt, up, cfg=None):
    """``diff`` RGB (h, w, 3) 0-1, ``mask`` the island texels, ``tilt`` the tangent normals' x and y (h, w, 2),
    ``up`` the surface's up component -1..1 (h, w) or None. -> (painted RGB, measures)."""
    c = {**PAINT, **(cfg or {})}
    soft = diff
    for _ in range(3):
        soft = _blur(soft, mask)
    mat, cent = materials(soft, mask, int(c["materials"]))
    mat = majority(mat, mask, int(c["clean"]))
    apart = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1)) >= c["border_de"]  # really different materials
    out = diff * (1 - c["flatten"]) + flat_within(diff, mat, mask, int(c["flat_passes"])) * c["flatten"]
    lum0 = diff @ np.array([0.2126, 0.7152, 0.0722])
    lum = out @ np.array([0.2126, 0.7152, 0.0722])
    for i in range(int(c["materials"])):
        sel = mat == i
        if sel.sum() < 16:
            continue
        mid = np.median(lum[sel])
        new = np.clip(mid + (lum[sel] - mid) * c["contrast"], 0.0, 1.0)
        out[sel] *= (new / np.maximum(lum[sel], 1e-4))[:, None]
    # cavity: the divergence of the tilt (central differences; the islands' edges left out)
    inner = mask & _shift(mask, 0, 1, False) & _shift(mask, 0, -1, False) & _shift(mask, 1, 0, False) & _shift(mask, -1, 0, False)
    div = (_shift(tilt[..., 0], 0, 1, 0) - _shift(tilt[..., 0], 0, -1, 0)) * 0.5 \
        + (_shift(tilt[..., 1], 1, 0, 0) - _shift(tilt[..., 1], -1, 0, 0)) * 0.5  # Blender's rows run up, as green does
    div = np.where(inner, _blur(div, inner), 0.0)
    scale = np.percentile(np.abs(div[inner]), 95) if inner.any() else 1.0
    cav = np.clip(div / max(scale, 1e-6), -1.0, 1.0)  # +1 a ridge, -1 a crease
    shade = 1.0 + np.where(cav < 0, cav * c["cavity"], cav * c["cavity"] * 0.5)
    # borders: a texel whose 4-neighbour on the same island is another material
    edge = np.zeros(mask.shape, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = _shift(mat, dy, dx, -1)
        edge |= mask & (nb >= 0) & (nb != mat) & apart[np.maximum(mat, 0), np.maximum(nb, 0)]
    shade = np.where(edge, shade * (1.0 - c["border"]), shade)
    if up is not None:
        shade *= 1.0 + c["top"] * np.clip(up, -1.0, 1.0)
    out *= shade[..., None]
    if c["sharpen"]:
        out = out + c["sharpen"] * (out - _blur(out, mask))
    out = np.where(mask[..., None], np.clip(out, 0.0, 1.0), diff)
    lum1 = out @ np.array([0.2126, 0.7152, 0.0722])
    return out, {"materials": int(c["materials"]), "border_texels": round(float(edge.sum() / max(mask.sum(), 1)), 4),
                 "contrast_before": round(float(np.std(lum0[mask])), 4), "contrast_after": round(float(np.std(lum1[mask])), 4)}
