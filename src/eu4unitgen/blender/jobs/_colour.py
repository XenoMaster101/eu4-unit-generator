"""Colour classes for the bakes (numpy only, no bpy: Blender's numpy runs it; tests/blender_rules_check.py too).

Track P1 (the user, 2026-10-24: every one-off a general rule): the bakes found the cloth by a fixed red window
(hue under 20 or over 330), right for the cataphract only. The cloth is now the subject's own: the dominant
saturated colour that isn't gold (``dominant_hue``), and every rule that looked for "red" looks for hues near it
(``near_hue``). The coat (a horse's own hair colour on its head and neck) is told apart, so a bay horse's coat isn't
taken for cloth.
"""
import numpy as np

LUM = np.array([0.299, 0.587, 0.114])
GOLD_HUE = (28.0, 58.0)  # yellow-orange: the gold class (gloss_map), never cloth


def hcl(c):
    """Hue (degrees, the hexagonal approximation the bakes always used), chroma (0-1 of the brightest channel) and
    luminance of sRGB 0-1 colours (``c[..., :3]``)."""
    c = np.asarray(c, dtype=np.float64)[..., :3]
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    return hue, (mx - mn) / np.maximum(mx, 1e-6), c @ LUM


def near_hue(hue, h0, tol):
    """Hues within ``tol`` degrees of ``h0`` (round the circle)."""
    d = np.abs((np.asarray(hue) - h0 + 180.0) % 360.0 - 180.0)
    return d < tol


def is_gold(hue, chroma, lum, chroma_min=0.3, lum_min=0.25):
    return (hue > GOLD_HUE[0]) & (hue < GOLD_HUE[1]) & (chroma > chroma_min) & (lum > lum_min)


def dominant_hue(c, mask, chroma_min=0.4, lum_min=0.06, bin_deg=10.0):
    """The most common hue among ``mask``'s saturated, non-gold colours, refined to the circular median within half
    a bin either side of the peak: (hue, the share of ``mask`` in that colour) or (None, 0)."""
    hue, chroma, lum = hcl(c)
    sel = mask & (chroma > chroma_min) & (lum > lum_min) & ~is_gold(hue, chroma, lum)
    if sel.sum() < 50:
        return None, 0.0
    h = hue[sel]
    counts = np.bincount((h // bin_deg).astype(int) % int(360 / bin_deg), minlength=int(360 / bin_deg))
    # a peak with its neighbours (a colour straddling a bin edge)
    smooth = counts + np.roll(counts, 1) + np.roll(counts, -1)
    k = int(np.argmax(smooth))
    centre = (k + 0.5) * bin_deg
    near = near_hue(h, centre, 1.5 * bin_deg)
    ang = np.radians(h[near])
    h0 = float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    return h0, float(near_hue(hue, h0, 20.0)[sel].sum() / max(mask.sum(), 1))


def cloth_and_coat(c, cov, posmap=None, bones=None):
    """The cloth's hue and, for a horse, its coat's: the coat is the dominant saturated hue of the head and neck
    (ahead of and above the neck's base) when it covers more than 40% of them (a bare head; a chanfron's gold and
    a neck's scale leave too little colour). The cloth is the dominant saturated non-gold hue elsewhere, ignoring
    the coat's hue when one is found. -> {"cloth": hue or None, "coat": hue or None, "conflict": bool}."""
    out = {"cloth": None, "coat": None, "conflict": False}
    rest = cov.copy()
    if posmap is not None and bones and "HorseNeckBase" in bones:
        nb = np.asarray(bones["HorseNeckBase"], dtype=np.float64)
        head = cov & (posmap[..., 2] < nb[2]) & (posmap[..., 1] > nb[1] - 1.0)
        h, share = dominant_hue(c, head, chroma_min=0.3)
        if h is not None and share > 0.4:
            out["coat"] = round(h, 1)
            hue, _, _ = hcl(c)
            rest = cov & ~near_hue(hue, h, 15.0)
    h, share = dominant_hue(c, rest)
    if h is None and out["coat"] is not None:  # nothing else saturated: the coat itself is the only colour
        h, share = dominant_hue(c, cov)
        out["conflict"] = True
    out["cloth"] = None if h is None else round(h, 1)
    out["cloth_share"] = round(share, 3)
    return out


def cloth_window(h0, tol=25.0):
    """The cloth test every bake rule uses: hues within ``tol`` of the cloth's (the old fixed red window, 330 to 20,
    is 355 +- 25)."""
    return lambda hue: near_hue(hue, 355.0 if h0 is None else h0, tol)


def crest_colour(co, rgb, keep, above):
    """The crest's colour test (Track P1: it was a fixed red, the cataphract's): the dominant saturated, non-gold
    hue among the points over ``above`` (the helmet's steel and gold are unsaturated or gold). -> a function of
    (hue, chroma) giving the crest's points, and the hue (None: no crest colour there)."""
    c = rgb / 255.0 if rgb.max() > 1.5 else rgb
    h0, share = dominant_hue(c, keep & (co[:, 1] > above))
    if h0 is None or share < 0.05:
        return (lambda hue_, chroma_: np.zeros(len(hue_), dtype=bool)), None
    return (lambda hue_, chroma_: near_hue(hue_, h0, 25.0) & (chroma_ > 0.4)), h0
