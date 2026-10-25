"""A unit's palette, read from its finished textures (Track P1; pure Python and Pillow: the app has no numpy).

The user's rule after test 34 ("the shield part of the design, not a clean sticker"; the helmet gold "same gold as
the chanfron"), made general: parts take their colours from the reference (their crop) and each colour is then
moved onto the nearest colour of the unit's own palette (the horse's when the rider rides one, else the subject's
own), so the parts and the body share one red, one gold, one steel.

  classes(diffuse, specular, primary)   the palette: gold, silver, cloth and the main matte colours, by the specular
                                        map's classes (gloss 12 gold, 14 silver; R the country mask, its diffuse a
                                        grey times ``primary``)
  snap(colour, palette)                 the nearest palette colour within ``max_de`` (CIE76 Lab), else the colour
  lab / rgb                             sRGB 0-255 <-> CIE Lab (D65)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

GLOSS = {"gold": 12, "silver": 14}


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lab(rgb) -> tuple[float, float, float]:
    r, g, b = (_lin(float(x)) for x in rgb[:3])
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def rgb(lab_) -> tuple[int, int, int]:
    L, a, b = lab_
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    x, y, z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    lin = (3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z)

    def g(c):
        c = max(0.0, min(1.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return tuple(int(round(255 * g(c))) for c in lin)


def delta_e(c1, c2) -> float:
    a, b = lab(c1), lab(c2)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _median(vals: list) -> tuple[int, int, int] | None:
    if not vals:
        return None
    return tuple(sorted(v[k] for v in vals)[len(vals) // 2] for k in range(3))


def classes(diffuse: Path | str, specular: Path | str | None = None, primary=None, size: int = 256) -> dict:
    """The palette of a finished texture set (module doc). Each class: its median colour and its texel share;
    ``matte``: the main matte colours (a coarse 4-level-per-channel histogram's largest bins, merged within 12
    Lab)."""
    d = Image.open(diffuse).convert("RGB").resize((size, size), Image.NEAREST)
    s = Image.open(specular).convert("RGBA").resize((size, size), Image.NEAREST) if specular else None
    dp = d.load()
    sp = s.load() if s else None
    groups: dict[str, list] = {"gold": [], "silver": [], "cloth": [], "matte": []}
    for y in range(0, size, 2):
        for x in range(0, size, 2):
            c = dp[x, y]
            if c == (0, 0, 0):  # outside every UV island (the gutters past the bake's padding)
                continue
            if sp is None:
                groups["matte"].append(c)
                continue
            r, _, _, a = sp[x, y]
            if r > 127 and primary is not None:
                groups["cloth"].append(tuple(min(255, int(c[k] * primary[k] / 255)) for k in range(3)))
            elif a == GLOSS["gold"]:
                groups["gold"].append(c)
            elif a == GLOSS["silver"]:
                groups["silver"].append(c)
            else:
                groups["matte"].append(c)
    total = sum(len(v) for v in groups.values()) or 1
    out = {k: {"colour": _median(v), "share": round(len(v) / total, 3)} for k, v in groups.items() if k != "matte"}
    bins: dict = {}
    for c in groups["matte"]:
        bins.setdefault(tuple(v // 64 for v in c), []).append(c)
    mattes = []
    for _, vals in sorted(bins.items(), key=lambda kv: -len(kv[1])):
        m = _median(vals)
        if len(vals) / total < 0.02:
            break
        if all(delta_e(m, q["colour"]) > 12 for q in mattes):
            mattes.append({"colour": m, "share": round(len(vals) / total, 3)})
    out["matte"] = mattes[:6]
    return out


def colours(pal: dict) -> list[tuple[int, int, int]]:
    """Every colour of a palette (``classes``), its classes first."""
    out = [pal[k]["colour"] for k in ("cloth", "gold", "silver") if pal.get(k, {}).get("colour")]
    return out + [m["colour"] for m in pal.get("matte", [])]


def snap(colour, palette: list, max_de: float = 25.0) -> tuple[int, int, int]:
    """The palette colour nearest ``colour`` when within ``max_de``, else ``colour`` itself."""
    best = min(palette, key=lambda p: delta_e(colour, p), default=None)
    return tuple(best) if best is not None and delta_e(colour, best) <= max_de else tuple(int(x) for x in colour[:3])


def lum(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
