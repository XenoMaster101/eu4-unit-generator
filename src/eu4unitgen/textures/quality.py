"""Texture quality measures (Q6, SPEC Track R): the base game's painted style and texel density as
numbers, so generated parts can be checked against them.

  density    sqrt(sum UV area in texels / sum 3D surface area): texels per mesh unit
  detail     mean std dev of luminance in 8x8 blocks fully inside the UVs (fine painted detail)
  shading    std dev of luminance after a blur of 16/512 of the width, over covered texels (broad
             painted light and shadow)
  normal     mean tilt of the tangent-space normal from flat, degrees
  gloss      mean spec alpha over covered texels

Measured on 1.37.5 (SPEC Track R "Quality measurements"): bodies 16.5-32.8 texels/unit, detail
12-13, shading 22-26; weapons and shields 9.8-26.9, detail 11-22, shading 4-17, normal 4-20
degrees, gloss 4-18.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter, ImageStat

# what a generated part should reach (base-game ranges, widened a little; shading has no upper bound:
# one texture may hold dark lining next to bright metal)
RANGES = {"density": (15.0, 40.0), "detail": (8.0, None), "shading": (4.0, None), "normal": (4.0, 20.0),
          "gloss": (2.0, 20.0)}


def _pixels(img: Image.Image) -> list:
    return list(img.get_flattened_data() if hasattr(img, "get_flattened_data") else img.getdata())


def density(u0: list[float], p: list[float], tri: list[int], size: tuple[int, int]) -> float:
    area3 = area_uv = 0.0
    for k in range(0, len(tri), 3):
        a, b, c = tri[k:k + 3]
        e1 = [p[b * 3 + j] - p[a * 3 + j] for j in range(3)]
        e2 = [p[c * 3 + j] - p[a * 3 + j] for j in range(3)]
        cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        area3 += 0.5 * math.sqrt(sum(x * x for x in cr))
        uv = [(u0[i * 2] * size[0], u0[i * 2 + 1] * size[1]) for i in (a, b, c)]
        area_uv += 0.5 * abs((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]))
    return math.sqrt(area_uv / area3) if area3 else 0.0


def coverage(u0: list[float], tri: list[int], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for k in range(0, len(tri), 3):
        draw.polygon([(u0[i * 2] * size[0], u0[i * 2 + 1] * size[1]) for i in tri[k:k + 3]], fill=255)
    return mask


def style(diffuse: Image.Image, normal: Image.Image | None, spec: Image.Image | None, mask: Image.Image) -> dict:
    lum = diffuse.convert("L")
    w, h = lum.size
    mk = _pixels(mask)
    blocks = [ImageStat.Stat(lum.crop((x, y, x + 8, y + 8))).stddev[0]
              for y in range(0, h - 7, 8) for x in range(0, w - 7, 8)
              if mask.crop((x, y, x + 8, y + 8)).getextrema()[0] == 255]
    blur = [v for v, k in zip(_pixels(lum.filter(ImageFilter.GaussianBlur(16 * w / 512))), mk) if k]
    mean = sum(blur) / len(blur) if blur else 0.0
    shading = math.sqrt(sum((v - mean) ** 2 for v in blur) / len(blur)) if blur else 0.0
    tilt = 0.0
    if normal is not None:
        tilts = []
        for (r, g, _b), k in zip(_pixels(normal.convert("RGB").resize(lum.size)), mk):
            if k:
                x, y = r / 127.5 - 1, g / 127.5 - 1
                tilts.append(math.degrees(math.asin(min(1.0, math.hypot(x, y)))))
        tilt = sum(tilts) / len(tilts) if tilts else 0.0
    gloss = None
    if spec is not None:
        alpha = [px[3] for px, k in zip(_pixels(spec.convert("RGBA").resize(lum.size)), mk) if k]
        gloss = sum(alpha) / len(alpha) if alpha else None
    return {"detail": sum(blocks) / len(blocks) if blocks else 0.0, "shading": shading, "normal": tilt, "gloss": gloss}


def measure(u0: list[float], p: list[float], tri: list[int], diffuse: Image.Image, normal: Image.Image | None,
            spec: Image.Image | None) -> dict:
    """All five measures for one mesh and its three textures."""
    out = style(diffuse, normal, spec, coverage(u0, tri, diffuse.size))
    out["density"] = density(u0, p, tri, diffuse.size)
    return out


def outside(m: dict) -> list[str]:
    """The measures outside RANGES, as short phrases."""
    found = []
    for key, (lo, hi) in RANGES.items():
        v = m.get(key)
        if v is None:
            continue
        if lo is not None and v < lo:
            found.append(f"{key} {v:.1f} < {lo:g}")
        elif hi is not None and v > hi:
            found.append(f"{key} {v:.1f} > {hi:g}")
    return found
