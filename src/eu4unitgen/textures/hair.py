"""The hair swatch of a derived body's textures (Q5, SPEC Track R).

Skinned additions (tail, tuft, ears) use the body's own textures (one shape, one material). The
base texture's hair is the soldier's (dark, glossy-looking on horns-like shapes; at map scale the
tail and ears barely read). A derived copy of the body's diffuse, normal and spec gets a painted
horsehair swatch in a texture area no base triangle uses (``derived_body.free_rect``):

  diffuse   warm brown strands running along the swatch's length; a lighter inner-ear patch
  normal    strand relief (tangent space, green up the image: the base game's convention)
  spec      no country-colour mask (RGB 0) and a low, matte gloss (alpha)

The swatch is split along its length: tail, tuft, outer ears, inner ears (``LAYOUT``).
"""
from __future__ import annotations

import math

from PIL import Image

LAYOUT = {"tail": (0.0, 0.62), "tuft": (0.64, 0.78), "ears": (0.80, 0.92), "ears_inner": (0.93, 1.0)}
HAIR = (112, 78, 50)  # warm brown horsehair (user: matte, hair-coloured, not glossy black)
INNER = (168, 128, 104)  # the lighter inner ear
MATTE = 6  # spec alpha (base game gloss 4-18)


def patches(rect: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Each addition's sub-rectangle of the swatch (image px), split along its length (y)."""
    x0, y0, x1, y1 = rect
    h = y1 - y0
    return {k: (x0 + 1, round(y0 + h * a) + 1, x1 - 1, round(y0 + h * b) - 1) for k, (a, b) in LAYOUT.items()}


def _hash(i: int) -> float:
    h = (i * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return (h & 0xFFFF) / 65535.0


def _strand(x: int, y: int) -> float:
    """Strand brightness in [-1, 1]: per-column strands that drift slowly along the length."""
    col = x + int(3 * math.sin(y / 23.0 + x * 0.7))
    base = _hash(col * 7919 + 17) * 2 - 1
    return 0.7 * base + 0.3 * math.sin(y / 9.0 + col * 1.3)


def paint_swatch(diffuse: Image.Image, normal: Image.Image, spec: Image.Image,
                 rect: tuple[int, int, int, int]) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Copies of the three base images with the hair swatch painted into ``rect``."""
    d, n, s = diffuse.convert("RGBA").copy(), normal.convert("RGBA").copy(), spec.convert("RGBA").copy()
    inner = patches(rect)["ears_inner"]
    x0, y0, x1, y1 = rect
    for y in range(y0, y1):
        for x in range(x0, x1):
            k = _strand(x, y)
            colour = INNER if inner[1] - 1 <= y < inner[3] + 1 else HAIR
            f = 1 + 0.22 * k
            d.putpixel((x, y), (*(min(255, max(0, round(c * f))) for c in colour), 255))
            slope = (_strand(x + 1, y) - _strand(x - 1, y)) * 0.35  # strand relief across the length
            nx = -slope / math.sqrt(1 + slope * slope)
            nz = 1 / math.sqrt(1 + slope * slope)
            n.putpixel((x, y), (round((nx * 0.5 + 0.5) * 255), 128, round((nz * 0.5 + 0.5) * 255), 255))
            s.putpixel((x, y), (0, 0, 0, MATTE))
    return d, n, s
