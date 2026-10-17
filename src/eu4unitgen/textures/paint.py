"""Painted part textures in the base game's style (Q2, SPEC Track R).

Base-game unit textures are hand-painted: fine material detail (8x8-block luminance std dev
11-22), broad painted light and shadow (16 px-blur std dev 4-26) and a wide value range. Our
parts had flat zone colours (detail 0). ``paint`` shades a part's diffuse from maps the Blender
job bakes on the part's own UVs (``build_parts`` with Cycles):

  ao       ambient occlusion: cavities and insides darker
  curv     curvature (Cycles pointiness, 0.5 = flat): convex edges lighter (painted wear)
  up       how much the surface faces up (world Z, 0..1): a soft top light, like vanilla's paint
  pattern  a material pattern computed in 3D (seamless across UV seams): wood grain, hammered
           bronze, brushed iron, horsehair strands, leather, paint

Country-colour zones keep their neutral grey shaded the same way (the shader multiplies the
colour into it; vanilla's grey under its masks is painted too).
"""
from __future__ import annotations

from io import BytesIO

from PIL import Image

from eu4unitgen.library.data import library

# material -> (pattern strength, AO strength, top-light strength, edge wear lightening, brush grain):
# library/materials.json "paint" (tuned against the base game's measured style: detail 11-20, shading 7-20)
STYLE = {name: tuple(v) for name, v in library("materials")["paint"].items()}


def _grain(i: int) -> float:
    """Deterministic per-texel brush grain in [-1, 1] (a triangular-ish hash, no randomness between runs)."""
    h = (i * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return ((h & 0xFFFF) + ((h >> 16) & 0xFFFF)) / 65535.0 - 1.0


def _pixels(img: Image.Image) -> list:
    """Pixel values in row order (Pillow 12 deprecates ``getdata`` for ``get_flattened_data``)."""
    return list(img.get_flattened_data() if hasattr(img, "get_flattened_data") else img.getdata())


def _map(data: bytes | Image.Image, size: tuple[int, int]) -> list[float]:
    img = data if isinstance(data, Image.Image) else Image.open(BytesIO(data))
    img = img.convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return [v / 255 for v in _pixels(img)]


def paint(diffuse: Image.Image, zones_map: Image.Image, zones: list, maps: dict) -> Image.Image:
    """``diffuse`` (RGBA, flat zone colours, any blazon already on it) shaded by the baked maps."""
    size = diffuse.size
    ao = _map(maps["ao"], size) if "ao" in maps else None
    curv = _map(maps["curv"], size) if "curv" in maps else None
    up = _map(maps["up"], size) if "up" in maps else None
    pat = _map(maps["pattern"], size) if "pattern" in maps else None
    zid = _pixels(zones_map.convert("L").resize(size, Image.Resampling.NEAREST))
    out = []
    for i, (r, g, b, a) in enumerate(_pixels(diffuse.convert("RGBA"))):
        z = zones[zid[i]] if zid[i] < len(zones) else zones[0]
        ps, aos, ups, edge, grain = STYLE.get(getattr(z, "material", "plain"), STYLE["plain"])
        f = 1.0
        if pat is not None:
            f *= 1 + ps * (pat[i] - 0.5) * 2
        if ao is not None:
            f *= 1 - aos * (1 - ao[i])
        if up is not None:
            f *= 1 - ups * 0.5 + ups * up[i]
        lift = edge * max(0.0, (curv[i] - 0.5) * 2) if curv is not None else 0.0
        lift += grain * _grain(i) * (0.4 + 0.6 * (r + g + b) / (3 * 128))  # stronger on lighter paint
        out.append((min(255, max(0, round(r * f + lift))), min(255, max(0, round(g * f + lift))),
                    min(255, max(0, round(b * f + lift))), a))
    img = Image.new("RGBA", diffuse.size)
    img.putdata(out)
    return img
