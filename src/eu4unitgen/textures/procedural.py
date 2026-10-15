"""Textures made from scratch (no base-game pixels) for procedural parts (Phase 3).

A part's UVs point into a horizontal strip of flat colour swatches, one per material zone
(e.g. dory: wood shaft, bronze head). Swatch i covers u in [i/n, (i+1)/n); the Blender job
puts each zone's faces at that swatch's centre (``swatch_uv``).
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from eu4unitgen.textures.dds import DDSFormat, full_mip_count, write_dds

Colour = tuple[int, int, int]
SIZE = 64
FLAT_NORMAL = (128, 128, 255)


def swatch_uv(index: int, count: int) -> tuple[float, float]:
    return ((index + 0.5) / count, 0.5)


def _strip(colours: list[tuple[int, ...]], mode: str) -> Image.Image:
    img = Image.new(mode, (SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    n = len(colours)
    for i, c in enumerate(colours):
        draw.rectangle([i * SIZE // n, 0, (i + 1) * SIZE // n - 1, SIZE - 1], fill=c)
    return img


def part_textures(zones: list[tuple[Colour, float]]) -> dict[str, bytes]:
    """zones: (diffuse colour, gloss percent) per swatch -> {diffuse, normal, specular} DDS bytes:
    DXT1 diffuse and flat normal, DXT5 spec (RGB 0: no country colour; A = gloss), full mips."""
    if not zones:
        raise ValueError("a part needs at least one colour zone")
    mips = full_mip_count(SIZE, SIZE)
    diffuse = _strip([(*c, 255) for c, _ in zones], "RGBA")
    normal = Image.new("RGBA", (SIZE, SIZE), (*FLAT_NORMAL, 255))
    spec = _strip([(0, 0, 0, round(g * 255 / 100)) for _, g in zones], "RGBA")
    return {"diffuse": write_dds(diffuse, DDSFormat("DXT1", SIZE, SIZE, mips)),
            "normal": write_dds(normal, DDSFormat("DXT1", SIZE, SIZE, mips)),
            "specular": write_dds(spec, DDSFormat("DXT5", SIZE, SIZE, mips))}
