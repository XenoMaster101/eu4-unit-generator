"""Decal masks (white on black) for bake_colour's ``emblem`` (SPEC M21/M22). Historical symbols only (rule 8).

usage: python -m eu4unitgen.textures.decals tetragrammatic <out.png>
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SERIF_BOLD = ("timesbd.ttf", "georgiab.ttf", "DejaVuSerif-Bold.ttf")  # Windows, then Linux


def tetragrammatic(out: Path, size: int = 512) -> Path:
    """The Byzantine tetragrammatic cross: a cross with flared ends and four betas in its quadrants, the left two
    mirrored (as on the Palaiologan banner)."""
    S = size
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    bar, c = int(S * 0.13), S // 2
    lo, hi = int(S * 0.02), int(S * 0.98)
    d.rectangle([c - bar // 2, lo, c + bar // 2, hi], fill=255)
    d.rectangle([lo, c - bar // 2, hi, c + bar // 2], fill=255)
    for a, b in ((lo, lo + bar // 3), (hi - bar // 3, hi)):  # flared ends
        d.rectangle([a, c - int(bar * 0.9), b, c + int(bar * 0.9)], fill=255)
        d.rectangle([c - int(bar * 0.9), a, c + int(bar * 0.9), b], fill=255)
    font = None
    for name in SERIF_BOLD:
        try:
            font = ImageFont.truetype(name, int(S * 0.36))
            break
        except OSError:
            continue
    q = (S - bar) // 2
    glyph = Image.new("L", (q, q), 0)
    gd = ImageDraw.Draw(glyph)
    box = gd.textbbox((0, 0), "B", font=font)
    gd.text(((q - (box[2] - box[0])) / 2 - box[0], (q - (box[3] - box[1])) / 2 - box[1]), "B", fill=255, font=font)
    far = c + bar // 2
    for (x, y), mirror in (((0, 0), True), ((far, 0), False), ((0, far), True), ((far, far), False)):
        im.paste(255, (x, y), ImageOps.mirror(glyph) if mirror else glyph)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return Path(out)


DECALS = {"tetragrammatic": tetragrammatic}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in DECALS:
        raise SystemExit(__doc__)
    print(DECALS[sys.argv[1]](Path(sys.argv[2])))
