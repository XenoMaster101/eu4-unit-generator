"""Built-in shield emblems for the Result screen's Colours tab (SPEC Track F, F10).

Plain geometric shapes drawn here, original to this app (no game, franchise or real-world
heraldry files; CLAUDE.md rule 8). Each is an RGBA image: the shape in one colour on transparency,
as a hoplon blazon expects (textures.procedural.blazon_layer). The user may also use their own PNG.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
EMBLEMS = {"star": "Star", "crescent": "Crescent", "lambda": "Lambda", "wheel": "Wheel", "trident": "Trident",
           "triskelion": "Triskelion"}


def draw(name: str, colour: tuple = (195, 150, 60)) -> Image.Image:
    if name not in EMBLEMS:
        raise ValueError(f"unknown emblem {name!r}; known: {', '.join(EMBLEMS)}")
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = tuple(int(v) for v in colour) + (255,)
    m = SIZE / 2
    if name == "star":  # eight rays
        pts = []
        for i in range(16):
            r = 92 if i % 2 == 0 else 34
            a = math.pi / 8 * i - math.pi / 2
            pts.append((m + r * math.cos(a), m + r * math.sin(a)))
        d.polygon(pts, fill=c)
    elif name == "crescent":
        d.ellipse((m - 80, m - 80, m + 80, m + 80), fill=c)
        d.ellipse((m - 50, m - 95, m + 100, m + 55), fill=(0, 0, 0, 0))
    elif name == "lambda":  # the Greek letter, as a plain thick chevron
        d.line([(m - 70, m + 85), (m, m - 85), (m + 70, m + 85)], fill=c, width=34, joint="curve")
    elif name == "wheel":
        d.ellipse((m - 85, m - 85, m + 85, m + 85), outline=c, width=22)
        d.ellipse((m - 20, m - 20, m + 20, m + 20), fill=c)
        for i in range(6):
            a = math.pi / 3 * i
            d.line([(m, m), (m + 80 * math.cos(a), m + 80 * math.sin(a))], fill=c, width=14)
    elif name == "trident":
        d.line([(m, m + 95), (m, m - 60)], fill=c, width=20)
        d.line([(m - 60, m - 20), (m + 60, m - 20)], fill=c, width=18)
        for x in (m - 60, m + 60):
            d.line([(x, m - 20), (x, m - 80)], fill=c, width=18)
        for x in (m - 60, m, m + 60):
            d.polygon([(x - 16, m - 75), (x + 16, m - 75), (x, m - 100)], fill=c)
    elif name == "triskelion":  # three arms spiralling out from the centre, a third of a turn apart
        for i in range(3):
            a = 2 * math.pi / 3 * i
            arc = [(m + r * math.cos(a + (r - 15) / 70 * 1.9), m + r * math.sin(a + (r - 15) / 70 * 1.9))
                   for r in range(15, 88, 3)]
            d.line(arc, fill=c, width=24, joint="curve")
        d.ellipse((m - 26, m - 26, m + 26, m + 26), fill=c)
    return img


def write(name: str, folder: Path, colour: tuple = (195, 150, 60)) -> Path:
    """Draw emblem ``name`` into ``folder`` (``emblem_<name>_<rrggbb>.png``) and return the path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"emblem_{name}_{''.join(f'{int(v):02x}' for v in colour)}.png"
    if not path.is_file():
        draw(name, colour).save(path)
    return path
