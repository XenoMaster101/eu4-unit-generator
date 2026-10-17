"""A reference image -> suggested finishes and colours (SPEC Track F, F5; GUI usability U7).

The image is a front view of one soldier, original or licensed (CLAUDE.md rule 8; the GUI says
so). Its background (the colour along the border, or transparent pixels) is left out; each
horizontal band of what remains (head, torso, legs: ``library/vocabulary.json`` "reference")
gives its main colour, which becomes a finish by hue, saturation and value. The results are
``core.prompt.Item``s the user accepts or removes like any other chip. No part detection: parts
come from templates and words.
"""
from __future__ import annotations

import colorsys

from PIL import Image

from eu4unitgen.core.prompt import Item
from eu4unitgen.library.data import library
from eu4unitgen.textures.recolour import dominant_colours


def _rules() -> dict:
    return library("vocabulary")["reference"]


def hsv(c: tuple) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c))
    return h * 360, s, v


def finish_for(c: tuple) -> str:
    """The finish a colour looks like (first rule that fits), else "colour"."""
    h, s, v = hsv(c)
    for r in _rules()["rules"]:
        if all(lo <= x <= hi for key, x in (("hue", h), ("sat", s), ("val", v)) if key in r for lo, hi in [r[key]]):
            return r["finish"]
    return "colour"


def figure(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """(RGB image, 'L' mask of the figure): transparent pixels and the border colour left out."""
    rgba = img.convert("RGBA")
    rgba.thumbnail((256, 256))
    rgb = rgba.convert("RGB")
    w, h = rgb.size
    border = [rgb.getpixel((x, y)) for x in range(w) for y in (0, h - 1)] + \
             [rgb.getpixel((x, y)) for y in range(h) for x in (0, w - 1)]
    bg = dominant_colours(Image.frombytes("RGB", (len(border), 1), bytes(v for c in border for v in c)), 1)[0]
    d2 = _rules()["background_distance"] ** 2
    alpha = rgba.getchannel("A")
    mask = Image.new("L", rgb.size, 0)
    px, al, m = rgb.load(), alpha.load(), mask.load()
    for y in range(h):
        for x in range(w):
            if al[x, y] >= 128 and sum((a - b) ** 2 for a, b in zip(px[x, y], bg)) > d2:
                m[x, y] = 255
    return rgb, mask


def band_colours(img: Image.Image) -> dict[str, tuple | None]:
    """{band: its figure's main colour, or None if the band has no figure pixels}."""
    rgb, mask = figure(img)
    box = mask.getbbox()
    out: dict[str, tuple | None] = {}
    for band, (lo, hi) in _rules()["bands"].items():
        if box is None:
            out[band] = None
            continue
        top = box[1] + int((box[3] - box[1]) * lo)
        bottom = max(top + 1, box[1] + int((box[3] - box[1]) * hi))
        half = (box[2] - box[0]) * _rules()["middle"] / 2  # the body, not the arms reaching out
        mid = (box[0] + box[2]) / 2
        left, right = int(mid - half), max(int(mid - half) + 1, int(mid + half))
        crop, mcrop = rgb.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))
        pixels = [c for c, k in zip(crop.get_flattened_data(), mcrop.get_flattened_data()) if k]
        if not pixels:
            out[band] = None
            continue
        strip = Image.frombytes("RGB", (len(pixels), 1), bytes(v for c in pixels for v in c))
        out[band] = tuple(dominant_colours(strip, 1)[0])
    return out


def suggestions(img: Image.Image) -> list[Item]:
    """The reference's suggestions as chips: a finish for the torso and legs, a crest colour from
    the head band (saturated colours that aren't skin or metal), the main unit colour."""
    rules = _rules()
    cols = band_colours(img)
    items: list[Item] = []
    for band, area in rules["areas"].items():
        c = cols.get(band)
        if c is None:
            continue
        f = finish_for(c)
        label = f"From image: {area} " + (f"rgb {c[0]} {c[1]} {c[2]}" if f == "colour" else f)
        items.append(Item("finish", f, label, area=area, colour=c if f == "colour" else None, words="(image)"))
    head = cols.get("head")
    if head and hsv(head)[1] >= rules["crest_saturation"] and finish_for(head) == "colour":
        items.append(Item("crest_colour", None, f"From image: crest rgb {head[0]} {head[1]} {head[2]}",
                          colour=head, words="(image)"))
    vivid = [c for c in cols.values() if c and finish_for(c) == "colour"]
    if vivid:
        main = max(vivid, key=lambda c: hsv(c)[1])
        items.append(Item("unit_colour", None, f"From image: main colour rgb {main[0]} {main[1]} {main[2]}",
                          colour=main, words="(image)"))
    return items
