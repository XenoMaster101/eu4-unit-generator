"""Good and bad example pictures for the input guide, drawn here (simple original drawings: nothing copied or
bundled). ``examples(kind)`` -> [(good, caption, PIL image)] for a soldier, a horse or a gun."""
from __future__ import annotations

import random

from PIL import Image, ImageDraw

SIZE = 240
BG = (226, 226, 230)
STEEL, CLOTH, SKIN, GOLD, WOOD, DARK = (150, 152, 160), (160, 40, 40), (222, 180, 150), (200, 160, 60), (120, 80, 45), (50, 45, 45)
COAT, HAIR = (140, 80, 40), (40, 30, 25)


def _canvas(busy: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(im)
    if busy:  # a reference sheet: props, lines, labels, a second figure's outline
        rnd = random.Random(7)
        for _ in range(14):
            x, y = rnd.randrange(SIZE), rnd.randrange(SIZE)
            d.rectangle([x, y, x + rnd.randrange(14, 50), y + rnd.randrange(8, 30)],
                        fill=tuple(rnd.randrange(60, 200) for _ in range(3)))
        for i in range(0, SIZE, 24):
            d.line([0, i, SIZE, i + 30], fill=(170, 160, 140), width=1)
        d.text((8, 6), "CHARACTER SHEET  v2", fill=(20, 20, 20))
        d.text((150, 222), "front / back", fill=(20, 20, 20))
    return im, d


def soldier(arms: str = "out", view: str = "front", busy: bool = False, crop: bool = False) -> Image.Image:
    im, d = _canvas(busy)
    k = 1.6 if crop else 1.0
    ox, oy = (SIZE / 2, 22) if not crop else (SIZE / 2, 60)

    def P(x, y):  # figure coordinates (0 at the head's centre line) to the canvas
        return ox + x * k, oy + y * k

    def line(a, b, w, c):
        d.line([P(*a), P(*b)], fill=c, width=int(w * k))
    if view == "side":
        d.polygon([P(-10, 40), P(12, 40), P(10, 128), P(-8, 128)], fill=STEEL)  # a narrow body: one arm hides the other
        line((0, 44), (4, 120), 9, CLOTH)
        line((-3, 128), (-6, 208), 10, DARK)
        line((5, 128), (8, 208), 10, DARK)
        d.ellipse([P(-11, 6), P(13, 34)], fill=SKIN)
        d.ellipse([P(-13, 2), P(13, 22)], fill=STEEL)
        line((18, -10), (18, 214), 4, WOOD)
        return im
    d.polygon([P(-26, 44), P(26, 44), P(18, 126), P(-18, 126)], fill=STEEL)  # the scale-armoured body
    d.polygon([P(-20, 120), P(20, 120), P(26, 150), P(-26, 150)], fill=CLOTH)  # the skirt
    line((-10, 150), (-14, 210), 11, DARK)
    line((10, 150), (14, 210), 11, DARK)
    if arms == "out":  # an A-pose: the arms clear of the body
        line((-26, 48), (-58, 118), 10, STEEL)
        line((26, 48), (58, 118), 10, STEEL)
        line((62, -12), (62, 214), 4, WOOD)  # the lance upright beside it
        d.polygon([P(-96, 80), P(-62, 80), P(-66, 132), P(-79, 150), P(-92, 132)], fill=CLOTH, outline=GOLD)  # shield out
    else:  # arms down against the body, the shield held over it
        line((-24, 48), (-22, 122), 10, STEEL)
        line((24, 48), (22, 122), 10, STEEL)
        d.polygon([P(-22, 70), P(14, 70), P(10, 120), P(-4, 140), P(-18, 120)], fill=CLOTH, outline=GOLD)
        line((24, -12), (24, 214), 4, WOOD)
    d.ellipse([P(-13, 6), P(13, 34)], fill=SKIN)
    d.polygon([P(-15, 18), P(-13, 2), P(0, -6), P(13, 2), P(15, 18)], fill=STEEL)  # the helmet
    line((0, -6), (0, -22), 4, GOLD)
    return im


def horse(view: str = "side", busy: bool = False, rider: bool = False, crop: bool = False) -> Image.Image:
    im, d = _canvas(busy)
    if view == "front":  # head-on: no length to read
        d.ellipse([88, 90, 152, 170], fill=COAT)
        d.polygon([(108, 40), (132, 40), (128, 110), (112, 110)], fill=COAT)
        for x in (96, 112, 128, 144):
            d.rectangle([x - 5, 160, x + 5, 222], fill=COAT)
        d.polygon([(104, 40), (114, 22), (120, 40)], fill=HAIR)
        return im
    s = 1.35 if crop else 1.0
    ox = -40 if crop else 0

    def B(box):
        return [box[0] * s + ox, box[1] * s, box[2] * s + ox, box[3] * s]
    d.ellipse(B([46, 96, 186, 150]), fill=COAT)  # the barrel
    d.polygon([(p[0] * s + ox, p[1] * s) for p in [(156, 104), (190, 70), (212, 66), (224, 84), (204, 92), (178, 128)]], fill=COAT)
    d.polygon([(p[0] * s + ox, p[1] * s) for p in [(162, 100), (190, 68), (180, 70), (156, 96)]], fill=HAIR)  # the mane
    for x in (66, 84, 156, 172):
        d.rectangle(B([x - 5, 140, x + 5, 214]), fill=COAT)
        d.rectangle(B([x - 6, 206, x + 6, 216]), fill=DARK)
    d.polygon([(p[0] * s + ox, p[1] * s) for p in [(54, 104), (36, 120), (30, 176), (44, 178), (50, 130)]], fill=HAIR)
    d.rectangle(B([96, 92, 140, 120]), fill=CLOTH, outline=GOLD)  # the saddle cloth
    if rider:
        d.rectangle([108, 50, 128, 96], fill=STEEL)
        d.ellipse([108, 30, 128, 50], fill=SKIN)
    return im


def gun(view: str = "three_quarter", busy: bool = False) -> Image.Image:
    im, d = _canvas(busy)
    if view == "front":
        d.ellipse([104, 104, 136, 136], fill=DARK)  # the muzzle head-on
        d.rectangle([92, 136, 148, 170], fill=WOOD)
        d.ellipse([56, 120, 76, 210], fill=WOOD)
        d.ellipse([164, 120, 184, 210], fill=WOOD)
        return im
    d.polygon([(40, 150), (176, 96), (184, 110), (52, 170)], fill=DARK)  # the barrel, receding
    d.polygon([(70, 160), (170, 128), (182, 150), (90, 188)], fill=WOOD)  # the carriage
    d.polygon([(150, 140), (226, 180), (220, 190), (144, 154)], fill=WOOD)  # the trail
    d.ellipse([60, 140, 116, 214], outline=WOOD, width=7)  # the near wheel
    d.ellipse([150, 116, 186, 170], outline=WOOD, width=6)  # the far wheel
    return im


def examples(kind: str) -> list[tuple[bool, str, Image.Image]]:
    """The guide's pictures for ``kind`` (soldier / rider, horse, artillery): good first."""
    if kind in ("soldier", "rider"):
        return [(True, "Front view, arms a little out, lance and shield clear of the body, plain background", soldier()),
                (False, "Arms down and the shield over the body: they melt into one shape", soldier(arms="down")),
                (False, "A side view: one arm and the shield are hidden", soldier(view="side")),
                (False, "A busy character sheet: labels and props get modelled too", soldier(busy=True)),
                (False, "Cropped: the feet and the lance tip are missing", soldier(crop=True))]
    if kind == "horse":
        return [(True, "Side view, all four legs and the tail in frame, plain background", horse()),
                (False, "Head-on: the body's length can't be read", horse(view="front")),
                (False, "With a rider: make the soldier separately", horse(rider=True)),
                (False, "Cropped: the head and hooves are cut off", horse(crop=True))]
    return [(True, "Three-quarter view: the barrel's length and both wheels show", gun()),
            (False, "Head-on: the barrel's length is hidden", gun(view="front")),
            (False, "A busy background", gun(busy=True))]
