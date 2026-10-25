"""The app's icon: a gold helmet (dome, brow band, nasal guard, a finial) with a red horsehair crest, on a dark crimson
rounded tile. Drawn at 1024 px and reduced, so every size is crisp. Writes ``src/eu4unitgen/gui/icon.png`` (256 px,
the window icon) and ``src/eu4unitgen/gui/eu4unitgen.ico`` (16-256 px, shortcuts and the Apps list).

usage: python packaging/icon/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
S = 1024
CRIMSON, CRIMSON_DARK = (112, 22, 26), (64, 10, 14)
GOLD, GOLD_LIGHT, GOLD_DARK = (214, 168, 74), (246, 212, 128), (140, 98, 36)
STEEL, STEEL_DARK = (196, 200, 206), (110, 116, 126)
PLUME, PLUME_LIGHT = (176, 28, 36), (226, 70, 72)


def draw() -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # the tile: a rounded square, darker towards the bottom
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    for i in range(S):
        t = i / S
        c = tuple(int(CRIMSON[k] * (1 - t) + CRIMSON_DARK[k] * t) for k in range(3))
        td.line([(0, i), (S, i)], fill=c + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([24, 24, S - 24, S - 24], radius=190, fill=255)
    im.paste(tile, (0, 0), mask)
    # the crest: a flat horsehair blade sweeping back from the finial, jagged on top
    crest = [(470, 250), (520, 170), (600, 128), (690, 120), (770, 150), (830, 210), (860, 290), (850, 380),
             (812, 452), (780, 400), (790, 330), (760, 280), (700, 250), (630, 250), (560, 270)]
    d.polygon(crest, fill=PLUME)
    for x0, y0, x1, y1 in ((560, 200, 800, 220), (600, 170, 830, 260), (650, 150, 845, 330)):
        d.line([(x0, y0 + 20), (x1, y1 + 20)], fill=PLUME_LIGHT, width=10)
    # the dome
    d.chord([250, 260, 774, 784], 180, 360, fill=STEEL)
    d.chord([250, 260, 774, 784], 200, 300, fill=(222, 226, 232))
    d.rectangle([250, 520, 774, 700], fill=STEEL)
    # the cheek pieces and the face opening
    d.polygon([(250, 520), (250, 800), (380, 860), (420, 600)], fill=STEEL_DARK)
    d.polygon([(774, 520), (774, 800), (644, 860), (604, 600)], fill=STEEL_DARK)
    d.polygon([(420, 600), (604, 600), (580, 860), (444, 860)], fill=(30, 8, 10))
    # gold: the brow band, the nasal guard, a front rib, the finial
    d.rectangle([236, 500, 788, 560], fill=GOLD)
    d.rectangle([236, 500, 788, 514], fill=GOLD_LIGHT)
    d.rectangle([236, 548, 788, 560], fill=GOLD_DARK)
    d.polygon([(486, 560), (538, 560), (530, 820), (494, 820)], fill=GOLD)
    d.rectangle([494, 280, 530, 500], fill=GOLD)
    d.rectangle([470, 236, 554, 290], fill=GOLD)
    d.ellipse([468, 196, 556, 256], fill=GOLD_LIGHT)
    for x in range(280, 760, 80):  # rivets on the brow band
        d.ellipse([x - 12, 518, x + 12, 542], fill=GOLD_DARK)
    shadow = im.split()[3].filter(ImageFilter.GaussianBlur(6))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste((0, 0, 0, 70), (8, 12), shadow)
    out.alpha_composite(im)
    return out


def main():
    im = draw()
    gui = ROOT / "src" / "eu4unitgen" / "gui"
    im.resize((256, 256), Image.LANCZOS).save(gui / "icon.png")
    im.save(gui / "eu4unitgen.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(gui / "eu4unitgen.ico")


if __name__ == "__main__":
    main()
