"""The input guide (Track P2.3): what makes a good reference, and warnings before a run when an image breaks the
rules. Pillow only.

The rules (from the horse and rider tracks):
  * a horse: a clean side view, the whole horse in frame, legs apart enough to see all four hooves;
  * a soldier (or rider): a front view standing, arms a little out from the body (an A-pose), weapons held clear of
    the body (the lance beside it, the shield out to the side), the whole figure in frame;
  * a gun (artillery, planned): a three-quarter view, the whole gun and its carriage in frame;
  * a plain, even background (a studio backdrop or a flat colour), no other figures, props or text;
  * at least about 768 pixels on the long side;
  * your own image or one you may use (the rights statement, rule 8).
"""
from __future__ import annotations

from pathlib import Path

import statistics

from PIL import Image

GUIDE = {
    "horse": ["A side view: the horse seen from its left or right, head to tail in frame.",
              "All four legs visible, a little apart; nothing in front of them.",
              "A plain, even background; no rider, people, props or text.",
              "Armour, caparison and trappings as you want them in game (they are read from the image).",
              "At least about 768 pixels wide."],
    "rider": ["A front view of the soldier standing, the whole figure in frame, feet included.",
              "Arms a little out from the body (an A-pose), so the arms and the body are apart.",
              "Weapons held clear of the body: a lance or spear upright beside it, a shield out to the side.",
              "A plain, even background; no horse, other figures, props or text.",
              "At least about 768 pixels tall."],
    "artillery": ["A three-quarter view of the gun: from the front and a little to one side, so the barrel's length "
                  "and the wheels both show.",
                  "The whole gun and carriage in frame; no crew (your soldiers become the crew).",
                  "A plain, even background; no people, props or text.",
                  "At least about 768 pixels wide."],
    "all": ["Use your own image or one you have the right to use, and say so in the rights statement."],
}
GUIDE["soldier"] = GUIDE["rider"]
# why each kind's picture matters, in one sentence (the New unit screen's first step)
WHY = {
    "soldier": "The 3D model is guessed from this one picture, so the app needs to see the whole figure from the front, "
               "with the arms apart from the body: that is how it tells the arms, the body, the weapon and the shield "
               "apart and fits them to the game's skeleton.",
    "horse": "The 3D model is guessed from this one picture. A side view shows the length of the body, the legs and "
             "the tail, which is what the game's horse skeleton needs.",
    "artillery": "A three-quarter view shows both the barrel's length and the wheels, which a front or side view "
                 "hides.",
}
WHY["rider"] = WHY["soldier"]
# what each kind's picture should look like, in a few words (shown with the warnings)
VIEW = {"soldier": "front view, arms a little out (A-pose)", "rider": "front view, arms a little out (A-pose)",
        "horse": "side view", "artillery": "three-quarter view"}


def horse_check(image: Path) -> list[str]:
    """What may go wrong with this image: each a sentence for the user (empty: looks fine)."""
    out = []
    im = Image.open(image)
    small = im.convert("RGBA")
    small.thumbnail((256, 256))
    w, h = small.size
    alpha = small.getchannel("A")
    if alpha.getextrema()[0] < 128:  # a cut-out: the background is already gone
        fg = alpha.point(lambda a: 255 if a >= 128 else 0)
    else:
        rgb = small.convert("RGB")
        px = rgb.load()
        b = max(2, round(0.03 * min(w, h)))
        # the background from the border: each row's left and right strips, each column's top and bottom ones
        # (a soft gradient or vignette stays background), blended across the image
        med = lambda pts: tuple(statistics.median(px[q][k] for q in pts) for k in range(3))  # noqa: E731
        left = [med([(x, y) for x in range(b)]) for y in range(h)]
        right = [med([(x, y) for x in range(w - b, w)]) for y in range(h)]
        top = [med([(x, y) for y in range(b)]) for x in range(w)]
        bottom = [med([(x, y) for y in range(h - b, h)]) for x in range(w)]
        # clean: the border changes little from pixel to pixel (no props, floor lines or scenery)
        ring = [(x, y) for y in range(h) for x in range(w) if min(x, y, w - 1 - x, h - 1 - y) < b]
        calm = sum(1 for x, y in ring if x + 1 < w and y + 1 < h and
                   sum(abs(px[x, y][k] - px[x + 1, y][k]) + abs(px[x, y][k] - px[x, y + 1][k]) for k in range(3)) < 24)
        calm /= len(ring)
        if calm < 0.75:  # clean AI renders: 89%; a reference sheet with text and props: 31%
            out.append(f"The background doesn't look clean: only {calm:.0%} of the image's border is plain. "
                       "A plain, even background reads best; the horse may be cut out badly.")
        fg = Image.new("L", (w, h))
        fgp = fg.load()
        for y in range(h):
            for x in range(w):
                u, v = x / (w - 1), y / (h - 1)
                bg = [((1 - u) * left[y][k] + u * right[y][k] + (1 - v) * top[x][k] + v * bottom[x][k]) / 2 for k in range(3)]
                fgp[x, y] = 255 if sum(abs(px[x, y][k] - bg[k]) for k in range(3)) >= 60 else 0
    box = fg.getbbox()
    if box is None:
        return out + ["No horse found: the image looks empty."]
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw < 1.1 * bh:
        out.append(f"This may not be a side view: the horse is {bw / bh:.2f} times as wide as it is tall (a side "
                   "view is about 1.2 to 1.6). The pipeline reads side views; a front or three-quarter view builds badly.")
    edges = sum((box[0] <= 1, box[1] <= 1, box[2] >= w - 1, box[3] >= h - 1))
    if edges >= 2:
        out.append("The horse touches the image's edges: it may be cropped (missing hooves, ears or tail).")
    return out


def _foreground(image: Path):
    """(mask L at <=256 px, size, the border's plainness 0-1) from the image's alpha, else from its border colours."""
    im = Image.open(image)
    full = im.size
    small = im.convert("RGBA")
    small.thumbnail((256, 256))
    a = small.getchannel("A")
    if a.getextrema()[0] < 128:
        return a.point(lambda v: 255 if v >= 128 else 0), full, 1.0
    warn = horse_check(image)
    calm = 0.5 if any("background" in w for w in warn) else 1.0
    return _border_mask(small.convert("RGB")), full, calm


def _border_mask(rgb: Image.Image) -> Image.Image:
    """The subject against a background read from the border (each row's left and right strips, each column's top
    and bottom ones, blended across the image), so a soft gradient or vignette stays background."""
    px = rgb.load()
    w, h = rgb.size
    b = max(2, round(0.03 * min(w, h)))
    med = lambda pts: tuple(statistics.median(px[q][k] for q in pts) for k in range(3))  # noqa: E731
    left = [med([(x, y) for x in range(b)]) for y in range(h)]
    right = [med([(x, y) for x in range(w - b, w)]) for y in range(h)]
    top = [med([(x, y) for y in range(b)]) for x in range(w)]
    bottom = [med([(x, y) for y in range(h - b, h)]) for x in range(w)]
    m = Image.new("L", (w, h))
    mp = m.load()
    for y in range(h):
        for x in range(w):
            u, v = x / (w - 1), y / (h - 1)
            bg = [((1 - u) * left[y][k] + u * right[y][k] + (1 - v) * top[x][k] + v * bottom[x][k]) / 2 for k in range(3)]
            mp[x, y] = 255 if sum(abs(px[x, y][k] - bg[k]) for k in range(3)) >= 60 else 0
    return m


def check(image: Path, kind: str) -> list[str]:
    """Warnings for an image (empty: it follows the guide as far as can be told), by what it should show: a horse's
    side view, a soldier's front A-pose, a gun's three-quarter view."""
    out = []
    full = Image.open(image).size
    if kind == "horse":
        out += horse_check(image)
    elif kind == "artillery":
        m, full, calm = _foreground(image)
        if calm < 1.0:
            out.append("The background doesn't look plain: the gun may be cut out badly.")
        box = m.getbbox()
        if box is None:
            return out + ["No gun found: the image looks empty."]
        ratio = (box[2] - box[0]) / max(1, box[3] - box[1])
        if not 1.0 <= ratio <= 2.6:
            out.append(f"This may not be a three-quarter view: the gun is {ratio:.1f} times as wide as it is tall (a "
                       "three-quarter view is about 1.2 to 2).")
    else:
        m, full, calm = _foreground(image)
        if calm < 1.0:
            out.append("The background doesn't look plain: the figure may be cut out badly. A plain, even background "
                       "works best.")
        box = m.getbbox()
        if box is None:
            return out + ["No figure found: the image looks empty."]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw > 1.1 * bh:
            out.append("This doesn't look like a standing front view: the figure is wider than it is tall.")
        elif bw < 0.38 * bh:  # good A-poses with a lance and shield: 0.55-0.71; arms down 0.24; a side view 0.15
            out.append("The figure is very narrow: the arms, weapon and shield seem to be close to the body, or it is "
                       "seen from the side. A front view with the arms a little out (an A-pose) builds much better.")
        w, h = m.size
        # three edges: a lance tip at the top and the feet at the bottom alone are common in good references
        if sum((box[0] <= 1, box[1] <= 1, box[2] >= w - 1, box[3] >= h - 1)) >= 3:
            out.append("The figure touches the image's edges: it may be cropped (feet, weapon tips or the helmet).")
    if max(full) < 700:
        out.append(f"The image is small ({full[0]} x {full[1]}); at least about 768 pixels on the long side reads "
                   "better.")
    return out
