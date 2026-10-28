"""Thumbnails for the Library and the builder: a render lifted onto a lit backdrop, so a dark model reads on any
screen. The render's own background is keyed out (a flat colour flooded in from the edges) when it has no alpha;
the model is cropped to its outline, brightened a little, and set on a radial backdrop with a soft ground shadow.
Cached by source, size and colours (``cache``). Pillow only."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def _rgb(hexcolour: str) -> tuple[int, int, int]:
    h = hexcolour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def cutout(im: Image.Image, tol: int = 18) -> Image.Image:
    """RGBA with the render's flat background transparent (only where it touches the edges)."""
    im = im.convert("RGBA")
    a = im.getchannel("A")
    if a.getextrema()[0] < 250:  # already has transparency
        return im
    rgb = im.convert("RGB")
    w, h = rgb.size
    corners = [rgb.getpixel(xy) for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    if max(max(abs(a - b) for a, b in zip(c, corners[0])) for c in corners) > 6:
        return im  # not a flat render background (a photo or a drawing): kept whole
    mark = (255, 0, 255)
    flood = rgb.copy()
    for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1)):
        if flood.getpixel(xy) != mark:
            ImageDraw.floodfill(flood, xy, mark, thresh=tol)
    from PIL import ImageChops
    diff = ImageChops.difference(flood, Image.new("RGB", (w, h), mark)).convert("L")
    m = diff.point(lambda v: 0 if v == 0 else 255)  # the flooded background -> transparent
    im.putalpha(m.filter(ImageFilter.GaussianBlur(0.8)))
    return im


def make(src: Path, size: tuple[int, int], inner: str, outer: str, lift: float | None = None) -> Image.Image:
    """``lift``: the brightening; by default enough to bring the model's mean brightness near 115 (1.0-1.9)."""
    w, h = size
    im = cutout(Image.open(src))
    box = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if box:
        im = im.crop(box)
    rgb, alpha = im.convert("RGB"), im.getchannel("A")
    if lift is None:
        from PIL import ImageStat
        mean = ImageStat.Stat(rgb.convert("L"), alpha.point(lambda v: 255 if v > 128 else 0)).mean[0] or 115
        lift = max(1.0, min(1.9, 115 / mean))
    rgb = ImageEnhance.Brightness(rgb).enhance(lift)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = ImageEnhance.Color(rgb).enhance(1.1)
    im = Image.merge("RGBA", (*rgb.split(), alpha))
    # the backdrop: a radial light from the top centre
    yy = Image.linear_gradient("L").resize((w, h))
    rad = Image.radial_gradient("L").resize((int(w * 1.6), int(h * 1.6))).crop(
        (int(w * 0.3), int(h * 0.15), int(w * 0.3) + w, int(h * 0.15) + h))
    mixmask = ImageOps.autocontrast(Image.blend(rad, yy, 0.25))
    back = Image.composite(Image.new("RGB", (w, h), _rgb(outer)), Image.new("RGB", (w, h), _rgb(inner)), mixmask)
    back = back.convert("RGBA")
    scale = min((w * 0.86) / im.width, (h * 0.84) / im.height)
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    x, y = (w - im.width) // 2, int(h * 0.9) - im.height
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # the ground shadow
    ImageDraw.Draw(sh).ellipse((x + im.width * 0.12, h * 0.86, x + im.width * 0.88, h * 0.94), fill=(40, 28, 12, 90))
    back.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    back.alpha_composite(im, (x, max(0, y)))
    return back.convert("RGB")


def cache(src: Path | str | None, size: tuple[int, int], inner: str, outer: str, folder: Path) -> str | None:
    """The thumbnail's path (made once), or None when there is no source picture."""
    if not src or not Path(src).is_file():
        return None
    st = Path(src).stat()
    key = hashlib.sha1(f"{Path(src).resolve()}|{st.st_mtime_ns}|{size}|{inner}|{outer}|4".encode()).hexdigest()[:20]
    out = Path(folder) / f"{key}.png"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        make(Path(src), size, inner, outer).save(out)
    return str(out)
