"""Texture edits on decoded images (SPEC §4, §10 Phase 2).

Diffuse operations take an effective mask (region minus the protect mask, see
``textures.regions``) and leave everything outside it untouched. Spec operations paint the
country-colour channels (R/G/B) and gloss (A). ``shade`` reproduces how PdxMeshColor combines
diffuse, spec and country colours (vanilla ``gfx/FX/pdxmesh.shader``), for previews.
"""
from __future__ import annotations

from PIL import Image, ImageChops, ImageStat

from eu4unitgen.textures.regions import COUNTRY_REGIONS, NOT, AND

Colour = tuple[int, int, int]
BLENDS = ("normal", "multiply", "overlay")
GLOSS_PRESETS = {"matte": 3, "cloth": 8, "metal": 13}  # percent of 255 spec alpha (SPEC §4)


class RecolourError(ValueError):
    pass


def effective(region: Image.Image, protect: Image.Image | None) -> Image.Image:
    return region if protect is None else AND(region, NOT(protect))


def _scaled(mask: Image.Image, factor: float) -> Image.Image:
    if factor >= 1:
        return mask
    return mask.point(lambda v: round(v * max(0.0, factor)))


def _apply(img: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    out = Image.composite(edited.convert("RGB"), img.convert("RGB"), mask).convert("RGBA")
    out.putalpha(img.convert("RGBA").getchannel("A"))
    return out


def hue_shift(img: Image.Image, degrees: float, mask: Image.Image | None = None) -> Image.Image:
    """Rotate hue by ``degrees`` (inside ``mask`` if given); saturation, value and alpha stay."""
    rgba = img.convert("RGBA")
    h, s, v = rgba.convert("RGB").convert("HSV").split()
    step = round(degrees / 360 * 256) % 256
    h = h.point(lambda x: (x + step) % 256)
    shifted = Image.merge("HSV", (h, s, v)).convert("RGB")
    if mask is None:
        out = shifted.convert("RGBA")
        out.putalpha(rgba.getchannel("A"))
        return out
    return _apply(rgba, shifted, mask)


def recolour(img: Image.Image, mask: Image.Image, colour: Colour, strength: float = 1.0,
             detail: float = 1.0) -> Image.Image:
    """Paint ``colour`` into the masked area, keeping each pixel's lightness relative to the
    area's mean lightness (folds, seams and shading survive). ``detail`` (0-1) scales how much of
    that lightness detail is kept: 1 all of it, 0 a flat colour (e.g. skin over mail rings)."""
    if not 0 <= strength <= 1:
        raise RecolourError(f"strength {strength} must be 0-1")
    if not 0 <= detail <= 1:
        raise RecolourError(f"detail {detail} must be 0-1")
    lum = img.convert("L")
    binary = mask.point(lambda v: 255 if v >= 128 else 0)
    if binary.getbbox() is None:
        return img.convert("RGBA")
    mean = max(1.0, ImageStat.Stat(lum, binary).mean[0])
    bands = [lum.point(lambda v, c=c: min(255, round(c * (1 + detail * (v / mean - 1))))) for c in colour]
    return _apply(img, Image.merge("RGB", bands), _scaled(mask, strength))


def tile(pattern: Image.Image, size: tuple[int, int], scale: float = 1.0) -> Image.Image:
    if scale <= 0:
        raise RecolourError(f"pattern scale {scale} must be > 0")
    p = pattern.convert("RGB")
    if scale != 1:
        p = p.resize((max(1, round(p.width * scale)), max(1, round(p.height * scale))), Image.Resampling.BICUBIC)
    out = Image.new("RGB", size)
    for y in range(0, size[1], p.height):
        for x in range(0, size[0], p.width):
            out.paste(p, (x, y))
    return out


def pattern_overlay(img: Image.Image, mask: Image.Image, pattern: Image.Image, blend: str = "multiply",
                    opacity: float = 1.0, scale: float = 1.0) -> Image.Image:
    if blend not in BLENDS:
        raise RecolourError(f"blend {blend!r} must be one of {BLENDS}")
    if not 0 <= opacity <= 1:
        raise RecolourError(f"opacity {opacity} must be 0-1")
    base = img.convert("RGB")
    layer = tile(pattern, base.size, scale)
    mixed = {"normal": layer, "multiply": ImageChops.multiply(base, layer),
             "overlay": ImageChops.overlay(base, layer)}[blend]
    return _apply(img, mixed, _scaled(mask, opacity))


def _channel(name: str) -> int:
    if name not in COUNTRY_REGIONS:
        raise RecolourError(f"spec channel {name!r} must be one of {COUNTRY_REGIONS}")
    return COUNTRY_REGIONS.index(name)


def spec_paint(spec: Image.Image, mask: Image.Image, channel: str, mode: str = "set") -> Image.Image:
    """Set (or clear) a country-colour channel of the spec map inside ``mask``."""
    if mode not in ("set", "clear"):
        raise RecolourError(f"spec_paint mode {mode!r} must be set or clear")
    bands = list(spec.convert("RGBA").split())
    i = _channel(channel)
    bands[i] = ImageChops.lighter(bands[i], mask) if mode == "set" else ImageChops.multiply(bands[i], NOT(mask))
    return Image.merge("RGBA", bands)


FLAT_NORMAL = (128, 128, 255)  # tangent-space "no bump" (RGB, OpenGL green-up as measured on vanilla)


def flatten(normal: Image.Image, mask: Image.Image, strength: float = 1.0) -> Image.Image:
    """Blend the normal map toward flat inside ``mask`` (alpha kept): e.g. bare skin painted over mail."""
    if not 0 <= strength <= 1:
        raise RecolourError(f"flatten strength {strength} must be 0-1")
    src = normal.convert("RGBA")
    flat = Image.merge("RGBA", (*(Image.new("L", src.size, c) for c in FLAT_NORMAL), src.getchannel("A")))
    return Image.composite(flat, src, _scaled(mask, strength))


def gloss(spec: Image.Image, mask: Image.Image, percent: float) -> Image.Image:
    """Set spec alpha (gloss) to ``percent`` of 255 inside ``mask``."""
    if not 0 <= percent <= 100:
        raise RecolourError(f"gloss {percent}% must be 0-100")
    bands = list(spec.convert("RGBA").split())
    bands[3] = Image.composite(Image.new("L", spec.size, round(percent * 255 / 100)), bands[3], mask)
    return Image.merge("RGBA", bands)


def dominant_colours(reference: Image.Image, count: int, min_distance: int = 60) -> list[Colour]:
    """The reference image's ``count`` most common, clearly different colours, most common first.
    Quantizes finely (median cut + k-means), then takes colours by pixel count, skipping any
    within ``min_distance`` (RGB) of one already taken."""
    if count < 1:
        return []
    small = reference.convert("RGB")
    small.thumbnail((256, 256))
    quant = small.quantize(colors=max(16, 4 * count), method=Image.Quantize.MEDIANCUT, kmeans=4)
    pal = quant.getpalette() or []
    ranked = [tuple(pal[3 * i:3 * i + 3]) for _, i in sorted(quant.getcolors() or [], reverse=True)]
    out: list[Colour] = []
    for c in ranked:
        if all(sum((a - b) ** 2 for a, b in zip(c, o)) >= min_distance ** 2 for o in out):
            out.append(c)
        if len(out) == count:
            return out
    for c in ranked:  # not enough distinct colours: fill with the most common remaining
        if len(out) == count:
            break
        if c not in out:
            out.append(c)
    while len(out) < count:
        out.append(out[-1] if out else (128, 128, 128))
    return out


def shade(diffuse: Image.Image, spec: Image.Image | None, colours: list[Colour], shader: str) -> Image.Image:
    """Diffuse as PdxMeshColor tints it (before lighting): for each of spec R, G, B
    ``col = lerp(col, col * s * C, s)``, i.e. ``col *= 1 - s + s*s*C``. Other shaders: unchanged."""
    out = diffuse.convert("RGB")
    if shader != "PdxMeshColor" or spec is None:
        return out
    bands = spec.convert("RGBA").resize(out.size).split()[:3]
    for s_band, colour in zip(bands, colours):
        factors = [s_band.point(lambda v, c=c: round(255 * (1 - v / 255 + (v / 255) ** 2 * c / 255)))
                   for c in colour]
        out = ImageChops.multiply(out, Image.merge("RGB", factors))
    return out
