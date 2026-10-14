"""Texture regions: soft masks ("L" images, 0-255) at the base texture's size (SPEC §10 Phase 2).

  country1/2/3  the spec map's R/G/B: where PdxMeshColor multiplies in the country's
                primary/secondary/tertiary colour (SPEC §4). Empty on other shaders.
  metal         spec alpha (gloss) >= METAL_GLOSS and low saturation, outside country regions.
  skin          auto-detected skin tones, outside country and metal. A heuristic (it can catch
                light leather); a protect PNG replaces it. On horses it is the hide: the same
                hues down to dark browns (HIDE_MIN_VALUE), so the coat is all one region.
  cloth         everything else.

Regions come from the *base* textures, so a recipe's operations all see the same regions.
Everything here uses whole-image Pillow operations (no per-pixel Python).
"""
from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter

REGIONS = ("country1", "country2", "country3", "metal", "skin", "cloth")
COUNTRY_REGIONS = REGIONS[:3]
COUNTRY_THRESHOLD = 16  # a spec channel above this counts as country-coloured
METAL_GLOSS = 14  # spec alpha; western helmets/armour sit above, cloth and skin below (1.37.5)
METAL_MAX_SAT = 90
SKIN_HUE = (250, 28)  # Pillow HSV hue 0-255, wrapping through red
SKIN_SAT = (35, 190)
SKIN_MIN_VALUE = 80
HIDE_MIN_VALUE = 20  # horses: dark coat shading is still hide
GUIDE_COLOURS = {"country1": (230, 40, 40), "country2": (40, 200, 40), "country3": (50, 80, 230),
                 "metal": (0, 220, 220), "skin": (240, 0, 240), "cloth": (90, 90, 90)}


def _threshold(band: Image.Image, lo: int, hi: int = 255) -> Image.Image:
    return band.point(lambda v: 255 if lo <= v <= hi else 0)


def AND(*masks: Image.Image) -> Image.Image:  # noqa: N802 - reads like the logic it is
    out = masks[0]
    for m in masks[1:]:
        out = ImageChops.multiply(out, m)
    return out


def OR(*masks: Image.Image) -> Image.Image:  # noqa: N802
    out = masks[0]
    for m in masks[1:]:
        out = ImageChops.lighter(out, m)
    return out


def NOT(mask: Image.Image) -> Image.Image:  # noqa: N802
    return ImageChops.invert(mask)


def blank(size: tuple[int, int], value: int = 0) -> Image.Image:
    return Image.new("L", size, value)


def auto_regions(diffuse: Image.Image, specular: Image.Image | None, colour_shader: bool,
                 hide: bool = False) -> dict[str, Image.Image]:
    """``hide``: the base is a horse; ``skin`` is its whole coat."""
    size = diffuse.size
    if specular is not None and specular.size != size:
        specular = specular.resize(size, Image.Resampling.BILINEAR)
    if specular is not None:
        r, g, b, a = specular.convert("RGBA").split()
    else:
        r = g = b = a = blank(size)
    if colour_shader:
        country = {"country1": r, "country2": g, "country3": b}
        any_country = OR(*(_threshold(c, COUNTRY_THRESHOLD + 1) for c in (r, g, b)))
    else:
        country = {k: blank(size) for k in COUNTRY_REGIONS}
        any_country = blank(size)
    h, s, v = diffuse.convert("RGB").convert("HSV").split()
    metal = AND(_threshold(a, METAL_GLOSS), _threshold(s, 0, METAL_MAX_SAT - 1), NOT(any_country))
    hue = OR(_threshold(h, SKIN_HUE[0]), _threshold(h, 0, SKIN_HUE[1]))
    skin = AND(hue, _threshold(s, *SKIN_SAT), _threshold(v, HIDE_MIN_VALUE if hide else SKIN_MIN_VALUE), NOT(any_country), NOT(metal))
    skin = skin.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    skin = AND(skin, NOT(any_country), NOT(metal))
    cloth = NOT(OR(any_country, metal, skin))
    return {**country, "metal": metal, "skin": skin, "cloth": cloth}


def protect_mask(regions: dict[str, Image.Image], skin: bool = True, metal: bool = False,
                 override: Image.Image | None = None) -> Image.Image:
    """What diffuse operations must leave alone. A base-provided PNG replaces the auto mask."""
    size = regions["cloth"].size
    if override is not None:
        return override.convert("L").resize(size, Image.Resampling.BILINEAR)
    parts = [regions[k] for k, on in (("skin", skin), ("metal", metal)) if on]
    return OR(*parts) if parts else blank(size)


def area(mask: Image.Image) -> float:
    """Mean coverage 0..1."""
    hist = mask.histogram()
    return sum(i * n for i, n in enumerate(hist)) / (255 * max(1, sum(hist)))


def guide(diffuse: Image.Image, regions: dict[str, Image.Image]) -> Image.Image:
    """The base diffuse with each region tinted (GUIDE_COLOURS), to paint masks over."""
    base = diffuse.convert("RGB")
    out = base.copy()
    for name in ("cloth", "metal", "skin", *COUNTRY_REGIONS):
        layer = Image.new("RGB", base.size, GUIDE_COLOURS[name])
        out = Image.composite(Image.blend(out, layer, 0.55), out, regions[name])
    return out
