"""Texture edits on decoded images (SPEC §4, §10 Phase 2).

Phase 2 step 2 only: a hue shift, used by the meshsettings in-game test. The spec-mask
painter, gloss presets, pattern overlays and reference palette come in step 4.
"""
from __future__ import annotations

from PIL import Image


def hue_shift(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate every pixel's hue by ``degrees``; saturation, value and alpha stay."""
    rgba = img.convert("RGBA")
    h, s, v = rgba.convert("RGB").convert("HSV").split()
    step = round(degrees / 360 * 256) % 256
    h = h.point(lambda x: (x + step) % 256)
    out = Image.merge("HSV", (h, s, v)).convert("RGB").convert("RGBA")
    out.putalpha(rgba.getchannel("A"))
    return out
