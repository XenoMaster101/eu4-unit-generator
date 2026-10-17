"""Textures made from scratch (no base-game pixels) for procedural parts (Phase 3).

Each part has colour zones (dory: wood shaft, bronze head). Three layouts:

* ``atlas`` (Q1): the part is unwrapped for real in Blender (islands packed, a power-of-two size
  chosen for ~32 texels per mesh unit, like the base-game bodies) and each triangle's zone is
  painted where its UVs fall (``atlas_images``).

* ``strip``: one flat swatch per zone side by side; every face of a zone has its UVs at that
  swatch's centre.
* ``disc`` (hoplon): zone 0 (the face) fills the circle inscribed in the square and is mapped
  by planar projection, so a blazon image can be painted on it; the other zones sit in the
  corners, outside the circle.

UV points are given in image coordinates (u right, v down from the top row, as PIL and DDS
store rows). io_pdx_mesh flips V on import and export, so the Blender job converts them with
``v_blender = 1 - v``; what Blender shows on the image is what the game shows.

A zone may take a country colour (``country1-3``): spec RGB is the PdxMeshColor mask
(SPEC §4), and the diffuse there is a neutral grey the shader multiplies by the colour. A
part with any country zone uses PdxMeshColor, else PdxMeshStandard.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw

from eu4unitgen.textures.dds import DDSFormat, full_mip_count, write_dds

Colour = tuple[int, int, int]
SIZE = {"strip": 64, "disc": 256, "atlas": 64}  # atlas: the job's chosen size replaces this
TEXEL_DENSITY = 32.0  # texels per mesh unit for atlas parts (western type1 body: 32.8; base-game weapons 17-27)
FLAT_NORMAL = (128, 128, 255)
COUNTRY_CHANNELS = ("country1", "country2", "country3")  # spec R, G, B
COUNTRY_GREY = (170, 170, 170)  # diffuse under a country zone: the shader multiplies it by the colour
DISC_CORNER = 0.06  # corner swatch centres (image units); (0.06, 0.06) lies 0.62 from the centre, outside the disc
DISC_CORNERS = ((DISC_CORNER, DISC_CORNER), (1 - DISC_CORNER, DISC_CORNER), (DISC_CORNER, 1 - DISC_CORNER),
                (1 - DISC_CORNER, 1 - DISC_CORNER))


@dataclass(frozen=True)
class Zone:
    colour: Colour  # diffuse (ignored under a country channel, which gets COUNTRY_GREY)
    gloss: float  # percent of 255 in spec alpha
    channel: str | None = None  # country1-3: the zone takes that country colour
    material: str = "plain"  # textures.paint pattern class (Q2)

    def __post_init__(self) -> None:
        if self.channel is not None and self.channel not in COUNTRY_CHANNELS:
            raise ValueError(f"zone channel {self.channel!r} must be one of {COUNTRY_CHANNELS}")


def shader_for(zones: list[Zone]) -> str:
    return "PdxMeshColor" if any(z.channel for z in zones) else "PdxMeshStandard"


def zone_uvs(count: int, layout: str) -> list[tuple[float, float]]:
    """Each zone's UV point in image coordinates. Disc: zone 0 is planar-mapped (its point is
    the centre, unused by faces that carry their own UVs); zones 1-4 are corners."""
    if layout in ("strip", "atlas"):  # atlas: placeholders, replaced by the job's unwrap
        return [((i + 0.5) / count, 0.5) for i in range(count)]
    if layout == "disc":
        if count > 1 + len(DISC_CORNERS):
            raise ValueError(f"the disc layout holds at most {1 + len(DISC_CORNERS)} zones")
        return [(0.5, 0.5), *DISC_CORNERS[:count - 1]]
    raise ValueError(f"unknown texture layout {layout!r}")


def _diffuse_of(z: Zone) -> Colour:
    return COUNTRY_GREY if z.channel else z.colour


def _spec_of(z: Zone) -> tuple[int, int, int, int]:
    rgb = [0, 0, 0]
    if z.channel:
        rgb[COUNTRY_CHANNELS.index(z.channel)] = 255
    return (*rgb, round(z.gloss * 255 / 100))


def _paint(zones: list[Zone], layout: str, fill) -> Image.Image:
    size = SIZE[layout]
    img = Image.new("RGBA", (size, size), fill(zones[0]))
    draw = ImageDraw.Draw(img)
    if layout == "strip":
        n = len(zones)
        for i, z in enumerate(zones):
            draw.rectangle([i * size // n, 0, (i + 1) * size // n - 1, size - 1], fill=fill(z))
        return img
    # disc: corners first (whole quadrants outside the circle), then the face circle on top
    half = size // 2
    quads = ((0, 0), (half, 0), (0, half), (half, half))
    for z, (x, y) in zip(zones[1:], quads):
        draw.rectangle([x, y, x + half - 1, y + half - 1], fill=fill(z))
    draw.ellipse([0, 0, size - 1, size - 1], fill=fill(zones[0]))
    return img


def blazon_layer(image: Image.Image, size: int, rotate: int = 0, mirror: bool = False) -> Image.Image:
    """A blazon fitted into the disc: scaled to the circle's square, rotated (degrees, counter-
    clockwise) and mirrored as asked, cut to the circle. RGBA; alpha = where it paints."""
    img = image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    if mirror:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rotate % 360:
        img = img.rotate(rotate % 360, resample=Image.Resampling.BICUBIC)
    circle = Image.new("L", (size, size), 0)
    ImageDraw.Draw(circle).ellipse([0, 0, size - 1, size - 1], fill=255)
    img.putalpha(ImageChops.multiply(img.getchannel("A"), circle))
    return img


def part_images(zones: list[Zone], layout: str = "strip", blazon: Image.Image | None = None,
                blazon_rotate: int = 0, blazon_mirror: bool = False) -> dict[str, Image.Image]:
    """{diffuse, normal, specular} as RGBA images. A blazon (disc layout only) is painted over
    the face in fixed colours: under it the country channel is cleared."""
    if not zones:
        raise ValueError("a part needs at least one colour zone")
    zone_uvs(len(zones), layout)  # checks the layout holds them
    diffuse = _paint(zones, layout, lambda z: (*_diffuse_of(z), 255))
    spec = _paint(zones, layout, _spec_of)
    if blazon is not None:
        if layout != "disc":
            raise ValueError("only disc-layout parts (the hoplon) take a blazon")
        layer = blazon_layer(blazon, SIZE[layout], blazon_rotate, blazon_mirror)
        diffuse.alpha_composite(layer)
        diffuse.putalpha(255)
        cleared = Image.new("RGBA", spec.size, (0, 0, 0, _spec_of(zones[0])[3]))
        spec = Image.composite(cleared, spec, layer.getchannel("A").point(lambda a: 255 if a >= 128 else 0))
    size = SIZE[layout]
    return {"diffuse": diffuse, "normal": Image.new("RGBA", (size, size), (*FLAT_NORMAL, 255)), "specular": spec}


def _wh(size) -> tuple[int, int]:
    """An atlas size: an int (square, old caches) or [w, h]."""
    return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))


def atlas_images(zones: list[Zone], size, tris: list[list[float]]) -> dict[str, Image.Image]:
    """{diffuse, normal, specular} for an unwrapped part: each triangle ([zone, u0, v0, u1, v1, u2,
    v2], image coordinates) painted in its zone's diffuse and spec, edges bled 2 px so mip levels
    don't pull in the background; the background is zone 0's."""
    w, h = _wh(size)
    diffuse = Image.new("RGBA", (w, h), (*_diffuse_of(zones[0]), 255))
    spec = Image.new("RGBA", (w, h), _spec_of(zones[0]))
    dd, sd = ImageDraw.Draw(diffuse), ImageDraw.Draw(spec)
    for t in tris:
        z = zones[int(t[0]) if 0 <= int(t[0]) < len(zones) else 0]
        pts = [(t[1 + 2 * k] * w, t[2 + 2 * k] * h) for k in range(3)]
        for draw, fill in ((dd, (*_diffuse_of(z), 255)), (sd, _spec_of(z))):
            draw.polygon(pts, fill=fill, outline=fill, width=2)
    return {"diffuse": diffuse, "normal": Image.new("RGBA", (w, h), (*FLAT_NORMAL, 255)), "specular": spec}


def zone_map(zones: list[Zone], layout: str, size, tris: list[list[float]] | None = None) -> Image.Image:
    """An 'L' image holding each texel's zone index, laid out like the part's textures (``size``: an
    int or [w, h])."""
    w, h = _wh(size)
    if tris is not None:
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        for t in tris:
            z = int(t[0]) if 0 <= int(t[0]) < len(zones) else 0
            draw.polygon([(t[1 + 2 * k] * w, t[2 + 2 * k] * h) for k in range(3)], fill=z, outline=z, width=2)
        return img
    size = w
    ids = [Zone((i, i, i), 0) for i in range(len(zones))]
    return _paint(ids, layout, lambda z: (z.colour[0], z.colour[0], z.colour[0], 255)).convert("L").resize(
        (size, size), Image.Resampling.NEAREST)


def encode_part_textures(images: dict[str, Image.Image]) -> dict[str, bytes]:
    """DXT1 diffuse and normal, DXT5 spec, full mip chains (vanilla's formats, SPEC §4)."""
    out = {}
    for kind, fmt in (("diffuse", "DXT1"), ("normal", "DXT1"), ("specular", "DXT5")):
        w, h = images[kind].size
        out[kind] = write_dds(images[kind], DDSFormat(fmt, w, h, full_mip_count(w, h)))
    return out


def part_textures(zones: list[Zone], layout: str = "strip", blazon: Image.Image | None = None,
                  blazon_rotate: int = 0, blazon_mirror: bool = False) -> dict[str, bytes]:
    """part_images, encoded as DDS."""
    return encode_part_textures(part_images(zones, layout, blazon, blazon_rotate, blazon_mirror))
