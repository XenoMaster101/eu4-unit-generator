"""DDS read/write for unit textures (SPEC §4).

Reading goes through Pillow. Writing matches vanilla unit textures: BC1/DXT1 (diffuse, normal)
or BC3/DXT5 (spec), not sRGB, no DX10 header, and a **full mip chain**. Pillow's DDS writer
only writes the top level, so each level is encoded with Pillow and the file header is ours.

Mip levels are downsampled one channel at a time: Pillow resizes RGBA with premultiplied
alpha, which would wipe the colour of spec-map areas whose alpha (gloss) is 0, and those
areas still hold country-colour masks (SPEC §4).
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from PIL import Image

FORMATS = ("DXT1", "DXT5")
_BLOCK_BYTES = {"DXT1": 8, "DXT5": 16}

# DDSD_CAPS | HEIGHT | WIDTH | PIXELFORMAT | MIPMAPCOUNT | LINEARSIZE: what vanilla writes (0xa1007)
_FLAGS = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
_DDPF_FOURCC = 0x4
_CAPS_TEXTURE, _CAPS_COMPLEX, _CAPS_MIPMAP = 0x1000, 0x8, 0x400000


class DDSError(ValueError):
    pass


@dataclass(frozen=True)
class DDSFormat:
    """What a generated texture must match: the base texture's header facts."""
    format: str
    width: int
    height: int
    mips: int

    @classmethod
    def of(cls, header: dict) -> "DDSFormat":
        """From a scanner header dict (``read_dds_header`` / base_models.json ``textures``)."""
        if header.get("dx10_header") or header.get("format") not in FORMATS:
            raise DDSError(f"unsupported base texture format {header}")
        return cls(header["format"], header["width"], header["height"], header["mips"])


def full_mip_count(width: int, height: int) -> int:
    return max(width, height).bit_length()


def read_header(data: bytes) -> dict:
    from eu4unitgen.library.scanner import read_dds_header

    return read_dds_header(data)


def read_dds(data: bytes) -> Image.Image:
    """Top mip level as RGBA."""
    if data[:4] != b"DDS ":
        raise DDSError("not a DDS file")
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGBA")


def _level_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "DDS", pixel_format=fmt)
    data = buf.getvalue()
    w, h = img.size
    expected = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _BLOCK_BYTES[fmt]
    body = data[128:]
    if data[84:88] == b"DX10" or len(body) != expected:
        raise DDSError(f"Pillow wrote an unexpected {fmt} level ({len(body)} bytes, expected {expected})")
    return body


def _downsample(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return Image.merge(img.mode, [band.resize(size, Image.Resampling.BOX) for band in img.split()])


def mip_chain(img: Image.Image, mips: int) -> list[Image.Image]:
    levels = [img]
    w, h = img.size
    for _ in range(mips - 1):
        w, h = max(1, w // 2), max(1, h // 2)
        levels.append(_downsample(levels[0], (w, h)))
    return levels


def _header(fmt: str, width: int, height: int, mips: int) -> bytes:
    linear = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * _BLOCK_BYTES[fmt]
    caps = _CAPS_TEXTURE | ((_CAPS_COMPLEX | _CAPS_MIPMAP) if mips > 1 else 0)
    pixel_format = struct.pack("<II4s5I", 32, _DDPF_FOURCC, fmt.encode("ascii"), 0, 0, 0, 0, 0)
    head = struct.pack("<7I", 124, _FLAGS, height, width, linear, 0, mips) + b"\0" * 44 + pixel_format
    head += struct.pack("<5I", caps, 0, 0, 0, 0)
    assert len(head) == 124
    return b"DDS " + head


def write_dds(img: Image.Image, target: DDSFormat) -> bytes:
    """Encode ``img`` as ``target`` (format, size, mip count). The image is resized to the
    target size if needed. DXT1 is encoded from RGB (vanilla diffuse/normal alpha is unused, 255)."""
    if target.format not in FORMATS:
        raise DDSError(f"can't write {target.format}; supported: {', '.join(FORMATS)}")
    if not 1 <= target.mips <= full_mip_count(target.width, target.height):
        raise DDSError(f"{target.mips} mips is impossible for {target.width}x{target.height}")
    mode = "RGB" if target.format == "DXT1" else "RGBA"
    img = img.convert(mode)
    if img.size != (target.width, target.height):
        img = _downsample(img, (target.width, target.height)) if img.width >= target.width else \
            Image.merge(mode, [b.resize((target.width, target.height), Image.Resampling.BICUBIC) for b in img.split()])
    body = b"".join(_level_bytes(level, target.format) for level in mip_chain(img, target.mips))
    return _header(target.format, target.width, target.height, target.mips) + body


def check_matches(data: bytes, target: DDSFormat) -> list[str]:
    """Problems that would make ``data`` differ from the base's format (empty when it matches)."""
    try:
        h = read_header(data)
    except ValueError as exc:
        return [str(exc)]
    problems = []
    if h.get("dx10_header"):
        problems.append("has a DX10 header")
    if h["format"] != target.format:
        problems.append(f"format {h['format']}, base is {target.format}")
    if (h["width"], h["height"]) != (target.width, target.height):
        problems.append(f"size {h['width']}x{h['height']}, base is {target.width}x{target.height}")
    if h["mips"] != target.mips:
        problems.append(f"{h['mips']} mip levels, base has {target.mips}")
    else:
        expected = 128 + sum(max(1, (max(1, target.width >> i) + 3) // 4) * max(1, (max(1, target.height >> i) + 3) // 4)
                             * _BLOCK_BYTES[target.format] for i in range(target.mips))
        if len(data) < expected:
            problems.append(f"truncated: {len(data)} bytes, expected {expected}")
    return problems
