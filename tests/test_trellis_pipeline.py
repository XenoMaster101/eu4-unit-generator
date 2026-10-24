"""The standard horse pipeline's pure helpers (ref3d/trellis.py, textures/decals.py; SPEC M22)."""
import io
import json
import struct
from pathlib import PureWindowsPath

import pytest
from PIL import Image

from eu4unitgen.ref3d import trellis
from eu4unitgen.textures import decals


def test_wsl_path():
    assert trellis.wsl_path(PureWindowsPath(r"C:\Users\x\a b\f.png")) == "/mnt/c/Users/x/a b/f.png"
    assert trellis.wsl_path(PureWindowsPath(r"D:\eu4unitgen_ai\TRELLIS.2")) == "/mnt/d/eu4unitgen_ai/TRELLIS.2"


def _glb(images):
    """A minimal .glb: one material whose base colour is the second image (the first a metal-roughness map)."""
    blobs = []
    for im in images:
        b = io.BytesIO()
        im.save(b, "PNG")
        blobs.append(b.getvalue())
    views, off = [], 0
    for b in blobs:
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b)})
        off += len(b)
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": off}], "bufferViews": views,
           "images": [{"bufferView": i, "mimeType": "image/png"} for i in range(len(blobs))],
           "textures": [{"source": 0}, {"source": 1}],
           "materials": [{"pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}, "baseColorTexture": {"index": 1}}}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    bin_ = b"".join(blobs)
    bin_ += b"\0" * (-len(bin_) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bin_), 0x004E4942) + bin_
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def test_glb_basecolor(tmp_path):
    glb = tmp_path / "m.glb"
    glb.write_bytes(_glb([Image.new("RGB", (4, 4), (0, 200, 0)), Image.new("RGB", (4, 4), (120, 60, 30))]))
    out = trellis.glb_basecolor(glb, tmp_path / "base.png")
    assert Image.open(out).convert("RGB").getpixel((1, 1)) == (120, 60, 30)


def test_tetragrammatic_decal(tmp_path):
    im = Image.open(decals.tetragrammatic(tmp_path / "d.png", 256)).convert("L")
    assert im.size == (256, 256)
    assert im.getpixel((128, 128)) == 255 and im.getpixel((128, 6)) == 255  # the cross, to its ends
    # the betas: the left two mirrors of the right two
    left, right = im.crop((0, 0, 110, 110)), im.crop((146, 0, 256, 110))
    assert left.getbbox() is not None and right.getbbox() is not None
    lw = left.crop(left.getbbox()).transpose(Image.FLIP_LEFT_RIGHT)
    rw = right.crop(right.getbbox())
    assert lw.size == rw.size
    diff = sum(abs(a - b) for a, b in zip(lw.getdata(), rw.getdata())) / (255 * lw.size[0] * lw.size[1])
    assert diff < 0.05


@pytest.mark.parametrize("bad", ["nope"])
def test_decal_names(bad):
    assert bad not in decals.DECALS and "tetragrammatic" in decals.DECALS
