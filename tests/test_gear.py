"""core.gear: horsehair blades, strand textures, kite shields, normal maps (SPEC M23)."""
from PIL import Image

from eu4unitgen.core import gear


def test_normal_from_height_signs():
    """Green up the image: a height rising to the right tilts the normal left (R < 128); rising downwards tilts it
    up the image (G > 128)."""
    right = Image.linear_gradient("L").rotate(90, expand=True)  # the top (0) to the left
    assert right.getpixel((10, 128)) < right.getpixel((200, 128))
    n = gear.normal_from_height(right, 2.0)
    assert n.getpixel((128, 128))[0] < 128 and abs(n.getpixel((128, 128))[1] - 128) <= 1
    down = Image.linear_gradient("L")  # 0 at the top, 255 at the bottom
    n = gear.normal_from_height(down, 2.0)
    assert n.getpixel((128, 128))[1] > 128 and abs(n.getpixel((128, 128))[0] - 128) <= 1


def _normals_facing(geo, centre):
    """The share of triangles whose normal points away from ``centre`` (a closed convex-ish solid: all)."""
    p, t = geo["p"], geo["tri"]
    good = 0
    for i in range(0, len(t), 3):
        a, b, c = (p[3 * t[i + k]:3 * t[i + k] + 3] for k in range(3))
        nrm = gear._cross(gear._add(b, a, -1), gear._add(c, a, -1))
        mid = [(a[k] + b[k] + c[k]) / 3 - centre[k] for k in range(3)]
        good += gear._dot(nrm, mid) > 0
    return good / (len(t) // 3)


def test_horsehair_blade_flat_and_jagged():
    line = [[0.0, 0.0, z * 0.5] for z in range(8)]
    geo = gear.horsehair_blade(line, [1.0] * 8, [0.1] * 8)
    p = geo["p"]
    xs, ys = p[0::3], p[1::3]
    assert max(xs) - min(xs) <= 0.2 + 1e-6  # thin across
    assert max(ys) - min(ys) > 1.0  # tall: the spikes over the height
    tops = sorted(set(round(y, 3) for y in ys if y > 0.5))
    assert len(tops) > 5  # a jagged top edge, not one height
    assert len(geo["uv"]) // 2 == len(p) // 3
    assert max(geo["tri"]) < len(p) // 3


def test_strand_texture_streaks_along_u():
    tex = gear.strand_texture((86, 14, 22), (150, 38, 52), size=128)
    d = tex["diffuse"].convert("L")
    row = [d.getpixel((x, 20)) for x in range(64, 128)]
    col = [d.getpixel((100, y)) for y in range(0, 64)]
    var = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)  # noqa: E731
    assert var(col) > var(row)  # streaks run along u: steadier along a row than down a column
    assert tex["normal"].size == (128, 128)


def test_kite_shield_shape_and_mask():
    geo, tex = gear.kite_shield(3.0, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], primary=(255, 77, 70), cfg={"tex": 128})
    p = geo["p"]
    ys, zs = p[1::3], p[2::3]
    assert abs((max(ys) - min(ys)) - 3.0) < 0.05
    assert 0.4 < (max(zs) - min(zs)) / 3.0 < 0.6  # about half as wide as tall
    # pointed bottom: the width near the bottom is small
    low = [z for y, z in zip(ys, zs) if y < -1.3]
    assert max(low) - min(low) < 0.5
    assert _normals_facing(geo, [-0.0, 0.0, 0.0]) > 0.6
    spec = tex["specular"]
    assert spec.getpixel((52, 80))[0] == 255  # the field on the mask (below the cross's bar, off the upright)
    r, g, b, _ = tex["diffuse"].getpixel((52, 80))
    assert abs(r - g) < 3 and abs(g - b) < 3  # the masked field is grey
