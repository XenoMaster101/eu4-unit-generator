"""core.gear: horsehair blades, strand textures, kite shields, normal maps (SPEC M23)."""
import math

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


def _sphere(r, n=16):
    """A closed UV sphere's triangles (corner triples) of radius ``r`` at the origin."""
    import math
    pts = [[[r * math.sin(math.pi * i / n) * math.cos(2 * math.pi * j / n), r * math.cos(math.pi * i / n),
             r * math.sin(math.pi * i / n) * math.sin(2 * math.pi * j / n)] for j in range(n)] for i in range(n + 1)]
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = pts[i][j], pts[i][(j + 1) % n], pts[i + 1][(j + 1) % n], pts[i + 1][j]
            tris += [(a, b, c), (a, c, d)]
    return tris


def test_raycast_far_hits_the_outer_shell():
    tris = _sphere(1.0) + _sphere(0.5)  # an inner shell too: the outer one wins
    t, n = gear.raycast_far(tris, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0)
    assert 0.95 < t <= 1.0 and n[0] > 0.9
    assert gear.raycast_far(tris, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.3) is None


def test_surface_band_and_lathe():
    m = gear.Mesh()
    ring = [[math.cos(a / 10 * math.tau), 0.0, math.sin(a / 10 * math.tau)] for a in range(10)]
    gear.surface_band(m, ring, ring, [0.05] * 10, 0.04, 0.02, closed=True)
    assert len(m.tri) // 3 == 10 * 6
    r = [math.hypot(m.p[i], m.p[i + 2]) for i in range(0, len(m.p), 3)]
    assert min(r) > 0.97 and max(r) < 1.05  # on the surface: its foot under it, its face over it
    k = len(m.tri)
    gear.lathe(m, [(0.3, 0.0), (0.2, 0.5), (0.3, 1.0)], [0.0, 2.0, 0.0], 8, (0.5, 0.5))
    assert (len(m.tri) - k) // 3 == 2 * 8 * 2 + 2 * 6  # two bands of quads, two octagon caps
    assert _normals_facing({"p": m.p, "tri": m.tri[k:]}, [0.0, 2.5, 0.0]) > 0.9


def test_kite_shield_worn_and_darker():
    _, tex = gear.kite_shield(3.0, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], primary=None, cfg={"tex": 128})
    d = tex["diffuse"].convert("RGB")
    field = [d.getpixel((x, y)) for x in range(40, 60) for y in range(70, 90)]
    reds = [p[0] for p in field]
    assert max(reds) - min(reds) > 6  # worn: not one flat colour
    assert sum(reds) / len(reds) < gear.KITE_COLOURS["field"][0]  # a little darker than the caparison's red
