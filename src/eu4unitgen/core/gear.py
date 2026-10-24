"""Own-geometry gear for mesh parts (SPEC M23): horsehair (plumes, crests, manes) and kite shields.

Pure Python and Pillow (the app has no numpy). Geometry is ``{"p", "tri", "uv"}`` (flat lists; uv in image
convention, v down) in the part's own frame; textures are RGBA images in the project's conventions: normal maps
tangent space with green up the image (flat 128 128 255), specular alpha the gloss and R the country colour's
mask (PdxMeshColor, SPEC §4), the masked diffuse a neutral grey whose product with the primary colour gives the
colour back.

Rules (the user, test 32):
  * horsehair is a flat blade (tall and thin, a brush along its line), never a round tube; its top edge is jagged;
    its texture is strands along its length, a dark base with lighter streaks, in the reference's colours;
  * a kite shield is a tall almond with a pointed bottom, slightly curved, a raised rim, a crisp painted emblem, a
    normal map for the rim and studs, the field on the country colour's mask.
"""
from __future__ import annotations

import math
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

V = list[float]


def _add(u: V, v: V, k: float = 1.0) -> V:
    return [u[i] + k * v[i] for i in range(3)]


def _mix(u: V, v: V, t: float) -> V:
    return [u[i] * (1 - t) + v[i] * t for i in range(3)]


def _dot(u: V, v: V) -> float:
    return sum(u[i] * v[i] for i in range(3))


def _norm(v: V) -> V:
    n = math.sqrt(_dot(v, v)) or 1.0
    return [x / n for x in v]


def _cross(u: V, v: V) -> V:
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _lum(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


class Mesh:
    """Triangles with per-corner uv; faces wound to face an expected direction."""

    def __init__(self):
        self.p, self.tri, self.uv = [], [], []

    def vert(self, pos: V, uv) -> int:
        self.p.extend(float(x) for x in pos)
        self.uv.extend(float(x) for x in uv)
        return len(self.p) // 3 - 1

    def face(self, idx, expect: V | None = None):
        """A triangle or quad (vertex ids), reversed if its normal points away from ``expect``."""
        q = list(idx)
        if expect is not None:
            a, b, c = (self.p[3 * i:3 * i + 3] for i in q[:3])
            if _dot(_cross(_add(b, a, -1), _add(c, a, -1)), expect) < 0:
                q.reverse()
        for k in range(1, len(q) - 1):
            self.tri.extend([q[0], q[k], q[k + 1]])

    def geometry(self) -> dict:
        return {"p": [round(x, 5) for x in self.p], "tri": self.tri, "uv": [round(x, 5) for x in self.uv]}


def chaikin(pts: list[V], values: list[float], passes: int) -> tuple[list[V], list[float]]:
    """Corner cutting on a polyline and a value along it; the ends kept."""
    for _ in range(passes):
        q, w = [pts[0]], [values[0]]
        for a, b, va, vb in zip(pts[:-1], pts[1:], values[:-1], values[1:]):
            q += [_mix(a, b, 0.25), _mix(a, b, 0.75)]
            w += [0.75 * va + 0.25 * vb, 0.25 * va + 0.75 * vb]
        pts, values = q + [pts[-1]], w + [values[-1]]
    return pts, values


# horsehair ----------------------------------------------------------------------------------------------------

HAIR = {"below": 0.3, "jag": (0.22, 0.45), "top_thin": 0.25, "seed": 7}


def horsehair_blade(line: list[V], heights: list[float], half_thick: list[float], side_hint: V = (1.0, 0.0, 0.0),
                    cfg: dict | None = None) -> dict:
    """A flat blade along ``line``: at each point it spans ``heights[i]`` across the line (``below`` of it under the
    line, the rest above, ``up`` = tangent x side), ``half_thick[i]`` either side (thinning to ``top_thin`` of that at
    the top). Between points a spike column: its top raised by a random ``jag`` share of the height and thinned to
    nothing, a jagged top edge. uv: u along the line (arc length), v across (top 0.02, bottom 0.98)."""
    c = {**HAIR, **(cfg or {})}
    rnd = random.Random(c["seed"])
    n = len(line)
    frames = []
    for i in range(n):
        t = _norm(_add(line[min(i + 1, n - 1)], line[max(i - 1, 0)], -1.0))
        side = _norm(_add(list(side_hint), t, -_dot(list(side_hint), t)))
        frames.append((t, side, _cross(t, side)))
    cols = []  # (centre, height, half thickness, top factor, top thinning, frame)
    for i in range(n):
        cols.append((line[i], heights[i], half_thick[i], 1.0, c["top_thin"], frames[i]))
        if i + 1 < n:
            f = tuple(_norm(_mix(frames[i][k], frames[i + 1][k], 0.5)) for k in range(3))
            cols.append((_mix(line[i], line[i + 1], 0.5), 0.5 * (heights[i] + heights[i + 1]),
                         0.5 * (half_thick[i] + half_thick[i + 1]), 1.0 + rnd.uniform(*c["jag"]), 0.0, f))
    length = [0.0]
    for a, b in zip(cols[:-1], cols[1:]):
        length.append(length[-1] + math.sqrt(_dot(_add(b[0], a[0], -1), _add(b[0], a[0], -1))))
    total = length[-1] or 1.0
    m = Mesh()
    rings = []
    for (ctr, h, th, top_k, thin, (t, side, up)), s in zip(cols, length):
        u = 0.02 + 0.96 * s / total
        bot = _add(ctr, up, -c["below"] * h)
        top = _add(ctr, up, (1 - c["below"]) * h * top_k)
        tt = th * thin
        rings.append((t, side, up, [m.vert(_add(bot, side, -th), (u, 0.98)), m.vert(_add(bot, side, th), (u, 0.98)),
                                    m.vert(_add(top, side, tt), (u, 0.02)), m.vert(_add(top, side, -tt), (u, 0.02))]))
    for (t, side, up, a), (_, _, _, b) in zip(rings[:-1], rings[1:]):
        m.face([a[0], a[3], b[3], b[0]], [-x for x in side])
        m.face([a[1], b[1], b[2], a[2]], side)
        m.face([a[3], b[3], b[2], a[2]], up)
        m.face([a[0], a[1], b[1], b[0]], [-x for x in up])
    t0, _, _, first = rings[0]
    m.face(first, [-x for x in t0])
    t1, _, _, last = rings[-1]
    m.face(last, t1)
    return m.geometry()


def strand_texture(dark, light, size: int = 256, seed: int = 11) -> dict[str, Image.Image]:
    """Horsehair strands along u (the blade's length): the dark base, lighter streaks of random length and
    brightness (strongest along the top edge, v 0), a darker root (v 1); matte. -> diffuse, specular, normal."""
    rnd = random.Random(seed)
    big = size * 2
    im = Image.new("RGB", (big, big), tuple(int(x) for x in dark))
    d = ImageDraw.Draw(im)
    for _ in range(int(big * 0.9)):
        y = rnd.random() ** 0.8 * big
        k = rnd.random() ** 1.6 * (1.0 - 0.55 * y / big)
        col = tuple(int(dark[i] + (light[i] - dark[i]) * k) for i in range(3))
        x0 = rnd.random() * 0.45 * big
        d.line([(x0, y), (big, y + rnd.uniform(-3, 3))], fill=col, width=rnd.choice((1, 1, 2, 3)))
    for _ in range(int(big * 0.25)):  # darker gaps between strands
        y = rnd.random() * big
        col = tuple(int(x * 0.7) for x in dark)
        d.line([(rnd.random() * 0.5 * big, y), (big, y)], fill=col, width=1)
    root = Image.linear_gradient("L").resize((big, big))  # 0 at the top, 255 at the bottom (the root)
    shade = Image.new("RGB", (big, big), tuple(int(x * 0.6) for x in dark))
    im = Image.composite(shade, im, root.point(lambda v: int(v * 0.55)))
    im = im.filter(ImageFilter.GaussianBlur(0.6)).resize((size, size), Image.LANCZOS)
    height = im.convert("L").filter(ImageFilter.GaussianBlur(0.8))
    return {"diffuse": im.convert("RGBA"), "specular": Image.new("RGBA", (size, size), (0, 0, 0, 3)),
            "normal": normal_from_height(height, 1.5)}


def normal_from_height(height: Image.Image, strength: float = 2.0) -> Image.Image:
    """A tangent-space normal map (green up the image) from a height map (L, bright high)."""
    # PIL's Kernel flips the kernel vertically, then correlates: dH/dx and dH/dy(down) as given below (checked in
    # tests/test_gear.py on ramps)
    k = 8.0 / strength
    gx = height.filter(ImageFilter.Kernel((3, 3), [1, 0, -1, 2, 0, -2, 1, 0, -1], scale=k, offset=128))  # 128 - dH/dx
    gy = height.filter(ImageFilter.Kernel((3, 3), [1, 2, 1, 0, 0, 0, -1, -2, -1], scale=k, offset=128))  # 128 + dH/dy
    b = Image.new("L", height.size, 255)
    return Image.merge("RGBA", (gx, gy, b, Image.new("L", height.size, 255)))


# kite shield --------------------------------------------------------------------------------------------------

KITE = {"aspect": 0.52, "widest": 0.18, "bulge": 0.1, "bulge_v": 0.04, "thickness": 0.08, "rim": 0.075, "rim_h": 0.05,
        "sides": 24, "tex": 512}
# the reference's colours (new_refs/cataphract_rider.png: the shield's field and its gold, sampled)
KITE_COLOURS = {"field": (85, 25, 27), "gold": (172, 118, 60), "stud": (196, 146, 86), "back": (72, 48, 32)}
KITE_GLOSS = {"field": 5, "gold": 12, "back": 3}


def kite_outline(H: float, W: float, n: int) -> list[tuple[float, float]]:
    """The kite's outline (x across, y up, centred), anticlockwise from the bottom point: a rounded top (half an
    ellipse over the widest point) and sides narrowing to a point (sin)."""
    yw = H * KITE["widest"]

    def half(y):
        if y >= yw:
            r = (y - yw) / (H / 2 - yw)
            return W / 2 * math.sqrt(max(0.0, 1 - r * r))
        s = (y + H / 2) / (yw + H / 2)
        return W / 2 * math.sin(math.pi / 2 * s)
    ys = [-H / 2 + (yw + H / 2) * i / (n // 2) for i in range(n // 2)] + \
         [yw + (H / 2 - yw) * math.sin(math.pi / 2 * i / (n // 2)) for i in range(n // 2 + 1)]
    right = [(half(y), y) for y in ys]
    left = [(-x, y) for x, y in reversed(right[1:-1])]
    return right + left


def _inset(outline, r: float, cap: float = 3.0):
    """The outline moved inward by ``r`` (mitred, at most ``cap`` x r at the point)."""
    n = len(outline)
    out = []
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = outline[i - 1], outline[i], outline[(i + 1) % n]
        n1 = (-(by - ay), bx - ax)  # inward for an anticlockwise outline: the left normal
        n2 = (-(cy - by), cx - bx)
        l1, l2 = math.hypot(*n1) or 1.0, math.hypot(*n2) or 1.0
        mx, my = n1[0] / l1 + n2[0] / l2, n1[1] / l1 + n2[1] / l2
        lm = math.hypot(mx, my) or 1.0
        mx, my = mx / lm, my / lm
        cosang = max(mx * n1[0] / l1 + my * n1[1] / l1, 1.0 / cap)
        out.append((bx + mx * r / cosang, by + my * r / cosang))
    return out


def _cross_emblem(draw: ImageDraw.ImageDraw, to_px, H: float, W: float):
    """The reference's emblem: a gold cross with slightly flared ends (its bar at the widest point) and a small
    cross in each quarter."""
    yb = H * KITE["widest"]
    bar = W * 0.13

    def rect(x0, y0, x1, y1):
        (a, b), (c, d) = to_px(x0, y1), to_px(x1, y0)
        draw.rectangle([a, b, c, d], fill=255)
    rect(-bar / 2, -H * 0.4, bar / 2, H * 0.44)  # upright
    rect(-W * 0.44, yb - bar / 2, W * 0.44, yb + bar / 2)  # bar
    f = bar * 0.85
    for x0, y0, x1, y1 in ((-f, H * 0.4, f, H * 0.44), (-f, -H * 0.4, f, -H * 0.36),
                           (-W * 0.44, yb - f, -W * 0.40, yb + f), (W * 0.40, yb - f, W * 0.44, yb + f)):
        rect(x0, y0, x1, y1)  # flared ends
    s, b2 = W * 0.09, W * 0.035
    for cx, cy in ((-W * 0.22, yb + H * 0.13), (W * 0.22, yb + H * 0.13), (-W * 0.2, yb - H * 0.14), (W * 0.2, yb - H * 0.14)):
        rect(cx - b2 / 2, cy - s, cx + b2 / 2, cy + s)
        rect(cx - s, cy - b2 / 2, cx + s, cy + b2 / 2)


def kite_shield(H: float, centre: V, out: V, up: V, across: V, primary=None, cfg: dict | None = None) -> tuple[dict, dict]:
    """A kite shield ``H`` tall centred at ``centre``, facing ``out`` (convex that way), its top along ``up``, its
    width along ``across`` (all in the part's frame). ``primary`` (0-255): the unit colour the field's mask is
    multiplied by (None: no mask, the field painted red). -> (geometry, {"diffuse", "specular", "normal"} images)."""
    c = {**KITE, **(cfg or {})}
    out, up, across = _norm(list(out)), _norm(list(up)), _norm(list(across))
    W = H * c["aspect"]
    outline = kite_outline(H, W, int(c["sides"]))
    inner = _inset(outline, c["rim"] * H / 2.0)
    scale = 0.96 / H

    def to_uv(x, y):
        return (0.5 + x * scale, 0.5 - y * scale)

    def pos(x, y, lift=0.0):
        z = c["bulge"] * W * (1 - (2 * x / W) ** 2) + c["bulge_v"] * H * (1 - (2 * y / H) ** 2) + lift
        return _add(_add(_add(list(centre), across, x), up, y), out, z)

    def outward(i, ring):  # the outline's outward normal in 3D at point i
        (ax, ay), (bx, by) = ring[i - 1], ring[(i + 1) % len(ring)]
        nx, ny = by - ay, -(bx - ax)
        return _add([across[k] * nx for k in range(3)], up, ny)

    m = Mesh()
    n = len(outline)
    rim_top_o = [m.vert(pos(x, y, c["rim_h"]), to_uv(x, y)) for x, y in outline]
    rim_top_i = [m.vert(pos(x, y, c["rim_h"]), to_uv(x, y)) for x, y in inner]
    rim_low_i = [m.vert(pos(x, y), to_uv(x, y)) for x, y in inner]
    back_uv = (0.03, 0.03)  # outside the kite: the back's leather patch
    wall_low_o = [m.vert(pos(x, y, -c["thickness"]), to_uv(x, y)) for x, y in outline]
    back_o = [m.vert(pos(x, y, -c["thickness"]), back_uv) for x, y in outline]
    cy = H * c["widest"] * 0.3
    rings = [rim_low_i]
    for s in (0.66, 0.33):
        rings.append([m.vert(pos(x * s, cy + (y - cy) * s), to_uv(x * s, cy + (y - cy) * s)) for x, y in inner])
    mid = m.vert(pos(0.0, cy), to_uv(0.0, cy))
    back_mid = m.vert(pos(0.0, cy, -c["thickness"]), back_uv)
    neg_out = [-x for x in out]
    for i in range(n):
        j = (i + 1) % n
        m.face([rim_top_o[i], rim_top_o[j], rim_top_i[j], rim_top_i[i]], out)  # the rim's top
        m.face([rim_top_i[i], rim_top_i[j], rim_low_i[j], rim_low_i[i]], [-x for x in outward(i, inner)])  # inner edge
        m.face([rim_top_o[i], wall_low_o[i], wall_low_o[j], rim_top_o[j]], outward(i, outline))  # outer edge
        for ra, rb in zip(rings[:-1], rings[1:]):
            m.face([ra[i], ra[j], rb[j], rb[i]], out)
        m.face([rings[-1][i], rings[-1][j], mid], out)
        m.face([back_o[i], back_o[j], back_mid], neg_out)
    # the texture, drawn at 4x and reduced (crisp edges)
    T = int(c["tex"])
    B = T * 4

    def to_px(x, y):
        u, v = to_uv(x, y)
        return (u * B, v * B)
    region = Image.new("L", (B, B), 0)  # 255 inside the rim (the field)
    ImageDraw.Draw(region).polygon([to_px(x, y) for x, y in inner], fill=255)
    emblem = Image.new("L", (B, B), 0)
    _cross_emblem(ImageDraw.Draw(emblem), to_px, H, W)
    emblem = ImageChops.multiply(emblem, region.filter(ImageFilter.MinFilter(9)))
    field_mask = ImageChops.subtract(region, emblem)
    gold = KITE_COLOURS["gold"]
    field = KITE_COLOURS["field"]
    if primary is not None:  # the field's mask: a grey whose product with the primary colour is the field's red
        p = [x / 255.0 for x in primary]
        g = int(round(min(1.0, _lum(field) / 255.0 / max(_lum(p), 1e-6)) * 255))
        field_px = (g, g, g)
    else:
        field_px = field
    diffuse = Image.new("RGB", (B, B), KITE_COLOURS["back"])
    ImageDraw.Draw(diffuse).polygon([to_px(x * 1.04, y * 1.03) for x, y in outline], fill=gold)  # the rim, its walls too
    diffuse.paste(field_px, (0, 0, B, B), field_mask)
    height = Image.new("L", (B, B), 0)
    hd = ImageDraw.Draw(height)
    hd.polygon([to_px(x * 1.04, y * 1.03) for x, y in outline], fill=200)
    hd.polygon([to_px(x, y) for x, y in inner], fill=0)
    height.paste(110, (0, 0, B, B), emblem)
    dd = ImageDraw.Draw(diffuse)
    studs = _inset(outline, c["rim"] * H / 4.0)
    rs = c["rim"] * H * 0.17 * scale * B
    for k, (x, y) in enumerate(studs):
        if k % 2:
            continue
        px, py = to_px(x, y)
        dd.ellipse([px - rs, py - rs, px + rs, py + rs], fill=KITE_COLOURS["stud"])
        for r, v in ((rs, 230), (rs * 0.6, 255)):
            hd.ellipse([px - r, py - r, px + r, py + r], fill=v)
    diffuse = diffuse.resize((T, T), Image.LANCZOS)
    height = height.filter(ImageFilter.GaussianBlur(B / T * 1.2)).resize((T, T), Image.LANCZOS)
    mask_small = field_mask.resize((T, T), Image.LANCZOS)
    gold_small = ImageChops.subtract(region, field_mask).resize((T, T), Image.LANCZOS)
    spec = Image.new("RGBA", (T, T), (0, 0, 0, KITE_GLOSS["back"]))
    rim_area = Image.new("L", (B, B), 0)
    ImageDraw.Draw(rim_area).polygon([to_px(x * 1.04, y * 1.03) for x, y in outline], fill=255)
    spec.paste((0, 0, 0, KITE_GLOSS["gold"]), (0, 0, T, T), rim_area.resize((T, T), Image.LANCZOS))
    spec.paste((255 if primary is not None else 0, 0, 0, KITE_GLOSS["field"]), (0, 0, T, T), mask_small.point(lambda v: 255 if v > 127 else 0))
    spec.paste((0, 0, 0, KITE_GLOSS["gold"]), (0, 0, T, T), gold_small.point(lambda v: 255 if v > 127 else 0))
    return m.geometry(), {"diffuse": diffuse.convert("RGBA"), "specular": spec, "normal": normal_from_height(height, 3.0)}
