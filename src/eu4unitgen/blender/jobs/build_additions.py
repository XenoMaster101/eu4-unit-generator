"""Blender job: build skinned additions for a derived body (runs in Blender only; SPEC Track R).

Imports the base body .mesh (with any removed parts already dropped; nothing exported) to measure
it, then builds each addition in the body's bind pose as rings along a centreline: every ring has
the shape's cross-section (``SIDES`` vertices), ring 0 is the root, and the far end is a flat cap,
a single point or a split (tufted) tip. Returns per addition, in PDX mesh space (Y up): positions,
smooth vertex normals, image-convention UVs inside the given texture patch, triangles and each
vertex's ring (tip points get the last ring + 1). Weights are set by the app (core.chain_weights)
and the file is written by core.derived_body; io_pdx_mesh never exports it.

Blender axes after import: Z up, the unit faces -Y (its back is +Y), its left is +X.

Shapes (plan after test 10, SPEC Track R):
  * ``tail``: a flared, tufted base over the tunic hem, a slim dock, then a hair plume (flattened
    ellipse, wider side to side) widening to ~40% of the length and tapering to a 5-point tip.
  * ``tuft``: a tied lock at the back of the head: a tight wrap near the root, a flattened brush,
    a 3-point tip; within pre-check P1's envelope (1.5 units at 45 degrees).
  * ``ears``: two horse ears on the sides of the crown (Q0): about 4:1, pinched base, widest a third
    up, a rounded tip; a lens-shaped section ~0.12 thick, cupped in front; tilted out and leaning
    back; placed from the head's own vertices (those mostly on ``head`` that a face uses).

args: {"body": abs .mesh, "additions": [{"name", "shape", "params": {...}}], "patch_px": [x0, y0, x1, y1],
       "tex": px, "result_path"}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402


# --------------------------------------------------------------------------- measuring the body

def used_vertices(body):
    return {i for poly in body.data.polygons for i in poly.vertices}


def back_profile(body, include_removed=False):
    """Blender z (quarter steps) -> the body's rearmost y (+Y = the unit's back) near the midline.
    Only vertices a face uses (a derived base keeps removed parts' vertices loose), unless
    ``include_removed``: the base as it was (type1 with its helmet)."""
    prof = {}
    used = used_vertices(body)
    for v in body.data.vertices:
        if v.index not in used and not include_removed:
            continue
        co = body.matrix_world @ v.co
        if abs(co.x) < 0.8:
            k = round(co.z * 4) / 4
            prof[k] = max(prof.get(k, -99.0), co.y)
    return prof


def back_y(prof, z):
    near = [k for k in prof if abs(k - z) <= 0.5]
    return max(prof[k] for k in near) if near else max(prof.values())


def head_box(body):
    """World-space box of the used vertices mostly on the ``head`` group."""
    g = body.vertex_groups.get("head")
    used = used_vertices(body)
    pts = []
    for v in body.data.vertices:
        if v.index not in used or not v.groups:
            continue
        best = max(v.groups, key=lambda x: x.weight)
        if g is not None and best.group == g.index and best.weight > 0.5:
            pts.append(body.matrix_world @ v.co)
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


# --------------------------------------------------------------------------- ring meshes

def lerp_profile(profile, t):
    """Piecewise-linear lookup in [(t, a, b), ...] (sorted by t)."""
    for (t0, a0, b0), (t1, a1, b1) in zip(profile, profile[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return a0 + (a1 - a0) * f, b0 + (b1 - b0) * f
    return profile[-1][1], profile[-1][2]


class RingMesh:
    """Rings of points along a centreline, closed into a manifold shell."""

    def __init__(self):
        self.bm = bmesh.new()
        self.rings = []  # lists of BMVerts
        self.ring_of = {}  # BMVert -> ring index
        self.around = {}  # BMVert -> position around its ring (0..1), for UVs

    def ring(self, points):
        verts = [self.bm.verts.new(p) for p in points]
        for k, v in enumerate(verts):
            self.ring_of[v] = len(self.rings)
            self.around[v] = k / len(verts)
        self.rings.append(verts)
        return verts

    def bands(self):
        for a, b in zip(self.rings, self.rings[1:]):
            n = len(a)
            for k in range(n):
                self.bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))

    def cap_root(self):
        self.bm.faces.new(list(reversed(self.rings[0])))

    def point_tip(self, tip):
        last = self.rings[-1]
        v = self.bm.verts.new(tip)
        self.ring_of[v], self.around[v] = len(self.rings), 0.5
        n = len(last)
        for k in range(n):
            self.bm.faces.new((last[k], last[(k + 1) % n], v))

    def split_tip(self, tips):
        """``len(tips)`` points, each fanned to 2 consecutive edges of the last ring (needs 2 x len(tips) sides)."""
        last = self.rings[-1]
        n = len(last)
        assert n == 2 * len(tips)
        for j, tip in enumerate(tips):
            v = self.bm.verts.new(tip)
            self.ring_of[v], self.around[v] = len(self.rings), (2 * j + 1) / n
            a, b, c = last[2 * j], last[2 * j + 1], last[(2 * j + 2) % n]
            self.bm.faces.new((a, b, v))
            self.bm.faces.new((b, c, v))

    def finish(self, name):
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces[:])
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces[:])
        self.bm.verts.index_update()
        rings = [0] * len(self.bm.verts)
        around = [0.0] * len(self.bm.verts)
        for v in self.bm.verts:
            rings[v.index], around[v.index] = self.ring_of[v], self.around[v]
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        return me, rings, around


def frame(direction):
    """(side, depth) axes across ``direction``: side stays near world X; depth completes the frame."""
    d = direction.normalized()
    side = Vector((1, 0, 0))
    side = (side - d * side.dot(d)).normalized()
    depth = d.cross(side).normalized()
    return side, depth


# --------------------------------------------------------------------------- shapes

TAIL_PROFILE = [(0.00, 0.80, 0.60), (0.06, 0.62, 0.48), (0.14, 0.34, 0.30), (0.26, 0.46, 0.30), (0.40, 0.60, 0.34),
                (0.55, 0.58, 0.32), (0.70, 0.48, 0.27), (0.82, 0.36, 0.22), (0.92, 0.26, 0.17), (1.00, 0.18, 0.12)]


def tail(body, p):
    sides, rings, length = 10, int(p.get("rings", 12)), p.get("length", 5.5)
    prof = back_profile(body)
    z0 = p.get("root_z", 8.0)
    pos = Vector((0, back_y(prof, z0) - p.get("sink", 0.15), z0))
    m = RingMesh()
    step = length / (rings - 1)
    for i in range(rings):
        t = i / (rings - 1)
        ang = math.radians(p.get("angle_root", 75) + (p.get("angle_tip", 20) - p.get("angle_root", 75)) * t)
        d = Vector((0, math.sin(ang), -math.cos(ang)))
        side, depth = frame(d)
        rx, ry = lerp_profile(TAIL_PROFILE, t)
        pts = []
        for k in range(sides):
            a = 2 * math.pi * k / sides
            if i == 0:
                jitter = 1 + 0.18 * math.cos(7 * a)  # tufted base
            elif t >= 0.2:
                jitter = 1 + 0.10 * math.cos(5 * a + i * 0.7)  # strand ridges
            else:
                jitter = 1.0
            pts.append(pos + (side * math.cos(a) * rx + depth * math.sin(a) * ry) * jitter)
        m.ring(pts)
        if i < rings - 1:
            pos = pos + d * step
    m.bands()
    m.cap_root()
    d = Vector((0, math.sin(math.radians(p.get("angle_tip", 20))), -math.cos(math.radians(p.get("angle_tip", 20)))))
    side, depth = frame(d)
    tips = []
    for j, (spread, extra) in enumerate(((-0.16, 0.45), (-0.08, 0.7), (0.0, 0.55), (0.09, 0.65), (0.17, 0.4))):
        a = 2 * math.pi * (2 * j + 1) / sides
        tips.append(pos + d * extra + (side * math.cos(a) * 0.1 + depth * math.sin(a) * 0.06) + side * spread)
    m.split_tip(tips)
    return m


TUFT_PROFILE = [(0.0, 0.28, 0.24), (0.15, 0.17, 0.15), (0.32, 0.22, 0.17), (0.62, 0.34, 0.17), (1.0, 0.28, 0.13)]


def tuft(body, p):
    sides, rings = 6, 5
    length = p.get("length", 1.5) - p.get("tip_extra", 0.25)
    # anchored on the base as it was (pre-check P1 validated the tuft on type1's helmet back); a
    # Greek helmet replacing it must put its back there too, so the tuft comes out of the helmet
    prof = back_profile(body, include_removed=p.get("anchor", "original") == "original")
    z0 = p.get("root_z", 15.2)
    root = Vector((0, back_y(prof, z0) - p.get("sink", 0.1), z0))
    ang = math.radians(p.get("angle", 45))
    d = Vector((0, math.sin(ang), -math.cos(ang)))
    side, depth = frame(d)
    m = RingMesh()
    for i in range(rings):
        t = i / (rings - 1)
        rx, ry = lerp_profile(TUFT_PROFILE, t)
        c = root + d * length * t
        m.ring([c + side * math.cos(2 * math.pi * k / sides) * rx + depth * math.sin(2 * math.pi * k / sides) * ry
                for k in range(sides)])
    m.bands()
    m.cap_root()
    end = root + d * length
    extra = p.get("tip_extra", 0.25)
    m.split_tip([end + d * extra * f + side * s for f, s in ((0.8, -0.14), (1.0, 0.0), (0.7, 0.14))])
    return m


# Horse ear (Q0, user feedback after test 11: the first ears read as cat ears and were paper-thin):
# (t along the length, width, depth). About 4:1, pinched at the base, widest a third of the way up,
# closing in narrowing rings to a rounded tip. Depth = the back's bulge; the front is cupped.
EAR_PROFILE = [(0.00, 0.30, 0.20), (0.18, 0.39, 0.19), (0.35, 0.42, 0.17), (0.55, 0.38, 0.14), (0.72, 0.30, 0.11),
               (0.85, 0.20, 0.08), (0.94, 0.12, 0.05)]  # Q0 round 2: shorter and broader mid-ear
EAR_CUP = 0.4  # the front's centre sits this share of the depth back from the edges: a cupped (concave) front


def ears(body, p):
    """Both ears in one mesh (two closed shells). Each ear: rings along a line from a base on the
    side of the crown, tilted ``tilt_deg`` out and leaning ``lean_deg`` back, its opening turned
    ``yaw_deg`` outwards, curving ``curve`` towards the midline by the tip; a lens-shaped section, convex behind and cupped in front, ~0.12 thick at
    the base (0.6 x the depth); narrowing rings close into a rounded tip."""
    lo, hi = head_box(body)
    half_w = (hi.x - lo.x) / 2
    cx, cy = (hi.x + lo.x) / 2, (hi.y + lo.y) / 2
    length = p.get("length", 1.15)
    tilt, lean = math.radians(p.get("tilt_deg", 13)), math.radians(p.get("lean_deg", 15))
    curve = p.get("curve", 0.12)  # the tips bend this far towards the midline (a gentle inward curve)
    m = RingMesh()
    for s in (-1.0, 1.0):
        base = Vector((cx + s * p.get("spread", 0.85) * half_w, cy + p.get("back", 0.05), hi.z - p.get("sink", 0.35)))
        d = Vector((s * math.sin(tilt), math.sin(lean), math.cos(tilt) * math.cos(lean))).normalized()  # +Y = back
        side, _ = frame(d)
        back = d.cross(side).normalized()
        if back.y < 0:  # depth axis towards the unit's back (+Y), so the cup opens to the front
            back = -back
        turn = Matrix.Rotation(-s * math.radians(p.get("yaw_deg", 25)), 3, d)  # opening turned out, like a horse's
        side, back = turn @ side, turn @ back
        start = len(m.rings)
        inward = Vector((-s, 0, 0))
        for t, w, dep in EAR_PROFILE:
            c = base + d * length * t + inward * curve * t * t
            outer = [(math.sin(a) * w / 2, math.cos(a) * dep) for a in (math.radians(x) for x in (-90, -45, 0, 45, 90))]
            inner = [(math.sin(a) * w / 2 * 0.72, math.cos(a) * dep * EAR_CUP)
                     for a in (math.radians(x) for x in (60, 0, -60))]
            m.ring([c + side * x + back * y for x, y in outer + inner])
        ear_rings = m.rings[start:]
        for ra, rb in zip(ear_rings, ear_rings[1:]):
            n = len(ra)
            for k in range(n):
                m.bm.faces.new((ra[k], ra[(k + 1) % n], rb[(k + 1) % n], rb[k]))
        m.bm.faces.new(list(reversed(ear_rings[0])))
        last = ear_rings[-1]
        centre = sum((v.co for v in last), Vector()) / len(last)
        tip = m.bm.verts.new(centre + (d * length + inward * curve * 1.88).normalized() * length * 0.05)  # rounded tip
        m.ring_of[tip], m.around[tip] = len(ear_rings), 0.5
        for k in range(len(last)):
            m.bm.faces.new((last[k], last[(k + 1) % len(last)], tip))
        for i, r in enumerate(ear_rings):  # each ear counts its rings from 0 (rigid: only the root ring matters)
            for v in r:
                m.ring_of[v] = i
    return m


SHAPES = {"tail": tail, "tuft": tuft, "ears": ears}


def pdx(co):
    """Blender (Z up) -> PDX mesh space (Y up): io_pdx_mesh's axis swap."""
    return [co[0], co[2], co[1]]


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    _pdx.clear_scene()
    before = set(bpy.data.objects)
    ie.import_meshfile(a["body"])
    body = next(o for o in bpy.data.objects if o not in before and o.type == "MESH")
    x0, y0, x1, y1 = a["patch_px"]
    tex = a["tex"]
    out = []
    for add in a["additions"]:
        x0, y0, x1, y1 = add.get("patch_px") or a["patch_px"]  # Q5: each addition's own part of the hair swatch
        inner = add.get("inner_patch_px")  # ears: the cupped front's vertices go to a lighter patch
        me, rings, around = SHAPES[add["shape"]](body, add.get("params", {})).finish(add["name"])
        last = max(rings) or 1
        # Q1: use only as much of the hair region as gives ~``density`` texels per mesh unit (all of it
        # when the region is too small for that, e.g. the tail, until a dedicated hair swatch exists).
        # Map at full size, measure the real density over the triangles, scale once.
        def uvs(f):
            out_uv = []
            for i in range(len(me.vertices)):
                if inner and around[i] >= 0.6:  # inner (front) ring points: the lighter inner-ear patch
                    ix0, iy0, ix1, iy1 = inner
                    out_uv.append((ix0 + 1 + (ix1 - ix0 - 2) * (around[i] - 0.6) / 0.4,
                                   iy0 + 1 + (iy1 - iy0 - 2) * (rings[i] / last)))
                else:
                    out_uv.append(((x0 + 1 + (x1 - x0 - 2) * (0.1 + 0.8 * around[i] * f)),
                                   (y0 + 1 + (y1 - y0 - 2) * (rings[i] / last) * f)))
            return out_uv

        def measured(uv):
            area3 = area_uv = 0.0
            for poly in me.polygons:
                vs = list(poly.vertices)
                for k in range(1, len(vs) - 1):
                    t = (vs[0], vs[k], vs[k + 1])
                    (ax, ay), (bx, by), (cx, cy) = (uv[i] for i in t)
                    area_uv += abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
            area3 = sum(poly.area for poly in me.polygons)
            return (area_uv / area3) ** 0.5 if area3 > 0 else 0.0

        d1 = measured(uvs(1.0))
        f = min(1.0, a.get("density", 32.0) / d1) if d1 > 0 else 1.0
        uv = uvs(f)
        p, n, u0 = [], [], []
        for v in me.vertices:
            p += pdx(v.co)
            n += pdx(v.normal)
            u0 += [uv[v.index][0] / tex, uv[v.index][1] / tex]
        tri = [i for poly in me.polygons for i in poly.vertices]
        out.append({"name": add["name"], "p": p, "n": n, "u0": u0, "tri": tri, "rings": rings})
    _pdx.write_result(a, {"additions": out})


main()
