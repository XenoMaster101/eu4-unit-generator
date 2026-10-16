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
  * ``ears``: two horse ears on top of the bare head, each a thin shell with a crescent cross-section
    cupped towards the front, pinched at the base, widest a quarter up, pointed; placed from the
    head's own vertices (those mostly on ``head`` that a face uses).

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


EAR_PROFILE = [(0.0, 0.30, 0.10), (0.25, 0.50, 0.13), (0.5, 0.46, 0.12), (0.75, 0.32, 0.09), (1.0, 0.16, 0.06)]


def ears(body, p):
    """Both ears in one mesh (two shells)."""
    lo, hi = head_box(body)
    half_w = (hi.x - lo.x) / 2
    cx, cy = (hi.x + lo.x) / 2, (hi.y + lo.y) / 2
    length, rings = p.get("length", 1.1), 5
    tilt, lean = math.radians(p.get("tilt_deg", 12)), math.radians(p.get("lean_deg", 8))
    m = RingMesh()
    for s in (-1.0, 1.0):
        base = Vector((cx + s * p.get("spread", 0.5) * half_w, cy + p.get("back", 0.05), hi.z - p.get("sink", 0.22)))
        d = Vector((s * math.sin(tilt), -math.sin(lean), math.cos(tilt) * math.cos(lean))).normalized()
        side, _ = frame(d)
        back = d.cross(side).normalized()
        if back.y < 0:  # depth axis towards the unit's back (+Y), so the cup opens to the front
            back = -back
        turn = Matrix.Rotation(-s * math.radians(p.get("yaw_deg", 25)), 3, d)  # opening turned out, like a horse's
        side, back = turn @ side, turn @ back
        start = len(m.rings)
        for i in range(rings):
            t = i / rings
            w, dep = lerp_profile(EAR_PROFILE, t)
            c = base + d * length * t
            outer = [(math.sin(a) * w / 2, math.cos(a) * dep) for a in (math.radians(x) for x in (-90, -45, 0, 45, 90))]
            inner = [(math.sin(a) * w / 2 * 0.75, math.cos(a) * dep * 0.35) for a in (math.radians(x) for x in (45, 0, -45))]
            m.ring([c + side * x + back * y for x, y in outer + inner])
        # this ear's own bands, root cap and point (RingMesh works on the last ring range)
        ear_rings = m.rings[start:]
        for a, b in zip(ear_rings, ear_rings[1:]):
            n = len(a)
            for k in range(n):
                m.bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
        m.bm.faces.new(list(reversed(ear_rings[0])))
        tip = m.bm.verts.new(base + d * length)
        m.ring_of[tip], m.around[tip] = rings, 0.5
        last = ear_rings[-1]
        for k in range(len(last)):
            m.bm.faces.new((last[k], last[(k + 1) % len(last)], tip))
        # rings of the second ear count from 0 again (rigid: only the root ring matters)
        for i, r in enumerate(ear_rings):
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
        me, rings, around = SHAPES[add["shape"]](body, add.get("params", {})).finish(add["name"])
        last = max(rings) or 1
        p, n, u0 = [], [], []
        for v in me.vertices:
            p += pdx(v.co)
            n += pdx(v.normal)
            u0 += [(x0 + 1 + (x1 - x0 - 2) * (0.1 + 0.8 * around[v.index])) / tex,
                   (y0 + 1 + (y1 - y0 - 2) * (rings[v.index] / last)) / tex]
        tri = [i for poly in me.polygons for i in poly.vertices]
        out.append({"name": add["name"], "p": p, "n": n, "u0": u0, "tri": tri, "rings": rings})
    _pdx.write_result(a, {"additions": out})


main()
