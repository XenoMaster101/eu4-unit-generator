"""Blender job: build procedural rigid parts and export each as a static .mesh (runs in Blender only).

Geometry is written in PDX mesh space (Y-up, the attach bone's local frame) and converted to
Blender (Z-up) with ``v()``; io_pdx_mesh converts back on export. Conventions (core/parts.py,
SPEC §10 Phase 3): hand weapons along +Z from a grip at the origin ("point": "down" mirrors
them along Z, like the cavalry sabre); shields in the hand's YZ plane, domed towards +X; head
parts in the ``head`` bone frame, the unit facing -Z.

UVs: each zone's faces sit on that zone's swatch (``zone_uvs``, image coordinates, v down);
the ``face`` zone of a disc layout is planar-mapped onto the image's inscribed circle.
io_pdx_mesh flips V on export, so image coordinates become Blender UVs with v -> 1 - v.

args: {"parts": [{"kind", "builder", "shape", "out_mesh", "shader", "textures": {diff, n, spec},
                  "zones": n, "layout", "zone_uvs": [[u, v], ...], "params": {...}}], "result_path": ...}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def v(x, y, z):
    """PDX (Y-up) -> Blender (Z-up), the inverse of io_pdx_mesh's SPACE_MATRIX swap."""
    return (x, z, y)


class Builder:
    def __init__(self, zone_uvs, flip_z=False):
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("map1")
        self.zone_uvs = zone_uvs
        self.flip_z = flip_z
        self.pdx = {}  # vert -> PDX coords (for planar UVs)

    def vert(self, x, y, z):
        if self.flip_z:
            z = -z
        vert = self.bm.verts.new(v(x, y, z))
        self.pdx[vert] = (x, y, z)
        return vert

    def face(self, verts, zone, planar=None):
        if self.flip_z:  # mirroring turns faces inside out; recalc_face_normals fixes closed shells, keep winding sane
            verts = list(reversed(verts))
        f = self.bm.faces.new(verts)
        for loop in f.loops:
            u, vv = planar(self.pdx[loop.vert]) if planar else self.zone_uvs[zone]
            loop[self.uv].uv = (u, 1.0 - vv)
        return f

    def ring(self, z, radius, sides, start=0.0, sx=1.0, sy=1.0):
        return [self.vert(radius * sx * math.cos(start + 2 * math.pi * i / sides),
                          radius * sy * math.sin(start + 2 * math.pi * i / sides), z) for i in range(sides)]

    def tube(self, a, b, zone):
        n = len(a)
        for i in range(n):
            self.face([a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]], zone)

    def cap(self, ring, tip_or_none, zone, reverse=False):
        if tip_or_none is None:
            self.face(list(reversed(ring)) if reverse else ring, zone)
            return
        n = len(ring)
        for i in range(n):
            tri = [ring[i], ring[(i + 1) % n], tip_or_none]
            self.face(list(reversed(tri)) if reverse else tri, zone)

    def finish(self, name):
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        # io_pdx_mesh computes tangents, which Blender only does for tris/quads: no n-gons
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces)
        mesh = bpy.data.meshes.new(name)
        self.bm.to_mesh(mesh)
        self.bm.free()
        return mesh


def build_spear(b, p):
    """Dory and sarissa. Shaft (zone 0, wood) from the butt to the head's base; leaf head and
    butt spike (zone 1, metal). Grip at z = 0, point at +Z."""
    r, sides = p["shaft_radius"], 8
    z0, tip = p["butt"], p["tip"]
    z1 = tip - p["head_length"]
    lo, hi = b.ring(z0, r, sides), b.ring(z1, r, sides)
    b.tube(lo, hi, 0)
    if p["spike"] > 0:
        b.cap(lo, b.vert(0, 0, z0 - p["spike"]), 1, reverse=True)  # sauroter
    else:
        b.cap(lo, None, 0, reverse=True)
    w, t = p["head_width"], max(0.06, r * 0.7)
    mid = z1 + (tip - z1) * 0.3
    blade = [b.vert(w, 0, mid), b.vert(0, t, mid), b.vert(-w, 0, mid), b.vert(0, -t, mid)]
    neck = b.ring(z1, r, 4, start=0.0)
    b.tube(neck, blade, 1)
    b.cap(blade, b.vert(0, 0, tip), 1)
    b.cap(hi, None, 0)


def build_sword(b, p):
    """Xiphos: leaf blade (zone 0) from the guard to the point at +Z; grip and pommel (zone 1)
    around the origin."""
    g, length, w = p["grip_length"], p["blade_length"], p["blade_width"]
    gr, half_t = 0.13, 0.07
    g0, g1 = -g * 0.5, g * 0.5
    grip_lo, grip_hi = b.ring(g0, gr, 6), b.ring(g1, gr, 6)
    b.tube(grip_lo, grip_hi, 1)
    b.cap(grip_lo, b.vert(0, 0, g0 - 0.25), 1, reverse=True)  # pommel
    b.cap(grip_hi, None, 1)
    gw, gt = p["guard_width"] * 0.5, 0.1  # guard: a flat bar across the blade's width (x)
    corners = [(-gw, -0.15), (gw, -0.15), (gw, 0.15), (-gw, 0.15)]
    lo = [b.vert(x, y, g1) for x, y in corners]
    hi = [b.vert(x, y, g1 + gt) for x, y in corners]
    b.tube(lo, hi, 0)
    b.cap(lo, None, 0, reverse=True)
    b.cap(hi, None, 0)
    base = g1 + gt
    # leaf: width along x, thickness along y; (fraction of length, width factor)
    sections = [(0.0, 0.7), (0.35, 0.8), (0.65, 1.0), (0.88, 0.6)]
    rings = [[b.vert(w * f * 0.5, 0, base + length * s), b.vert(0, half_t, base + length * s),
              b.vert(-w * f * 0.5, 0, base + length * s), b.vert(0, -half_t, base + length * s)]
             for s, f in sections]
    b.cap(rings[0], None, 0, reverse=True)
    for a, c in zip(rings, rings[1:]):
        b.tube(a, c, 0)
    b.cap(rings[-1], b.vert(0, 0, base + length), 0)


def build_shield(b, p):
    """Hoplon: a closed shell. Outer dome (zone 0, the face, planar UVs) and flat rim (zone 1)
    towards +X; the edge (zone 1); the inside (zone 2) back to ``back_x``. Face in the YZ plane
    around (centre_y, centre_z)."""
    radius, rim, depth, th = p["radius"], p["rim"], p["depth"], p["thickness"]
    bx, cy, cz = p["back_x"], p["centre_y"], p["centre_z"]
    rf = max(radius - rim, radius * 0.5)
    rings_n, sides = 6, 28
    xf = bx + th  # front plane of the rim

    def outer_x(r):
        return xf + depth * (1 - (r / rf) ** 2)

    def ring_at(r, x):
        return [b.vert(x, cy + r * math.sin(2 * math.pi * i / sides), cz + r * math.cos(2 * math.pi * i / sides))
                for i in range(sides)]

    def planar(q):  # seen from outside (+X) the viewer's right is PDX +Z (PDX space is mirrored), up is +Y
        return 0.5 + (q[2] - cz) / (2 * rf), 0.5 - (q[1] - cy) / (2 * rf)

    def quads(a, c, zone, uv=None):
        for i in range(sides):
            b.face([a[i], a[(i + 1) % sides], c[(i + 1) % sides], c[i]], zone, uv)

    def fan(centre, ring, zone, uv=None, reverse=False):
        for i in range(sides):
            tri = [centre, ring[i], ring[(i + 1) % sides]]
            b.face(list(reversed(tri)) if reverse else tri, zone, uv)

    radii = [rf * (i + 1) / rings_n for i in range(rings_n)]
    outer = [ring_at(r, outer_x(r)) for r in radii]
    inner = [ring_at(r, outer_x(r) - th) for r in radii]
    edge_front, edge_back = ring_at(radius, xf), ring_at(radius, bx)
    fan(b.vert(outer_x(0), cy, cz), outer[0], 0, planar)
    for a, c in zip(outer, outer[1:]):
        quads(a, c, 0, planar)
    quads(outer[-1], edge_front, 1)  # rim, front
    quads(edge_front, edge_back, 1)  # edge
    quads(edge_back, inner[-1], 2)  # rim, back
    for a, c in zip(reversed(inner), list(reversed(inner))[1:]):
        quads(a, c, 2)
    fan(b.vert(outer_x(0) - th, cy, cz), inner[0], 2, reverse=True)


def build_crest(b, p):
    """A horsehair fin (zone 0) on an arc over the helmet, in the head bone frame. Fore-aft: the
    arc runs along z; transverse: along x. ``centre_z`` moves it forward (-) or back (+)."""
    base, height = p["base"], p["height"]
    span, half = p["span_deg"], p["thickness"] * 0.5
    radius, steps = p["radius"], 10
    transverse = p.get("orientation") == "transverse"
    cz = p.get("centre_z", 0.0)

    def at(side, y, along):
        x, z = (along, side) if transverse else (side, along)
        return b.vert(x, y, z + cz)

    bottom, top = [], []
    for i in range(steps + 1):
        a = math.radians(-span + 2 * span * i / steps)
        along, y = radius * math.sin(a), base - radius * (1 - math.cos(a))
        h = height * (1 - 0.35 * abs(math.sin(a)))
        bottom.append((at(-half, y, along), at(half, y, along)))
        top.append((at(-half, y + h, along), at(half, y + h, along)))
    for i in range(steps):
        (bl0, br0), (bl1, br1) = bottom[i], bottom[i + 1]
        (tl0, tr0), (tl1, tr1) = top[i], top[i + 1]
        b.face([bl0, bl1, tl1, tl0], 0)   # one side
        b.face([br0, tr0, tr1, br1], 0)   # other side
        b.face([tl0, tl1, tr1, tr0], 0)   # top
        b.face([bl0, br0, br1, bl1], 0)   # bottom
    for (bl, br), (tl, tr) in ((bottom[0], top[0]), (bottom[-1], top[-1])):
        b.face([bl, tl, tr, br], 0)       # end caps


BUILDERS = {"spear": build_spear, "sword": build_sword, "shield": build_shield, "crest": build_crest}


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    out = []
    for part in a["parts"]:
        _pdx.clear_scene()
        params = part.get("params", {})
        builder = Builder(part["zone_uvs"], flip_z=params.get("point") == "down")
        BUILDERS[part["builder"]](builder, params)
        mesh = builder.finish(part["shape"])
        obj = bpy.data.objects.new(part["shape"].removesuffix("Shape"), mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(_pdx.pdx_material(part["shape"] + "_mat", part.get("shader", "PdxMeshStandard"),
                                                part["textures"]))
        ie.export_meshfile(part["out_mesh"], exp_mesh=True, exp_skel=False, exp_locs=False)
        out.append({"kind": part["kind"], "out_mesh": part["out_mesh"], "verts": len(mesh.vertices),
                    "faces": len(mesh.polygons)})
    _pdx.write_result(a, {"parts": out})


main()
