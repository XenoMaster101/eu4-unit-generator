"""Blender job: build procedural rigid parts and export each as a static .mesh (runs in Blender only).

Geometry is written in PDX mesh space (Y-up, the attach bone's local frame) and converted to
Blender (Z-up) with ``v()``; io_pdx_mesh converts back on export. Conventions measured from
vanilla (SPEC §10 Phase 3): hand weapons run along +Z with the grip at the origin (western
halbard z -10.8..13.9, point at +Z); head parts are built in the ``head`` bone frame (aligned
with the mesh axes; the western helmet top is ~2.15 units above the joint).

args: {"parts": [{"kind", "shape", "out_mesh", "shader", "textures": {diff, n, spec},
                  "zones": n, "params": {...}}], "result_path": ...}
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
    def __init__(self, zones: int):
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("map1")
        self.zones = zones

    def vert(self, x, y, z):
        return self.bm.verts.new(v(x, y, z))

    def face(self, verts, zone: int):
        f = self.bm.faces.new(verts)
        u = (zone + 0.5) / self.zones
        for loop in f.loops:
            loop[self.uv].uv = (u, 0.5)
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


def build_dory(b: Builder, p: dict) -> None:
    """Shaft (zone 0, wood) along +Z, leaf head and butt spike (zone 1, bronze). Grip at z = 0."""
    r, sides = p.get("shaft_radius", 0.17), 8
    z0, z1 = p.get("butt", -11.5), p.get("head_base", 11.0)
    lo, hi = b.ring(z0, r, sides), b.ring(z1, r, sides)
    b.tube(lo, hi, 0)
    b.cap(lo, b.vert(0, 0, z0 - p.get("spike", 1.3)), 1, reverse=True)  # sauroter
    w, t, tip = p.get("head_width", 0.45), p.get("head_thickness", 0.12), p.get("tip", 15.0)
    mid = z1 + (tip - z1) * 0.3
    blade = [b.vert(w, 0, mid), b.vert(0, t, mid), b.vert(-w, 0, mid), b.vert(0, -t, mid)]
    neck = b.ring(z1, r, 4, start=0.0)
    b.tube(neck, blade, 1)
    b.cap(blade, b.vert(0, 0, tip), 1)
    b.cap(hi, None, 0)


def build_crest(b: Builder, p: dict) -> None:
    """A front-to-back horsehair fin (zone 0) on an arc over the helmet, in the head bone frame."""
    base, height = p.get("base", 1.9), p.get("height", 1.2)
    span, half = p.get("span_deg", 70.0), p.get("thickness", 0.14)
    radius, steps = p.get("radius", 1.4), 10
    bottom, top = [], []
    for i in range(steps + 1):
        a = math.radians(-span + 2 * span * i / steps)
        z, y = radius * math.sin(a), base - radius * (1 - math.cos(a))
        h = height * (1 - 0.35 * abs(math.sin(a)))
        bottom.append((b.vert(-half, y, z), b.vert(half, y, z)))
        top.append((b.vert(-half, y + h, z), b.vert(half, y + h, z)))
    for i in range(steps):
        (bl0, br0), (bl1, br1) = bottom[i], bottom[i + 1]
        (tl0, tr0), (tl1, tr1) = top[i], top[i + 1]
        b.face([bl0, bl1, tl1, tl0], 0)   # left side
        b.face([br0, tr0, tr1, br1], 0)   # right side
        b.face([tl0, tl1, tr1, tr0], 0)   # top
        b.face([bl0, br0, br1, bl1], 0)   # bottom
    for (bl, br), (tl, tr) in ((bottom[0], top[0]), (bottom[-1], top[-1])):
        b.face([bl, tl, tr, br], 0)       # end caps


BUILDERS = {"dory": build_dory, "crest": build_crest}


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    out = []
    for part in a["parts"]:
        _pdx.clear_scene()
        builder = Builder(part.get("zones", 1))
        BUILDERS[part["kind"]](builder, part.get("params", {}))
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
