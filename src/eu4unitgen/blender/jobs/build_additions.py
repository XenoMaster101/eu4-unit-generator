"""Blender job: build skinned additions for a derived body (runs in Blender only; SPEC Track R).

Imports the base body .mesh in place (read from the install, nothing copied, nothing exported)
to measure its back, then builds each addition as a closed tapered tube in the body's bind pose:
rings from the root (0) to the tip, ``SIDES`` vertices per ring, both ends capped. Returns per
addition, in PDX mesh space (Y up): positions, smooth vertex normals, image-convention UVs inside
the given texture patch, triangles and each vertex's ring. Weights are set by the app
(core.chain_weights) and the file is written by core.derived_body; io_pdx_mesh never exports it.

Shapes (spike R0/P1, signed off):
  * ``tail``: root at the back of the pelvis (z 8.0 Blender, 0.15 inside), 10 rings over 5 units,
    arcing from 50 to 15 degrees off vertical towards the back, radius 0.45 -> 0.12, with a flared
    base (rings 0-1 at ``flare`` radii) over the tunic hem.
  * ``tuft``: root at the back of the head (z 15.2, 0.1 inside), straight at ``angle`` degrees
    off vertical towards the back, ``length`` units, radius 0.33 -> 0.14.

args: {"body": abs .mesh, "additions": [{"name", "shape", "params": {...}}], "patch_px": [x0, y0, x1, y1],
       "tex": px, "result_path"}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
from mathutils import Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402

SIDES = 8


def back_profile(body):
    """Blender z (quarter steps) -> the body's rearmost y (+Y = the unit's back) near the midline.
    Only vertices a face uses: a derived base keeps removed parts' vertices loose."""
    prof = {}
    used = {i for poly in body.data.polygons for i in poly.vertices}
    for v in body.data.vertices:
        if v.index not in used:
            continue
        co = body.matrix_world @ v.co
        if abs(co.x) < 0.8:
            k = round(co.z * 4) / 4
            prof[k] = max(prof.get(k, -99.0), co.y)
    return prof


def back_y(prof, z):
    near = [k for k in prof if abs(k - z) <= 0.5]
    return max(prof[k] for k in near) if near else max(prof.values())


def tail_line(prof, p):
    rings, length, z0 = int(p.get("rings", 10)), p.get("length", 5.0), p.get("root_z", 8.0)
    pos = Vector((0, back_y(prof, z0) - p.get("sink", 0.15), z0))
    step = length / (rings - 1)
    pts, radii = [], []
    for i in range(rings):
        t = i / (rings - 1)
        pts.append(pos.copy())
        radii.append(p.get("radius", 0.45) + (p.get("tip_radius", 0.12) - p.get("radius", 0.45)) * t)
        ang = math.radians(p.get("angle_root", 50) + (p.get("angle_tip", 15) - p.get("angle_root", 50)) * t)
        pos = pos + Vector((0, math.sin(ang), -math.cos(ang))) * step
    for i, r in enumerate(p.get("flare", [0.75, 0.6])):
        radii[i] = r
    return pts, radii


def tuft_line(prof, p):
    rings, length, z0 = int(p.get("rings", 5)), p.get("length", 1.5), p.get("root_z", 15.2)
    root = Vector((0, back_y(prof, z0) - p.get("sink", 0.1), z0))
    ang = math.radians(p.get("angle", 45))
    pts, radii = [], []
    for i in range(rings):
        t = i / (rings - 1)
        pts.append(root + Vector((0, math.sin(ang), -math.cos(ang))) * (length / (rings - 1)) * i)
        radii.append(p.get("radius", 0.33) + (p.get("tip_radius", 0.14) - p.get("radius", 0.33)) * t)
    return pts, radii


SHAPES = {"tail": tail_line, "tuft": tuft_line}


def tube(pts, radii):
    """Closed tapered tube; returns (bmesh, ring index per vertex in creation order)."""
    bm = bmesh.new()
    rings = []
    for i, (c, r) in enumerate(zip(pts, radii)):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        side = Vector((1, 0, 0))
        up = d.cross(side).normalized()
        rings.append([bm.verts.new(c + (side * math.cos(a) + up * math.sin(a)) * r)
                      for a in (2 * math.pi * k / SIDES for k in range(SIDES))])
    for a, b in zip(rings, rings[1:]):
        for k in range(SIDES):
            bm.faces.new((a[k], a[(k + 1) % SIDES], b[(k + 1) % SIDES], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm, [i for i in range(len(pts)) for _ in range(SIDES)]


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
    prof = back_profile(body)
    x0, y0, x1, y1 = a["patch_px"]
    tex = a["tex"]
    out = []
    for add in a["additions"]:
        pts, radii = SHAPES[add["shape"]](prof, add.get("params", {}))
        bm, rings = tube(pts, radii)
        me = bpy.data.meshes.new(add["name"])
        bm.to_mesh(me)
        bm.free()
        last = max(rings) or 1
        p, n, u0 = [], [], []
        for v in me.vertices:
            p += pdx(v.co)
            n += pdx(v.normal)
            t = rings[v.index] / last
            around = (v.index % SIDES) / SIDES
            u0 += [(x0 + 1 + (x1 - x0 - 2) * (0.1 + 0.8 * around)) / tex, (y0 + 1 + (y1 - y0 - 2) * t) / tex]
        tri = [i for poly in me.polygons for i in poly.vertices]
        out.append({"name": add["name"], "p": p, "n": n, "u0": u0, "tri": tri, "rings": rings})
    _pdx.write_result(a, {"additions": out})


main()
