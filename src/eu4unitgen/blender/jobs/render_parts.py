"""Blender job: render a fit preview, a body with its parts on their bones, and part thumbnails
(runs in Blender only).

Every mesh is imported with io_pdx_mesh from a temporary folder the app fills (the body's
.mesh and the parts' .mesh files, each beside DDS textures named as its material names them).
The body stays in its bind pose (T-pose). A part goes where the engine puts it at rest: its
bone's bind pose ``M`` (PDX mesh space, 4x4 row-major). io_pdx_mesh maps PDX (x, y, z) to
Blender (x, z, y), a mirror S, so a part's Blender matrix is S M S.

The unit faces PDX -Z = Blender -Y: the "front" camera looks from -Y, the "side" camera from +X
(the unit's left, the shield side). Workbench, flat studio light, textures.

args: {"body": .mesh or null, "parts": [{"mesh", "matrix": [16], "name"}], "views": ["front", "side"],
       "size": px, "thumbs": bool, "out_dir": ..., "result_path": ...}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402

S = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def import_mesh(ie, path):
    before = set(bpy.data.objects)
    ie.import_meshfile(path)
    return [o for o in bpy.data.objects if o not in before]


def bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def setup_scene(size):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = False
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE"
    shading.show_backface_culling = False
    shading.show_cavity = True
    world = scene.world or bpy.data.worlds.new("preview")
    scene.world = world
    world.color = (0.16, 0.16, 0.17)
    cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    scene.camera = cam
    return scene, cam


def render(scene, cam, objs, direction, path):
    """Orthographic render of ``objs`` from ``direction`` (camera -> scene, Blender axes)."""
    lo, hi = bounds(objs)
    centre, d = (lo + hi) / 2, Vector(direction).normalized()
    extent = (hi - lo).length
    cam.location = centre - d * (extent * 2 + 10)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()  # a camera looks down -Z, up is +Y
    cam.data.clip_end = extent * 4 + 40
    # frame: the box's spread across the view plane
    right, up = d.cross(Vector((0, 0, 1))), Vector((0, 0, 1))
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    right.normalize()
    up = right.cross(d).normalized()
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    w = max(abs((c - centre).dot(right)) for c in corners) * 2
    h = max(abs((c - centre).dot(up)) for c in corners) * 2
    cam.data.ortho_scale = max(w, h, 0.5) * 1.08
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


VIEWS = {"front": (0, 1, 0), "side": (-1, 0, 0), "three_quarter": (-0.7, 0.7, -0.35)}


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    _pdx.clear_scene()
    scene, cam = setup_scene(a.get("size", 360))
    body_objs = import_mesh(ie, a["body"]) if a.get("body") else []
    for o in body_objs:
        if o.type == "ARMATURE":
            o.hide_render = True
    parts = []
    for p in a.get("parts", []):
        objs = import_mesh(ie, p["mesh"])
        m = Matrix([p["matrix"][i * 4:(i + 1) * 4] for i in range(4)])
        for o in objs:
            if o.parent is None:
                o.matrix_world = S @ m @ S
        parts.append((p["name"], objs))
    everything = body_objs + [o for _, objs in parts for o in objs]
    out = {"views": {}, "thumbs": {}, "imported": {name: [f"{o.name}:{o.type}" for o in objs] for name, objs in parts}}
    for view in a.get("views", ["front", "side"]):
        out["views"][view] = render(scene, cam, everything, VIEWS[view], os.path.join(a["out_dir"], f"view_{view}.png"))
    if a.get("thumbs"):
        for o in body_objs:
            o.hide_render = True
        for name, objs in parts:
            for other, oo in parts:
                for o in oo:
                    o.hide_render = other != name
            out["thumbs"][name] = render(scene, cam, objs, VIEWS["three_quarter"],
                                         os.path.join(a["out_dir"], f"thumb_{name}.png"))
    _pdx.write_result(a, out)


main()
