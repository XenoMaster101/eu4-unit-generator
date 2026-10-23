"""Blender job (M21): several textured OBJs (preview axes, the same frame) rendered through one fixed
orthographic camera per view, so their silhouettes line up pixel for pixel. Workbench, studio light,
each model's own texture, transparent background.

args: {"models": {name: {"obj", "texture"}}, "views": {view: [azimuth deg, elevation deg]} (azimuth 90:
       from +X, the horse's left), "size": px, "out_dir", "result_path"}
Result: {name: {view: png}}; the camera fits the union of all models' bounds.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def load(path, texture):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path)
    obj = next(o for o in bpy.data.objects if o not in before and o.type == "MESH")
    obj.data.shade_smooth()  # as the game shades it (the low mesh's normals are smooth)
    mat = bpy.data.materials.new(os.path.basename(path))
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture)
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.nodes.active = tex
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    objs = {k: load(m["obj"], m["texture"]) for k, m in a["models"].items()}
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in objs.values() for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre, radius = (lo + hi) / 2, (hi - lo).length / 2
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = int(a.get("size", 900))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type, cam.data.ortho_scale = "ORTHO", 2.1 * radius
    out = {k: {} for k in objs}
    for view, (az, el) in a["views"].items():
        az, el = math.radians(az), math.radians(el)
        d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        cam.location = centre + d * 4 * radius
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        for k, o in objs.items():
            for other in objs.values():
                other.hide_render = other is not o
            p = os.path.join(a["out_dir"], f"{k}_{view}.png")
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True)
            out[k][view] = p
    _pdx.write_result(a, out)


main()
