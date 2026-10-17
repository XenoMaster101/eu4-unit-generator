"""Blender job: render whole units posed, for the gallery (SPEC Track F, F9; runs in Blender only).

A unit is a tree of nodes (export.gallery.stage): a .mesh (with its DDS textures beside it), the
.anim its state plays (or none: rigid parts), its pdxmesh scale, and children attached to its
bones. Every rig plays its animation; at ``frame`` each child goes where its parent's bone is,
as test 11's posed renders placed parts: the bone's posed matrix times the offset from the bone's
rest to the bind pose the .mesh records (io_pdx_mesh maps PDX (x, y, z) to Blender (x, z, y),
the mirror S). An attached entity keeps its own pdxmesh scale, not times its parent's
(``inherit_scale`` false, the default): parts at 0.5, like vanilla weapons, sat right on 0.5
bodies in in-game tests 7-8, and a 0.5 rider on a 0.48 horse looks like vanilla's riders.

args: {"units": [{"name": ..., "root": node}], "frame": 1, "view": [x, y, z] (camera -> scene),
       "size": px, "inherit_scale": false, "out_dir": ..., "result_path": ...}
node: {"name", "mesh", "anim": path | null, "scale": float, "hidden": bool (a helper mesh: placed, not drawn),
       "attach": [{"bone", "bind": [12] | null, "node"}]}
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402
import check_deform as cd  # noqa: E402

S = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
_scope = {"rig": None}  # the one rig io_pdx_mesh may find by bone name (None: none, build a new one)


def scope_rig_lookup(ie):
    """io_pdx_mesh reuses any armature in the file that shares a bone name with what it imports
    (get_rig_from_bone_name). A unit holds several skeletons (horse and rider share names), so a
    mesh import must always build its own rig, and an animation must go to the rig we name."""
    def lookup(name):
        rig = _scope["rig"]
        return rig if rig is not None and name in rig.data.bones else None
    ie.get_rig_from_bone_name = lookup


def load(ie, node, out):
    """Import ``node``'s tree; ``out`` collects (node, objects, rig) depth first."""
    before = set(bpy.data.objects)
    _scope["rig"] = None
    ie.import_meshfile(node["mesh"])
    objs = [o for o in bpy.data.objects if o not in before]
    rig = next((o for o in objs if o.type == "ARMATURE"), None)
    if rig is not None and node.get("anim"):
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        _scope["rig"] = rig
        try:
            cd.import_anim(ie, rig, node["anim"])
        finally:
            _scope["rig"] = None
    out.append((node, objs, rig))
    for a in node.get("attach", []):
        load(ie, a["node"], out)


def place(node, world, loaded, inherit):
    """Put ``node``'s root objects at ``world`` (scaled by its pdxmesh scale), then its children
    on its bones as posed at the current frame."""
    entry = next(e for e in loaded if e[0] is node)
    _, objs, rig = entry
    m = world @ Matrix.Scale(node.get("scale", 1.0), 4)
    for o in objs:
        if o.parent is None:
            o.matrix_world = m
    bpy.context.view_layer.update()
    for a in node.get("attach", []):
        child = a["node"]
        if rig is None or a["bone"] not in rig.pose.bones:
            print(f"render_unit: {node['name']} has no bone {a['bone']}; {child['name']} left at its parent's origin")
            place(child, m if inherit else world, loaded, inherit)
            continue
        posed = rig.matrix_world @ rig.pose.bones[a["bone"]].matrix
        offset = Matrix.Identity(4)
        if a.get("bind"):
            b = a["bind"]
            rest_local = S @ Matrix([b[0:4], b[4:8], b[8:12], [0, 0, 0, 1]]) @ S
            offset = rig.data.bones[a["bone"]].matrix_local.inverted() @ rest_local
        at = posed @ offset
        if not inherit:  # take the parent's scale back out
            loc, rot, _ = at.decompose()
            at = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        place(child, at, loaded, inherit)


def points(loaded):
    co = []
    for node, objs, _ in loaded:
        for o in objs:
            if o.type != "MESH" or node.get("hidden"):
                continue
            if o.parent is not None and o.parent.type == "ARMATURE":
                co += cd.evaluated(o)
            else:
                co += [o.matrix_world @ Vector(c) for c in o.bound_box]
    return co


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    scope_rig_lookup(ie)
    out = {"tiles": {}, "seconds": {}}
    for unit in a["units"]:
        start = time.time()
        _pdx.clear_scene()
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_WORKBENCH"
        sh = scene.display.shading
        sh.light, sh.color_type, sh.show_backface_culling, sh.show_cavity = "STUDIO", "TEXTURE", False, True
        world_ = scene.world or bpy.data.worlds.new("gallery")
        scene.world = world_
        world_.color = tuple(a.get("background", (0.8, 0.8, 0.8)))
        cam = bpy.data.objects.new("gallery_cam", bpy.data.cameras.new("gallery_cam"))
        scene.collection.objects.link(cam)
        cam.data.type = "ORTHO"
        scene.camera = cam
        loaded = []
        load(ie, unit["root"], loaded)
        scene.frame_set(a.get("frame", 1))
        place(unit["root"], Matrix.Identity(4), loaded, a.get("inherit_scale", False))
        bpy.context.view_layer.update()
        for node, objs, _ in loaded:
            for o in objs:
                if o.type != "MESH" or node.get("hidden"):
                    o.hide_render = True
        path = os.path.join(a["out_dir"], f"{unit['name']}.png")
        out["tiles"][unit["name"]] = cd.shoot(scene, cam, points(loaded), a.get("view", (-0.7, 0.7, -0.35)), path,
                                              a.get("size", 256))
        out["seconds"][unit["name"]] = round(time.time() - start, 2)
    _pdx.write_result(a, out)


main()
