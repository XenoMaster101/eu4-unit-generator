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
       "size": px, "camera": null | {fixed framing, see shoot_fixed}, "inherit_scale": false, "thumbs": px or 0 (each attached rigid part alone, by entity
       name, for the Parts cards), "dump": bool (each unit's posed meshes as JSON in the root mesh's PDX space:
       {node name: {"p", "tri"}}, for measuring e.g. where a rider's legs hang), "out_dir": ..., "result_path": ...}
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


def skinned(o) -> bool:
    """A mesh deformed by a rig (io_pdx_mesh skins with an Armature modifier, not always a parent)."""
    return (o.parent is not None and o.parent.type == "ARMATURE") or any(m.type == "ARMATURE" for m in o.modifiers)


def frame(loaded, pad=0.12):
    """What the camera frames: the animated bodies (soldier, horse, crew) with a margin, so a
    6 m sarissa or a lance doesn't shrink the soldier to a speck (it runs off the edge instead).
    A unit with no animated body is framed whole."""
    co = []
    for node, objs, _ in loaded:
        for o in objs:
            if o.type == "MESH" and not node.get("hidden") and skinned(o):
                co += cd.evaluated(o)
    if not co:
        return points(loaded)
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    m = (hi - lo) * pad
    lo, hi = lo - m, hi + m
    return [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]


def points(loaded):
    co = []
    for node, objs, _ in loaded:
        for o in objs:
            if o.type != "MESH" or node.get("hidden"):
                continue
            if skinned(o):
                co += cd.evaluated(o)
            else:
                co += [o.matrix_world @ Vector(c) for c in o.bound_box]
    return co


def dump_posed(unit, loaded, path, bone=None):
    """The posed, drawn meshes in the root mesh's own space (PDX axes, its pdxmesh scale taken out);
    with ``bone``, as if that bone of the root's rig were at rest (the animation can turn the root)."""
    import json
    inv = (S @ Matrix.Scale(unit["root"].get("scale", 1.0), 4)).inverted()  # S is its own inverse
    rig = loaded[0][2]
    if bone and rig is not None and bone in rig.pose.bones:
        rest = rig.matrix_world @ rig.data.bones[bone].matrix_local
        posed = rig.matrix_world @ rig.pose.bones[bone].matrix
        inv = inv @ rest @ posed.inverted()
    out = {}
    for node, objs, _ in loaded:
        if node.get("hidden"):
            continue
        p, tri = [], []
        for o in objs:
            if o.type != "MESH":
                continue
            ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            me.calc_loop_triangles()
            base = len(p) // 3
            for v in me.vertices:
                p += list(inv @ (o.matrix_world @ v.co))
            tri += [base + i for lt in me.loop_triangles for i in lt.vertices]
            ev.to_mesh_clear()
        if p:
            out[node["name"]] = {"p": p, "tri": tri}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return path


def shoot_fixed(sc, cam, c, path):
    """One framing for every unit (not fitted to each unit's bounds), so units render at the same
    scale and can be compared with each other and with an in-game screenshot at a known zoom.
    ``c``: {"view": camera -> scene, "target": the point at the image centre, "ortho_scale": scene
    units across the image's wider side, "width", "height", "shadow": bool}."""
    d = Vector(c["view"]).normalized()
    target = Vector(c.get("target", (0, 0, 0)))
    cam.location = target - d * 40
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = 100
    cam.data.ortho_scale = c["ortho_scale"]
    sc.display.shading.show_shadows = bool(c.get("shadow", False))
    sc.render.resolution_x, sc.render.resolution_y = c["width"], c["height"]
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def main() -> None:
    a = _pdx.args()
    ie = _pdx.io_module()
    scope_rig_lookup(ie)
    out = {"tiles": {}, "seconds": {}, "thumbs": {}, "dumps": {}}
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
        if a.get("dump"):
            bone = a["dump"].get("bone") if isinstance(a["dump"], dict) else None
            out["dumps"][unit["name"]] = dump_posed(unit, loaded, os.path.join(a["out_dir"], f"{unit['name']}_posed.json"), bone)
        if a.get("camera"):
            out["tiles"][unit["name"]] = shoot_fixed(scene, cam, a["camera"], path)
        else:
            out["tiles"][unit["name"]] = cd.shoot(scene, cam, frame(loaded), a.get("view", (-0.7, 0.7, -0.35)),
                                                  path, a.get("size", 256))
        if a.get("thumbs"):  # each rigid part alone, as posed, for the Parts cards
            rigid = [(n, objs) for n, objs, rig in loaded if n is not unit["root"] and not n.get("anim")
                     and not n.get("hidden") and n["name"] not in out["thumbs"]]
            for n, objs in rigid:
                for other, oo, _ in loaded:
                    for o in oo:
                        o.hide_render = o.type != "MESH" or other is not n
                co = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
                if co:
                    out["thumbs"][n["name"]] = cd.shoot(scene, cam, co, a.get("view", (-0.7, 0.7, -0.35)),
                                                        os.path.join(a["out_dir"], f"thumb_{n['name']}.png"),
                                                        a["thumbs"])
        out["seconds"][unit["name"]] = round(time.time() - start, 2)
    _pdx.write_result(a, out)


main()
