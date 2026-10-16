"""Test 10 pre-check P2 (Blender only; throwaway): build the derived type1 body (vanilla body +
flared gradient tail + 1.5-unit tuft rigid on head, one shape, one material), export it with
io_pdx_mesh, then re-import the export in a clean scene and play a spear_shield anim on it.

The additions' UVs sit in the face texture's hair patch (pixels 105-115, 30-40 of 512; no
country-colour mask), so the body keeps vanilla's three textures. Nothing is written outside
``out_dir``; no dummy bones in the exported skeleton.

args: {"eu4", "out_dir", "patch_px": [x0, y0, x1, y1], "tex": 512, "result_path"}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))

import bpy  # noqa: E402

import _pdx  # noqa: E402
import spike_job as sj  # noqa: E402

ADDITIONS = (("tail_flared", "C"), ("pony_tuft15", "F"))


def import_plain(ie, path):
    before = set(bpy.data.objects)
    ie.import_meshfile(path)
    new = [o for o in bpy.data.objects if o not in before]
    return next(o for o in new if o.type == "ARMATURE"), next(o for o in new if o.type == "MESH"), new


def set_patch_uvs(obj, uv_name, patch, tex, flip_v):
    """Every loop of ``obj`` gets a UV inside the patch, spread along the part's length (rings)."""
    x0, y0, x1, y1 = patch
    me = obj.data
    layer = me.uv_layers.get(uv_name) or me.uv_layers.new(name=uv_name)
    rings = list(obj["ring_of"])
    n = max(rings) or 1
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            t = rings[vi] / n
            px = x0 + 1 + (x1 - x0 - 2) * 0.5 + ((vi % sj.SIDES) / sj.SIDES - 0.5) * (x1 - x0 - 2) * 0.8
            py = y0 + 1 + (y1 - y0 - 2) * t
            u, v = px / tex, py / tex
            layer.data[li].uv = (u, 1 - v) if flip_v else (u, v)


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    units = os.path.join(a["eu4"], "gfx", "models", "Units")
    out = {}
    _pdx.clear_scene()
    rig, body, _ = import_plain(ie, os.path.join(units, sj.BODY + ".mesh"))
    out["vanilla_bones"] = [b.name for b in rig.data.bones]
    uv_name = body.data.uv_layers[0].name
    out["uv_layers"] = [l.name for l in body.data.uv_layers]
    # Blender's v vs the file's: compare one vanilla vertex's Blender UV with what the app reads
    # from the file (done outside); record a sample here
    loop0 = body.data.loops[0]
    out["uv_sample"] = {"vertex_co": list(body.data.vertices[loop0.vertex_index].co),
                        "uv": list(body.data.uv_layers[0].data[0].uv)}
    prof = sj.back_profile(body)
    parts = []
    for part, var in ADDITIONS:
        pts, radii = sj.centreline(part, prof)
        o = sj.tube(part, pts, radii)
        sj.weight(o, rig, part, var)
        set_patch_uvs(o, uv_name, a["patch_px"], a["tex"], flip_v=a.get("flip_v", True))
        parts.append(o)
        out[f"{part}_verts"] = len(o.data.vertices)
    # one object, one material: join the additions into the body (vertex groups merge by name)
    for o in parts:
        for m in list(o.modifiers):
            o.modifiers.remove(m)
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    out["joined_verts"] = len(body.data.vertices)
    out["joined_materials"] = [m.name for m in body.data.materials]
    out["joined_groups"] = sorted(g.name for g in body.vertex_groups)
    path = os.path.join(a["out_dir"], "hr_p2_derived_body.mesh")
    ie.export_meshfile(path, exp_mesh=True, exp_skel=True, exp_locs=True)
    out["exported"] = path
    # re-import the export alone and play an anim on it
    _pdx.clear_scene()
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)
    rig2, body2, new = import_plain(ie, path)
    out["reimport"] = {"objects": sorted(f"{o.name}:{o.type}" for o in new), "verts": len(body2.data.vertices),
                       "bones": [b.name for b in rig2.data.bones],
                       "groups": sorted(g.name for g in body2.vertex_groups),
                       "materials": [m.name for m in body2.data.materials]}
    bpy.context.view_layer.objects.active = rig2
    rig2.select_set(True)
    sj.import_anim(ie, rig2, os.path.join(units, "animation", "spear_shield_attack.anim"))
    scene = bpy.context.scene
    scene.frame_set((scene.frame_start + scene.frame_end) // 2)
    dg = bpy.context.evaluated_depsgraph_get()
    co, _ = sj.evaluated(body2, dg)
    out["reimport"]["anim_frames"] = [scene.frame_start, scene.frame_end]
    out["reimport"]["posed_bbox"] = [[min(c[i] for c in co) for i in range(3)], [max(c[i] for c in co) for i in range(3)]]
    # a render of the re-imported, posed, textured body (textures from the install's folder)
    sh = scene.display.shading
    scene.render.engine = "BLENDER_WORKBENCH"
    sh.light, sh.color_type = "STUDIO", "TEXTURE"
    world = scene.world or bpy.data.worlds.new("p2")
    scene.world = world
    world.color = (0.82, 0.84, 0.86)
    cam = bpy.data.objects.new("p2_cam", bpy.data.cameras.new("p2_cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    scene.camera = cam
    for img in bpy.data.images:  # textures resolve beside the .mesh; read vanilla's in place for the render
        name = os.path.basename(img.filepath.replace("//", ""))
        if name and os.path.isfile(os.path.join(units, name)):
            img.filepath = os.path.join(units, name)
            img.reload()
    rig2.hide_render = True
    for o in new:
        if o.type == "EMPTY":
            o.hide_render = True
    lo, hi = sj.bounds([body2], evaluated_pts=True)
    for view, d in (("side", (-1, 0, 0)), ("rear", (0, -1, 0)), ("rear_quarter", (0.6, -0.7, -0.35))):
        sj.frame_cam(cam, d, lo, hi)
        sj.render(scene, os.path.join(a["out_dir"], f"p2_reimport_{view}.png"), 420)
    _pdx.write_result(a, out)


if __name__ == "__main__":
    main()
