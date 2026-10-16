"""P2 re-check (Blender only): re-import the derived body written by core.derived_body, report what
Blender sees, play spear_shield_attack on it and measure the additions per frame (R0 metrics).
Optional: textured renders (step 4). Writes only to out_dir."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))
import bpy  # noqa: E402
import _pdx  # noqa: E402
import spike_job as sj  # noqa: E402

a = _pdx.args(); ie = _pdx.io_module(); _pdx.clear_scene()
units = os.path.join(a["eu4"], "gfx", "models", "Units")
before = set(bpy.data.objects)
ie.import_meshfile(a["mesh"])
new = [o for o in bpy.data.objects if o not in before]
rig = next(o for o in new if o.type == "ARMATURE"); body = next(o for o in new if o.type == "MESH")
out = {"objects": sorted(f"{o.name}:{o.type}" for o in new), "verts": len(body.data.vertices),
       "polys": len(body.data.polygons), "bones": [b.name for b in rig.data.bones],
       "materials": [m.name for m in body.data.materials]}
bpy.context.view_layer.objects.active = rig; rig.select_set(True)
sj.import_anim(ie, rig, os.path.join(units, "animation", "spear_shield_attack.anim"))
sc = bpy.context.scene
out["anim_frames"] = [sc.frame_start, sc.frame_end]
# the added vertices are the last ones (indices >= base count); their positions must move with the anim
base_n = a["base_vertices"]
rest = [body.matrix_world @ v.co for v in body.data.vertices]
moved = 0
for f in (sc.frame_start, (sc.frame_start + sc.frame_end) // 2, sc.frame_end):
    sc.frame_set(f)
    co, _ = sj.evaluated(body, bpy.context.evaluated_depsgraph_get())
    moved = max(moved, max((co[i] - rest[i]).length for i in range(base_n, len(co))))
out["added_max_move"] = moved
_pdx.write_result(a, out)

# ---- step 4: textured check render. io_pdx_mesh resolves textures beside the .mesh (our build
# folder has none), so build a plain material from the install's diffuse, read in place.
if a.get("render"):
    diff_path = os.path.join(units, "western_generic_01_type1_diffuse.dds")
    img = bpy.data.images.load(diff_path, check_existing=True)
    out["texture"] = {"path": diff_path, "size": list(img.size), "has_data": bool(img.has_data or img.pixels[:4])}
    mat = bpy.data.materials.new("p2b_check")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage"); tex.image = img
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nodes.active = tex
    body.data.materials.clear(); body.data.materials.append(mat)
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "TEXTURE"
    world = sc.world or bpy.data.worlds.new("p2b"); sc.world = world; world.color = (0.82, 0.84, 0.86)
    cam = bpy.data.objects.new("p2b_cam", bpy.data.cameras.new("p2b_cam")); sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"; sc.camera = cam
    for o in new:
        if o.type != "MESH":
            o.hide_render = True
    out["renders"] = []
    for f, tag in ((sc.frame_start, "f1"), ((sc.frame_start + sc.frame_end) // 2, "mid")):
        sc.frame_set(f)
        lo, hi = sj.bounds([body], evaluated_pts=True)
        for view, d in (("side", (-1, 0, 0)), ("rear", (0, -1, 0)), ("rear_quarter", (0.6, -0.7, -0.35))):
            sj.frame_cam(cam, d, lo, hi)
            out["renders"].append(sj.render(sc, os.path.join(a["out_dir_override"], f"p2b_{tag}_{view}.png"), 420))
    _pdx.write_result(a, out)
