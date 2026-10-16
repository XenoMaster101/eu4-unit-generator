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
