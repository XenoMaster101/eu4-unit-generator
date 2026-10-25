"""P2 diagnosis (Blender only): export the UNMODIFIED vanilla type1 body (import -> export, no
edits) to see whether the exporter splits every face corner by itself. Writes only to out_dir."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))
import bpy  # noqa: E402
import _pdx  # noqa: E402
a = _pdx.args(); ie = _pdx.io_module(); _pdx.clear_scene()
ie.import_meshfile(os.path.join(a["eu4"], "gfx", "models", "Units", "western_generic_01_type1.mesh"))
body = next(o for o in bpy.data.objects if o.type == "MESH")
me = body.data
info = {"verts": len(me.vertices), "has_custom_normals": me.has_custom_normals, "loops": len(me.loops)}
path = os.path.join(a["out_dir"], "hr_p2_vanilla_roundtrip.mesh")
ie.export_meshfile(path, exp_mesh=True, exp_skel=True, exp_locs=True)
info["exported"] = path
_pdx.write_result(a, info)
