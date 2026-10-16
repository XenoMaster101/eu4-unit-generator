"""Q0 (Blender only): head close-ups of a derived body in the bind pose, textured with the base
game's diffuse (read in place), optionally with a helmet part at its bone's bind pose. Writes
only to out_dir. args: {"body", "diffuse", "helmet": {"mesh", "bind": [12]} | null, "views",
"tag", "size", "out_dir", "result_path"}"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import _pdx  # noqa: E402
import check_deform as cd  # noqa: E402
S = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
VIEWS = {"front": (0, 1, 0), "side": (-1, 0, 0), "rear_quarter": (0.6, -0.7, -0.35), "three_quarter": (-0.6, 0.7, -0.3)}
a = _pdx.args(); ie = _pdx.io_module(); _pdx.clear_scene()
rig, body, new = cd.import_body(ie, a["body"])
sc, cam = cd.setup_render({"renders": {"diffuse": a["diffuse"]}}, body, new)
if a.get("helmet"):
    before = set(bpy.data.objects); ie.import_meshfile(a["helmet"]["mesh"])
    bind = a["helmet"]["bind"]; m = Matrix([bind[0:4], bind[4:8], bind[8:12], [0, 0, 0, 1]])
    for o in [o for o in bpy.data.objects if o not in before]:
        if o.parent is None: o.matrix_world = S @ m @ S
head = (rig.matrix_world @ rig.data.bones["head"].matrix_local).translation
box = [head + Vector((-1.9, -1.9, -1.2)), head + Vector((1.9, 1.9, 2.9))]
out = {v: cd.shoot(sc, cam, box, VIEWS[v], os.path.join(a["out_dir"], f"{a['tag']}_{v}.png"), a.get("size", 280))
       for v in a.get("views", ["front", "side", "rear_quarter"])}
_pdx.write_result(a, out)
