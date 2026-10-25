"""Plan 11 research (Blender only): render a body's head close up (front, side, rear, top,
rear-quarter) in the bind pose, textured with the install's diffuse read in place. Writes only to out_dir.
args: {"mesh", "diffuse", "out_dir", "tag", "size", "result_path"}"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "spike_r0"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import _pdx  # noqa: E402
import spike_job as sj  # noqa: E402
a = _pdx.args(); ie = _pdx.io_module(); _pdx.clear_scene()
before = set(bpy.data.objects); ie.import_meshfile(a["mesh"])
new = [o for o in bpy.data.objects if o not in before]
body = next(o for o in new if o.type == "MESH")
img = bpy.data.images.load(a["diffuse"], check_existing=True)
mat = bpy.data.materials.new("check"); mat.use_nodes = True
nodes = mat.node_tree.nodes; bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
tex = nodes.new("ShaderNodeTexImage"); tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"]); nodes.active = tex
body.data.materials.clear(); body.data.materials.append(mat)
sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading; sh.light, sh.color_type = "STUDIO", "TEXTURE"; sh.show_backface_culling = False
world = sc.world or bpy.data.worlds.new("w"); sc.world = world; world.color = (0.82, 0.84, 0.86)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); cam.data.type = "ORTHO"; sc.camera = cam
for o in new:
    if o.type != "MESH": o.hide_render = True
box = a.get("box") or [[-1.6, -1.8, 12.9], [1.6, 1.8, 17.0]]  # default: the head (Blender axes; the unit faces -Y)
lo, hi = Vector(box[0]), Vector(box[1])
out = {}
for view, d in (("front", (0, 1, 0)), ("side", (-1, 0, 0)), ("rear", (0, -1, 0)), ("top", (0, 0.001, -1)),
                ("rear_quarter", (0.6, -0.7, -0.4))):
    sj.frame_cam(cam, d, lo, hi)
    out[view] = sj.render(sc, os.path.join(a["out_dir"], f"{a['tag']}_{view}.png"), a.get("size", 300))
_pdx.write_result(a, out)
