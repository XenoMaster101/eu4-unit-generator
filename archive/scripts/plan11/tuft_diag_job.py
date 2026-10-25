"""Plan 11 diagnosis (Blender only): at one frame, which addition vertices the deformation check
counts as inside the base body, and the nearest base face's main bone. Writes nothing but the result."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src", "eu4unitgen", "blender", "jobs"))
import bpy  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
import _pdx  # noqa: E402
import check_deform  # noqa: E402
cd = vars(check_deform)
a = _pdx.args(); ie = _pdx.io_module(); _pdx.clear_scene()
rig, body, new = cd["import_body"](ie, a["body"])
bpy.context.view_layer.objects.active = rig; rig.select_set(True)
cd["import_anim"](ie, rig, a["anim"])
out = {}
for frame in a["frames"]:
    bpy.context.scene.frame_set(frame)
    co = cd["evaluated"](body)
    n = a["base_vertices"]
    polys = [tuple(p.vertices) for p in body.data.polygons if all(v < n for v in p.vertices)]
    tree = BVHTree.FromPolygons(co, polys)
    groups = {g.index: g.name for g in body.vertex_groups}
    def main_group(v):
        gs = body.data.vertices[v].groups
        return groups[max(gs, key=lambda g: g.weight).group] if gs else None
    rows = []
    for i in range(a["start"], a["start"] + a["count"]):
        loc, nrm, fi, dist = tree.find_nearest(co[i])
        if (co[i] - loc).dot(nrm) < 0:
            face = polys[fi]
            rows.append({"v": i - a["start"], "ring": a["rings"][i - a["start"]], "depth": round(dist, 3),
                         "near": sorted({main_group(v) for v in face}), "z": round(co[i].z, 2)})
    out[str(frame)] = rows
_pdx.write_result(a, out)

# optional close-up renders at the given frames (head box)
if a.get("render"):
    from mathutils import Vector  # noqa: E402
    img = bpy.data.images.load(a["diffuse"], check_existing=True)
    mat = bpy.data.materials.new("check"); mat.use_nodes = True
    nodes = mat.node_tree.nodes; bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage"); tex.image = img
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"]); nodes.active = tex
    body.data.materials.clear(); body.data.materials.append(mat)
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading; sh.light, sh.color_type, sh.show_backface_culling = "STUDIO", "TEXTURE", False
    world = sc.world or bpy.data.worlds.new("w"); sc.world = world; world.color = (0.82, 0.84, 0.86)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); cam.data.type = "ORTHO"; sc.camera = cam
    for o in new:
        if o.type != "MESH": o.hide_render = True
    out["renders"] = []
    for frame in a["frames"]:
        sc.frame_set(frame)
        co = cd["evaluated"](body)
        head = [co[i] for i in range(a["start"], a["start"] + a["count"])]
        c = sum(head, Vector()) / len(head)
        for view, d in (("side", (-1, 0, 0)), ("rear", (0, -1, 0)), ("top", (0, 0.001, -1))):
            out["renders"].append(cd["shoot"](sc, cam, [c + Vector((-1.8, -1.8, -1.8)), c + Vector((1.8, 1.8, 1.8))], d,
                                              os.path.join(a["out_dir_override"], f"tuftdiag_f{frame}_{view}.png"), 300))
_pdx.write_result(a, out)
