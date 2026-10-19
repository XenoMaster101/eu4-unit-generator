"""Blender job (Track C, test 15 round 3, step 4): horse proportions against vanilla's, and
silhouette overlays.

args: {"vanilla": {"p", "tri", "bones"} (core.fresh_mesh.base_arrays),
       "meshes": {label: path} (a fresh_horse geometry .json, PDX space; or an .obj in preview axes
       (PDX (x, y, z) = Blender (x, z, y)) as Blender's OBJ exporter writes them: file (x, y, -z)),
       "limits": {metric: ratio} | null (overrides _horse_measure.LIMITS),
       "rider": {"path": render_unit dump, "node": name part (default "rider"), "seat_y"} | null (the
       posed rider in horse space, through Saddle_Node: how much of its lower legs each horse hides),
       "overlays": [{"name", "draw": [[label | "vanilla", [r, g, b, a]], ...], "view": "side" | "front" | "top"}],
       "out_dir", "size": px, "result_path"}
Result: {"vanilla": proportions, "meshes": {label: {"proportions", "compare", "flags": [metric]}},
         "overlays": {name: path}}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _horse_measure as hm  # noqa: E402
import _pdx  # noqa: E402

# camera direction (camera -> scene) and up, PDX space: side = from the horse's left (+X)
VIEWS = {"side": ((-1, 0, 0), (0, 1, 0)), "front": ((0, 0, 1), (0, 1, 0)), "top": ((0, -1, 0), (0, 0, -1))}


def load(path):
    """(positions n x 3, triangles m x 3) in PDX space."""
    if path.lower().endswith(".json"):
        d = json.loads(open(path, encoding="utf-8").read())
        return np.array(d["p"], dtype=np.float64).reshape(-1, 3), np.array(d["tri"], dtype=np.int64).reshape(-1, 3)
    co, tris = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            co.append((x, y, -z))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return np.array(co, dtype=np.float64), np.array(tris, dtype=np.int64)


def mesh_object(name, co, tris, rgba):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in co], [], [tuple(t) for t in tris])
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    me.materials.append(mat)
    return obj


def render(scene, cam, view, out, size, centre, extent):
    d, up = (Vector(v) for v in VIEWS[view])
    cam.location = Vector(centre) - d * 60
    # explicit basis: to_track_quat's up is Blender's Z, but these meshes are in PDX space (Y up)
    x = up.cross(-d).normalized()
    cam.rotation_euler = Matrix((x, up, -d)).transposed().to_euler()
    cam.data.ortho_scale = extent
    scene.render.resolution_x, scene.render.resolution_y = size, size
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    return out


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    bones = van["bones"]
    vco = np.array(van["p"], dtype=np.float64).reshape(-1, 3)
    vtri = np.array(van["tri"], dtype=np.int64).reshape(-1, 3)
    legs = None
    if a.get("rider"):
        rd = json.loads(open(a["rider"]["path"], encoding="utf-8").read())
        node = next(k for k in rd if a["rider"].get("node", "rider") in k)
        legs = hm.rider_legs(np.array(rd[node]["p"]).reshape(-1, 3), float(a["rider"].get("seat_y", 12.36)))
    vp = hm.proportions(vco, vtri, bones)
    if legs is not None:
        vp["rider_leg"] = hm.visible_from_side(vco, vtri, legs)
    res = {"vanilla": vp, "meshes": {}, "overlays": {}}
    loaded = {"vanilla": (vco, vtri)}
    for label, path in a["meshes"].items():
        co, tris = load(path)
        loaded[label] = (co, tris)
        p = hm.proportions(co, tris, bones)
        if legs is not None:
            p["rider_leg"] = hm.visible_from_side(co, tris, legs)
        cmp_ = hm.compare(p, vp, a.get("limits"))
        res["meshes"][label] = {"proportions": p, "compare": cmp_, "flags": [k for k, c in cmp_.items() if c["flag"]]}
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light, sh.color_type, sh.show_xray, sh.xray_alpha = "FLAT", "MATERIAL", True, 0.55
    scene.world = scene.world or bpy.data.worlds.new("w")
    scene.world.color = (1, 1, 1)
    scene.view_settings.view_transform = "Standard"  # the background as given, not Filmic grey
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    cam.data.type, cam.data.clip_end = "ORTHO", 200
    scene.camera = cam
    for ov in a.get("overlays", []):
        for o in list(bpy.data.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        pts = []
        for label, rgba in ov["draw"]:
            co, tris = loaded[label]
            mesh_object(label, co, tris, tuple(rgba))
            pts.append(co)
        allp = np.vstack(pts)
        centre = (allp.min(axis=0) + allp.max(axis=0)) / 2
        extent = float(np.ptp(allp, axis=0).max()) * 1.08
        path = os.path.join(a["out_dir"], f"{ov['name']}.png")
        res["overlays"][ov["name"]] = render(scene, cam, ov.get("view", "side"), path, a.get("size", 700),
                                             centre.tolist(), extent)
    _pdx.write_result(a, res)


main()
