"""Blender job (Track C, test 15 round 3): a fresh horse's normal map and colour baked with Cycles
from its high-detail source onto the final low mesh.

PDX mesh space throughout (as fresh_horse), so the tangent frame Blender bakes in is the one
core.fresh_mesh.write_fresh writes (``ta`` from the same positions, normals and UVs). Blender's
default bake axes are the base game's convention (green up the image; SPEC Q3).
  * low: the fresh_horse geometry JSON (positions, the exported normals as custom normals, UVs
    flipped back to Blender's v-up), triangles with their ``part``;
  * high: the source surface after every edit (fresh_horse's ``4_trimmed`` debug stage: the fine
    voxel solid, warped) plus the low mesh's own added parts (chanfron, panels, stirrups: part
    >= 1), so rays from those find themselves and bake flat instead of the body behind them;
  * normal: tangent space, selected to active, rays from ``cage`` outside, ``distance`` deep;
    our own parts' texels flat afterwards (they have no detail source);
  * ao: ambient occlusion of the low mesh itself (the panels darken the body under them);
  * colour: the painted diffuse times (1 - ao_strength + ao_strength x ao); with ``paint`` (true or _paint's
    settings) then the game-painted pass (_paint: contrast per material, creases, material borders, light from above).
Reports each map's path, the normal map's mean tilt from flat (base game 4-20 deg) on the texels
the UVs cover, and how much of the low surface found the high one.

Match check (before baking): the low body (part 0) and the high surface must be the same shape
up to the reduction; every reshape of one has to be applied to the other (fresh_horse's
``bake_source_out``). Each low body face is sampled (centre and its three mid-edge points) and the
nearest high point found: faces further than ``mismatch`` are flagged, grouped into connected
areas (centre, size, mean and worst gap, whether the low surface is above or below), and their
texels written as a mask. With ``mismatch_flatten`` their texels bake flat.

args: {"low": geometry .json, "high": [.obj (preview axes, as fresh_horse's debug_dump writes)],
       "diffuse": painted .png, "size": px, "cage", "distance", "samples", "ao_strength",
       "preview": bool (lit EEVEE renders with and without the normal map),
       "high_smooth": iterations (the high surface smoothed before baking), "strength": the tangent
       normals' x and y scaled by it (towards flat) and renormalised, "outlier": degrees (steeper
       texels are rays that hit the wrong surface: flat),
       "preview_only": bool (M19: only the preview renders, from the given "diffuse" and "normal"; "high" unused),
       "mismatch": gap (default 0.3: about the coarse voxel, below it the reduction's own error) over which a low face is flagged, "mismatch_flatten": bool,
       "out_dir", "name", "result_path"}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _paint  # noqa: E402
import _pdx  # noqa: E402


def load_obj(path):
    """(positions, triangles) in PDX space from a preview-axes OBJ (file (x, y, -z), see
    horse_proportions.load)."""
    co, tris = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            co.append((x, y, -z))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return np.array(co, dtype=np.float64), np.array(tris, dtype=np.int64)


def mesh_object(name, co, tris):
    me = bpy.data.meshes.new(name)
    me.from_pydata(co.tolist(), [], tris.tolist())
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def low_object(geo):
    co = np.array(geo["p"], dtype=np.float64).reshape(-1, 3)
    tris = np.array(geo["tri"], dtype=np.int64).reshape(-1, 3)
    obj = mesh_object("low", co, tris)
    me = obj.data
    me.shade_smooth()
    uv = me.uv_layers.new(name="UV")
    u0 = np.array(geo["u0"], dtype=np.float64).reshape(-1, 2)
    loops = np.array([lp.vertex_index for lp in me.loops])
    uvs = np.stack([u0[loops, 0], 1.0 - u0[loops, 1]], axis=1)  # the JSON's image convention -> Blender's
    uv.data.foreach_set("uv", uvs.ravel())
    n = np.array(geo["n"], dtype=np.float64).reshape(-1, 3)
    me.normals_split_custom_set_from_vertices([tuple(v / max(np.linalg.norm(v), 1e-9)) for v in n])
    return obj, co, tris


def match_check(lco, ltri, body, high, limit):
    """Per low face (``body`` faces only): the gap to the high surface (module doc). -> (flagged
    per face, report)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons([tuple(v) for v in high[0]], [tuple(t) for t in high[1]])
    tri = lco[ltri]
    pts = np.stack([tri.mean(axis=1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
                    (tri[:, 2] + tri[:, 0]) / 2], axis=1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    gap = np.zeros(len(ltri))
    side = np.zeros(len(ltri))
    for f in np.flatnonzero(body):
        ds, ss = [], []
        for q in pts[f]:
            loc = bvh.find_nearest(Vector(q))[0]
            d = np.array(loc) - q if loc is not None else np.zeros(3)
            ds.append(np.linalg.norm(d) if loc is not None else 9.0)
            ss.append(-np.dot(d, fn[f]))  # > 0: the low surface is outside (above) the high one
        gap[f], side[f] = max(ds), np.mean(ss)
    flagged = body & (gap > limit)
    # connected areas of flagged faces (sharing a vertex)
    parent = list(range(len(ltri)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_vertex = {}
    for f in np.flatnonzero(flagged):
        for v in ltri[f]:
            if v in by_vertex:
                parent[root(f)] = root(by_vertex[v])
            else:
                by_vertex[v] = f
    groups = {}
    for f in np.flatnonzero(flagged):
        groups.setdefault(root(f), []).append(f)
    areas = []
    for fs in sorted(groups.values(), key=len, reverse=True):
        c = tri[fs].reshape(-1, 3)
        areas.append({"faces": len(fs), "centre": [round(float(x), 2) for x in c.mean(axis=0)],
                      "size": [round(float(x), 2) for x in np.ptp(c, axis=0)],
                      "gap_mean": round(float(gap[fs].mean()), 3), "gap_max": round(float(gap[fs].max()), 3),
                      "low_is": "above" if side[fs].mean() > 0 else "below"})
    report = {"limit": limit, "gap_mean": round(float(gap[body].mean()), 3),
              "gap_p95": round(float(np.percentile(gap[body], 95)), 3), "gap_max": round(float(gap[body].max()), 3),
              "flagged_faces": int(flagged.sum()), "areas": areas[:12], "ok": not flagged.any()}
    return flagged, report


def bake_target(obj, name, size):
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.nodes.active = node
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return img


def pixels(img):
    a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def save(arr, name, out_dir):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(np.clip(arr, 0, 1).astype(np.float32).ravel())
    path = os.path.join(out_dir, f"{name}.png")
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()
    return path


def coverage_mask(obj, size, faces=None):
    """Texels the low mesh's UVs cover (baked emission of white on black); with ``faces`` (a
    boolean per face), only those faces' texels."""
    img = bake_target(obj, "mask", size)
    mat = obj.data.materials[0]
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (1, 1, 1, 1)
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")  # the material's own (active) output
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    if faces is not None:
        black = bpy.data.materials.new("mask_off")
        black.use_nodes = True
        bn = black.node_tree.nodes
        bn.remove(next(n for n in bn if n.type == "BSDF_PRINCIPLED"))
        bt = bn.new("ShaderNodeTexImage")
        bt.image = img
        bn.active = bt
        obj.data.materials.append(black)
        obj.data.polygons.foreach_set("material_index", [0 if f else 1 for f in faces])
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = 0
    bpy.ops.object.bake(type="EMIT")
    if faces is not None:
        obj.data.polygons.foreach_set("material_index", [0] * len(obj.data.polygons))
    return pixels(img)[..., 0] > 0.5


VIEWS = {"side": ((-1, 0, 0), (0, 1, 0)), "three_quarter": ((-0.6, -0.45, 0.65), (0, 1, 0))}  # camera -> scene, PDX


def preview(low, diffuse, normal, out_dir, name, size=600):
    """Lit EEVEE renders of the low mesh with the baked colour, with and without the normal map
    (Workbench shows no normal maps). -> {view_variant: path}."""
    from mathutils import Matrix, Vector
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.view_settings.view_transform = "Standard"
    scene.world = scene.world or bpy.data.worlds.new("w")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.37, 1)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = Vector((-0.5, 0.8, 0.4)).to_track_quat("-Z", "Y").to_euler()  # light -> scene, PDX
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    cam.data.type, cam.data.ortho_scale, cam.data.clip_end = "ORTHO", 24.0, 200
    scene.camera = cam
    mat = bpy.data.materials.new("preview")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.6
    dif = nodes.new("ShaderNodeTexImage")
    dif.image = bpy.data.images.load(diffuse)
    links.new(dif.outputs["Color"], bsdf.inputs["Base Color"])
    nimg = bpy.data.images.load(normal)
    nimg.colorspace_settings.name = "Non-Color"
    ntex = nodes.new("ShaderNodeTexImage")
    ntex.image = nimg
    nmap = nodes.new("ShaderNodeNormalMap")
    links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    low.data.materials.clear()
    low.data.materials.append(mat)
    for o in bpy.data.objects:
        if o.type == "MESH" and o is not low:
            o.hide_render = True
    low.hide_render = False
    centre = Vector((0, 7.0, -1.0))
    paths = {}
    for view, (d, up) in VIEWS.items():
        d, up = Vector(d).normalized(), Vector(up)
        x = up.cross(-d).normalized()
        up = (-d).cross(x).normalized()
        cam.location = centre - d * 60
        cam.rotation_euler = Matrix((x, up, -d)).transposed().to_euler()
        for variant, on in (("flat", False), ("normal", True)):
            if on:
                links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
            else:
                for lk in [lk for lk in links if lk.to_socket == bsdf.inputs["Normal"]]:
                    links.remove(lk)
            path = os.path.join(out_dir, f"{name}_preview_{view}_{variant}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            paths[f"{view}_{variant}"] = path
    return paths


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(a.get("samples", 32))
    size = int(a.get("size", 512))
    geo = json.loads(open(a["low"], encoding="utf-8").read())
    low, lco, ltri = low_object(geo)
    if a.get("preview_only"):  # M19: renders only, with given maps ("diffuse", "normal"): nothing baked
        _pdx.write_result(a, {"preview": preview(low, a["diffuse"], a["normal"], a["out_dir"], a["name"])})
        return
    parts = np.array(geo.get("part") or [0] * len(ltri))
    hco, htri = [], []
    for path in a["high"]:
        c, t = load_obj(path)
        htri.append(t + sum(len(x) for x in hco))
        hco.append(c)
    ln = np.array(geo["n"], dtype=np.float64).reshape(-1, 3)
    added = ltri[parts >= 1]  # the low mesh's own added parts: they bake themselves
    hco.append(lco)
    htri.append(added)
    high = mesh_object("high", np.vstack(hco[:-1]), np.vstack(htri[:-1]) if len(htri) > 1 else np.zeros((0, 3), int))
    if a.get("high_smooth"):  # the fine voxel solid's stair-steps would bake as speckle
        bpy.context.view_layer.objects.active = high
        mod = high.modifiers.new("smooth", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(a["high_smooth"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    hv = np.empty(len(high.data.vertices) * 3)
    high.data.vertices.foreach_get("co", hv)
    hf = np.array([tuple(p.vertices) for p in high.data.polygons], dtype=np.int64).reshape(-1, 3)
    flagged, match = match_check(lco, ltri, parts == 0, (hv.reshape(-1, 3), hf), float(a.get("mismatch", 0.3)))
    own = mesh_object("own_parts", lco, added)
    own.data.shade_smooth()  # with the low mesh's normals, so its rays find exactly their own surface
    own.data.normals_split_custom_set_from_vertices([tuple(v / max(np.linalg.norm(v), 1e-9)) for v in ln])
    bpy.ops.object.select_all(action="DESELECT")
    for o in (high, own):
        o.select_set(True)
    bpy.context.view_layer.objects.active = high
    bpy.ops.object.join()
    high.data.shade_smooth()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = low
    low.select_set(True)
    mask = coverage_mask(low, size)
    part_masks = {int(k): coverage_mask(low, size, parts == k) for k in np.unique(parts)}
    mis_mask = coverage_mask(low, size, flagged) if flagged.any() else np.zeros_like(mask)
    match["flagged_texels_share"] = round(float(mis_mask.sum() / max(mask.sum(), 1)), 4)
    high.select_set(True)
    bk = scene.render.bake
    bk.use_selected_to_active = True
    bk.cage_extrusion = float(a.get("cage", 0.12))
    bk.max_ray_distance = float(a.get("distance", 0.6))
    bk.margin = 6
    bk.normal_space = "TANGENT"
    out = {"tris": {"low": int(len(ltri)), "high": int(sum(len(t) for t in htri)), "added_parts": int(len(added))},
           "match": match}
    nimg = bake_target(low, "normal", size)
    bpy.ops.object.bake(type="NORMAL")
    nrm = pixels(nimg)
    # AO on the low mesh alone: on the high one it is taken inside the closed solid wherever the
    # low surface lies under it (black islands); the panels still shade the body under them
    bk.use_selected_to_active = False
    high.select_set(False)
    high.hide_render = True
    aimg = bake_target(low, "ao", size)
    bpy.ops.object.bake(type="AO")
    ao = pixels(aimg)[..., 0]
    up = None
    if a.get("paint"):  # the painted pass's light from above: the low surface's own normal, object space (Y up)
        bk.normal_space = "OBJECT"
        oimg = bake_target(low, "objn", size)
        bpy.ops.object.bake(type="NORMAL")
        up = pixels(oimg)[..., 1] * 2 - 1
        bk.normal_space = "TANGENT"
    vec = nrm[..., :3] * 2 - 1
    k = float(a.get("strength", 1.0))  # blended towards flat, renormalised
    if k != 1.0:
        vec[..., :2] *= k
        vec /= np.maximum(np.linalg.norm(vec, axis=2, keepdims=True), 1e-6)
        nrm[..., :3] = vec * 0.5 + 0.5
    # rays that hit the wrong surface (another part of the body, a back face) bake steep tilts:
    # past ``outlier`` degrees (after the strength) the texel is taken as a miss, flat
    raw = np.degrees(np.arccos(np.clip(vec[..., 2], -1, 1)))
    wild = mask & (raw > float(a.get("outlier", 45.0)))
    out["outlier_texels_share"] = round(float(wild.sum() / max(mask.sum(), 1)), 4)
    vec[wild] = (0.0, 0.0, 1.0)
    nrm[wild, :3] = (0.5, 0.5, 1.0)
    # our own parts have no detail source: flat (the source head can poke through the chanfron,
    # which sits 0.06 off the low head)
    for kpart, m in part_masks.items():
        if kpart >= 1:
            vec[m] = (0.0, 0.0, 1.0)
            nrm[m, :3] = (0.5, 0.5, 1.0)
    if a.get("mismatch_flatten"):
        vec[mis_mask] = (0.0, 0.0, 1.0)
        nrm[mis_mask, :3] = (0.5, 0.5, 1.0)
    tilt = np.degrees(np.arccos(np.clip(vec[..., 2] / np.maximum(np.linalg.norm(vec, axis=2), 1e-6), -1, 1)))
    out["normal_tilt_deg"] = {"mean": round(float(tilt[mask].mean()), 2), "p95": round(float(np.percentile(tilt[mask], 95)), 2),
                              "by_part": {k: round(float(tilt[m].mean()), 2) for k, m in part_masks.items() if m.any()}}
    out["ao"] = {"mean": round(float(ao[mask].mean()), 3), "p05": round(float(np.percentile(ao[mask], 5)), 3)}
    # rays that found nothing bake the low surface's own normal: flat
    out["flat_texels_share"] = round(float((tilt[mask] < 0.5).mean()), 3)
    name, od = a["name"], a["out_dir"]
    nrm[..., 3] = 1.0
    paths = {"normal": save(nrm, f"{name}_normal", od)}
    aoc = np.dstack([ao, ao, ao, np.ones_like(ao)])
    paths["ao"] = save(aoc, f"{name}_ao", od)
    mm = np.dstack([mis_mask, mask * 0.25, np.zeros_like(ao), np.ones_like(ao)]).astype(np.float32)
    paths["mismatch"] = save(mm, f"{name}_mismatch", od)  # red: flagged texels; dark green: the rest
    dimg = bpy.data.images.load(a["diffuse"])
    dimg.colorspace_settings.name = "Non-Color"  # the painted values as they are
    diff = pixels(dimg).copy()
    s = float(a.get("ao_strength", 0.45))
    diff[..., :3] *= (1 - s + s * ao)[..., None]
    if a.get("paint"):  # the game-painted pass (_paint)
        cfg = a["paint"] if isinstance(a["paint"], dict) else None
        diff[..., :3], out["paint"] = _paint.game_paint(diff[..., :3], mask, vec[..., :2], up, cfg)
    diff[..., 3] = 1.0
    paths["diffuse"] = save(diff, f"{name}_diffuse", od)
    out["paths"] = paths
    if a.get("preview"):
        out["preview"] = preview(low, paths["diffuse"], paths["normal"], od, name)
    _pdx.write_result(a, out)


main()
