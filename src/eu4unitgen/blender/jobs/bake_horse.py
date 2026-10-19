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
  * colour: the painted diffuse times (1 - ao_strength + ao_strength x ao).
Reports each map's path, the normal map's mean tilt from flat (base game 4-20 deg) on the texels
the UVs cover, and how much of the low surface found the high one.

args: {"low": geometry .json, "high": [.obj (preview axes, as fresh_horse's debug_dump writes)],
       "diffuse": painted .png, "size": px, "cage", "distance", "samples", "ao_strength",
       "high_smooth": iterations (the high surface smoothed before baking), "strength": the tangent
       normals' x and y scaled by it (towards flat) and renormalised, "outlier": degrees (steeper
       texels are rays that hit the wrong surface: flat),
       "out_dir", "name", "result_path"}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

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
    parts = np.array(geo.get("part") or [0] * len(ltri))
    hco, htri = [], []
    for path in a["high"]:
        c, t = load_obj(path)
        htri.append(t + sum(len(x) for x in hco))
        hco.append(c)
    added = ltri[parts >= 1]  # the low mesh's own added parts: they bake themselves
    hco.append(lco)
    htri.append(added)
    high = mesh_object("high", np.vstack(hco[:-1]), np.vstack(htri[:-1]) if len(htri) > 1 else np.zeros((0, 3), int))
    if a.get("high_smooth"):  # the fine voxel solid's stair-steps would bake as speckle
        bpy.context.view_layer.objects.active = high
        mod = high.modifiers.new("smooth", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(a["high_smooth"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    own = mesh_object("own_parts", lco, added)
    own.data.shade_smooth()  # with the low mesh's normals, so its rays find exactly their own surface
    ln = np.array(geo["n"], dtype=np.float64).reshape(-1, 3)
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
    high.select_set(True)
    bk = scene.render.bake
    bk.use_selected_to_active = True
    bk.cage_extrusion = float(a.get("cage", 0.12))
    bk.max_ray_distance = float(a.get("distance", 0.6))
    bk.margin = 6
    bk.normal_space = "TANGENT"
    out = {"tris": {"low": int(len(ltri)), "high": int(sum(len(t) for t in htri)), "added_parts": int(len(added))}}
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
    dimg = bpy.data.images.load(a["diffuse"])
    dimg.colorspace_settings.name = "Non-Color"  # the painted values as they are
    diff = pixels(dimg).copy()
    s = float(a.get("ao_strength", 0.45))
    diff[..., :3] *= (1 - s + s * ao)[..., None]
    diff[..., 3] = 1.0
    paths["diffuse"] = save(diff, f"{name}_diffuse", od)
    out["paths"] = paths
    _pdx.write_result(a, out)


main()
