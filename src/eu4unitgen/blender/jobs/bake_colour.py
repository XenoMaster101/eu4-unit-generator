"""Blender job (M21): a textured model's base colour baked onto a fresh horse's low mesh.

PDX mesh space (as fresh_horse and bake_horse). The high surface is fresh_horse's ``textured_out``
(the generated model with its own UVs, fitted and reshaped; preview axes), shown by its base-colour
image as emission and baked selected to active: rays from ``cage`` outside the low surface,
``distance`` deep, islands padded by ``margin`` texels. Texels no ray reached are reported.

args: {"low": geometry .json, "high": .obj with UVs (preview axes), "image": base colour .png,
       "size": px, "cage", "distance", "margin", "out_dir", "name", "result_path"}
Result: {"diffuse": path, "reached": share of the covered texels a ray found the high surface for}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy
from mathutils import Matrix  # noqa: E402 - Blender module

import _pdx  # noqa: E402

FLIP_Z = Matrix.Diagonal((1, 1, -1, 1))  # preview axes (x, y, -z) -> PDX


def low_object(geo):
    co = np.array(geo["p"], dtype=np.float64).reshape(-1, 3)
    tris = np.array(geo["tri"], dtype=np.int64).reshape(-1, 3)
    me = bpy.data.meshes.new("low")
    me.from_pydata(co.tolist(), [], tris.tolist())
    me.update()
    obj = bpy.data.objects.new("low", me)
    bpy.context.scene.collection.objects.link(obj)
    me.shade_smooth()
    uv = me.uv_layers.new(name="UV")
    u0 = np.array(geo["u0"], dtype=np.float64).reshape(-1, 2)
    loops = np.array([lp.vertex_index for lp in me.loops])
    uv.data.foreach_set("uv", np.stack([u0[loops, 0], 1.0 - u0[loops, 1]], axis=1).ravel())
    obj["part"] = list(geo.get("part") or [0] * len(tris))
    return obj


def high_object(path, image):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path)
    obj = next(o for o in bpy.data.objects if o not in before and o.type == "MESH")
    obj.matrix_world = Matrix.Identity(4)  # the importer's axis change is an object rotation: the file's own (x, y, -z)
    obj.data.transform(FLIP_Z)
    obj.data.flip_normals()
    mat = bpy.data.materials.new("high")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(next(n for n in nodes if n.type == "BSDF_PRINCIPLED"))
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image)
    emit = nodes.new("ShaderNodeEmission")
    links.new(tex.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], next(n for n in nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def target(obj, name, size, colour):
    img = bpy.data.images.new(name, size, size, alpha=True)
    img.generated_color = colour
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.nodes.active = node
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return img


def part_mask(low, part, size):
    """The texels of the low mesh's faces of ``part`` (the geometry's per-triangle ``part``)."""
    img = bpy.data.images.new(f"mask{part}", size, size, alpha=True)
    mats = []
    for on in (False, True):
        m = bpy.data.materials.new(f"mask{part}_{on}")
        m.use_nodes = True
        nodes = m.node_tree.nodes
        nodes.remove(next(n for n in nodes if n.type == "BSDF_PRINCIPLED"))
        e = nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (1, 1, 1, 1) if on else (0, 0, 0, 1)
        m.node_tree.links.new(e.outputs["Emission"], next(n for n in nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
        t = nodes.new("ShaderNodeTexImage")
        t.image = img
        nodes.active = t
        mats.append(m)
    low.data.materials.clear()
    for m in mats:
        low.data.materials.append(m)
    low.data.polygons.foreach_set("material_index", [1 if p == part else 0 for p in low["part"]])
    bk = bpy.context.scene.render.bake
    bk.use_selected_to_active, bk.margin = False, 0
    for o in bpy.data.objects:
        o.select_set(o is low)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type="EMIT")
    low.data.polygons.foreach_set("material_index", [0] * len(low.data.polygons))
    return pixels(img)[..., 0] > 0.5


def pixels(img):
    a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.device = "CPU"
    low = low_object(json.loads(open(a["low"], encoding="utf-8").read()))
    high = high_object(a["high"], a["image"])
    size = int(a.get("size", 512))
    bk = sc.render.bake
    # coverage: the low mesh's own texels (no margin), then the colour with rays to the high surface;
    # a magenta clear colour marks texels no ray reached
    cov_img = target(low, "cover", size, (0, 0, 0, 1))
    emit = low.data.materials[0].node_tree
    e = emit.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    emit.links.new(e.outputs["Emission"], next(n for n in emit.nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
    bk.use_selected_to_active, bk.margin = False, 0
    for o in bpy.data.objects:
        o.select_set(o is low)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type="EMIT")
    cov = pixels(cov_img)[..., 0] > 0.5
    painted = {}
    for part, (lo, hi) in (a.get("part_colours") or {}).items():  # pieces of our own, coloured, not baked
        painted[int(part)] = (part_mask(low, int(part), size) & cov, np.array(lo) / 255.0, np.array(hi) / 255.0)
    img = target(low, "colour", size, (1, 0, 1, 1))
    bk.use_selected_to_active, bk.use_cage = True, False
    bk.cage_extrusion, bk.max_ray_distance = float(a.get("cage", 0.4)), float(a.get("distance", 0.8))
    bk.margin, bk.margin_type = 0, "EXTEND"
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type="EMIT")
    px = pixels(img)
    for mask, lo, hi in painted.values():  # hair: streaks between the two colours along v
        rng = np.random.default_rng(0)
        streak = np.repeat(rng.random(size // 4 + 1), 4)[:size]
        t = np.clip(streak[None, :] * 0.8 + 0.2 * rng.random((size, size)), 0, 1)
        col = lo[None, None] + (hi - lo)[None, None] * t[..., None]  # a byte sRGB image's pixels are sRGB
        px[mask, :3] = col[mask]
    miss = cov & (px[..., 0] > 0.99) & (px[..., 1] < 0.01) & (px[..., 2] > 0.99)
    # the misses and the gutters filled from the nearest reached texel, then padded ``margin`` texels
    good = cov & ~miss
    from collections import deque
    filled = good.copy()
    q = deque(zip(*np.nonzero(good)))
    reach = np.where(good, 0, -1)
    margin = int(a.get("margin", 8))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < size and 0 <= xx < size and not filled[yy, xx]:
                if not cov[yy, xx] and reach[y, x] >= margin:
                    continue
                filled[yy, xx] = True
                reach[yy, xx] = 0 if cov[yy, xx] else reach[y, x] + 1
                px[yy, xx] = px[y, x]
                q.append((yy, xx))
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    path = os.path.join(a["out_dir"], f"{a['name']}_diffuse.png")
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()
    _pdx.write_result(a, {"diffuse": path, "reached": round(float(good.sum() / max(cov.sum(), 1)), 4),
                          "covered": round(float(cov.mean()), 4)})


main()
