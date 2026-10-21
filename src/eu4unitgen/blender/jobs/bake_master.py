"""Blender job (runs inside Blender only), Track M finish line step 2: the posed master's colour and classes baked
onto the game mesh.

Usage: blender --background --python bake_master.py -- args.json

Opens game_master's reposed.blend (every part with its albedo as an emission, Blender's mirrored frame), loads the
low mesh (fresh_horse's geometry JSON, PDX space: mirrored into the scene's frame, faces and normals turned to
match) and bakes, selected to active with a cage ``cage`` out and rays ``distance`` deep:
  diffuse   the parts' albedo (EMIT);
  classes   each part's class colour (EMIT, one sample): which texels are scale, cloth, trim, chanfron, coat ...
            (the panels' gold read from the baked colour: gold texels are trim);
  specular  RGB 0 (no unit colours), alpha the class's gloss (_horse_paint's values).
Texels no ray reached keep the nearest baked texel (islands bled outwards). The ``fit`` (fresh_horse's scale and
offset of the source model) is applied to the parts first, so they lie where the low mesh was made from them.

args: {"reposed_blend", "low", "fit": {"scale", "offset"}, "size", "cage", "distance", "classes": {name: rgb},
       "out_dir", "name", "result_path"}
"""
import json
import sys
import time
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Matrix  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _horse_paint as hp  # noqa: E402

GLOSS = {"coat": hp.GLOSS[hp.COAT], "leg": hp.GLOSS[hp.COAT], "scale": hp.GLOSS[hp.SCALE], "cloth": hp.GLOSS[hp.CLOTH],
         "trim": hp.GLOSS[hp.TRIM], "chanfron": hp.GLOSS[hp.CHANFRON], "leather": hp.GLOSS[hp.LEATHER],
         "tail": hp.GLOSS[hp.HAIR], "plume": hp.GLOSS[hp.CLOTH]}
DECAL = {"decal_eye": "chanfron"}  # other decals are gold ornament (trim)


def part_class(name):
    """The class of a master part, by its object name (master_assemble's names)."""
    if name.startswith("body_"):
        return name[5:]
    if name.endswith("_tassels") or name.startswith("shell_trim"):
        return "trim"
    if name.startswith(("panel_", "chest_")):
        return "cloth"
    if name.startswith("decal_"):
        return DECAL.get(name, "trim")
    return {"plates": "scale", "backing": "scale", "shell_leather": "leather", "tail": "tail", "plume": "plume"}.get(name, "coat")


def emission(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def target(obj, name, size, colour):
    img = bpy.data.images.new(name, size, size, alpha=False)
    if not colour:
        img.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new(f"{name}_target")
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


def bleed(arr, filled, passes=64):
    """Unreached texels take their nearest reached neighbour's value (islands grown outwards)."""
    arr, filled = arr.copy(), filled.copy()
    for _ in range(passes):
        if filled.all():
            break
        grow = np.zeros_like(filled)
        acc = np.zeros_like(arr)
        cnt = np.zeros(filled.shape)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            f = np.roll(np.roll(filled, dy, 0), dx, 1)
            v = np.roll(np.roll(arr, dy, 0), dx, 1)
            take = f & ~filled
            acc[take] += v[take]
            cnt[take] += 1
            grow |= take
        arr[grow] = acc[grow] / cnt[grow][:, None]
        filled |= grow
    return arr


def save_png(arr, path, colour):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(Path(path).stem, w, h, alpha=True)
    if not colour:
        img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :arr.shape[2]] = arr
    img.pixels.foreach_set(np.clip(rgba, 0, 1).ravel())
    img.filepath_raw, img.file_format = str(path), "PNG"
    img.save()


def main():
    t0 = time.time()
    a = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=a["reposed_blend"])
    sc = bpy.context.scene
    parts = [ob for ob in sc.objects if ob.type == "MESH"]
    # fresh_horse's fit of the source model (PDX: x *= s, then + offset), in the scene's mirrored frame
    s, off = a["fit"]["scale"], a["fit"]["offset"]
    M = Matrix.Translation((-off[0], off[1], off[2])) @ Matrix.Scale(s, 4)
    for ob in parts:
        ob.data.transform(M)
        ob.data.update()
    # the low mesh, mirrored into the scene's frame
    geo = json.loads(Path(a["low"]).read_text(encoding="utf-8"))
    co = np.array(geo["p"], dtype=np.float64).reshape(-1, 3) * np.array([-1.0, 1.0, 1.0])
    tris = np.array(geo["tri"], dtype=np.int64).reshape(-1, 3)[:, ::-1]
    me = bpy.data.meshes.new("low")
    me.from_pydata(co.tolist(), [], tris.tolist())
    me.update()
    low = bpy.data.objects.new("low", me)
    sc.collection.objects.link(low)
    me.shade_smooth()
    uv = me.uv_layers.new(name="UV")
    u0 = np.array(geo["u0"], dtype=np.float64).reshape(-1, 2)
    loops = np.array([lp.vertex_index for lp in me.loops])
    uv.data.foreach_set("uv", np.stack([u0[loops, 0], 1.0 - u0[loops, 1]], axis=1).ravel())
    n = np.array(geo["n"], dtype=np.float64).reshape(-1, 3) * np.array([-1.0, 1.0, 1.0])
    me.normals_split_custom_set_from_vertices([tuple(v / max(np.linalg.norm(v), 1e-9)) for v in n])
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 4
    bk = sc.render.bake
    bk.use_selected_to_active, bk.cage_extrusion, bk.max_ray_distance, bk.margin = True, a["cage"], a["distance"], 0
    for o in sc.objects:
        o.select_set(o in parts)
    bpy.context.view_layer.objects.active = low
    low.select_set(True)
    size = int(a["size"])
    res = {}
    # colour
    img = target(low, "diffuse", size, True)
    bpy.ops.object.bake(type="EMIT")
    col = pixels(img)
    # what the rays reached: a bake of white on every part
    white = emission("white", (1.0, 1.0, 1.0))
    keep = {ob.name: list(ob.data.materials) for ob in parts}
    for ob in parts:
        ob.data.materials.clear()
        ob.data.materials.append(white)
    img = target(low, "reached", size, False)
    bpy.ops.object.bake(type="EMIT")
    reached = pixels(img)[..., 0] > 0.5
    # classes
    names = list(a["classes"])
    for ob in parts:
        c = part_class(ob.name)
        ob.data.materials.clear()
        ob.data.materials.append(emission(f"cls_{ob.name}", tuple(np.array(a["classes"][c]) / 255.0)))
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    img = target(low, "classes", size, False)
    bpy.ops.object.bake(type="EMIT")
    cls_rgb = pixels(img)[..., :3]
    pal = np.array([a["classes"][c] for c in names]) / 255.0
    cls = np.argmin(((cls_rgb[..., None, :] - pal[None, None]) ** 2).sum(-1), axis=-1)
    # the panels' gold is trim (the colour says so: hue 25-65 deg, saturation > 0.35, value > 0.45 in sRGB)
    srgb = col[..., :3]  # a byte image's pixels are as stored: sRGB
    mx, mn = srgb.max(-1), srgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = srgb[..., 0], srgb[..., 1], srgb[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    gold = (hue > 25) & (hue < 65) & (sat > 0.35) & (mx > 0.45) & (cls == names.index("cloth"))
    cls[gold] = names.index("trim")
    for ob in parts:
        ob.data.materials.clear()
        for m in keep[ob.name]:
            ob.data.materials.append(m)
    # bleed, save: the image's rows run bottom-up in Blender; PNGs are saved as Blender stores them
    col = bleed(col[..., :3], reached)
    gloss = np.array([GLOSS.get(c, 5) for c in names])[cls].astype(np.float64)
    gl = bleed(gloss[..., None], reached)[..., 0]
    cl_rgb = bleed(pal[cls], reached)
    save_png(col, out / f"{a['name']}_diffuse.png", True)
    spec = np.zeros(col.shape[:2] + (4,), np.float32)
    spec[..., 3] = gl / 255.0
    h, w = spec.shape[:2]
    simg = bpy.data.images.new("spec", w, h, alpha=True)
    simg.colorspace_settings.name = "Non-Color"
    simg.alpha_mode = "STRAIGHT"
    simg.pixels.foreach_set(spec.ravel())
    simg.filepath_raw, simg.file_format = str(out / f"{a['name']}_specular.png"), "PNG"
    simg.save()
    save_png(cl_rgb, out / f"{a['name']}_classes.png", False)
    share = {c: round(float((cls[reached] == i).mean()), 4) for i, c in enumerate(names) if (cls[reached] == i).any()}
    res = {"diffuse": str(out / f"{a['name']}_diffuse.png"), "specular": str(out / f"{a['name']}_specular.png"),
           "classes": str(out / f"{a['name']}_classes.png"), "reached": round(float(reached.mean()), 4),
           "class_share": share, "seconds": round(time.time() - t0, 1)}
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f)


main()
