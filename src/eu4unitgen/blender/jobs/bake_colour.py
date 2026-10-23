"""Blender job (M21): a textured model's base colour baked onto a fresh horse's low mesh.

PDX mesh space (as fresh_horse and bake_horse). The high surface is fresh_horse's ``textured_out``
(the generated model with its own UVs, fitted and reshaped; preview axes), shown by its base-colour
image as emission and baked selected to active: rays from ``cage`` outside the low surface,
``distance`` deep, islands padded by ``margin`` texels. Texels no ray reached are reported.

args: {"low": geometry .json, "high": .obj with UVs (preview axes), "image": base colour .png,
       "size": px, "cage", "distance", "margin", "part_colours": {part: [lo rgb, hi rgb]} (our own pieces,
       painted), "gloss": {"silver", "gold", "matte", thresholds} (writes <name>_specular.png),
       "colour_render": the raw model's lit render (the colour transferred to it), "colour_strength",
       "out_dir", "name", "result_path"}
Result: {"diffuse": path, "reached": share of the covered texels a ray found the high surface for, ...}
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


def srgb_to_lab(c):
    """sRGB (0-1, ... x 3) -> CIE Lab (D65)."""
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.9505, 1.0, 1.089])
    m = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    c = np.clip(xyz @ m.T, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def render_pixels(path):
    """The subject's pixels of a render (not the corner's background colour), sRGB 0-1."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"  # the stored sRGB values as they are
    im = pixels(img).astype(np.float64)
    if im[..., 3].min() < 0.99:
        sel = im[..., 3] > 0.5
    else:
        sel = np.abs(im[..., :3] - im[2, 2, :3]).sum(-1) > 0.05
    return im[sel][:, :3]


def colour_transfer(px, mask, render, strength=(0.5, 1.0, 1.0), sd_cap=1.1):
    """M21 (the user: the game horse looked flat and orange next to the raw model): the texels' Lab mean
    and spread matched to the raw model's lit render (what the user approved), per channel by
    ``strength`` (L, a, b). Lightness half way, its spread lifted at most ``sd_cap``: the render's
    lightness carries the metal's reflections (the cataphract's scale up to 227 in the render, 147 in its
    base colour) and at full strength the scale went nearly white (the user, test 26's gallery).
    -> the stats before and after."""
    ref = srgb_to_lab(render_pixels(render))
    lab = srgb_to_lab(px[mask, :3])
    mu_t, sd_t, mu_r, sd_r = lab.mean(0), lab.std(0) + 1e-6, ref.mean(0), ref.std(0) + 1e-6
    k = np.array(strength if np.ndim(strength) else [strength] * 3, dtype=np.float64)
    ratio = sd_r / sd_t
    ratio[0] = min(ratio[0], sd_cap)
    new = (lab - mu_t) * (1 + k * (ratio - 1)) + mu_t + k * (mu_r - mu_t)
    px[mask, :3] = lab_to_srgb(new)
    return {"texture_lab_mean": mu_t.round(1).tolist(), "render_lab_mean": mu_r.round(1).tolist(),
            "texture_lab_sd": sd_t.round(1).tolist(), "render_lab_sd": sd_r.round(1).tolist()}


def gloss_map(px, cov, gloss):
    """The specular map's gloss (alpha; RGB 0: no unit colour) by colour class of the base colour:
    silver (bright, colourless), gold (bright, yellow-orange, saturated), the rest matte (coat, cloth,
    leather, hair). TRELLIS.2's metallic map isn't used (horse 2: 0.65 over coat and hair alike)."""
    c = px[..., :3]
    mx, mn = c.max(-1), c.min(-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    lum = c @ np.array([0.299, 0.587, 0.114])
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    silver = cov & (chroma < gloss.get("silver_chroma", 0.18)) & (lum > gloss.get("silver_lum", 0.38))
    gold = cov & ~silver & (hue > 28) & (hue < 58) & (chroma > 0.35) & (lum > gloss.get("gold_lum", 0.36))
    out = np.full(cov.shape, float(gloss.get("matte", 5)))
    out[silver] = gloss.get("silver", 14)
    out[gold] = gloss.get("gold", 12)
    return out, {"silver": round(float(silver.sum() / max(cov.sum(), 1)), 3),
                 "gold": round(float(gold.sum() / max(cov.sum(), 1)), 3)}


def texel_positions(geo, size):
    """Each texel's 3D point on the low mesh (PDX), rows as Blender's pixels (bottom up), and the mask
    of texels a triangle covers."""
    p = np.array(geo["p"], dtype=np.float64).reshape(-1, 3)
    uv = np.array(geo["u0"], dtype=np.float64).reshape(-1, 2) * [1, -1] + [0, 1]  # -> Blender's v up
    uv *= size
    pos = np.zeros((size, size, 3))
    hit = np.zeros((size, size), dtype=bool)
    for t in np.array(geo["tri"], dtype=np.int64).reshape(-1, 3):
        a, b, c = uv[t]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        q = np.stack([xs + 0.5, ys + 0.5], -1)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (q[..., 0] - c[0]) + (c[0] - b[0]) * (q[..., 1] - c[1])) / d
        w1 = ((c[1] - a[1]) * (q[..., 0] - c[0]) + (a[0] - c[0]) * (q[..., 1] - c[1])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        pts = w0[..., None] * p[t[0]] + w1[..., None] * p[t[1]] + w2[..., None] * p[t[2]]
        pos[ys[inside], xs[inside]] = pts[inside]
        hit[ys[inside], xs[inside]] = True
    return pos, hit


def head_band(px, posmap, hit, poll, axis, width=0.6, radius=3.0):
    """M21 (the user, the cataphract: the red blobs by the eyes are a red cloth band round the head behind
    the gold chanfron): a clean red stripe all the way round the head, ``width`` wide, just behind the
    chanfron's rear edge (where its gold texels start along the head's axis, poll to muzzle), in the
    head's own red; red texels ahead of it on the chanfron turn the chanfron's gold (its edge sharp).
    Only when the head shows both gold and red. -> what was done."""
    c = px[..., :3]
    mx, mn = c.max(-1), c.min(-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    lum = c @ np.array([0.299, 0.587, 0.114])
    r_, g_, b_ = c[..., 0], c[..., 1], c[..., 2]
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (g_ - b_), 2 * r_ - g_ - b_)) + 360) % 360
    axis = np.asarray(axis, dtype=np.float64)
    rel = posmap - np.asarray(poll, dtype=np.float64)
    t = rel @ axis
    rad = np.linalg.norm(rel - t[..., None] * axis, axis=-1)
    near = hit & (rad < radius)
    gold = near & (t > -0.5) & (hue > 28) & (hue < 58) & (chroma > 0.3) & (lum > 0.25)
    red = near & (t > -1.5) & ((hue < 15) | (hue > 340)) & (chroma > 0.45)
    if gold.sum() < 200 or red.sum() < 50:
        return {"band": False, "gold_texels": int(gold.sum()), "red_texels": int(red.sum())}
    t0, t1 = float(np.percentile(t[gold], 5)), float(np.percentile(t[gold], 99)) + 0.3  # the chanfron's ends
    red_c, gold_c = np.median(c[red & (t < t1)], axis=0), np.median(c[gold], axis=0)
    # the collar: a clean strip of even width behind the chanfron's rear edge, which is traced as a smooth curve
    # (at each height across the head, h along the axis' perpendicular in the side plane: where the gold
    # starts, smoothed): distances to the nearest gold texel made a ragged, combed band (the chanfron's gold is
    # speckled with dark ornament), and a flat cut at the poll hardly showed. The ears and the top of the poll
    # stay their own colour (a red cap otherwise).
    face = gold & (t > 0.0) & (rad < 0.75 * radius)  # the chanfron's own gold, ahead of the poll
    up = np.cross(axis, [1.0, 0.0, 0.0])
    up /= np.linalg.norm(up)
    h = rel @ up
    bins = np.arange(np.percentile(h[face], 1) - 0.5, np.percentile(h[face], 99) + 0.5, 0.2)
    edge = np.full(len(bins), np.nan)
    for i, b0 in enumerate(bins):
        sel = face & (h >= b0) & (h < b0 + 0.2)
        if sel.sum() > 20:
            edge[i] = np.percentile(t[sel], 5)
    ok = ~np.isnan(edge)
    edge = np.interp(np.arange(len(bins)), np.flatnonzero(ok), edge[ok])
    edge = np.convolve(np.pad(edge, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    t_edge = np.interp(h, bins + 0.1, edge)
    band = hit & ~face & (t >= t_edge - width) & (t < t_edge) & (rad < radius)         & (h >= bins[0]) & (h <= bins[-1] + 0.2) & (posmap[..., 1] < float(poll[1]) - 0.2)
    stray = red & (t >= t0) & (t <= t1) & (rad < 0.75 * radius)  # on the head only (the axis runs on past it)
    px[band, :3] = red_c
    px[stray, :3] = gold_c
    return {"band": True, "rear_edge_t": round(t0, 3), "band_texels": int(band.sum()), "red_recoloured": int(stray.sum()),
            "red": [int(x * 255) for x in red_c], "gold": [int(x * 255) for x in gold_c]}


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
    result = {}
    if a.get("head_band"):
        hb_ = a["head_band"]
        posmap, hit = texel_positions(json.loads(open(a["low"], encoding="utf-8").read()), size)
        result["head_band"] = head_band(px, posmap, hit & filled, hb_["poll"], hb_["axis"], float(hb_.get("width", 0.6)))
    if a.get("gloss") is not None:  # from the colour as baked (before any transfer)
        g, result["gloss_shares"] = gloss_map(px, filled, a["gloss"])
        spec = bpy.data.images.new("specular", size, size, alpha=True)
        spec.colorspace_settings.name = "Non-Color"
        sp = np.zeros((size, size, 4), dtype=np.float32)
        sp[..., 3] = np.clip(g, 0, 255) / 255.0
        spec.pixels.foreach_set(sp.ravel())
        result["specular"] = os.path.join(a["out_dir"], f"{a['name']}_specular.png")
        spec.filepath_raw, spec.file_format = result["specular"], "PNG"
        spec.save()
    if a.get("colour_render"):
        result["colour_transfer"] = colour_transfer(px, filled, a["colour_render"], a.get("colour_strength", (0.5, 1.0, 1.0)))
    img.pixels.foreach_set(px.ravel())
    path = os.path.join(a["out_dir"], f"{a['name']}_diffuse.png")
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()
    _pdx.write_result(a, {"diffuse": path, "reached": round(float(good.sum() / max(cov.sum(), 1)), 4),
                          "covered": round(float(cov.mean()), 4), **result})


main()
