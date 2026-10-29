"""Blender job: an imported model (.glb, .gltf or .fbx: the user's own or a CC0 / CC BY download) made into what
TRELLIS.2 hands the build (the user, 2026-10-29: "the imported model replaces the TRELLIS.2 step"):

  1. imported; a rig, shape keys and other modifiers applied as the model stands, then dropped (no skins, no
     animations: the build rigs it on the game's skeleton);
  2. its meshes joined, its transforms applied;
  3. a soldier stood up (its tallest side up) and turned to face the front (Blender's -Y, as TRELLIS.2's do): from
     its feet (the toes, the lowest slice, stick out forwards of the shins) unless ``turn`` (degrees about up) is
     given; turned in quarter turns;
  4. centred and scaled so its largest side is 1 (TRELLIS.2's unit cube);
  5. one material with one base-colour texture: kept if it already is (one material, an image on its base colour,
     UVs); plain-coloured materials put on a small palette texture (``palette``: exact, no bake); else the
     materials' colours baked (Cycles, diffuse colour only) onto new UVs at ``bake_size`` (twice that over 150k
     triangles);
  6. written as a .glb (``out_glb``);
  7. the reference picture (the cut-out the build's part crops, the outline check and the result sheet read): an
     orthographic render on a transparent ground from the kind's view (``views``: soldier front, horse side, gun
     three-quarter), ``out_dir``/input_rgba.png.

args: {"model", "out_glb", "kind", "turn": deg | null, "bake_size": 2048, "out_dir", "size": 1024, "result_path"}
Result: {"triangles", "materials", "baked": bool, "rig_applied": bool, "stood_up": [axis] | None, "turn": deg,
         "turn_from": "feet" | "asked" | "kept", "toes": offset, "dimensions", "reference"}
"""
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).parent))
from _pdx import args as job_args, clear_scene, write_result  # noqa: E402

TOES_MIN = 0.01  # toes out less than this share of the height say nothing about facing: the model is left as it is
PERSON_MIN = 1.4  # a standing person is at least this much taller than wide (arms out at 35 degrees: about 1.6-2)
ALPHA_MIN = 0.5
VIEWS = {"soldier": (0.0, 0.0), "horse": (90.0, 0.0), "gun": (45.0, 20.0)}  # azimuth, elevation (render_model's)


def _import(path: Path) -> list:
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise SystemExit(f"can't import {path.suffix} (a .glb or .fbx)")
    return list(bpy.data.objects)


def model_meshes(objs) -> list:
    """The model's own meshes: those in the scene, shown, and not a rig's bone shape. Blender's glTF reader adds an
    "Icosphere" (radius 1, outside the scene) to draw a rig's bones: joined in, it made Mambo a ball 2 across with her
    textures baked onto it from inside (2026-10-29)."""
    scene = set(bpy.context.scene.objects)
    shapes = {pb.custom_shape for o in objs if o.type == "ARMATURE" for pb in o.pose.bones if pb.custom_shape}
    return [o for o in objs if o.type == "MESH" and o in scene and o not in shapes and not o.hide_get()
            and not o.hide_render]


def baked_meshes(objs) -> tuple[list, bool]:
    """Every mesh as it stands (modifiers, a rig's pose and shape keys applied), in world space; the rest removed."""
    dg = bpy.context.evaluated_depsgraph_get()
    rig = any(o.type == "ARMATURE" for o in objs)
    out = []
    for o in model_meshes(objs):
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
        me.transform(o.matrix_world)
        new = bpy.data.objects.new(o.name + "_baked", me)
        bpy.context.scene.collection.objects.link(new)
        out.append(new)
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    if not out:
        raise SystemExit("no mesh in the model")
    return out, rig


def clear_faces(obj) -> int:
    """A see-through material's faces that are clear where they sit (the texture's alpha at the face's middle under
    ``ALPHA_MIN``) removed: the bake reads colour without alpha, so a transparent overlay baked as black over
    Mambo's face (2026-10-29); its visible faces (brows, blush) stay. -> faces removed."""
    me = obj.data
    if not me.uv_layers:
        return 0
    uv = me.uv_layers.active.data
    alpha = {}
    for i, m in enumerate(me.materials):
        if m is None or not m.use_nodes or getattr(m, "blend_method", "OPAQUE") == "OPAQUE":
            continue
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        link = bsdf.inputs["Alpha"].links if bsdf else None
        node = link[0].from_node if link else None
        if node is None or node.type != "TEX_IMAGE" or node.image is None:
            continue
        img = node.image
        w, h = img.size
        if not w or not h:
            continue
        px = list(img.pixels)
        alpha[i] = (w, h, px)
    if not alpha:
        return 0
    drop = []
    for p in me.polygons:
        if p.material_index not in alpha:
            continue
        w, h, px = alpha[p.material_index]
        u = sum(uv[li].uv[0] for li in p.loop_indices) / p.loop_total
        v = sum(uv[li].uv[1] for li in p.loop_indices) / p.loop_total
        x, y = int((u % 1.0) * (w - 1)), int((v % 1.0) * (h - 1))
        if px[4 * (y * w + x) + 3] < ALPHA_MIN:
            drop.append(p.index)
    if drop:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in drop], context="FACES")
        bm.to_mesh(me)
        bm.free()
    return len(drop)


def join(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def extent(obj):
    co = [v.co for v in obj.data.vertices]
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    return lo, hi


def stand_up(obj):
    """A person's tallest side to +Z (a Z-up export read as Y-up lies on its back). -> the axis it was, or None."""
    lo, hi = extent(obj)
    d = hi - lo
    tall = max(range(3), key=lambda k: d[k])
    if tall == 2:
        return None
    rot = Matrix.Rotation(math.radians(90), 4, "X" if tall == 1 else "Y")
    obj.data.transform(rot)
    lo, hi = extent(obj)
    co = [v.co for v in obj.data.vertices]  # head up: the arms and shoulders, the widest, belong in the upper half
    mid = (lo.z + hi.z) / 2
    up_w = max((abs(c.x) + abs(c.y) for c in co if c.z > mid), default=0)
    low_w = max((abs(c.x) + abs(c.y) for c in co if c.z <= mid), default=0)
    if low_w > 1.4 * up_w:
        obj.data.transform(Matrix.Rotation(math.radians(180), 4, "X"))
    return ["x", "y", "z"][tall]


def facing_from_feet(obj) -> tuple[float, float]:
    """The turn about Z (a quarter turn) that faces a standing person to -Y, from its feet: the lowest 3% of its
    height (the soles and toes) against the slice at 6-12% (the shins): toes stick out forwards. -> (degrees, how
    far the toes stick out, as a share of the height)."""
    lo, hi = extent(obj)
    h = hi.z - lo.z
    co = [v.co for v in obj.data.vertices]
    sole = [c for c in co if c.z < lo.z + 0.03 * h]
    shin = [c for c in co if lo.z + 0.06 * h < c.z < lo.z + 0.12 * h]
    if not sole or not shin:
        return 0.0, 0.0
    ms = Vector((sum(c.x for c in sole) / len(sole), sum(c.y for c in sole) / len(sole), 0))
    mh = Vector((sum(c.x for c in shin) / len(shin), sum(c.y for c in shin) / len(shin), 0))
    d = ms - mh
    if d.length < 1e-9:
        return 0.0, 0.0
    ang = math.degrees(math.atan2(d.x, -d.y))  # 0 when the toes point to -Y
    return float(-round(ang / 90.0) * 90.0) % 360.0, d.length / h


def normalise(obj):
    lo, hi = extent(obj)
    c, s = (lo + hi) / 2, 1.0 / max(hi - lo)
    obj.data.transform(Matrix.Scale(s, 4) @ Matrix.Translation(-c))


def one_texture(obj) -> bool:
    """One material whose base colour is an image, and UVs."""
    mats = [m for m in obj.data.materials if m]
    if len(mats) != 1 or not obj.data.uv_layers or not mats[0].use_nodes:
        return False
    bsdf = next((n for n in mats[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    link = bsdf and bsdf.inputs["Base Color"].links
    return bool(link and link[0].from_node.type == "TEX_IMAGE" and link[0].from_node.image)


def plain_colours(obj) -> list | None:
    """Each material's colour when none uses an image or the mesh's colour attributes (plain values only), else
    None."""
    if obj.data.color_attributes:
        return None
    out = []
    for m in obj.data.materials:
        if m is None:
            out.append((0.6, 0.6, 0.6))
            continue
        if m.use_nodes:
            if any(n.type in ("TEX_IMAGE", "VERTEX_COLOR", "ATTRIBUTE") for n in m.node_tree.nodes):
                return None
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            c = bsdf.inputs["Base Color"].default_value if bsdf else m.diffuse_color
        else:
            c = m.diffuse_color
        out.append(tuple(c[:3]))
    return out or [(0.6, 0.6, 0.6)]


def palette(obj, colours: list, cell: int = 16) -> None:
    """Plain-coloured materials onto one texture without a bake: each material a flat cell of a small palette image,
    its faces' UVs all at the cell's middle (exact colours; a bake of a dense mesh's thousands of small UV islands
    left black texels: 289k triangles, 2026-10-29)."""
    n = max(1, len(colours))
    side = math.ceil(math.sqrt(n))
    size = side * cell
    img = bpy.data.images.new("palette", size, size, alpha=False)
    px = [0.0] * (size * size * 4)
    for i, c in enumerate(colours):
        cx, cy = (i % side) * cell, (i // side) * cell
        for y in range(cy, cy + cell):
            for x in range(cx, cx + cell):
                k = 4 * (y * size + x)
                px[k:k + 4] = [c[0], c[1], c[2], 1.0]
    img.pixels = px
    img.pack()
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    uv = obj.data.uv_layers.new(name="UV")
    for p in obj.data.polygons:
        i = min(p.material_index, n - 1)
        u, v = ((i % side) + 0.5) / side, ((i // side) + 0.5) / side
        for li in p.loop_indices:
            uv.data[li].uv = (u, v)
    _one_material(obj, img)


def _one_material(obj, img) -> None:
    mat = bpy.data.materials.new("imported")
    mat.use_nodes = True
    t = mat.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = img
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mat.node_tree.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def bake_colours(obj, size: int) -> None:
    """The materials' base colours (images, vertex colours or plain values) baked onto new UVs, one material."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    src_uv = obj.data.uv_layers.active.name if obj.data.uv_layers else None
    new = obj.data.uv_layers.new(name="baked")
    obj.data.uv_layers.active = new
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(80), island_margin=0.001)  # few, large islands
    bpy.ops.object.mode_set(mode="OBJECT")
    img = bpy.data.images.new("baked_colour", size, size, alpha=False)
    if not obj.data.materials:
        m = bpy.data.materials.new("plain")
        m.use_nodes = True
        obj.data.materials.append(m)
    for m in obj.data.materials:
        if m is None:
            continue
        m.use_nodes = True
        nt = m.node_tree
        if src_uv:  # the old textures still read the old UVs
            for n in nt.nodes:
                if n.type == "TEX_IMAGE" and not n.inputs["Vector"].links:
                    uvn = nt.nodes.new("ShaderNodeUVMap")
                    uvn.uv_map = src_uv
                    nt.links.new(uvn.outputs["UV"], n.inputs["Vector"])
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img
        nt.nodes.active = t
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=8, use_clear=True, uv_layer="baked")
    for uv in [u for u in obj.data.uv_layers if u.name != "baked"]:
        obj.data.uv_layers.remove(uv)
    img.pack()
    _one_material(obj, img)


def render_reference(obj, kind: str, out_dir: Path, size: int) -> str:
    """The model on a transparent ground from its kind's view, orthographic, its own material, plain light."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(-20))
    scene.collection.objects.link(sun)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.1
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    az, el = (math.radians(x) for x in VIEWS.get(kind, (0.0, 0.0)))
    direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    cam.location = direction * 3.0
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "input_rgba.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)


def main() -> None:
    a = job_args()
    clear_scene()
    objs, rig = baked_meshes(_import(Path(a["model"])))
    cleared = sum(clear_faces(o) for o in objs)
    obj = join(objs)
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    res = {"rig_applied": rig, "clear_faces_removed": cleared, "materials": len([m for m in me.materials if m]), "stood_up": None}
    if a.get("kind") == "soldier":
        res["stood_up"] = stand_up(obj)
    if a.get("turn") is not None:
        turn, res["turn_from"] = float(a["turn"]) % 360.0, "asked"
    elif a.get("kind") == "soldier":
        turn, res["toes"] = facing_from_feet(obj)
        res["turn_from"] = "feet"
        if res["toes"] < TOES_MIN:  # no feet to go by (Mambo, a ball, 2026-10-29: turned 90 degrees on nothing)
            turn, res["turn_from"] = 0.0, "kept (no clear feet)"
    else:
        turn, res["turn_from"] = 0.0, "kept"
    if turn:
        obj.data.transform(Matrix.Rotation(math.radians(turn), 4, "Z"))
    res["turn"] = turn
    normalise(obj)
    obj.data.update()
    plain = None if one_texture(obj) else plain_colours(obj)
    res["colours"] = "kept" if one_texture(obj) else "palette" if plain else "baked"
    res["baked"] = res["colours"] == "baked"
    if plain:
        palette(obj, plain)
    elif res["baked"]:  # a dense mesh's islands are small: a bigger texture
        bake_colours(obj, int(a.get("bake_size", 2048)) * (2 if len(obj.data.polygons) > 150000 else 1))
    lo, hi = extent(obj)
    res["dimensions"] = [round(x, 4) for x in (hi - lo)]
    res["triangles"] = len(obj.data.polygons)
    if a.get("kind") == "soldier":  # a soldier is fitted to a human skeleton: say so early if it isn't one
        d = res["dimensions"]
        ratio = d[2] / max(d[0], d[1], 1e-9)
        res["person_ratio"] = round(ratio, 2)
        if ratio < PERSON_MIN:
            res["warning"] = (f"This model isn't shaped like a standing person (only {ratio:.1f} times as tall as it is "
                              "wide): a soldier is fitted to the game's human skeleton, so it will fold and squash.")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    Path(a["out_glb"]).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(a["out_glb"]), export_format="GLB", use_selection=True,
                              export_animations=False, export_skins=False, export_morph=False, export_yup=True)
    res["reference"] = render_reference(obj, a.get("kind", "soldier"), Path(a["out_dir"]), int(a.get("size", 1024)))
    write_result(a, res)


main()
