"""Blender job (runs inside Blender only), Track M finish line step 2: the master posed for the game.

Usage: blender --background --python game_master.py -- args.json

Opens the master's scene (master_assemble's master.blend), moves every part from the master's skeleton and
L's pose onto the vanilla horse skeleton at rest (_ref3d_repose: the barrel and what lies on it with the
trunk, the neck, head and legs along their bones; a part that isn't the body follows the body part it lies
on, never a leg), and writes:
  reposed.blend  the scene posed, each part's albedo material assigned (its class-ID material kept by name):
                 what the colour bake reads;
  high.obj       every part (modifiers applied, decals left out) joined: the normal bake's high surface;
  solid.obj      that surface but the cloth (panels, chest drapes, tassels) as one closed piece (a voxel remesh,
                 ``voxel``): fresh_horse's source model;
  pieces.json    the panels and chest drapes as low closed slabs (``piece_rows`` x ``piece_cols``, ``piece_thick``
                 thick; each vertex's anchor at its column's top): fresh_horse joins them after its remesh (a
                 voxel remesh erases 4 mm cloth: game v007 had no caparison);
  check.png      a side view of the posed master (Workbench).
OBJs in preview axes (file (x, y, -z) of PDX space, as fresh_horse and bake_horse read them).

args: {"master_blend", "skeleton", "body": {"shape", "skeleton", "cameras", "poses"}, "voxel",
       "grids": {panel object: [rows, cols]}, "piece_rows", "piece_cols", "piece_thick", "out_dir", "result_path"}
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Matrix  # noqa: E402 - Blender module
from mathutils.kdtree import KDTree  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "ai" / "jobs"))
from _ref3d_repose import PART_MODE, Repose  # noqa: E402 - numpy only
from _ref3d_scene import body_in_view, look_camera, render, set_engine_workbench, to_pdx, to_scene, vertex_part  # noqa: E402

PREVIEW = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # PDX (x, y, z) -> Blender (x, z, y)


def export_obj(ob, path):
    """``ob`` (PDX coordinates) as an OBJ in preview axes."""
    prev = ob.copy()
    prev.data = ob.data.copy()
    bpy.context.scene.collection.objects.link(prev)
    prev.data.transform(PREVIEW)
    prev.data.flip_normals()  # the axis swap is a mirror: without this the faces come out inside out (game v006)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    prev.select_set(True)
    bpy.context.view_layer.objects.active = prev
    bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True, export_uv=False, export_materials=False)
    bpy.data.objects.remove(prev, do_unlink=True)


def albedo_of(ob):
    """An emission material of ``ob``'s lit material's base colour (its image texture when it has one)."""
    lit = ob.active_material
    m = bpy.data.materials.new(f"{ob.name}_bake")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    em.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    if lit is not None and lit.use_nodes:
        bsdf = next((n for n in lit.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bc = bsdf.inputs["Base Color"]
            img = next((lk.from_node.image for lk in bc.links if lk.from_node.type == "TEX_IMAGE"), None)
            if img is not None:
                tx = nt.nodes.new("ShaderNodeTexImage")
                tx.image = img
                nt.links.new(tx.outputs["Color"], em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = tuple(bc.default_value)
    return m


def slab(G, body_kd, rows, cols, thick):
    """A grid (R x C x 3, PDX) as a closed slab: ``rows`` x ``cols`` samples, a back sheet ``thick`` towards the
    body, the borders closed. -> (vertices, faces, anchors)."""
    R, C = G.shape[:2]
    ri = np.unique(np.linspace(0, R - 1, rows).round().astype(int))
    ci = np.unique(np.linspace(0, C - 1, cols).round().astype(int))
    S = G[np.ix_(ri, ci)]
    r, c = S.shape[:2]
    du = np.gradient(S, axis=1)
    dv = np.gradient(S, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)
    near = np.array([body_kd.find(tuple(p))[0] for p in S.reshape(-1, 3)]).reshape(S.shape)
    if np.sum(n * (S - near)) < 0:  # the normals face away from the body
        n = -n
    front, back = S.reshape(-1, 3), (S - n * thick).reshape(-1, 3)
    V = np.vstack([front, back])
    k = r * c
    F = []
    for i in range(r - 1):
        for j in range(c - 1):
            a, b, d, e = i * c + j, i * c + j + 1, (i + 1) * c + j, (i + 1) * c + j + 1
            F += [(a, d, e), (a, e, b), (k + a, k + e, k + d), (k + a, k + b, k + e)]
    ring = [(0, j) for j in range(c)] + [(i, c - 1) for i in range(1, r)] + [(r - 1, j) for j in range(c - 2, -1, -1)] + \
           [(i, 0) for i in range(r - 2, 0, -1)]
    for (i0, j0), (i1, j1) in zip(ring, ring[1:] + ring[:1]):
        a, b = i0 * c + j0, i1 * c + j1
        F += [(a, b, k + b), (a, k + b, k + a)]
    F = np.array(F)
    # the faces' winding: outward along n on the front sheet (checked on the first triangle)
    t = V[F[0]]
    if np.cross(t[1] - t[0], t[2] - t[0]) @ n[0, 0] < 0:
        F = F[:, ::-1]
    top = np.tile(S[0], (r, 1, 1)).reshape(-1, 3)
    return V, F, np.vstack([top, top])


def main():
    t0 = time.time()
    a = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=a["master_blend"])
    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    body = a["body"]
    rp = Repose(sk, body, "L")
    _, _, _, V, F, part = body_in_view({"body": body, "view_name": "L"}, sk)
    vpart = vertex_part(F, part, len(V))
    vmode = np.array([PART_MODE.get(int(p), "trunk") for p in vpart])
    # what isn't the body follows the body part it lies on (never a leg: panels hang over the legs, they
    # don't swing with them)
    keep = np.isin(vpart, [0, 1, 2, 3, 4])
    kd = KDTree(int(keep.sum()))
    idx = np.nonzero(keep)[0]
    for i in idx:
        kd.insert(tuple(V[i]), int(i))
    kd.balance()
    res = {"parts": {}}
    meshes = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
    for ob in meshes:
        me = ob.data
        co = np.array([v.co[:] for v in me.vertices])
        X = to_pdx(co)
        if ob.name.startswith("body_") and len(X) == len(V):
            mode = vmode
            how = "body parts"
        elif ob.name == "tail" or ob.name.startswith(("panel_", "chest_")):
            # the tail hangs from the trunk; the panels, chest drapes and their tassels hang from it too (master
            # v010: in L's pose the lowered head was the nearest body part to the chest drape's top)
            mode = np.full(len(X), "trunk")
            how = "trunk"
        else:
            near = np.array([kd.find(tuple(x))[1] for x in X]) if len(X) else np.zeros(0, int)
            mode = vmode[near] if len(X) else np.zeros(0, object)
            how = "nearest body part"
        Y = rp.apply(X, mode) if len(X) else X
        me.vertices.foreach_set("co", to_scene(Y).astype(np.float32).ravel())
        me.update()
        ms = {str(m): int((mode == m).sum()) for m in np.unique(mode)}
        res["parts"][ob.name] = {"vertices": len(X), "by": how, "modes": ms}
        # its albedo for the colour bake: an emission of its lit material's base colour or image (master.blend
        # keeps only the lit materials: the albedo ones had no users when it was saved)
        alb = albedo_of(ob)
        ob.data.materials.clear()
        ob.data.materials.append(alb)
    for m in bpy.data.materials:
        m.use_fake_user = True
    # the cloth as low slabs for fresh_horse (PDX, posed)
    bco = to_pdx(np.array([v.co[:] for v in bpy.data.objects["body_coat"].data.vertices]))
    bkd = KDTree(len(bco))
    for i, p in enumerate(bco):
        bkd.insert(tuple(p), i)
    bkd.balance()
    pieces = []
    for name, (rows, cols) in (a.get("grids") or {}).items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        G = to_pdx(np.array([v.co[:] for v in ob.data.vertices]))
        if len(G) != (rows + 1) * cols:
            res.setdefault("pieces_skipped", []).append(name)
            continue
        V, Fp, anc = slab(G.reshape(rows + 1, cols, 3), bkd, a.get("piece_rows", 7), a.get("piece_cols", 12),
                          a.get("piece_thick", 0.06))
        pieces.append({"name": name, "v": V.tolist(), "f": Fp.tolist(), "anchor": anc.tolist(), "hem": float(G[-cols:, 1].min())})
    (out / "pieces.json").write_text(json.dumps(pieces), encoding="utf-8")
    res["pieces"] = {"path": str(out / "pieces.json"), "count": len(pieces), "tris": sum(len(q["f"]) for q in pieces)}
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "reposed.blend"), compress=True)
    # the high surface: every part but the decals (flat cards), modifiers applied, joined
    dg = bpy.context.evaluated_depsgraph_get()
    allV, allF, off = [], [], 0
    cloth = np.zeros(0, bool)
    clothF = []
    for ob in meshes:
        if ob.name.startswith("decal_"):
            continue
        ev = ob.evaluated_get(dg)
        m2 = ev.to_mesh()
        m2.calc_loop_triangles()
        vv = np.array([v.co[:] for v in m2.vertices])
        tt = np.array([t.vertices[:] for t in m2.loop_triangles], dtype=np.int64)
        ev.to_mesh_clear()
        if len(tt) == 0:
            continue
        allV.append(to_pdx(vv @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]))
        allF.append(tt + off)
        clothF.append(np.full(len(tt), ob.name.startswith(("panel_", "chest_"))))
        off += len(vv)
    HV, HF = np.vstack(allV), np.vstack(allF)
    me = bpy.data.meshes.new("high")
    me.from_pydata(HV.tolist(), [], HF.tolist())
    me.update()
    high = bpy.data.objects.new("high", me)
    bpy.context.scene.collection.objects.link(high)
    export_obj(high, out / "high.obj")
    res["high"] = {"path": str(out / "high.obj"), "tris": int(len(HF))}
    # one closed piece for fresh_horse: everything but the cloth (joined later as slabs)
    keepF = HF[~np.concatenate(clothF)]
    me2 = bpy.data.meshes.new("solid_src")
    me2.from_pydata(HV.tolist(), [], keepF.tolist())
    me2.update()
    solid = bpy.data.objects.new("solid", me2)
    bpy.context.scene.collection.objects.link(solid)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = solid
    solid.select_set(True)
    mod = solid.modifiers.new("voxel", "REMESH")
    mod.mode, mod.voxel_size, mod.use_remove_disconnected = "VOXEL", float(a.get("voxel", 0.1)), False
    bpy.ops.object.modifier_apply(modifier=mod.name)
    export_obj(solid, out / "solid.obj")
    res["solid"] = {"path": str(out / "solid.obj"), "tris": len(solid.data.polygons), "voxel": a.get("voxel", 0.1)}
    # a look: the posed master from its left
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH":
            o.hide_render = o.name in ("high", "solid")
    sc = bpy.context.scene
    set_engine_workbench(sc)
    lo, hi = HV.min(axis=0), HV.max(axis=0)
    c = to_scene(((lo + hi) / 2)[None])[0]
    r = 0.5 * float(np.linalg.norm(hi - lo))
    for name, az in (("check_left", 0), ("check_34", 40)):
        cam = look_camera(f"cam_{name}", c, az, 5, 1.1 * r / math.tan(math.radians(17)), 50.0)
        render(sc, cam, out / f"{name}.png", 1400, 1000)
    res["renders"] = [str(out / "check_left.png"), str(out / "check_34.png")]
    res["bones_rest"] = {k: list(map(float, v)) for k, v in rp.Br.items()}
    res["seconds"] = round(time.time() - t0, 1)
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f, default=float)


main()
