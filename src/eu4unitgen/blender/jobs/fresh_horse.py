"""Blender job (Track C, test 15, steps 4-7): a new horse body on the vanilla horse skeleton.

Works in PDX mesh space throughout (Y up, the horse facing -Z, its left side at +X), so the
vanilla horse (given as arrays, not imported) and its bones are where the .mesh and .anim say.

source "model": a generated model file (glTF: Y up, facing +Z; PDX = (x, y, -z)), then
  clean-up: keep the largest piece; fit (uniform scale by nose-to-tail length, hooves on the
  ground, centred), then optionally the landmark fit; cut the rider: everything above ``cut.y``
  between ``cut.z0`` and ``cut.z1`` (the saddle region; head, neck and tail stay); keep the
  largest piece again; fill holes; optionally a voxel remesh into one closed manifold solid.
source "control": a crude horse from capsules along the vanilla bones, voxel-remeshed into one
  closed surface (a pipeline control only, never shipped).
Then both: an even-sized remesh to about ``target_tris`` (QuadriFlow, then a collapse trim),
one piece kept, Smart UV Project, and weights by
nearest-surface interpolation from the vanilla horse (Data Transfer), smoothed ``smooth``
times, limited to ``influences`` bones per vertex, normalised.

args: {"vanilla": {"p", "tri", "weights": [{bone: w}], "bones": {name: [x, y, z]}},
       "source": {"kind": "model", "path"} | {"kind": "control"},
       "cut": {"y", "z0", "z1"} | null, "voxel": size | null (model: voxel remesh after the cut),
       "anatomy": bool (model: the landmark fit after the box fit),
       "target_tris", "quads" (QuadriFlow's target; default target_tris / 2), "relax", "smooth", "influences",
       "geometry_out": .json path, "preview_out": .obj path (Blender axes, for renders), "result_path"}
Result: counts per stage, the fit (scale, offset) and back heights (ours vs vanilla).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import _pdx  # noqa: E402


def mesh_object(name, verts, tris):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(t) for t in tris])
    me.validate()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def activate(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def keep_largest(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    seen, best = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, island = [v], []
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            island.append(cur)
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        if len(island) > len(best):
            best = island
    keep = {v.index for v in best}
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def count_pieces(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    seen, n = set(), 0
    for v in bm.verts:
        if v.index in seen:
            continue
        n += 1
        stack = [v]
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
    bm.free()
    return n


def import_model(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"can't import {path}")
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    activate(objs[0])
    for o in objs:
        o.select_set(True)
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # the importer turned glTF (x, y, z) into Blender (x, -z, y); PDX wants (x, y, -z) of glTF
    m = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    obj.data.transform(m)
    obj.data.flip_normals()  # the mirror turned the faces inside out
    obj.data.update()
    for o in [o for o in bpy.data.objects if o not in before and o is not obj]:
        bpy.data.objects.remove(o, do_unlink=True)
    return obj


def bounds(obj):
    vs = [v.co for v in obj.data.vertices]
    lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    return lo, hi


def top_at(obj, z0, z1, x_lim):
    """The highest surface point of ``obj`` in the slab z0..z1, |x| < x_lim (None if empty)."""
    ys = [v.co.y for v in obj.data.vertices if z0 <= v.co.z <= z1 and abs(v.co.x) < x_lim]
    return max(ys) if ys else None


def fit(obj, vanilla):
    vlo, vhi = bounds(vanilla)
    lo, hi = bounds(obj)
    s = (vhi.z - vlo.z) / max(hi.z - lo.z, 1e-6)
    obj.data.transform(Matrix.Scale(s, 4))
    lo, hi = bounds(obj)
    off = Vector((-(lo.x + hi.x) / 2, vlo.y - lo.y, vlo.z - lo.z))
    obj.data.transform(Matrix.Translation(off))
    obj.data.update()
    return s, list(off)


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _rotate_yz(co, pivot, angle):
    """``co`` turned by ``angle`` (radians) about the X axis through ``pivot`` (the side plane)."""
    d = co - pivot
    c, s = math.cos(angle), math.sin(angle)
    return pivot + Vector((d.x, d.y * c - d.z * s, d.y * s + d.z * c))


def hooves(obj, mid_z):
    """Per leg (front/back x left/right) the lowest region's centre (vertices within 0.6 of its floor)."""
    out = {}
    for end in ("Front", "Back"):
        for side in ("Left", "Right"):
            vs = [v.co for v in obj.data.vertices
                  if ((v.co.z < mid_z) == (end == "Front")) and ((v.co.x > 0) == (side == "Left")) and v.co.y < 6]
            if not vs:
                continue
            floor = min(v.y for v in vs)
            low = [v for v in vs if v.y < floor + 0.6]
            out[f"{end}{side}"] = Vector((sum(v.x for v in low) / len(low), floor, sum(v.z for v in low) / len(low)))
    return out


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def nose(o):
    """The centre of the front 3 units of the mesh above y 9 (the head, not a reaching hoof)."""
    pts = [v.co for v in o.data.vertices if v.co.y > 9]
    front = min(c.z for c in pts)
    pts = [c for c in pts if c.z < front + 3.0]
    return sum(pts, Vector()) / len(pts)


def anatomy_fit(obj, vanilla, bones):
    """Landmark fit after the box fit (test 15: TripoSG's proportions and pose aren't the vanilla
    skeleton's). The box fit gets the length right (nose and legs line up along the body: measured
    side profiles) but the generated horse stands lower: scale the height (and width) to the croup,
    turn the head and neck about the neck base onto the vanilla nose, and swing each leg about
    its top joint until its hoof is under the vanilla heel (straightens a lifted or striding leg)."""
    B = {k: Vector(v) for k, v in bones.items()}
    log = {}
    croup_z = (B["Saddle_Node"].z + 3.0, B["Saddle_Node"].z + 4.0)
    sy = top_at(vanilla, *croup_z, 1.0) / top_at(obj, *croup_z, 1.0)
    for v in obj.data.vertices:
        v.co.y *= sy
        v.co.x *= sy
    log.update(scale_xy=sy)
    pivot = B["HorseNeckBase"]
    ours_n, van_n = nose(obj), nose(vanilla)
    turn = _wrap(math.atan2(van_n.y - pivot.y, van_n.z - pivot.z) - math.atan2(ours_n.y - pivot.y, ours_n.z - pivot.z))
    reach = (van_n - pivot).length / max((ours_n - pivot).length, 1e-6)
    for v in obj.data.vertices:
        w = _smooth((pivot.z - v.co.z) / 2.5)  # 0 behind the neck base, 1 from 2.5 in front of it
        if w > 0:
            d = (v.co - pivot) * (1 + (reach - 1) * w)
            d.x = v.co.x - pivot.x
            v.co = _rotate_yz(pivot + d, pivot, turn * w)
    log.update(nose={"ours": list(ours_n), "vanilla": list(van_n)}, head_turn_deg=math.degrees(turn), head_reach=reach)
    mid_z = (B["HorseFrontLeftUp"].z + B["HorseBackLeftUp"].z) / 2
    legs = {}
    for key, hoof in hooves(obj, mid_z).items():
        top, heel = B[f"Horse{key}Up"], B[f"Horse{key}Heel"]
        turn = _wrap(math.atan2(0.0 - top.y, heel.z - top.z) - math.atan2(hoof.y - top.y, hoof.z - top.z))
        left, front = key.endswith("Left"), key.startswith("Front")
        for v in obj.data.vertices:
            if (v.co.x > 0) != left or (v.co.z < mid_z) != front:
                continue
            w = _smooth((top.y - 1.0 - v.co.y) / 2.5)  # 0 above the joint, 1 from 3.5 below it
            if w > 0:
                v.co = _rotate_yz(v.co, top, turn * w)
        legs[key] = {"hoof": list(hoof), "turn_deg": math.degrees(turn)}
    log["legs"] = legs
    obj.data.update()
    return log


def cut_rider(obj, cut):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    gone = [v for v in bm.verts if v.co.y > cut["y"] and cut["z0"] <= v.co.z <= cut["z1"]]
    bmesh.ops.delete(bm, geom=gone, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(gone)


def fill_holes(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = [e for e in bm.edges if e.is_boundary]
    before = len(edges)
    if edges:
        res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
        bmesh.ops.triangulate(bm, faces=res["faces"])
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return before


def capsule(name, a, b, r):
    a, b = Vector(a), Vector(b)
    d = b - a
    length = max(d.length, 1e-3)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=r, depth=length, location=(a + b) / 2)
    cyl = bpy.context.active_object
    cyl.rotation_mode = "QUATERNION"
    cyl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts = [cyl]
    for c in (a, b):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=r, location=c)
        parts.append(bpy.context.active_object)
    return parts


def control_horse(bones):
    """Capsules along the vanilla bones (radius by body part), joined and voxel-remeshed."""
    B = {k: Vector(v) for k, v in bones.items()}
    parts = []
    body_a = B["HorseSpineEnd"] + Vector((0, -1.6, 0.6))
    body_b = B["HorseSpineTop"] + Vector((0, -1.8, -1.2))
    parts += capsule("body", body_a, body_b, 2.7)
    parts += capsule("neck", B["HorseNeckBase"] + Vector((0, -0.8, 0)), B["HorseHead"], 1.25)
    head_tip = B["HorseHead"] + Vector((0, -2.6, -3.0))
    parts += capsule("head", B["HorseHead"], head_tip, 0.95)
    for side in ("Left", "Right"):
        for end in ("Front", "Back"):
            chain = [f"Horse{end}{side}{p}" for p in ("Up", "Down", "Knee", "Heel")]
            pts = [B[c] for c in chain] + [Vector((B[chain[-1]].x, 0.15, B[chain[-1]].z))]
            for i in range(len(pts) - 1):
                parts += capsule(f"leg{i}", pts[i], pts[i + 1], 0.75 if i == 0 else 0.42)
    tail = [B[t] for t in ("HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd")]
    for i in range(len(tail) - 1):
        parts += capsule(f"tail{i}", tail[i], tail[i + 1], 0.45 - 0.08 * i)
    activate(parts[0])
    for p in parts:
        p.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mod = obj.modifiers.new("remesh", "REMESH")
    mod.mode, mod.voxel_size = "VOXEL", 0.22
    bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "control_horse"
    return obj


def decimate(obj, target, log=None, quads=None):
    """Down to about ``target`` triangles with even-sized faces: QuadriFlow quads (target / 2),
    split into triangles. Collapse decimation leaves long slivers across flat areas, and an edge
    spanning several bones stretches badly when they bend (test 15, control horse). QuadriFlow
    needs a closed manifold input (a voxel remesh gives one) and otherwise does nothing; the
    final collapse then does all the work, which ``quadriflow_ran`` reports."""
    activate(obj)
    log = log if log is not None else {}
    log["before_quadriflow"] = tri_count(obj)
    status = bpy.ops.object.quadriflow_remesh(target_faces=max(100, int(quads or target // 2)), use_preserve_sharp=False,
                                              use_preserve_boundary=False, use_mesh_symmetry=False,
                                              smooth_normals=True, mode="FACES", seed=0)
    log["quadriflow"] = sorted(status)
    log["after_quadriflow"] = tri_count(obj)
    log["quadriflow_ran"] = log["after_quadriflow"] != log["before_quadriflow"]  # it refuses non-manifold input silently
    mod = obj.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if tri_count(obj) > target:  # QuadriFlow lands near, not on, the target
        mod = obj.modifiers.new("trim", "DECIMATE")
        mod.decimate_type, mod.ratio = "COLLAPSE", target / tri_count(obj)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return tri_count(obj)


def smart_uv(obj):
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")


def transfer_weights(obj, vanilla, smooth, influences):
    for g in vanilla.vertex_groups:
        obj.vertex_groups.new(name=g.name)
    activate(obj)
    mod = obj.modifiers.new("weights", "DATA_TRANSFER")
    mod.object = vanilla
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.modifier_apply(modifier=mod.name)
    rows = [{obj.vertex_groups[g.group].name: g.weight for g in v.groups if g.weight > 0}
            for v in obj.data.vertices]
    neighbours = [[] for _ in rows]
    for e in obj.data.edges:
        a, b = e.vertices
        neighbours[a].append(b)
        neighbours[b].append(a)
    for _ in range(int(smooth or 0)):  # joint smoothing: each vertex halfway to its neighbours' mean
        new = []
        for i, row in enumerate(rows):
            if not neighbours[i]:
                new.append(row)
                continue
            mean = {}
            for j in neighbours[i]:
                for k, w in rows[j].items():
                    mean[k] = mean.get(k, 0.0) + w / len(neighbours[i])
            new.append({k: 0.5 * row.get(k, 0.0) + 0.5 * mean.get(k, 0.0) for k in set(row) | set(mean)})
        rows = new
    for v, row in zip(obj.data.vertices, rows):
        top = sorted(((k, w) for k, w in row.items() if w >= 0.01), key=lambda kv: -kv[1])[:int(influences)]
        total = sum(w for _, w in top) or 1.0
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        for k, w in top:
            obj.vertex_groups[k].add([v.index], w / total, "REPLACE")


def export(obj, geometry_out, preview_out):
    me = obj.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.data
    names = {g.index: g.name for g in obj.vertex_groups}
    # split vertices by UV seam: one PDX vertex per (vertex, uv) pair
    key_to_index, p, n, u0, weights, tri = {}, [], [], [], [], []
    normals = [v.normal.copy() for v in me.vertices]
    for lt in me.loop_triangles:
        face = []
        for li in lt.loops:
            vi = me.loops[li].vertex_index
            u, v = uv[li].uv
            key = (vi, round(u, 6), round(v, 6))
            if key not in key_to_index:
                key_to_index[key] = len(p) // 3
                co = me.vertices[vi].co
                p += [co.x, co.y, co.z]
                nn = normals[vi]
                n += [nn.x, nn.y, nn.z]
                u0 += [u, 1.0 - v]  # Blender UV -> image convention (SPEC: io_pdx_mesh flips V)
                w = {names[g.group]: g.weight for g in me.vertices[vi].groups if g.weight > 0}
                total = sum(w.values()) or 1.0
                weights.append({k: x / total for k, x in w.items()})
            face.append(key_to_index[key])
        tri += face
    unweighted = sum(1 for w in weights if not w)
    with open(geometry_out, "w", encoding="utf-8") as f:
        json.dump({"p": p, "n": n, "u0": u0, "tri": tri, "weights": weights}, f)
    if preview_out:  # PDX -> Blender axes as io_pdx_mesh shows meshes: (x, y, z) -> (x, z, y)
        prev = obj.copy()
        prev.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(prev)
        prev.data.transform(Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
        prev.data.flip_normals()
        activate(prev)
        bpy.ops.wm.obj_export(filepath=preview_out, export_selected_objects=True, export_uv=True,
                              export_materials=False)
    return len(p) // 3, len(tri) // 3, unweighted


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    vp = van["p"]
    vanilla = mesh_object("vanilla", [vp[i:i + 3] for i in range(0, len(vp), 3)],
                          [van["tri"][i:i + 3] for i in range(0, len(van["tri"]), 3)])
    for vi, row in enumerate(van["weights"]):
        for bone, w in row.items():
            g = vanilla.vertex_groups.get(bone) or vanilla.vertex_groups.new(name=bone)
            g.add([vi], w, "REPLACE")
    res = {"stages": {}}
    src = a["source"]
    if src["kind"] == "model":
        obj = import_model(src["path"])
        res["stages"]["imported"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        keep_largest(obj)
        res["stages"]["largest_piece"] = {"tris": tri_count(obj)}
        s, off = fit(obj, vanilla)
        res["fit"] = {"scale": s, "offset": off}
        if a.get("anatomy"):
            res["fit"]["anatomy"] = anatomy_fit(obj, vanilla, van["bones"])
        cut = a.get("cut")
        if cut:
            res["stages"]["rider_cut"] = {"vertices_removed": cut_rider(obj, cut)}
            keep_largest(obj)
            res["stages"]["rider_cut"]["tris"] = tri_count(obj)
        res["stages"]["holes_filled"] = {"boundary_edges": fill_holes(obj), "tris": tri_count(obj)}
        if a.get("voxel"):  # one closed manifold solid (QuadriFlow needs one): thin sheets (skirts, fringes,
            activate(obj)   # reins) thickened, inner faces and the cut's opening gone
            mod = obj.modifiers.new("solid", "REMESH")
            mod.mode, mod.voxel_size = "VOXEL", float(a["voxel"])
            bpy.ops.object.modifier_apply(modifier=mod.name)
            res["stages"]["voxel_remesh"] = {"voxel": a["voxel"], "tris": tri_count(obj), "pieces": count_pieces(obj)}
            keep_largest(obj)
    else:
        obj = control_horse(van["bones"])
        res["stages"]["built"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        res["fit"] = {"scale": 1.0, "offset": [0, 0, 0], "note": "built on the skeleton: no fit"}
    lo, hi = bounds(obj)
    vlo, vhi = bounds(vanilla)
    res["bounds"] = {"ours": [list(lo), list(hi)], "vanilla": [list(vlo), list(vhi)]}
    saddle = van["bones"]["Saddle_Node"]
    res["back"] = {}
    for label, (z0, z1) in (("withers", (saddle[2] - 3.4, saddle[2] - 2.4)), ("croup", (saddle[2] + 3.0, saddle[2] + 4.0)),
                            ("saddle", (saddle[2] - 0.5, saddle[2] + 0.5))):
        res["back"][label] = {"ours": top_at(obj, z0, z1, 1.0), "vanilla": top_at(vanilla, z0, z1, 1.0)}
    res["saddle_node_y"] = saddle[1]
    if a.get("relax"):  # even out voxel stair-steps before the reduction
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(a["relax"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    log = {}
    res["stages"]["decimated"] = {"tris": decimate(obj, int(a["target_tris"]), log, a.get("quads")), **log}
    if count_pieces(obj) > 1:  # remeshing can leave a stray shell
        keep_largest(obj)
        res["stages"]["decimated"]["tris_one_piece"] = tri_count(obj)
    smart_uv(obj)
    transfer_weights(obj, vanilla, a.get("smooth", 2), a.get("influences", 3))
    verts, tris, unweighted = export(obj, a["geometry_out"], a.get("preview_out"))
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": count_pieces(obj)}
    _pdx.write_result(a, res)


main()
