"""Blender job (Track C, test 15, steps 4-7): a new horse body on the vanilla horse skeleton.

Works in PDX mesh space throughout (Y up, the horse facing -Z, its left side at +X), so the
vanilla horse (given as arrays, not imported) and its bones are where the .mesh and .anim say.

source "model": a generated model file (glTF: Y up, facing +Z; PDX = (x, y, -z)), then
  clean-up: keep the largest piece; fit (uniform scale by nose-to-tail length, hooves on the
  ground, centred), then optionally the reshape onto the vanilla rest pose and a fine voxel
  solid; the rider cut off with a boolean box above the saddle line (head, neck and tail stay);
  optionally a one-sided bulge pushed in, the skirts cut at a hem (legs and tail kept), the
  caparison pushed off the tail; a coarse voxel solid (a voxel remesh needs closed input, and
  booleans on a closed solid keep it closed).
source "control": a crude horse from capsules along the vanilla bones, voxel-remeshed into one
  closed surface (a pipeline control only, never shipped).
Then both: an even-sized remesh to about ``target_tris`` (QuadriFlow, then a collapse trim),
one piece kept, Smart UV Project, and weights by
nearest-surface interpolation from the vanilla horse (Data Transfer), smoothed ``smooth``
times, limited to ``influences`` bones per vertex, normalised.

args: {"vanilla": {"p", "tri", "weights": [{bone: w}], "bones": {name: [x, y, z]}},
       "source": {"kind": "model", "path"} | {"kind": "control"} | {"kind": "procedural", "fit_from": OBJ | null,
       "voxel", "relax"} (bake-off horse B: _horse_proc's own body fitted to the reference),
       "cut": {"y", "z0", "z1"} | null, "voxel": size | null (model: voxel remesh after the cut),
       "reshape": bool (model: height to the croup, then _horse_shape.reshape onto the vanilla rest pose),
       "leg_top": y up to which legs are tracked, "cut".y may be "saddle" (vanilla's back height there),
       "lump": {"z0", "z1", "y0", "y1", "margin"} (mirror-based removal of a one-sided bulge),
       "voxel_fine": size (a closed solid before the edits), "hem_y", "leg_radius" (skirts raised to the
       hem), "neck_allow" (the neck thinned to vanilla's thickness x this), "flank_allow" (each flank
       narrowed to vanilla's half-width x this), "leg_allow" (thin legs thickened to vanilla's radius x this), "tail_gap" (the caparison
       pushed off the tail),
       "chanfron": add_chanfron's keywords (a plate made from scratch, merged in, 100% HorseHead),
       "tail": {"build": "auto" | true | false, "cut": stub cut radius, _horse_barding.tail's sides, rings,
       radii, flat} (round 3: with "auto", built when _horse_measure.tail_state says the generated
       one is missing, fused or a stub; a fused tail or stub is cut off first, before the bake source
       is taken),
       "barding": {"style": "drape" (_horse_barding.drape: draped cloth) | "slab" (panels(), test 16's), + its keywords} (round 3: hanging panels, weighted from the body
       without legs), "stirrups": {"tread", "strap_top"} (round 3: 100% HorseSpineTop),
       "paint": {"palette": {silver, red, gold, coat}, "out_dir", "name", "size", "hem_y" (the coat below it;
       default the skirt hem_y)} (step 8: textures),
       "chain_weights": bool (head, leg and tail regions weighted by distance to their bone chains),
       "target_tris", "quads" (QuadriFlow's target; default target_tris / 2), "relax", "smooth", "influences",
       "geometry_out": .json path, "preview_out": .obj path (Blender axes, for renders),
       "bake_source_out": .obj path (preview axes; model only): the fine surface for bake_horse, with
       every reshape the low mesh gets (the edits, the tail cleared, the saddle matched), "result_path"}
Result: counts per stage, the fit (scale, offset) and back heights (ours vs vanilla); "steep_reshapes":
the reshapes whose steepness max is over STEEP (steps the bake would show).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
from mathutils import Matrix, Vector  # noqa: E402 - Blender module

import numpy as np  # noqa: E402 - Blender's numpy

import _horse_barding as hb  # noqa: E402
import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402

STEEP = 2.0  # _horse_shape.steepness max over which a reshape is reported (smooth warps here: about 1.5)


def mesh_object(name, verts, tris):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(t) for t in tris])
    me.validate()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def activate(obj):
    bpy.context.view_layer.update()
    for o in bpy.context.view_layer.objects:
        if o is not None:
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


def scale_height(obj, vanilla, bones):
    """Height and width scaled so the croup (the back 3-4 behind Saddle_Node, |x| < 1) is vanilla's:
    the box fit gets the length right, but the generated horse stands lower (side profiles)."""
    z = bones["Saddle_Node"][2]
    s = top_at(vanilla, z + 3.0, z + 4.0, 1.0) / top_at(obj, z + 3.0, z + 4.0, 1.0)
    for v in obj.data.vertices:
        v.co.x *= s
        v.co.y *= s
    obj.data.update()
    return s


def delete_vertices(obj, indices):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    gone = set(indices)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index in gone], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(gone)


def solidify(obj, voxel):
    """A voxel remesh into one closed manifold solid (thin sheets thickened, inner faces gone),
    largest piece kept. Its input must be closed: holes let the inside leak and the body vanishes."""
    activate(obj)
    mod = obj.modifiers.new("solid", "REMESH")
    mod.mode, mod.voxel_size = "VOXEL", float(voxel)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    out = {"voxel": voxel, "tris": tri_count(obj), "pieces": count_pieces(obj)}
    keep_largest(obj)
    out["tris_kept"] = tri_count(obj)
    out["normals_flipped"] = outward(obj)
    return out


def outward(obj):
    """Face normals made consistent and pointing out of the (closed) solid; booleans and the
    voxel remesh read inside-out meshes inverted. -> whether they had pointed in."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = sum(f.normal.dot(f.calc_center_median()) for f in bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    after = sum(f.normal.dot(f.calc_center_median()) for f in bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return (before > 0) != (after > 0)


def box(name, lo, hi):
    bpy.ops.mesh.primitive_cube_add(size=1, location=[(a + b) / 2 for a, b in zip(lo, hi)])
    o = bpy.context.active_object
    o.name = name
    o.scale = [b - a for a, b in zip(lo, hi)]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o


def join(objs):
    activate(objs[0])
    for o in objs:
        o.select_set(True)
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def boolean(obj, cutter, operation="DIFFERENCE"):
    activate(obj)
    mod = obj.modifiers.new("bool", "BOOLEAN")
    mod.operation, mod.object, mod.solver = operation, cutter, "EXACT"
    mod.use_self, mod.use_hole_tolerant = True, True  # cutters made of overlapping capsules
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def cut_rider_box(obj, cut):
    """The rider removed with a boolean: a box above ``cut.y`` between ``cut.z0`` and ``cut.z1``
    (head, neck and tail stay). On a closed solid the result is closed, with a flat cap at
    exactly ``cut.y``: the back the rider sits on."""
    before = tri_count(obj)
    boolean(obj, box("rider_cutter", (-20, cut["y"], cut["z0"]), (20, 60, cut["z1"])))
    return before - tri_count(obj)


def trim_skirts(obj, bones, hem_y, leg_radius):
    """Skirts and caparison end at ``hem_y`` (above the knees and hocks) so the legs swing free:
    a boolean cut of everything below the hem except capsules of ``leg_radius`` round the vanilla
    leg chains (the legs are on them after the reshape) and round the hanging tail."""
    B = bones
    cutter = box("skirt_cutter", (-20, -5, -40), (20, hem_y, 40))
    keep = []
    for leg in hs.LEGS:
        pts = hs.leg_chain({k: np.array(v) for k, v in B.items()}, leg)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            if p0[1] < hem_y + 1:
                keep += capsule("leg_keep", p0.tolist(), p1.tolist(), leg_radius)
    tail = [B[t] for t in ("HorseTailMiddle", "HorseTailTip", "HorseTailEnd")]
    for p0, p1 in zip(tail[:-1], tail[1:]):
        keep += capsule("tail_keep", p0, p1, 0.6)
    keeper = join(keep)
    for o in (keeper, cutter):
        activate(o)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    boolean(cutter, keeper)
    before = tri_count(obj)
    boolean(obj, cutter)
    return before - tri_count(obj)


def clear_tail(obj, bones, gap):
    """The caparison pushed out to ``gap`` beyond the hanging tail (TailMiddle down; tail radius
    0.45): every vertex off the centre line (|x| > 0.45) closer than that moves straight away from
    the tail, so the tail and the caparison stay apart after the voxel solid."""
    B = {k: np.array(v) for k, v in bones.items()}
    co = hs.verts(obj)
    dist, point = hs.nearest_on_chains(co, [hs.chain(B, ["HorseTailMiddle", "HorseTailTip", "HorseTailEnd"])])
    move = (dist < 0.45 + gap) & (np.abs(co[:, 0]) > 0.45)
    away = co[move] - point[move]
    norm = np.linalg.norm(away, axis=1, keepdims=True)
    co[move] = point[move] + away / np.maximum(norm, 1e-6) * (0.45 + gap)
    hs.set_verts(obj, co)
    return int(move.sum())


def surface_top(obj, z, half=0.5, x=0.0):
    """The highest surface point straight below y 60 over (x, z +- ``half``): rays cast down on
    the faces (vertex samples miss a flat cap made of large faces). None if nothing is hit."""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    hits = []
    for dz in (-half, -half / 2, 0.0, half / 2, half):
        loc = bvh.ray_cast(Vector((x, 60.0, z + dz)), Vector((0, -1, 0)))[0]
        if loc is not None:
            hits.append(loc.y)
    return max(hits) if hits else None


def match_saddle(obj, vanilla, bones, tolerance=0.05, limit=1.0, also=()):
    """Our back at the saddle (surface height on the centre line within 0.5 of Saddle_Node along
    z) onto vanilla's, by a smooth vertical shift of the back (full within 1.5 of the saddle
    along z, none from 3.5; only above y 9). Saddle_Node itself never moves. A shift over
    ``limit`` is refused (something else is wrong). The same shift field moves ``also`` (the
    bake source: it must keep matching the low mesh). -> (before, after, vanilla, note)."""
    z = bones["Saddle_Node"][2]
    want = surface_top(vanilla, z)
    before = surface_top(obj, z)
    if before is None or abs(want - before) > limit:
        return before, before, want, f"refused: a shift of {want - (before or 0):.2f} is over {limit}"
    for _ in range(4):
        delta = want - surface_top(obj, z)
        if abs(delta) <= tolerance / 2:
            break
        for o in (obj, *also):
            for v in o.data.vertices:
                if v.co.y > 9.0:
                    w = 1.0 - _smooth((abs(v.co.z - z) - 1.5) / 2.0)
                    v.co.y += delta * w * _smooth((v.co.y - 9.0) / 1.5)
            o.data.update()
    return before, surface_top(obj, z), want, "matched"


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


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


def read_obj(path):
    """(positions, triangles) in PDX space from a preview-axes OBJ (as debug_dump writes them)."""
    co, tris = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            co.append((x, y, -z))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return np.array(co, dtype=np.float64), np.array(tris, dtype=np.int64)


def procedural_horse(van, src):
    """Bake-off horse B: _horse_proc's tubes on the vanilla skeleton, fitted to the reference
    model (``fit_from``: a preview-axes OBJ of it on the vanilla rest pose, e.g. a model build's
    3_rider_cut stage), joined into one closed surface by a voxel remesh (``voxel``), relaxed
    (``relax``). -> (object, fit log)."""
    import _horse_proc as hp
    vco = np.array(van["p"], dtype=np.float64).reshape(-1, 3)
    vtri = np.array(van["tri"], dtype=np.int64).reshape(-1, 3)
    fit = {"ratios": None, "raw": None, "topline": None}
    if src.get("fit_from"):
        rco, rtri = read_obj(src["fit_from"])
        fit.update(hp.fit_ratios(rco, rtri, vco, vtri, van["bones"]))
        fit["topline"] = hp.topline(rco, rtri, vco, vtri, van["bones"])
    # the template's reins and tail are left out of the samples (by their dominant bone)
    skip = {"Rem", "Stigbyglar", *hs.TAIL}
    keep = []
    for t in vtri:
        score = {}
        for vi in t:
            for bone, w in van["weights"][vi].items():
                score[bone] = score.get(bone, 0.0) + w
        keep.append(max(score, key=score.get) not in skip if score else True)
    built = hp.build(vco, vtri, van["bones"], fit["ratios"], fit["topline"], tri_keep=np.array(keep))
    objs = [mesh_object(f"proc_{i}", v.tolist(), t.tolist()) for i, (v, t) in enumerate(built["tubes"])]
    obj = join(objs)
    solidify(obj, float(src.get("voxel", 0.15)))
    if src.get("relax", 6):
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(src.get("relax", 6))
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "procedural_horse"
    return obj, {"scale": 1.0, "offset": [0, 0, 0], "note": "built on the skeleton", **fit, "build": built["log"]}


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


def add_chanfron(obj, bones, lift=0.06, thickness=0.05, start=0.12, stop=0.86, width=0.78, nose=0.45,
                 rows=8, cols=6, corner=0.35, facing=None):
    """A chanfron merged into the mesh as its own piece (part = 1; 100% HorseHead after the
    transfer): a plate made from scratch, not the head's own faces (those gave a ragged outline,
    test 16). The head's axis runs from HorseHead to the muzzle (the body point furthest along
    vanilla's poll-to-muzzle direction); the plate covers ``start``..``stop`` of it, ``rows`` x
    ``cols`` points, each cast down onto the face from above and lifted ``lift`` off it. Its
    half-width is ``width`` x the head's half-width there (rays across), narrowing to ``nose`` x
    at the muzzle end, the corners rounded (``corner``); ``thickness`` with a rim.
    -> its triangle count."""
    from mathutils.bvhtree import BVHTree
    head = Vector(bones["HorseHead"])
    axis0 = Vector((0.0, -1.5, -3.3)).normalized()
    co = hs.verts(obj)
    rel = co - np.array(head)
    tip = Vector(co[int(np.argmax(rel @ np.array(axis0)))])
    axis = tip - head
    length = axis.length
    axis.normalize()
    side = Vector((1.0, 0.0, 0.0))
    up = axis.cross(side).normalized()
    if up.y < 0:
        up = -up
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    grid = []
    for i in range(rows):
        u = start + (stop - start) * i / (rows - 1)
        c = head + axis * (u * length)
        hit = bvh.ray_cast(c + side * 4.0, -side, 8.0)[0]
        half = max((hit - c).dot(side), 0.2) if hit is not None else 0.6
        taper = width + (nose - width) * ((u - start) / (stop - start)) ** 1.5
        row = []
        for j in range(cols):
            v = -1.0 + 2.0 * j / (cols - 1)
            # rounded corners: the outer columns pull in at the two ends
            e = min((u - start) / (stop - start), (stop - u) / (stop - start)) / corner
            pull = 1.0 if abs(v) < 0.99 else 0.75 + 0.25 * min(e, 1.0)
            pos = c + side * (v * half * taper * pull)
            h = bvh.ray_cast(pos + up * 4.0, -up, 8.0)
            if h[0] is None:
                row.append(None)
                continue
            row.append(h[0] + h[1] * lift)
        grid.append(row)
    bm = bmesh.new()
    vs = [[bm.verts.new(p) if p is not None else None for p in row] for row in grid]
    for i in range(rows - 1):
        for j in range(cols - 1):
            q = [vs[i][j], vs[i][j + 1], vs[i + 1][j + 1], vs[i + 1][j]]
            if all(x is not None for x in q):
                bm.faces.new(q)
    bm.normal_update()
    if bm.faces and sum(f.normal.dot(up) for f in bm.faces) < 0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("chanfron")
    bm.to_mesh(me)
    bm.free()
    ch = bpy.data.objects.new("chanfron", me)
    bpy.context.scene.collection.objects.link(ch)
    activate(ch)
    mod = ch.modifiers.new("solid", "SOLIDIFY")
    mod.thickness, mod.offset, mod.use_rim = thickness, 1.0, True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    mod = ch.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    tris = tri_count(ch)
    for o, value in ((obj, 0), (ch, 1)):
        if o.data.attributes.get("part") is None:
            attr = o.data.attributes.new("part", "INT", "FACE")
            attr.data.foreach_set("value", [value] * len(o.data.polygons))
    activate(obj)
    ch.select_set(True)
    bpy.ops.object.join()
    return tris


def add_barding(obj, bones, barding, stirrups=None):
    """Round 3: the hanging panels and the stirrups (_horse_barding) joined in as their own
    pieces, with face attributes ``part`` and ``hem`` and a point attribute ``anchor`` (where each
    vertex takes its body weights from). -> the builders' logs."""
    body = hs.verts(obj)
    style = barding.get("style", "slab")
    kw = {k: v for k, v in barding.items() if k != "style"}
    built = [hb.drape(body, bones, **kw) if style == "drape" else hb.panels(body, bones, **kw)]
    if stirrups:
        built.append(hb.stirrups(**stirrups))
    if obj.data.attributes.get("part") is None:
        attr = obj.data.attributes.new("part", "INT", "FACE")
        attr.data.foreach_set("value", [0] * len(obj.data.polygons))
    log = {}
    for name, b in zip(("panels", "stirrups"), built):
        join_piece(obj, name, b)
        log[name] = b["log"]
    return log


def join_piece(obj, name, b):
    """A built piece (_horse_barding arrays) joined into ``obj`` as its own piece, with face
    attributes ``part`` and ``hem``, point attributes ``anchor`` and, for the tail, ``chain``
    (vertices without one read 0)."""
    if obj.data.attributes.get("part") is None:
        attr = obj.data.attributes.new("part", "INT", "FACE")
        attr.data.foreach_set("value", [0] * len(obj.data.polygons))
    me = bpy.data.meshes.new(name)
    me.from_pydata(b["v"].tolist(), [], b["f"].tolist())
    me.update()
    me.attributes.new("part", "INT", "FACE").data.foreach_set("value", list(b["part"]))
    me.attributes.new("hem", "FLOAT", "FACE").data.foreach_set("value", list(b["hem"]))
    me.attributes.new("anchor", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", b["anchor"].ravel().tolist())
    if "chain" in b:
        me.attributes.new("chain", "FLOAT", "POINT").data.foreach_set("value", list(b["chain"]))
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    activate(obj)
    o.select_set(True)
    bpy.ops.object.join()


def cut_tail_stub(obj, bones, radius):
    """A generated tail stub cut off (a capsule from TailMiddle past TailEnd: behind the rump, so
    the rump itself stays) before our own tail is built in its place. -> triangles before, after
    (the cut faces can make it more)."""
    B = {k: Vector(v) for k, v in bones.items()}
    end = B["HorseTailEnd"] + (B["HorseTailEnd"] - B["HorseTailTip"]).normalized() * 1.5
    before = tri_count(obj)
    boolean(obj, join(capsule("tail_cutter", B["HorseTailMiddle"], end, radius)))
    keep_largest(obj)
    return [before, tri_count(obj)]


def tail_anchor(obj, bones, inside=0.4, out=0.05):
    """Where our tail sits: our rump's surface point nearest vanilla's TailTop (on the centre
    line). -> (root: that point ``inside`` under the surface, buried so no gap opens when the tail
    swings; offset: from TailTop to the point ``out`` above the surface, which the tail's hanging
    nodes move by, so the tail lies on our rump instead of starting in the air behind it)."""
    from mathutils.bvhtree import BVHTree
    top = Vector(bones["HorseTailTop"])
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    loc, normal, _, _ = bvh.find_nearest(Vector((0.0, top.y, top.z)))
    if loc is None:
        return list(bones["HorseSpineEnd"]), [0.0, 0.0, 0.0]
    loc.x, normal.x = 0.0, 0.0
    normal.normalize()
    return list(loc - normal * inside), list(loc + normal * out - top)


def pin_barding(obj, bones):
    """Panel vertices take the weights of the body vertex nearest their anchor, without leg, tail
    or rein bones (a panel hangs from the body and doesn't swing with the legs: round 2's lesson);
    stirrups and straps 100% HorseSpineTop (as vanilla's, ~97%); the tail along its chain
    (_horse_barding.tail_weights). -> how many vertices."""
    from mathutils.kdtree import KDTree
    me = obj.data
    parts = face_parts(obj)
    kind = {}
    for poly, part in zip(me.polygons, parts):
        for vi in poly.vertices:
            kind[vi] = max(kind.get(vi, 0), part)
    body = [vi for vi in range(len(me.vertices)) if kind.get(vi, 0) == 0]
    tree = KDTree(len(body))
    for vi in body:
        tree.insert(me.vertices[vi].co, vi)
    tree.balance()
    anchor = [0.0] * (3 * len(me.vertices))
    me.attributes["anchor"].data.foreach_get("vector", anchor)
    chain = [0.0] * len(me.vertices)
    if me.attributes.get("chain") is not None:
        me.attributes["chain"].data.foreach_get("value", chain)
    skip = set(hs.TAIL) | {"Rem"} | {f"Horse{leg}{p}" for leg in hs.LEGS for p in ("Up", "Down", "Knee", "Heel")}
    group = {g.index: g.name for g in obj.vertex_groups}
    n = 0
    for vi, k in kind.items():
        if k < hb.CLOTH_PANEL:
            continue
        if k in (hb.IRON, hb.LEATHER):
            row = {"HorseSpineTop": 1.0}
        elif k == hb.TAIL:
            row = hb.tail_weights(chain[vi])
        else:
            _, src, _ = tree.find(Vector(anchor[3 * vi:3 * vi + 3]))
            row = {group[g.group]: g.weight for g in me.vertices[src].groups if group[g.group] not in skip and g.weight > 0}
            if not row:
                row = {"HorseSpineTop": 1.0}
        total = sum(row.values())
        for g in list(me.vertices[vi].groups):
            obj.vertex_groups[g.group].remove([vi])
        for b, w in row.items():
            (obj.vertex_groups.get(b) or obj.vertex_groups.new(name=b)).add([vi], w / total, "REPLACE")
        n += 1
    return n


def face_parts(obj):
    attr = obj.data.attributes.get("part")
    if attr is None:
        return [0] * len(obj.data.polygons)
    vals = [0] * len(obj.data.polygons)
    attr.data.foreach_get("value", vals)
    return vals


def pin_chanfron(obj, bone="HorseHead"):
    """Every vertex of a part-1 face weighted 100% to ``bone``. -> how many."""
    parts = face_parts(obj)
    pinned = {v for poly, part in zip(obj.data.polygons, parts) if part == 1 for v in poly.vertices}
    group = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
    for vi in pinned:
        for g in list(obj.data.vertices[vi].groups):
            obj.vertex_groups[g.group].remove([vi])
        group.add([vi], 1.0, "REPLACE")
    return len(pinned)


def paint_textures(obj, bones, palette, hem_y, out_dir, name, size):
    """Step 8: the textures (_horse_paint) as PNGs in ``out_dir``. -> paths and coverage."""
    import _horse_paint as hp
    me = obj.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.data
    parts = face_parts(obj)
    co = hs.verts(obj)
    tris = np.array([lt.vertices[:] for lt in me.loop_triangles])
    uvs = np.array([[uv[li].uv[:] for li in lt.loops] for lt in me.loop_triangles])
    normals = np.array([lt.normal[:] for lt in me.loop_triangles])
    centres = co[tris].mean(axis=1)
    part = np.array([parts[lt.polygon_index] for lt in me.loop_triangles])
    hem_attr = me.attributes.get("hem")
    face_hem = None
    if hem_attr is not None:
        vals = [0.0] * len(me.polygons)
        hem_attr.data.foreach_get("value", vals)
        face_hem = np.array([vals[lt.polygon_index] for lt in me.loop_triangles])
    g = obj.vertex_groups.get("HorseHead")
    head_w = np.zeros(len(me.vertices))
    if g is not None:
        for v in me.vertices:
            for e in v.groups:
                if e.group == g.index:
                    head_w[v.index] = e.weight
    region = hp.regions(centres, normals, part, bones, hem_y, face_hem, head_w[tris].mean(axis=1))
    diff, spec, normal, coverage = hp.paint(uvs, tris, co, normals, region, palette, bones, size)
    paths = {}
    for kind, arr in (("diffuse", diff), ("specular", spec), ("normal", normal)):
        h, w = arr.shape[:2]
        rgba = arr if arr.shape[2] == 4 else np.concatenate([arr, np.full((h, w, 1), 255, np.uint8)], axis=2)
        img = bpy.data.images.new(f"{name}_{kind}", w, h, alpha=True)
        img.pixels.foreach_set((rgba[::-1].astype(np.float32) / 255.0).ravel())  # Blender rows run up
        path = os.path.join(out_dir, f"{name}_{kind}.png")
        img.filepath_raw, img.file_format = path, "PNG"
        img.save()
        paths[kind] = path
    counts = {int(r): int((region == r).sum()) for r in np.unique(region)}
    return {"paths": paths, "coverage": coverage, "faces_by_region": counts}


def smart_uv(obj):
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")


def transfer_weights(obj, vanilla, smooth, influences, bones=None, stats=None):
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
    if bones is not None:  # A: chain weights in the head, leg and tail regions; the torso keeps the surface's
        co = hs.verts(obj)
        chain, counts = hs.chain_weights(co, bones)
        cleaned = hs.clean_surface_rows(co, rows, chain, bones)
        rows = [c if c is not None else r for c, r in zip(chain, rows)]
        if stats is not None:
            stats["chain_regions"], stats["surface_rows_cleaned"] = counts, cleaned
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
    parts = face_parts(obj)
    tri_part = [parts[lt.polygon_index] for lt in me.loop_triangles]
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
        json.dump({"p": p, "n": n, "u0": u0, "tri": tri, "weights": weights, "part": tri_part}, f)
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


def debug_dump(a, obj, stage):
    """With ``debug_dir``: the mesh after ``stage`` as an OBJ (preview axes), for looking at."""
    if a.get("debug_dir"):
        write_obj(obj, os.path.join(a["debug_dir"], f"{stage}.obj"))


def write_obj(obj, path):
    """``obj`` as an OBJ file in preview axes (PDX (x, y, z) -> Blender (x, z, y))."""
    prev = obj.copy()
    prev.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(prev)
    prev.data.transform(Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
    activate(prev)
    bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                          export_uv=False, export_materials=False)
    bpy.data.objects.remove(prev, do_unlink=True)


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
        if a.get("reshape"):
            res["fit"]["height_scale"] = scale_height(obj, vanilla, van["bones"])
            res["fit"]["reshape"] = hs.reshape(obj, vanilla, van["bones"], leg_top=float(a.get("leg_top", 6.0)))
        debug_dump(a, obj, "1_reshaped")
        if a.get("voxel_fine"):  # closed before any edit (TripoSG's pieces and open sheets)
            res["stages"]["solid_fine"] = solidify(obj, a["voxel_fine"])
        debug_dump(a, obj, "2_solid_fine")
        cut = a.get("cut")
        if cut:
            if cut.get("y") == "saddle":  # the cap at vanilla's back height at the saddle
                z = van["bones"]["Saddle_Node"][2]
                cut = {**cut, "y": top_at(vanilla, z - 0.5, z + 0.5, 1.0)}
            res["stages"]["rider_cut"] = {"y": cut["y"], "tris_removed": cut_rider_box(obj, cut)}
        debug_dump(a, obj, "3_rider_cut")
        lump = a.get("lump")
        if lump:
            res["stages"]["lump_flattened"] = hs.flatten_lump(
                obj, lump["z0"], lump["z1"], lump["y0"], lump["y1"], lump.get("margin", 0.3))
        if a.get("hem_y"):
            res["stages"]["skirts_trimmed"] = {"hem_y": a["hem_y"], "tris_removed": trim_skirts(
                obj, van["bones"], float(a["hem_y"]), float(a.get("leg_radius", 0.9)))}
            keep_largest(obj)
        if a.get("neck_allow"):  # round 3: the neck no thicker than vanilla's x neck_allow
            res["stages"]["neck_warp"] = hs.neck_warp(obj, vanilla, van["bones"], float(a["neck_allow"]))
        if a.get("flank_allow"):  # round 3: each flank no wider than vanilla's x flank_allow
            res["stages"]["flank_warp"] = hs.flank_warp(obj, vanilla, van["bones"], float(a["flank_allow"]))
        if a.get("leg_allow"):  # round 3: legs no thinner than vanilla's x leg_allow, before the coarse voxels
            res["stages"]["leg_inflate"] = hs.leg_inflate(obj, vanilla, van["bones"], float(a.get("hem_y") or 6.0),
                                                          float(a["leg_allow"]))
        tail = a.get("tail")
        build_tail = False
        if tail:  # round 3: our own tail when the generated one is missing, fused or a stub
            import _horse_measure as hm
            state = hm.tail_state(hs.verts(obj), hs.faces(obj), van["bones"])
            build_tail = tail.get("build", "auto") is True or (tail.get("build", "auto") == "auto" and state["state"] != "free")
            res["tail"] = {"source": state, "build": build_tail}
            if build_tail and state["state"] != "missing":
                res["tail"]["stub_cut_tris"] = cut_tail_stub(obj, van["bones"], float(tail.get("cut", 0.8)))
        debug_dump(a, obj, "4_trimmed")
        # the bake source: the fine surface after every edit so far; each later reshape of the
        # body moves it too (the voxel solid, relax and reduction don't: those are what the
        # normal map makes up for)
        high = obj.copy()
        high.data = obj.data.copy()
        high.name = "bake_source"  # not linked to the scene: the joins and selections below never see it
        if a.get("tail_gap"):
            res["stages"]["tail_cleared"] = {"vertices": clear_tail(obj, van["bones"], float(a["tail_gap"])),
                                             "bake_source_vertices": clear_tail(high, van["bones"], float(a["tail_gap"]))}
        if a.get("voxel"):  # merge what the edits made overlap; closed in, closed out
            res["stages"]["solid"] = solidify(obj, a["voxel"])
        else:
            res["stages"]["holes_filled"] = {"boundary_edges": fill_holes(obj), "tris": tri_count(obj)}
        debug_dump(a, obj, "5_solid")
    elif src["kind"] == "procedural":
        high, build_tail = None, False
        obj, res["fit"] = procedural_horse(van, src)
        res["stages"]["built"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        high = obj.copy()  # the bake source: the smooth surface before the reduction
        high.data = obj.data.copy()
        high.name = "bake_source"
        if a.get("tail"):
            import _horse_measure as hm
            state = hm.tail_state(hs.verts(obj), hs.faces(obj), van["bones"])
            build_tail = a["tail"].get("build", "auto") is True or (a["tail"].get("build", "auto") == "auto" and state["state"] != "free")
            res["tail"] = {"source": state, "build": build_tail}
        debug_dump(a, obj, "5_solid")
    else:
        high, build_tail = None, False
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
    if src["kind"] == "model" and a.get("cut"):
        before, after, want, note = match_saddle(obj, vanilla, van["bones"], also=[high])
        res["saddle"] = {"before": before, "after": after, "vanilla": want, "note": note,
                         "saddle_node_y": van["bones"]["Saddle_Node"][1]}
    if a.get("chanfron"):
        res["chanfron"] = {"tris": add_chanfron(obj, van["bones"], **a["chanfron"])}
    if build_tail:  # found on the body alone, before the panels join it
        root, offset = tail_anchor(obj, van["bones"])
    if a.get("barding"):
        res["barding"] = add_barding(obj, van["bones"], a["barding"], a.get("stirrups"))
    if build_tail:
        b = hb.tail(van["bones"], root, offset=offset,
                    **{k: v for k, v in a["tail"].items() if k in ("sides", "rings", "radii", "flat")})
        join_piece(obj, "tail", b)
        res["tail"]["built"] = {**b["log"], "offset": [round(x, 2) for x in offset]}
    smart_uv(obj)
    wstats = {}
    transfer_weights(obj, vanilla, a.get("smooth", 2), a.get("influences", 3),
                     van["bones"] if a.get("chain_weights") else None, wstats)
    res["weights"] = wstats
    if a.get("chanfron"):
        res["chanfron"]["pinned_vertices"] = pin_chanfron(obj)
    if a.get("barding") or build_tail:  # panels, stirrups and the tail: their own weights
        res.setdefault("barding", {})["pinned_vertices"] = pin_barding(obj, van["bones"])
    if a.get("paint"):
        pt = a["paint"]
        res["textures"] = paint_textures(obj, van["bones"], pt["palette"], float(pt.get("hem_y", a.get("hem_y") or 0.0)),
                                         pt["out_dir"], pt["name"], int(pt.get("size", 512)))
    verts, tris, unweighted = export(obj, a["geometry_out"], a.get("preview_out"))
    if high is not None and a.get("bake_source_out"):
        write_obj(high, a["bake_source_out"])
        res["bake_source"] = {"path": a["bake_source_out"], "tris": tri_count(high)}
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": count_pieces(obj)}
    # a reshape that moves neighbours apart steeply leaves steps on the fine surface: invisible
    # after the reduction, but the normal bake picks them up
    res["steep_reshapes"] = {k: v["steepness"] for k, v in res["stages"].items()
                             if isinstance(v, dict) and v.get("steepness", {}).get("max", 0) > STEEP}
    _pdx.write_result(a, res)


main()
