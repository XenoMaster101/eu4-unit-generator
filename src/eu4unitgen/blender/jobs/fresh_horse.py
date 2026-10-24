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
source "direct" (M21, TRELLIS.2 models): the model's own shape kept: no reshape onto vanilla's rest
  pose, no coarse voxels or QuadriFlow (see ``direct``; args "merge", "pose", "close_voxel",
  "uv_margin", "fitted_out", "textured_out").
source "control": a crude horse from capsules along the vanilla bones, voxel-remeshed into one
  closed surface (a pipeline control only, never shipped).
Then both: an even-sized remesh to about ``target_tris`` (QuadriFlow, then a collapse trim),
one piece kept, Smart UV Project, and weights by
nearest-surface interpolation from the vanilla horse (Data Transfer), smoothed ``smooth``
times, limited to ``influences`` bones per vertex, normalised.

args: {"vanilla": {"p", "tri", "weights": [{bone: w}], "bones": {name: [x, y, z]}},
       "source": {"kind": "model", "path"} | {"kind": "control"} | {"kind": "procedural", "fit_from": OBJ | null,
       "voxel", "relax"} (round 4's horse B: sections traced from vanilla's; superseded) |
       {"kind": "anatomy", "fit_from": OBJ | null, "params": {name: value}, "voxel", "relax"} (round 5's
       horse B: _horse_anatomy's own parametric horse driven by the reference),
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
       "readability": {"mode": "auto" (default) | "reference", "leg_share", "depth_floor", "hem_steps"}
       (round 6, the user's rule: at the game camera the legs show at least leg_share (default
       _horse_read.LEG_SHARE) of vanilla's visible length; "auto" first makes an anatomy barrel
       shallower, never under depth_floor (default 1.05) x vanilla's depth, then shortens a draped
       hem along hem_steps until it passes; "reference" leaves the reference's look alone and
       only reports),
       "paint": {"palette": {silver, red, gold, coat}, "out_dir", "name", "size", "hem_y" (the coat below it;
       default the skirt hem_y)} (step 8: textures),
       "chain_weights": bool (head, leg and tail regions weighted by distance to their bone chains),
       "target_tris", "quads" (QuadriFlow's target; default target_tris / 2), "relax", "smooth", "influences",
       "geometry_out": .json path, "preview_out": .obj path (Blender axes, for renders),
       "textured_out": .obj path (model only: the model with its own UVs after the fit and reshape, for
       bake_colour; M21),
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
import _horse_read as hr  # noqa: E402
import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402

STEEP = 2.0  # _horse_shape.steepness max over which a reshape is reported (smooth warps here: about 1.5)
DEPTH_FLOOR = 1.05  # the user's rule (round 6): readability never makes a barrel shallower than this x vanilla's
HEM_STEPS = tuple(-0.25 * i for i in range(9))  # drape hem_length 0 .. -2, tried in turn while the legs don't show (assumed)


class Readability:
    """The round 6 rule's settings and measurements (fresh_horse args "readability")."""

    def __init__(self, cfg, van, barding):
        cfg = cfg or {}
        self.auto = cfg.get("mode", "auto") == "auto"
        self.share = float(cfg.get("leg_share", hr.LEG_SHARE))
        self.depth_floor = float(cfg.get("depth_floor", DEPTH_FLOOR))
        self.hem_steps = [float(h) for h in cfg.get("hem_steps", HEM_STEPS)]
        self.bones = van["bones"]
        vco = np.array(van["p"], dtype=np.float64).reshape(-1, 3)
        vtri = np.array(van["tri"], dtype=np.int64).reshape(-1, 3)
        self.vanilla_arrays = (vco, vtri)
        self.vanilla = hr.leg_visible(vco, vtri, self.bones)["length"]
        self.vanilla_depth = None
        self.barding = barding if barding and barding.get("style") == "drape" else None
        self.log = {"mode": "auto" if self.auto else "reference", "leg_share": self.share,
                    "vanilla_leg": self.vanilla, "steps": []}

    def depth_ratio(self, depth):
        if self.vanilla_depth is None:
            import _horse_measure as hm
            v = self.vanilla_arrays
            self.vanilla_depth = hm.summary(hm.proportions(v[0], v[1], self.bones))["depth"]
        return depth / self.vanilla_depth

    def legs(self, co, tri, part=None, hem_length=None, extra=()):
        """Visible leg length with the drape (at ``hem_length``, default its own) hung on this body,
        and the ``extra`` pieces (_horse_barding arrays: the stirrups, the tail) beside it."""
        co, tri = np.asarray(co, dtype=np.float64), np.asarray(tri, dtype=np.int64)
        part = np.zeros(len(tri), dtype=np.int64) if part is None else np.asarray(part)
        pieces = list(extra)
        if self.barding is not None:
            kw = {k: v for k, v in self.barding.items() if k != "style"}
            if hem_length is not None:
                kw["hem_length"] = hem_length
            pieces.append(hb.drape(co, self.bones, **kw))  # on the body alone, as add_barding hangs it
        for b in pieces:
            co, tri = np.vstack([co, b["v"]]), np.vstack([tri, np.asarray(b["f"]) + len(co)])
            part = np.concatenate([part, np.asarray(b["part"])])
        return hr.leg_visible(co, tri, self.bones, part)["length"]

    def passes(self, length):
        return length >= self.share * self.vanilla

    def step(self, **kw):
        kw["leg_ratio"] = round(kw["leg"] / self.vanilla, 3)
        kw["passes"] = self.passes(kw["leg"])
        self.log["steps"].append(kw)
        return kw["passes"]


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


def shallower(ha, bones, seat, params, read):
    """The readability rule's first step (round 6): while the legs (with the caparison hung at its
    own hem) don't show enough, the barrel 0.05 x vanilla's depth shallower, never under
    ``read.depth_floor``; measured on the anatomy's tubes, before the remesh. -> (params, built)."""
    params = dict(params)
    for _ in range(8):
        built = ha.build(bones, seat, params)
        co, tri = ha._joined(built["tubes"])
        ratio = read.depth_ratio(ha.measure(built["tubes"], bones)["depth"])
        ok = read.step(stage="depth", depth=round(ratio, 3), barrel_depth=round(built["params"]["barrel_depth"], 4),
                       leg=read.legs(co, tri))
        if ok or ratio <= read.depth_floor + 0.005:
            return params, built
        want = max(read.depth_floor, ratio - 0.05)
        params["barrel_depth"] = built["params"]["barrel_depth"] * want / ratio
    return params, built


def anatomy_horse(van, vanilla, src, read=None):
    """Bake-off horse B, round 5: _horse_anatomy's own parametric horse on the vanilla skeleton
    (from vanilla only the bones and the seat height: its back's surface at Saddle_Node), its
    parameters driven by the reference model (``fit_from``: a preview-axes OBJ of it on the vanilla
    rest pose) and ``params`` overrides, joined by a voxel remesh (``voxel``) and relaxed
    (``relax``); with ``read`` (a Readability in auto mode) made shallower first (``shallower``).
    -> (object, fit log)."""
    import _horse_anatomy as ha
    bones = van["bones"]
    seat = surface_top(vanilla, bones["Saddle_Node"][2])
    params, fit = src.get("params") or {}, {"seat": seat}
    if src.get("fit_from"):
        rco, rtri = read_obj(src["fit_from"])
        fit.update(ha.fit(bones, seat, rco, rtri, params))
        params = fit["params"]
    if read is not None and read.auto:
        params, built = shallower(ha, bones, seat, params, read)
    else:
        built = ha.build(bones, seat, params)
    fit["achieved"] = ha.measure(built["tubes"], bones)
    objs = [mesh_object(f"anat_{i}", v.tolist(), t.tolist()) for i, (v, t) in enumerate(built["tubes"])]
    obj = join(objs)
    solidify(obj, float(src.get("voxel", 0.15)))
    if src.get("relax", 4):
        activate(obj)
        mod = obj.modifiers.new("relax", "SMOOTH")
        mod.factor, mod.iterations = 0.5, int(src.get("relax", 4))
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = "anatomy_horse"
    fit["params"] = {k: round(v, 4) for k, v in built["params"].items()}
    return obj, {"scale": 1.0, "offset": [0, 0, 0], "note": "the app's own anatomy on the skeleton", **fit,
                 "build": built["log"]}


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


def shorter_hem(obj, barding, read, extra=()):
    """The readability rule's second step (round 6): the drape's hem_length, from its own along
    ``read.hem_steps`` (each shorter), the first whose legs show enough on this body (the reduced
    one, with its chanfron, and the ``extra`` pieces still to join: _horse_barding arrays, e.g. the
    stirrups and the tail); in reference mode, or when none does, the shortest tried. -> hem_length."""
    own = float(barding.get("hem_length", 0.25))
    tri, part = [], []
    for poly, p in zip(obj.data.polygons, face_parts(obj)):
        v = poly.vertices
        tri += [(v[0], v[i], v[i + 1]) for i in range(1, len(v) - 1)]
        part += [p] * (len(v) - 2)
    co = hs.verts(obj)
    hem = own
    for hem in [own] + [h for h in read.hem_steps if h < own]:
        if read.step(stage="hem", hem_length=hem, leg=read.legs(co, tri, part, hem, extra)) or not read.auto:
            break
    return hem


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
    """Panel vertices take the trunk-bone weights of the body vertex nearest their anchor, blended
    over neighbouring anchors (a panel hangs from the body and doesn't swing with the legs: round 2's
    lesson; nor twist between columns: round 5's);
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
    group = {g.index: g.name for g in obj.vertex_groups}
    # panels: only the trunk's bones (round 5: with the head's and legs' too, and a different mix in
    # each column, the cloth between columns twisted when the spine bent: 21 of idle's 30 flips),
    # blended over the neighbouring columns' anchors
    trunk = {"HorseSpineTop", "HorseSpineMiddle", "HorseSpineEnd", "HorseNeckBase", "Root"}
    anchors = {}
    for vi, k in kind.items():
        if k in (hb.CLOTH_PANEL, hb.SCALE_PANEL):
            key = tuple(round(x, 3) for x in anchor[3 * vi:3 * vi + 3])
            if key not in anchors:
                _, src, _ = tree.find(Vector(key))
                row = {group[g.group]: g.weight for g in me.vertices[src].groups if group[g.group] in trunk and g.weight > 0}
                anchors[key] = row or {"HorseSpineTop": 1.0}
    keys = list(anchors)
    pos = np.array(keys) if keys else np.zeros((0, 3))
    blended = {}
    for i, key in enumerate(keys):
        w = np.exp(-(np.linalg.norm(pos - pos[i], axis=1) / 1.5) ** 2)
        acc = {}
        for j in np.flatnonzero(w > 0.05):
            tot = sum(anchors[keys[j]].values())
            for b, x in anchors[keys[j]].items():
                acc[b] = acc.get(b, 0.0) + w[j] * x / tot
        blended[key] = acc
    n = 0
    for vi, k in kind.items():
        if k < hb.CLOTH_PANEL:
            continue
        if k in (hb.IRON, hb.LEATHER):
            row = {"HorseSpineTop": 1.0}
        elif k == hb.TAIL:
            row = hb.tail_weights(chain[vi])
        else:
            row = dict(sorted(blended[tuple(round(x, 3) for x in anchor[3 * vi:3 * vi + 3])].items(), key=lambda kv: -kv[1])[:3])
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


def write_obj(obj, path, uv=False):
    """``obj`` as an OBJ file in preview axes (PDX (x, y, z) -> Blender (x, z, y)); with ``uv``, its UVs too."""
    prev = obj.copy()
    prev.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(prev)
    prev.data.transform(Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
    activate(prev)
    bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                          export_uv=uv, export_materials=False)
    bpy.data.objects.remove(prev, do_unlink=True)


def merge_seams(obj, dist):
    """TRELLIS.2's mesh is split at its UV seams: coincident vertices welded (the UVs stay per loop)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hooves(co, mid_z, ground=0.8):
    """Each leg's hoof (the centre of its vertices under ``ground``), the legs told apart by the
    side (x) and by fore (z < mid_z) or hind."""
    low = co[co[:, 1] < co[:, 1].min() + ground]
    out = {}
    for leg, side, fore in (("FrontLeft", 1, True), ("FrontRight", -1, True), ("BackLeft", 1, False), ("BackRight", -1, False)):
        sel = low[((low[:, 0] > 0) if side > 0 else (low[:, 0] <= 0)) & ((low[:, 2] < mid_z) if fore else (low[:, 2] >= mid_z))]
        out[leg] = sel.mean(axis=0) if len(sel) else None
    return out


def seat_top(obj, z, half=1.0, step=0.25):
    """Where a rider sits: the lowest of the surface heights hit by rays straight down the centre
    line from z - half to z + half (the dip between a saddle's front and back rims; on a bare back,
    the back)."""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    tops = []
    for z0 in np.arange(z - half, z + half + 1e-6, step):
        hit = bvh.ray_cast(Vector((0.0, 100.0, float(z0))), Vector((0.0, -1.0, 0.0)))
        if hit[0] is not None:
            tops.append(hit[0].y)
    return float(min(tops))


def face_forward(obj):
    """A generated model turned about the vertical to face -Z as the vanilla horse (TRELLIS.2 picks its
    own facing: horse 2 faced -Z, the cataphract +-X): its long horizontal axis (principal axis of x, z)
    onto z, then the head's end (the end fifth that reaches higher: the head and ears over the croup)
    to -z. -> the turn in degrees."""
    co = hs.verts(obj)
    xz = co[:, [0, 2]] - co[:, [0, 2]].mean(0)
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    ang = math.atan2(vt[0][0], vt[0][1])  # the axis' angle from +z towards +x
    c, s_ = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    co = co @ rot.T
    z0, z1 = co[:, 2].min(), co[:, 2].max()
    fifth = (z1 - z0) / 5
    if co[co[:, 2] > z1 - fifth, 1].max() > co[co[:, 2] < z0 + fifth, 1].max():  # the head at +z: turned round
        co[:, [0, 2]] *= -1
        ang += math.pi
    hs.set_verts(obj, co)
    return round((math.degrees(ang) + 180) % 360 - 180, 1)


def side_area(obj, cell=0.15, density=4.0):
    """The side silhouette's area (y, z), from surface samples on a ``cell`` grid."""
    me = obj.data
    me.calc_loop_triangles()
    co = hs.verts(obj)
    tri = np.array([lt.vertices[:] for lt in me.loop_triangles], dtype=np.int64)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / (cell * cell) * density).astype(np.int64))
    idx = np.repeat(np.arange(len(tri)), n)
    rng = np.random.default_rng(0)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    cells = np.unique(np.floor(pts[:, 1:] / cell).astype(np.int64), axis=0)
    return len(cells) * cell * cell


def slim(obj, vanilla, bones, target=1.15, low=0.75, points=None):
    """M21 (the user: both horses look bigger than vanilla's): a model whose side silhouette is over
    ``target`` x vanilla's is made smaller about the seat, so the rider still sits where vanilla's does:
    the body above the belly (vanilla's leg tops) scaled by s about the seat point, x and z with it, and
    the legs below stretched to keep the hooves on the ground. s = sqrt(target / ratio), at least
    ``low``. -> the ratio before and after and s."""
    before = side_area(obj) / side_area(vanilla)
    s = float(np.clip(math.sqrt(target / before), low, 1.0))
    if s < 0.999:
        z = bones["Saddle_Node"][2]
        seat = seat_top(obj, z)
        belly = min(bones["HorseFrontLeftDown"][1], bones["HorseBackLeftDown"][1])
        co = hs.verts(obj)
        out = co.copy()
        out[:, 0] = co[:, 0] * s
        out[:, 2] = z + (co[:, 2] - z) * s
        top = co[:, 1] >= belly
        out[top, 1] = seat + (co[top, 1] - seat) * s
        new_belly = seat + (belly - seat) * s
        out[~top, 1] = co[~top, 1] * new_belly / belly
        hs.set_verts(obj, out)
        if points is not None:  # other points (the head's poll) moved the same way
            for q in points:
                q[0], q[2] = q[0] * s, z + (q[2] - z) * s
                q[1] = seat + (q[1] - seat) * s if q[1] >= belly else q[1] * new_belly / belly
    return {"side_area_ratio": round(before, 3), "scale": round(s, 3),
            "after": round(side_area(obj) / side_area(vanilla), 3) if s < 0.999 else round(before, 3)}


def direct_fit(obj, vanilla, bones):
    """M21: one uniform scale (the croup's height over the ground: vanilla's), the hooves on the
    ground, centred in x and with the hooves' midpoint at vanilla's heels' midpoint in z. The
    model's own proportions are kept."""
    co = hs.verts(obj)
    co -= [0.5 * (co[:, 0].min() + co[:, 0].max()), co[:, 1].min(), 0.5 * (co[:, 2].min() + co[:, 2].max())]
    hs.set_verts(obj, co)
    z = bones["Saddle_Node"][2]
    # the croup band on the model: the same share of its hoof-to-hoof span behind its middle as vanilla's
    heels = np.array([bones[f"Horse{l}Heel"] for l in ("FrontLeft", "FrontRight", "BackLeft", "BackRight")])
    vmid, vspan = heels[:, 2].mean(), heels[2:, 2].mean() - heels[:2, 2].mean()
    h = hooves(co, 0.0)
    fz = np.mean([h[k][2] for k in ("FrontLeft", "FrontRight") if h[k] is not None])
    bz = np.mean([h[k][2] for k in ("BackLeft", "BackRight") if h[k] is not None])
    mmid, mspan = 0.5 * (fz + bz), bz - fz
    rel0, rel1 = (z + 3.0 - vmid) / vspan, (z + 4.0 - vmid) / vspan
    mine = top_at(obj, mmid + rel0 * mspan, mmid + rel1 * mspan, 1.0 * mspan / vspan)
    s = top_at(vanilla, z + 3.0, z + 4.0, 1.0) / mine
    co = hs.verts(obj) * s
    off = np.array([0.0, -co[:, 1].min(), vmid - mmid * s])
    hs.set_verts(obj, co + off)
    # then the rider's seat: the top at Saddle_Node (the model's saddle, or its back) at vanilla's, about
    # the ground under the hooves' midpoint (the croup's scale reported beside it)
    seat = seat_top(vanilla, z) / seat_top(obj, z)
    pivot = np.array([0.0, 0.0, vmid])
    hs.set_verts(obj, (hs.verts(obj) - pivot) * seat + pivot)
    return {"scale": float(s * seat), "croup_scale": float(s), "seat_factor": float(seat),
            "offset": off.tolist(), "hoof_span": {"model": float(mspan * s * seat), "vanilla": float(vspan)}}


def head_shift(co, bones, head):
    """``co`` with the head and neck moved by ``head["d"]``: fully from the poll forward, faded out
    along z to the neck base (only above the neck base's height less 3)."""
    nb = np.array(bones["HorseNeckBase"])
    t = smoothstep(-(co[:, 2] - nb[2]), 0.0, -(head["poll"][2] - nb[2]))
    up = co[:, 1] > nb[1] - 3.0
    return co + np.where(up[:, None], t[:, None] * np.array(head["d"]), 0.0)


def direct_pose(obj, vanilla, bones, legs=True, head=True, belly=None, apply_head=True, head_at=None):
    """M21's minimal posing: each lower leg slid sideways (x, z) so its hoof stands under vanilla's
    heel, faded out from the knee up to the belly; the head and neck moved as one rigid piece onto
    vanilla's poll, faded out along the neck to its base. Nothing is turned or reshaped. Reports
    each part's move."""
    co = hs.verts(obj)
    rep = {}
    if legs:
        h = hooves(co, 0.5 * (bones["HorseFrontLeftHeel"][2] + bones["HorseBackLeftHeel"][2]))
        knee = min(bones["HorseFrontLeftKnee"][1], bones["HorseBackLeftKnee"][1])
        belly = belly if belly is not None else min(bones["HorseFrontLeftDown"][1], bones["HorseBackLeftDown"][1])
        names = [k for k in h if h[k] is not None]
        centres = np.array([[h[k][0], h[k][2]] for k in names])
        near = np.argmin(((co[:, None, [0, 2]] - centres[None]) ** 2).sum(-1), axis=1)
        w = 1.0 - smoothstep(co[:, 1], knee, belly)
        move = np.zeros_like(co)
        for i, k in enumerate(names):
            heel = np.array(bones[f"Horse{k}Heel"])
            d = np.array([heel[0] - h[k][0], 0.0, heel[2] - h[k][2]])
            sel = near == i
            move[sel] = d * w[sel, None]
            rep[k] = {"hoof": [round(float(x), 2) for x in h[k]], "moved": round(float(np.linalg.norm(d)), 3)}
        co = co + move
    if head:
        vco = hs.verts(vanilla)
        nb = np.array(bones["HorseNeckBase"])
        _, vaxis, vpoll, _ = hs.head_features(vco, nb)
        _, axis, poll, _ = hs.head_features(co, nb) if head_at is None else (None, head_at[0], head_at[1], None)
        d = vpoll - poll
        if apply_head:
            co = head_shift(co, bones, {"poll": poll, "d": d})
        tilt = math.degrees(math.acos(float(np.clip(np.dot(axis, vaxis), -1, 1))))
        rep["head"] = {"poll": [round(float(x), 2) for x in poll], "vanilla_poll": [round(float(x), 2) for x in vpoll],
                       "moved": round(float(np.linalg.norm(d)), 3), "axis_vs_vanilla_deg": round(tilt, 1),
                       "turned": 0.0, "d": d.tolist(), "axis": [round(float(x), 4) for x in axis]}
    hs.set_verts(obj, co)
    return rep


def _grow(m):
    out = m.copy()
    for ax in range(3):
        out |= np.roll(m, 1, ax) | np.roll(m, -1, ax)
    return out


def solid_fill(obj, vs, closing=1, opening=0, protect=None, region=None, carve=(), carve_out=(), carve_in=()):
    """``obj`` replaced by the solid it encloses, by flood fill: the voxels (size ``vs``) its surface
    passes through, closed by ``closing`` voxels (gaps up to twice that sealed), then everything the
    outside can't reach from the box's edge is solid; the closing taken back off; the solid's voxel
    faces as one closed mesh. A voxel remesh of an open surface makes a hollow shell wherever the
    outside reaches in (the cataphract: volume 74 against horse 2's 379; its low mesh came out as
    lace); filled, 627. -> {"volume", "grid"}."""
    me = obj.data
    me.calc_loop_triangles()
    co = hs.verts(obj)
    tri = np.array([lt.vertices[:] for lt in me.loop_triangles], dtype=np.int64)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / (vs * vs) * 8).astype(np.int64))
    idx = np.repeat(np.arange(len(tri)), n)
    rng = np.random.default_rng(0)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = np.vstack([co, a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])])
    pad = closing + 2
    lo = pts.min(0) - pad * vs
    shape = np.ceil((pts.max(0) + pad * vs - lo) / vs).astype(int) + 1
    occ = np.zeros(shape, dtype=bool)
    ijk = ((pts - lo) / vs).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    def solid_at(r):
        """Not reachable from the box's edge once the surface is grown by ``r`` voxels, grown back."""
        o = occ
        for _ in range(r):
            o = _grow(o)
        ext = np.zeros_like(o)
        ext[0], ext[-1], ext[:, 0], ext[:, -1], ext[:, :, 0], ext[:, :, -1] = True, True, True, True, True, True
        ext &= ~o
        while True:
            nxt = _grow(ext) & ~o
            if nxt.sum() == ext.sum():
                break
            ext = nxt
        ins = ~ext
        for _ in range(r):
            ins &= ~_grow(~ins)
        return ins
    inside = solid_at(1)
    if closing > 1:
        # what only the wider closing seals is kept where it is thick (a hollow body the outside got into:
        # the cataphract's) and dropped where it is a thin slab (the gap between two legs standing close:
        # horse 2's hind legs welded at a 0.2 closing and tore in the attack)
        gap = solid_at(closing) & ~inside
        thick = gap.copy()
        for _ in range(closing):
            thick &= ~_grow(~thick)
        for _ in range(closing + 1):
            thick = _grow(thick) & gap
        inside |= thick
    if opening:  # thin pieces off (mane slabs, flaps, a bit or rein under the muzzle): shrunk away, regrown
        core = inside.copy()
        for _ in range(opening):
            core &= ~_grow(~core)
        for _ in range(opening):
            core = _grow(core) & inside
        ii = np.indices(inside.shape)

        def box(b):
            lo_b, hi_b = (np.array(b[0]) - lo) / vs, (np.array(b[1]) - lo) / vs
            return np.all([(ii[d] >= lo_b[d]) & (ii[d] <= hi_b[d]) for d in range(3)], axis=0)
        if region is not None:  # only there (the head and neck: the caparison's thin skirts stay)
            core |= inside & ~box(region)
        if protect is not None:  # the ears: thin, kept
            core |= inside & box(protect)
        inside = core
    if carve:  # M23: slots cut out of the solid (boxes, world units): the rider's skirt split between the legs
        ii = np.indices(inside.shape)
        for b in carve:
            lo_b, hi_b = (np.array(b[0]) - lo) / vs, (np.array(b[1]) - lo) / vs
            inside &= ~np.all([(ii[d] >= lo_b[d]) & (ii[d] <= hi_b[d]) for d in range(3)], axis=0)
    for c in carve_out:  # M23: everything above ``y0`` farther than ``radius`` from a vertical axis (the collar round a head)
        ii = np.indices(inside.shape)
        wx, wy, wz = (lo[d] + ii[d] * vs for d in range(3))
        inside &= ~((wy > c["y0"]) & (np.hypot(wx - c["centre"][0], wz - c["centre"][1]) > c["radius"]))
    for c in carve_in:  # M23: everything under ``y1`` within ``radius`` of a vertical axis (the plume's gap over a helmet)
        ii = np.indices(inside.shape)
        wx, wy, wz = (lo[d] + ii[d] * vs for d in range(3))
        inside &= ~((wy < c["y1"]) & (np.hypot(wx - c["centre"][0], wz - c["centre"][1]) < c["radius"]))
    # the solid's boundary faces (between an inside and an outside voxel) as quads on the corner grid
    S = np.array(inside.shape) + 1
    vid = lambda i, j, k: (i * S[1] + j) * S[2] + k  # noqa: E731
    quads = []
    for ax in range(3):
        pad_in = np.pad(inside, [(1, 1) if d == ax else (0, 0) for d in range(3)])
        lo_side, hi_side = np.take(pad_in, range(0, pad_in.shape[ax] - 1), ax), np.take(pad_in, range(1, pad_in.shape[ax]), ax)
        for sign, mask in ((1, lo_side & ~hi_side), (-1, ~lo_side & hi_side)):
            p = np.argwhere(mask)  # face at index p[ax] on axis ax (between cells p-1 and p)
            o1, o2 = [d for d in range(3) if d != ax]
            corners = []
            for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = p.copy()
                q[:, o1] += d1
                q[:, o2] += d2
                corners.append(vid(q[:, 0], q[:, 1], q[:, 2]))
            f = np.stack(corners, 1)
            if (sign > 0) != ((o1, o2) in ((1, 2), (2, 0), (0, 1))):
                f = f[:, ::-1]
            quads.append(f)
    f = np.vstack(quads)
    used, inv = np.unique(f.ravel(), return_inverse=True)
    i, rem = np.divmod(used, S[1] * S[2])
    j, k = np.divmod(rem, S[2])
    verts = lo + np.stack([i, j, k], 1) * vs
    new = bpy.data.meshes.new("solid")
    new.from_pydata(verts.tolist(), [], inv.reshape(-1, 4).tolist())
    new.validate()
    old = obj.data
    obj.data = new
    bpy.data.meshes.remove(old)
    return {"volume": round(float(inside.sum() * vs ** 3), 1), "grid": [int(x) for x in inside.shape]}


SOLID = {}  # collapse_to's solid_fill settings for this run (direct sets them)


def collapse_to(obj, target, close=None):
    """Quadric (collapse) decimation straight to ``target`` triangles: the silhouette kept. With
    ``close``, a voxel remesh that fine first: TRELLIS.2's surface has thousands of non-manifold
    edges and open sheets, and the collapse stalls on them (horse 2: at 36-61k triangles whatever
    the clean-up); at 0.06 the strands fall into thousands of tiny pieces that stall it too (~31k); 0.1 is about 0.5% of
    the horse's length."""
    activate(obj)
    if close:
        # a closed solid first; its closing and opening are distances (in voxels at this size): the
        # cataphract's gaps let the outside in at a 0.1 closing (a shell, volume 211), not at 0.2 (598)
        vs_ = float(close)
        SOLID.setdefault("log", []).append(solid_fill(
            obj, vs_, closing=max(1, int(round(SOLID.get("closing", 0.2) / vs_))),
            opening=int(round(SOLID.get("opening", 0.0) / vs_)), protect=SOLID.get("protect"),
            region=SOLID.get("region"), carve=SOLID.get("carve", ()), carve_out=SOLID.get("carve_out", ()),
            carve_in=SOLID.get("carve_in", ())))
        mod = obj.modifiers.new("close", "REMESH")
        mod.mode, mod.voxel_size, mod.adaptivity = "VOXEL", float(close), 0.0
        bpy.ops.object.modifier_apply(modifier=mod.name)
        keep_largest(obj)  # the voxels leave tassels and cloth edges as their own pieces
        if SOLID.get("relax"):  # the voxel steps relaxed before the collapse (they showed as lumps)
            mod = obj.modifiers.new("relax", "SMOOTH")
            mod.factor, mod.iterations = 0.5, int(SOLID["relax"])
            bpy.ops.object.modifier_apply(modifier=mod.name)
    mod = obj.modifiers.new("tri", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    for _ in range(8):  # one pass can stall far above the target (horse 2: 277k -> 61k)
        if tri_count(obj) <= target * 1.02:
            break
        mod = obj.modifiers.new("collapse", "DECIMATE")
        mod.decimate_type, mod.ratio, mod.use_collapse_triangulate = "COLLAPSE", target / tri_count(obj), True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if target < 10 ** 8:  # zero-length edges and slivers the collapse leaves (the deformation check divides by them)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-3)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    return tri_count(obj)


def packed_uv(obj, margin):
    """Smart UV Project (89 degrees: the fewest islands), the islands at one texel density, then packed
    tightly (any rotation, concave shapes) for M21's 1024 bake. Horse 2 (2,404 triangles): 59% of the
    image at a 0.001 gap (one texel at 1024), 71% at no gap (neighbouring islands bleed into each other)."""
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = True  # else the pack sees no UVs selected in background
    bpy.ops.uv.smart_project(angle_limit=math.radians(89), island_margin=0.0)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, rotate_method="ANY", margin_method="FRACTION", margin=margin,
                            shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")


def vertex_colours(obj):
    """Per vertex: the model's base colour (0-255 sRGB) at its loops' UVs, averaged, and its
    luminance. The base colour is the image on the imported material's Base Color."""
    me = obj.data
    bsdf = next(n for n in me.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    img = bsdf.inputs["Base Color"].links[0].from_node.image
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[..., :3]
    uv = np.empty(len(me.loops) * 2, dtype=np.float64)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    lv = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    col = px[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    acc, n = np.zeros((len(me.vertices), 3)), np.zeros(len(me.vertices))
    np.add.at(acc, lv, col)
    np.add.at(n, lv, 1)
    rgb = 255 * acc / np.maximum(n, 1)[:, None]
    return rgb, rgb @ np.array([0.299, 0.587, 0.114])


def flood(cand, seeds, edges):
    """The candidate vertices connected to ``seeds`` through candidate vertices."""
    keep = cand[edges[:, 0]] & cand[edges[:, 1]]
    adj = [[] for _ in range(len(cand))]
    for p, q in edges[keep]:
        adj[p].append(q)
        adj[q].append(p)
    out = np.zeros(len(cand), dtype=bool)
    stack = list(np.flatnonzero(seeds & cand))
    out[stack] = True
    while stack:
        for u in adj[stack.pop()]:
            if not out[u]:
                out[u] = True
                stack.append(u)
    return out


def hair(obj, bones, poll, tol=40.0, behind_hip=0.8, grow=3, max_share=0.4):
    """M21: TRELLIS.2's tail and mane on the posed model, by the tail's own colour and place (no fixed
    colour rule: horse 2's tail is grey, the cataphract's black beside silver scale; TRELLIS.2 bakes
    shade into its colours, so darkness took horse 2's shadowed coat).
      * The tail's colour: the median of the rear-most vertices (within 0.3 of each slice's back) at the
        height of vanilla's tail tip +-1.5, near the centre line: what hangs furthest back there is tail.
      * The buttock: the back of the body's sides (|x| over 1.3) between y 5 and 11.
      * The tail: vertices within ``tol`` (RGB distance) of its colour, ``behind_hip`` behind vanilla's
        hind leg tops, no further forward than 0.3 ahead of the buttock, above the ground, connected to
        those rear-most seeds; grown ``grow`` rings over such vertices; refused over ``max_share`` of the
        model (TRELLIS.2 spends many vertices on strands: horse 2's tail is 24% of them).
      * The mane: vertices within ``tol`` of the tail's colour in the upper half of each neck slice
        between the poll and the neck base.
    -> (tail mask, mane mask, info with its colours and size)."""
    co = hs.verts(obj)
    rgb, lum = vertex_colours(obj)
    z_hip = max(bones["HorseBackLeftUp"][2], bones["HorseBackRightUp"][2])
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    behind = co[:, 2] > z_hip + behind_hip
    tip = bones["HorseTailTip"][1]
    seeds = np.zeros(len(co), dtype=bool)
    for y0 in np.arange(tip - 1.5, tip + 1.5, 0.5):
        sl = behind & (co[:, 1] >= y0) & (co[:, 1] < y0 + 0.5) & (np.abs(co[:, 0]) < 1.5)
        if sl.any():
            seeds |= sl & (co[:, 2] > co[sl, 2].max() - 0.3)
    info = {"tail_seed_vertices": int(seeds.sum())}
    tail = np.zeros(len(co), dtype=bool)
    mane = np.zeros(len(co), dtype=bool)
    if not seeds.any():
        info["tail_refused"] = True
        return tail, mane, info
    ref = np.median(rgb[seeds], axis=0)
    like = np.linalg.norm(rgb - ref, axis=1) < tol
    sides = (np.abs(co[:, 0]) > 1.3) & (co[:, 1] > 5.0) & (co[:, 1] < 11.0)
    butt = float(np.percentile(co[sides, 2], 99.5)) if sides.any() else z_hip
    zone = behind & (co[:, 2] > butt - 0.3) & (co[:, 1] > 1.5)
    tail = flood(like & zone, seeds & like, e)
    for _ in range(grow):  # the strands' darker or lighter streaks
        nxt = tail.copy()
        nxt[e[tail[e[:, 0]], 1]] = True
        nxt[e[tail[e[:, 1]], 0]] = True
        tail = nxt & zone & (np.linalg.norm(rgb - ref, axis=1) < 1.5 * tol)
    refused = tail.mean() > max_share
    if refused:
        tail[:] = False
    nb = np.array(bones["HorseNeckBase"])
    for z0 in np.arange(poll[2] + 0.8, nb[2] + 1.0, 0.4):
        sl = (co[:, 2] >= z0) & (co[:, 2] < z0 + 0.4) & (co[:, 1] > 8.0) & (np.abs(co[:, 0]) < 1.8)
        if sl.any():
            mane |= sl & like & (co[:, 1] > 0.5 * (co[sl, 1].min() + co[sl, 1].max()))
    info.update({"colour_reference": [int(round(x)) for x in ref], "tol": tol, "buttock_z": round(butt, 2),
                 "tail_vertices": int(tail.sum()), "tail_share": round(float(tail.mean()), 3),
                 "tail_refused": bool(refused), "mane_vertices": int(mane.sum())})
    if tail.any():
        t = co[tail]
        # the tube's size from the cut tail: half its depth (z) and width (x), robust to stray strands
        info["tail_half_depth"] = round(float(np.percentile(t[:, 2], 97) - np.percentile(t[:, 2], 3)) / 2, 3)
        info["tail_half_width"] = round(float(np.percentile(np.abs(t[:, 0]), 97)), 3)
        info["tail_colour"] = [int(round(x)) for x in np.median(rgb[tail], axis=0)]
        info["tail_colour_p25_p75"] = [[int(round(x)) for x in np.percentile(rgb[tail], q, axis=0)] for q in (25, 75)]
    return tail, mane, info


def smooth_region(obj, mask, iterations=40, factor=0.5, fade=2):
    """``mask``'s vertices smoothed hard (Laplacian), the ``fade`` rings round it gently: spiky
    strands pulled into one solid mass that keeps its colours (its UVs ride along)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    ring = set(np.flatnonzero(mask).tolist())
    core = [bm.verts[i] for i in ring]
    edge = []
    for _ in range(fade):
        nxt = {e.other_vert(bm.verts[i]).index for i in ring for e in bm.verts[i].link_edges} - ring
        edge += [bm.verts[i] for i in nxt]
        ring |= nxt
    for _ in range(iterations):
        bmesh.ops.smooth_vert(bm, verts=core, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
        if edge:
            bmesh.ops.smooth_vert(bm, verts=edge, factor=factor * 0.3, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def fill_open(obj):
    """Boundary loops closed (the cut tail's root) before the voxel close. -> boundary edges found."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loose = [e for e in bm.edges if e.is_boundary]
    bmesh.ops.holes_fill(bm, edges=loose, sides=0)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(loose)


def region_uv(obj, bones, margin, method="CONFORMAL", split=(), smooth=4, weights=None):
    """M21: a few large islands instead of Smart UV's many. Each face's region: the tail piece; a leg
    (below the belly, by the nearest vanilla heel column); the head (ahead of vanilla's head bone +
    0.5); else the trunk with the neck. Each region split into left and right (a leg at its heel's
    x). Seams on the region borders, angle-based unwrap, one texel density, packed."""
    me = obj.data
    parts = face_parts(obj)
    belly = min(bones["HorseFrontLeftDown"][1], bones["HorseBackLeftDown"][1])
    legs = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")
    heels = np.array([[bones[f"Horse{k}Heel"][0], bones[f"Horse{k}Heel"][2]] for k in legs])
    head_z = bones["HorseHead"][2] + 0.5
    panel_y = (6.0, 12.0)
    panel_z = (max(bones["HorseFrontLeftUp"][2], bones["HorseFrontRightUp"][2]) + 1.0,
               min(bones["HorseBackLeftUp"][2], bones["HorseBackRightUp"][2]) - 1.0)
    region = []
    for poly, part in zip(me.polygons, parts):
        c = poly.center
        if part == hb.TAIL:
            zone, centre = "tail", 0.0
        elif c.y < belly and c.z > head_z:
            k = int(np.argmin(((heels - [c.x, c.z]) ** 2).sum(1)))
            zone, centre = legs[k], heels[k][0]
        elif c.z < head_z:
            zone, centre = "head", 0.0
        elif (abs(c.x) > 0.9 and panel_y[0] < c.y < panel_y[1] and panel_z[0] < c.z < panel_z[1]):
            zone, centre = "panel", 0.0  # the main caparison panel's place: its own island (decals stay sharp)
        else:
            zone, centre = "trunk", 0.0
            for i, z in enumerate(sorted(split)):  # the trunk cut across at these z (front to back)
                if c.z > z:
                    zone = f"trunk{i + 1}"
        region.append(f"{zone}_{'L' if c.x > centre else 'R'}")
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    # borders straightened: each face takes its edge neighbours' majority region a few times over (face by
    # face, the left/right and leg-top borders zig-zag, and jagged islands pack with wide gaps)
    nbrs = [[g.index for e in f.edges for g in e.link_faces if g.index != f.index] for f in bm.faces]
    for _ in range(smooth):
        new = list(region)
        for i, ns in enumerate(nbrs):
            if region[i].startswith("tail") or not ns:
                continue
            votes = {}
            for j in ns:
                if not region[j].startswith("tail"):
                    votes[region[j]] = votes.get(region[j], 0) + 1
            best = max(votes, key=votes.get) if votes else region[i]
            if votes.get(best, 0) >= 2 and best != region[i]:
                new[i] = best
        region = new
    seams = 0
    for e in bm.edges:
        fs = e.link_faces
        e.seam = len(fs) == 2 and region[fs[0].index] != region[fs[1].index]
        seams += e.seam
    bm.to_mesh(me)
    bm.free()
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = True  # else the pack sees no UVs selected in background
    bpy.ops.uv.unwrap(method=method, margin=0.0)
    bpy.ops.uv.average_islands_scale()
    if weights:  # more texels for some regions (the trunk: its panels carry decals), each island about its centre
        bpy.ops.object.mode_set(mode="OBJECT")
        uv = np.empty(len(me.loops) * 2, dtype=np.float64)
        me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        for name in set(region):
            k = weights.get(name.rsplit("_", 1)[0])
            if not k:
                continue
            loops = [li for f, r in zip(me.polygons, region) if r == name for li in f.loop_indices]
            c = uv[loops].mean(0)
            uv[loops] = c + (uv[loops] - c) * k
        me.uv_layers.active.data.foreach_set("uv", uv.ravel())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, rotate_method="ANY", margin_method="FRACTION", margin=margin,
                            shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    # the pack keeps the islands' shared aspect, so its block can leave a strip of the square empty:
    # stretched to fill it (texel density a little anisotropic, no texels wasted)
    uv = np.empty(len(me.loops) * 2, dtype=np.float64)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    lo, hi = uv.min(0), uv.max(0)
    pad = margin / 2
    uv = pad + (uv - lo) / np.maximum(hi - lo, 1e-9) * (1 - 2 * pad)
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())
    counts = {}
    for r in region:
        counts[r] = counts.get(r, 0) + 1
    return {"seam_edges": int(seams), "regions": counts, "packed_block": [round(float(x), 3) for x in hi - lo]}


def uv_area(obj):
    """The share of the unit square the UVs cover."""
    me = obj.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.data
    s = 0.0
    for lt in me.loop_triangles:
        a, b, c = (uv[i].uv for i in lt.loops)
        s += abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2
    return s


def hair_voxels(obj, hair_pts, model_pts, bones, reach):
    """The closed (voxel) surface's vertices to cut with the tail: nearest a tail vertex of the model,
    or within ``reach`` of one behind the hips (the darker strands the mask missed)."""
    from mathutils.kdtree import KDTree

    def tree_of(*sets):
        t = KDTree(sum(len(x) for x in sets))
        k = 0
        for pts in sets:
            for q in pts:
                t.insert(q, k)
                k += 1
        t.balance()
        return t
    both, hair_only = tree_of(hair_pts, model_pts), tree_of(hair_pts)
    z_cut = max(bones["HorseBackLeftUp"][2], bones["HorseBackRightUp"][2]) + 0.8
    cut = []
    for i, q in enumerate(hs.verts(obj)):
        if both.find(q)[1] < len(hair_pts):
            cut.append(i)
        elif q[2] > z_cut and hair_only.find(q)[2] < reach:
            cut.append(i)
    return cut


def clear_tail_bones(obj):
    """The body's vertices (part 0) lose any tail-bone weight (our tail is its own piece; vanilla's tail
    bones run past a TRELLIS.2 horse's buttock, and the chain weights gave the rump to them: it swung
    with the tail in idle); the rest renormalised, HorseSpineEnd if nothing is left. -> vertices changed."""
    me = obj.data
    kind = {}
    for poly, part in zip(me.polygons, face_parts(obj)):
        for vi in poly.vertices:
            kind[vi] = max(kind.get(vi, 0), part)
    tail = {obj.vertex_groups[n].index for n in hb.TAIL_NODES[1:] if n in obj.vertex_groups}
    names = {g.index: g.name for g in obj.vertex_groups}
    n = 0
    for v in me.vertices:
        if kind.get(v.index, 0) != 0 or not any(g.group in tail and g.weight > 0 for g in v.groups):
            continue
        row = {names[g.group]: g.weight for g in v.groups if g.group not in tail and g.weight > 0}
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        total = sum(row.values())
        for b, w in (row.items() if total > 0 else [("HorseSpineEnd", 1.0)]):
            (obj.vertex_groups.get(b) or obj.vertex_groups.new(name=b)).add([v.index], w / (total or 1.0), "REPLACE")
        n += 1
    return n


def direct(a, van, vanilla):
    """M21 (the user, 2026-10-23): a TRELLIS.2 model to the game mesh with its own shape kept: no
    reshape onto vanilla's rest pose. Import, weld the seams, the largest piece, direct_fit,
    direct_pose; with ``hair``, TRELLIS.2's tail cut off (it tore in idle) and the mane's strands
    smoothed into a solid crest; ``textured_out`` (the posed model with its own UVs: the bake
    source); a fine voxel close and collapse decimation to ``target_tris``; our tail on vanilla's
    tail bones; region UVs (``uv`` "smart": Smart UV); chain weights (the tail along its bones)."""
    bones = van["bones"]
    src = a["source"]
    res = {"stages": {}}
    obj = import_model(src["path"])
    merge_seams(obj, float(src.get("merge", 1e-5)))
    res["stages"]["imported"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
    keep_largest(obj)
    res["stages"]["largest_piece"] = {"tris": tri_count(obj)}
    turn = face_forward(obj)
    res["fit"] = direct_fit(obj, vanilla, bones)
    res["fit"]["turned_deg"] = turn
    # the head found before the slim (after it, the cataphract's chest and tassels stood ahead of its head and
    # were taken for it: the poll at y 9.4, the axis upright), its poll moved with the model; above the belly
    # the slim is uniform, so the axis holds
    _, head_axis, head_poll, _ = hs.head_features(hs.verts(obj), np.array(bones["HorseNeckBase"]))
    head_poll = np.array(head_poll, dtype=np.float64)
    if a.get("slim", 1.15):
        res["fit"]["slim"] = slim(obj, vanilla, bones, float(a.get("slim", 1.15)), points=[head_poll])
    if a.get("fitted_out"):  # the model fitted, not posed: the raw model in the game's frame, for comparisons
        write_obj(obj, a["fitted_out"], uv=True)
    pose = a.get("pose") or {}
    # the legs moved (their hooves under vanilla's heels); the head and neck keep their shape and place:
    # the head is moved onto vanilla's poll only while the weights are taken (below)
    res["pose"] = direct_pose(obj, vanilla, bones, legs=pose.get("legs", True), apply_head=False,
                              head_at=(head_axis, head_poll))
    hc = a.get("hair") or {}
    build_tail = build_mane = False
    if hc:
        tail, mane, res["hair"] = hair(obj, bones, np.array(res["pose"]["head"]["poll"]), float(hc.get("tol", 40.0)),
                                       max_share=float(hc.get("max_share", 0.4)))
        build_tail = bool(hc.get("tail", True) and tail.any())
        build_mane = bool(hc.get("mane", True) and mane.any())
        if build_mane and hc.get("mane") == "smooth":  # the first try (v004): too weak on a dense mesh
            smooth_region(obj, mane, int(hc.get("mane_iterations", 40)))
            build_mane = False
    if a.get("textured_out"):  # the bake source: the tail cut off (its root left open: rays there miss and
        # take their neighbours' colour)
        tex = obj.copy()
        tex.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(tex)
        if build_tail:
            delete_vertices(tex, np.flatnonzero(tail).tolist())
        write_obj(tex, a["textured_out"], uv=True)
        debug_dump(a, tex, "hair")
        bpy.data.objects.remove(tex, do_unlink=True)
    if build_tail or build_mane:
        hair_pts, model_pts = hs.verts(obj)[tail], hs.verts(obj)[~tail]
        mane_pts = hs.verts(obj)[mane] if build_mane else None
    while obj.data.uv_layers:  # the model's own UVs hold the collapse at their seams (~60k): new ones after
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    src_mesh = obj.data.copy()
    target = int(a["target_tris"])
    poll = res["pose"]["head"]["poll"]
    SOLID.update({"opening": float(a.get("opening", 0.2)), "relax": int(a.get("relax", 3)), "closing": float(a.get("closing", 0.2)),
                  # the ears: from just under the poll up, round it
                  "protect": [[poll[0] - 1.2, poll[1] - 0.3, poll[2] - 1.2], [poll[0] + 1.2, poll[1] + 3.0, poll[2] + 1.2]],
                  # the opening's reach: the head and neck, ahead of the withers and over the chest (fragments
                  # are there: mane flaps, the bit under the muzzle; the cataphract's skirts went with a
                  # whole-body opening)
                  "region": [[-1e9, bones["HorseNeckBase"][1] - 3.5, -1e9], [1e9, 1e9, bones["HorseNeckBase"][2] + 1.0]]})

    def low(vs):
        """The low mesh from the posed model at voxel ``vs``: closed, the hair handled, collapsed."""
        if not (build_tail or build_mane):
            return collapse_to(obj, target, vs)
        # closed first with the hair on (cut from TRELLIS.2's open surface, the tail's ragged root - 4,472
        # boundary edges on horse 2 - leaked through the voxels and the hindquarters vanished), then the
        # tail's voxels off and the few clean holes filled
        collapse_to(obj, 10 ** 9, vs)
        if build_tail:
            cut = hair_voxels(obj, hair_pts, model_pts, bones, float(hc.get("tail_reach", 0.35)))
            delete_vertices(obj, cut)
            keep_largest(obj)
            res["hair"]["voxel_vertices_cut"] = len(cut)
            res["hair"]["boundary_edges_filled"] = fill_open(obj)
        if build_mane:
            # the mane's voxels (within reach of its strands) smoothed hard into one crest: cut off, the
            # strands took the neck's top half with them (v006: the neck thin and low); smoothed on the
            # dense model, they barely moved (v004); on the even voxels the smoothing spans the notches
            from mathutils.kdtree import KDTree
            t = KDTree(len(mane_pts))
            for i, q in enumerate(mane_pts):
                t.insert(q, i)
            t.balance()
            reach = float(hc.get("mane_reach", 0.6))
            near = np.array([t.find(q)[2] < reach for q in hs.verts(obj)])
            smooth_region(obj, near, int(hc.get("mane_iterations", 60)), 0.5, int(hc.get("mane_fade", 3)))
            res["hair"]["mane_voxels_smoothed"] = int(near.sum())
        return collapse_to(obj, target, None)

    # the close made coarser (x1.5 a try, from the same input) while the collapse stalls over the target
    # x 1.1: sheets a voxel thick don't collapse (the cataphract's caparison at 0.1: ~8k triangles; at 0.15
    # they merge and it reaches 2,500)
    vs, tries = float(a.get("close_voxel", 0.1)), []
    for k in range(3):
        n = low(vs)
        tries.append({"voxel": round(vs, 3), "tris": n})
        if n <= target * 1.1 or k == 2:
            break
        old = obj.data
        obj.data = src_mesh.copy()
        bpy.data.meshes.remove(old)
        vs *= 1.5
    res["stages"]["decimated"] = {"tris": n, "close_voxel": round(vs, 3), "close_tries": tries, "solid": SOLID.get("log")}
    if count_pieces(obj) > 1:
        keep_largest(obj)
        res["stages"]["decimated"]["tris_one_piece"] = tri_count(obj)
    if build_tail:
        root, offset = tail_anchor(obj, bones)
        tube = dict(hc.get("tube", {}))
        if "radii" not in tube:  # hb.tail's profile scaled so its widest ring is the cut tail's half depth
            k = float(np.clip(res["hair"]["tail_half_depth"] / 0.62, 1.0, 2.5))
            tube["radii"] = [[f, r * k] for f, r in ((0.0, 0.3), (0.25, 0.36), (0.55, 0.62), (0.85, 0.5), (1.0, 0.18))]
            tube["flat"] = float(np.clip(res["hair"]["tail_half_width"] / res["hair"]["tail_half_depth"], 0.5, 1.0))
            tube["sides"] = 8
        b = hb.tail(bones, root, offset=offset, **tube)
        join_piece(obj, "tail", b)
        res["hair"]["tube"] = {**b["log"], "offset": [round(x, 2) for x in offset]}
    obj.data.uv_layers.new(name="UV")
    if a.get("uv") == "smart":
        packed_uv(obj, float(a.get("uv_margin", 0.001)))
        res["uv"] = {}
    else:
        res["uv"] = region_uv(obj, bones, float(a.get("uv_margin", 0.002)), a.get("uv_method", "CONFORMAL"),
                              [bones["Saddle_Node"][2] + dz for dz in a.get("uv_trunk_cuts", [])], int(a.get("uv_smooth", 4)),
                              a.get("uv_weights", {"trunk": 1.4, "panel": 2.0}))
    res["uv"]["area"] = round(uv_area(obj), 4)
    rest = hs.verts(obj)
    if pose.get("head", "weights") == "weights":
        hs.set_verts(obj, head_shift(rest, bones, res["pose"]["head"]))
        res["pose"]["head"]["note"] = "weights only: the geometry isn't moved"
    wstats = {}
    transfer_weights(obj, vanilla, a.get("smooth", 12), a.get("influences", 3),
                     bones if a.get("chain_weights", True) else None, wstats)
    hs.set_verts(obj, rest)
    if build_tail:
        wstats["tail_pinned"] = pin_barding(obj, bones)
        wstats["body_tail_weights_cleared"] = clear_tail_bones(obj)
    res["weights"] = wstats
    saddle = bones["Saddle_Node"]
    res["back"] = {"seat": {"ours": seat_top(obj, saddle[2]), "vanilla": seat_top(vanilla, saddle[2])}}
    verts, tris, unweighted = export(obj, a["geometry_out"], a.get("preview_out"))
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": count_pieces(obj)}
    _pdx.write_result(a, res)


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
    if a["source"]["kind"] == "direct":
        return direct(a, van, vanilla)
    res = {"stages": {}}
    src = a["source"]
    read = Readability(a.get("readability"), van, a["barding"]) if a.get("barding") else None
    if src["kind"] == "model":
        obj = import_model(src["path"])
        if src.get("merge"):
            merge_seams(obj, float(src["merge"]))
        res["stages"]["imported"] = {"tris": tri_count(obj), "pieces": count_pieces(obj)}
        keep_largest(obj)
        res["stages"]["largest_piece"] = {"tris": tri_count(obj)}
        if a.get("fit", True):
            s, off = fit(obj, vanilla)
        else:  # already on the vanilla skeleton (Track M's game version): a box fit would move it off the bones
            s, off = 1.0, [0.0, 0.0, 0.0]
        res["fit"] = {"scale": s, "offset": off}
        if a.get("reshape"):
            res["fit"]["height_scale"] = scale_height(obj, vanilla, van["bones"])
            res["fit"]["reshape"] = hs.reshape(obj, vanilla, van["bones"], leg_top=float(a.get("leg_top", 6.0)))
        debug_dump(a, obj, "1_reshaped")
        if a.get("textured_out"):  # M21: the model's own UVs after the fit and reshape, for its texture's bake
            write_obj(obj, a["textured_out"], uv=True)
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
    elif src["kind"] in ("procedural", "anatomy"):
        high, build_tail = None, False
        obj, res["fit"] = (anatomy_horse(van, vanilla, src, read) if src["kind"] == "anatomy" else procedural_horse(van, src))
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
    if a.get("pieces"):  # Track M: the master's cloth as closed low slabs, fitted like the source model
        s_, off_ = res["fit"]["scale"], np.array(res["fit"]["offset"])
        res["pieces"] = {}
        for q in json.loads(open(a["pieces"], encoding="utf-8").read()):
            v = np.array(q["v"]) * s_ + off_
            f = np.array(q["f"], dtype=np.int64)
            join_piece(obj, q["name"], {"v": v, "f": f, "part": [q.get("part", hb.CLOTH_PANEL)] * len(f),
                                        "hem": [q["hem"] * s_ + off_[1]] * len(f), "anchor": np.array(q["anchor"]) * s_ + off_})
            res["pieces"][q["name"]] = {"tris": int(len(f))}
    if build_tail:  # found on the body alone, before the panels join it
        root, offset = tail_anchor(obj, van["bones"])
        b = hb.tail(van["bones"], root, offset=offset,
                    **{k: v for k, v in a["tail"].items() if k in ("sides", "rings", "radii", "flat")})
    if a.get("barding"):
        barding = dict(a["barding"])
        if read is not None and read.barding is not None:  # with what else hangs off the body
            extra = ([hb.stirrups(**a["stirrups"])] if a.get("stirrups") else []) + ([b] if build_tail else [])
            barding["hem_length"] = shorter_hem(obj, barding, read, extra)
        res["barding"] = add_barding(obj, van["bones"], barding, a.get("stirrups"))
    if read is not None:
        res["readability"] = read.log
    if build_tail:
        join_piece(obj, "tail", b)
        res["tail"]["built"] = {**b["log"], "offset": [round(x, 2) for x in offset]}
    smart_uv(obj)
    wstats = {}
    transfer_weights(obj, vanilla, a.get("smooth", 2), a.get("influences", 3),
                     van["bones"] if a.get("chain_weights") else None, wstats)
    res["weights"] = wstats
    if a.get("chanfron"):
        res["chanfron"]["pinned_vertices"] = pin_chanfron(obj)
    elif a.get("pieces") and any(q.get("part") == 1 for q in json.loads(open(a["pieces"], encoding="utf-8").read())):
        res["head_pieces_pinned"] = pin_chanfron(obj)  # the ears and plume: rigid with the head
    if a.get("barding") or a.get("pieces") or build_tail:  # panels, stirrups and the tail: their own weights
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


if __name__ == "__main__":  # fresh_rider imports the helpers
    main()
