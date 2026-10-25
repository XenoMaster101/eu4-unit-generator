"""SPEC Track R, spike R0 (Blender only; throwaway). Can a tail and a ponytail be skinned to the
unmodified western_generic_01_type1 skeleton?

Per animation: import the vanilla body (io_pdx_mesh, from the install), build placeholder tubes
(tail, ponytail) in three weighting variants (A auto/heat, B rigid, C gradient), import the
vanilla .anim, then measure every frame and render. Nothing is exported.

Blender axes after import: Z up, the unit faces -Y (toes at -Y), its back is +Y, its left +X.

args: {"eu4": install, "anims": [...], "out_dir": ..., "contact_frames": 8, "size": px,
       "worst_size": px, "map_px": px, "only_metrics": bool, "result_path": ...}
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eu4unitgen",
                                "blender", "jobs"))

import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

import _pdx  # noqa: E402

BODY = "western_generic_01_type1"
MISSING_IN_MESH = ["attack_L_hand", "attack_R_hand"]  # animated by the .anim, absent from the mesh
PARTS = {
    # root bone; bones heat weighting may use (A); gradient (C): (root bone, tip bone, tip share)
    "tail": {"root": "hip", "auto": ["hip", "lower_spine", "mid_spine", "L_thigh", "R_thigh"],
             "grad": ("hip", "lower_spine", 0.5), "colour": (0.95, 0.45, 0.1, 1), "variants": "ABC"},
    "ponytail": {"root": "head", "auto": ["head", "top_spine", "mid_spine", "L_shoulder", "R_shoulder"],
                 "grad": ("head", "top_spine", 0.7), "colour": (0.1, 0.75, 0.95, 1), "variants": "ABC"},
    # rerun (alternative 1): nape-length tuft, rigid on head (D = rigid, like B)
    "pony_short": {"root": "head", "colour": (0.1, 0.75, 0.95, 1), "variants": "D"},
    # rerun (alternative 2): long, 0.6 clearance, chain head -> top_spine -> mid_spine (E = chain projection)
    # P1 (test 10 pre-check): the signed-off tail with a flared base; last ponytail round (F, G)
    "tail_flared": {"root": "hip", "grad": ("hip", "lower_spine", 0.5), "colour": (0.95, 0.45, 0.1, 1),
                    "variants": "C"},
    "pony_tuft15": {"root": "head", "colour": (0.1, 0.75, 0.95, 1), "variants": "F"},
    "pony_bun": {"root": "head", "colour": (0.1, 0.75, 0.95, 1), "variants": "G"},
    "pony_long": {"root": "head", "chain": ["head", "top_spine", "mid_spine"], "colour": (0.1, 0.75, 0.95, 1),
                  "variants": "E"},
}
SIDES = 8


# --------------------------------------------------------------------------- scene

def import_body(ie, units):
    before = set(bpy.data.objects)
    ie.import_meshfile(os.path.join(units, BODY + ".mesh"))
    new = [o for o in bpy.data.objects if o not in before]
    rig = next(o for o in new if o.type == "ARMATURE")
    body = next(o for o in new if o.type == "MESH")
    for o in new:
        if o.type == "EMPTY":
            o.hide_render = True
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for n in MISSING_IN_MESH:
        b = rig.data.edit_bones.new(n)
        b.head, b.tail = (0, 0, 0), (0, 0, 0.5)
        b.parent = rig.data.edit_bones["root"]
    bpy.ops.object.mode_set(mode="OBJECT")
    for n in MISSING_IN_MESH:
        rig.data.bones[n].use_deform = False
    return rig, body


def back_profile(body):
    """z -> the body's rearmost y near the midline (|x| < 0.8), in rest pose."""
    prof = {}
    for v in body.data.vertices:
        co = body.matrix_world @ v.co
        if abs(co.x) < 0.8:
            k = round(co.z * 4) / 4
            prof[k] = max(prof.get(k, -99), co.y)
    return prof


def back_y(prof, z):
    ks = [k for k in prof if abs(k - z) <= 0.5]
    return max(prof[k] for k in ks) if ks else max(prof.values())


def centreline(part, prof):
    """Points (rings) and radii of the placeholder tube, in Blender world space at rest."""
    pts, radii = [], []
    if part == "tail_flared":
        # as the signed-off tail, plus a flared base: wider root rings that cover the tunic hem
        pts, radii = centreline("tail", prof)
        radii[0], radii[1] = 0.75, 0.6
        return pts, radii
    if part == "pony_tuft15":
        rings, length, z0 = 5, 1.5, 15.2
        root = Vector((0, back_y(prof, z0) - 0.1, z0))
        ang = math.radians(45)
        for i in range(rings):
            t = i / (rings - 1)
            pts.append(root + Vector((0, math.sin(ang), -math.cos(ang))) * (length / (rings - 1)) * i)
            radii.append(0.33 + (0.14 - 0.33) * t)
        return pts, radii
    if part == "pony_bun":
        # a knot at the nape under the helmet rim: short, pointing straight back
        z0 = 14.9
        root = Vector((0, back_y(prof, z0) - 0.1, z0))
        for i, (d, r) in enumerate(((0.0, 0.3), (0.25, 0.48), (0.55, 0.5), (0.8, 0.38), (0.95, 0.15))):
            pts.append(root + Vector((0, d, -0.05 * i)))
            radii.append(r)
        return pts, radii
    if part == "tail":
        rings, length = 10, 5.0
        z0 = 8.0
        root = Vector((0, back_y(prof, z0) - 0.15, z0))  # root ring sunk 0.15 into the pelvis
        p = root.copy()
        step = length / (rings - 1)
        for i in range(rings):
            t = i / (rings - 1)
            pts.append(p.copy())
            radii.append(0.45 + (0.12 - 0.45) * t)
            ang = math.radians(50 + (15 - 50) * t)  # from vertical, toward the back: arcs out then hangs
            p = p + Vector((0, math.sin(ang), -math.cos(ang))) * step
    elif part == "pony_short":
        rings, length, z0 = 6, 2.0, 15.2
        root = Vector((0, back_y(prof, z0) - 0.1, z0))
        step = length / (rings - 1)
        ang = math.radians(25)  # from vertical, toward the back
        for i in range(rings):
            t = i / (rings - 1)
            pts.append(root + Vector((0, math.sin(ang), -math.cos(ang))) * step * i)
            radii.append(0.35 + (0.15 - 0.35) * t)
    else:
        rings, length = 8, 4.0
        z0 = 15.2
        clearance = 0.6 if part == "pony_long" else 0.25
        root = Vector((0, back_y(prof, z0) - 0.1, z0))
        step = length / (rings - 1)
        for i in range(rings):
            t = i / (rings - 1)
            z = z0 - step * i
            r = 0.35 + (0.15 - 0.35) * t
            # hangs down the back, clear of it by 0.25 beyond the root ring
            y = root.y if i == 0 else max(root.y, back_y(prof, z) + clearance + r)
            pts.append(Vector((0, y, z)))
            radii.append(r)
    return pts, radii


def tube(name, pts, radii):
    """Closed tapered tube (caps at both ends): manifold, so heat weighting works."""
    bm = bmesh.new()
    rings = []
    for i, (c, r) in enumerate(zip(pts, radii)):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        side = Vector((1, 0, 0))
        up = d.cross(side).normalized()
        ring = [bm.verts.new(c + (side * math.cos(a) + up * math.sin(a)) * r)
                for a in (2 * math.pi * k / SIDES for k in range(SIDES))]
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(SIDES):
            bm.faces.new((a[k], a[(k + 1) % SIDES], b[(k + 1) % SIDES], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    # ring index per vertex (creation order) and its position along the length (0..1)
    obj["ring_of"] = [i for i in range(len(pts)) for _ in range(SIDES)]
    return obj


def weight(obj, rig, part, variant):
    cfg = PARTS[part]
    n_rings = max(obj["ring_of"]) + 1
    if variant == "A":
        keep = {b.name: b.use_deform for b in rig.data.bones}
        for b in rig.data.bones:
            b.use_deform = b.name in cfg["auto"]
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        for b in rig.data.bones:
            b.use_deform = keep[b.name]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
        empty = [v.index for v in obj.data.vertices if not v.groups or sum(g.weight for g in v.groups) < 1e-6]
        return {"unweighted_verts": len(empty)}
    obj.parent = rig
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    if variant == "E":
        # chain projection: a vertex's height between two consecutive joints of the chain sets
        # its blend of those two bones (above the first joint: all first; below the last: all last)
        joints = [(b, (rig.matrix_world @ rig.data.bones[b].head_local).z) for b in cfg["chain"]]
        groups = {b: obj.vertex_groups.new(name=b) for b, _ in joints}
        for v in obj.data.vertices:
            z = (obj.matrix_world @ v.co).z
            if z >= joints[0][1]:
                w = {joints[0][0]: 1.0}
            elif z <= joints[-1][1]:
                w = {joints[-1][0]: 1.0}
            else:
                for (b0, z0), (b1, z1) in zip(joints, joints[1:]):
                    if z1 <= z <= z0:
                        t = (z0 - z) / (z0 - z1)
                        w = {b0: 1 - t, b1: t}
                        break
            for b, x in w.items():
                if x > 0:
                    groups[b].add([v.index], x, "REPLACE")
        return {"unweighted_verts": 0}
    if variant in ("B", "D", "F", "G"):
        g = obj.vertex_groups.new(name=cfg["root"])
        g.add([v.index for v in obj.data.vertices], 1.0, "REPLACE")
    else:
        a, b, share = cfg["grad"]
        ga, gb = obj.vertex_groups.new(name=a), obj.vertex_groups.new(name=b)
        for v in obj.data.vertices:
            t = obj["ring_of"][v.index] / (n_rings - 1)
            ga.add([v.index], 1 - share * t, "REPLACE")
            if t > 0:
                gb.add([v.index], share * t, "REPLACE")
    return {"unweighted_verts": 0}


def weight_summary(obj):
    out = {}
    for v in obj.data.vertices:
        for g in v.groups:
            name = obj.vertex_groups[g.group].name
            out[name] = out.get(name, 0) + g.weight
    n = len(obj.data.vertices)
    return {k: round(w / n, 3) for k, w in sorted(out.items(), key=lambda kv: -kv[1]) if w / n >= 0.005}


# --------------------------------------------------------------------------- metrics

def evaluated(obj, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    co = [obj.matrix_world @ v.co for v in me.vertices]
    tris = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return co, tris


def bone_matrix(rig, name):
    return rig.matrix_world @ rig.pose.bones[name].matrix


def tri_normal(co, t):
    return (co[t[1]] - co[t[0]]).cross(co[t[2]] - co[t[0]]).normalized()


class Rest:
    def __init__(self, obj, rig, part, depsgraph):
        self.co, self.tris = evaluated(obj, depsgraph)
        self.edges = sorted({tuple(sorted((t[i], t[(i + 1) % 3]))) for t in self.tris for i in range(3)})
        self.len = [(self.co[a] - self.co[b]).length for a, b in self.edges]
        mb = bone_matrix(rig, PARTS[part]["root"])
        inv = mb.inverted()
        self.local = [inv @ c for c in self.co]
        rot = inv.to_3x3()
        self.normals = [(rot @ tri_normal(self.co, t)).normalized() for t in self.tris]
        self.ring_of = list(obj["ring_of"])
        # anatomy check: the tail stays behind the hip joint, the ponytail behind top_spine
        self.ref = "hip" if part == "tail" else "top_spine"
        rm = bone_matrix(rig, self.ref)
        self.back_local = (rm.to_3x3().inverted() @ Vector((0, 1, 0))).normalized()  # world +Y = the unit's back


def measure(obj, rig, part, rest, depsgraph, body_bvh):
    co, _ = evaluated(obj, depsgraph)
    ratios = sorted((co[a] - co[b]).length / l0 for (a, b), l0 in zip(rest.edges, rest.len) if l0 > 1e-6)
    p95 = lambda q: ratios[min(len(ratios) - 1, int(q * (len(ratios) - 1)))]  # noqa: E731
    mb = bone_matrix(rig, PARTS[part]["root"])
    inv = mb.inverted()
    rot = inv.to_3x3()
    flips = sum(1 for t, n0 in zip(rest.tris, rest.normals) if (rot @ tri_normal(co, t)).dot(n0) < 0)
    root_idx = [i for i, r in enumerate(rest.ring_of) if r == 0]
    drift = max((inv @ co[i] - rest.local[i]).length for i in root_idx)
    pen, depth, pen_rings = 0, 0.0, set()
    for i, c in enumerate(co):
        if rest.ring_of[i] == 0:
            continue  # the root ring is sunk into the body by design
        loc, nrm, _, dist = body_bvh.find_nearest(c)
        if loc is not None and (c - loc).dot(nrm) < 0:
            pen += 1
            depth = max(depth, dist)
            pen_rings.add(rest.ring_of[i])
    n_out = sum(1 for r in rest.ring_of if r != 0)
    rm = bone_matrix(rig, rest.ref)
    back = (rm.to_3x3() @ rest.back_local).normalized()
    behind = min((c - rm.translation).dot(back) for i, c in enumerate(co) if rest.ring_of[i] >= 2)
    # facing-frame version: the unit's back direction from root (never pitches), origin at the
    # reference joint, horizontal only. The bone-axis version above misreads a forward lunge.
    rootm = bone_matrix(rig, "root")
    face_back = rootm.to_3x3() @ (bone_matrix_rest_root_inv(rig) @ Vector((0, 1, 0)))
    face_back.z = 0
    face_back.normalize()
    behind_facing = min((c - rm.translation).dot(face_back) for i, c in enumerate(co) if rest.ring_of[i] >= 2)
    tip = [c for i, c in enumerate(co) if rest.ring_of[i] == max(rest.ring_of)]
    root = [c for i, c in enumerate(co) if rest.ring_of[i] == 0]
    tipc, rootc = sum(tip, Vector()) / len(tip), sum(root, Vector()) / len(root)
    v = tipc - rootc
    elev = math.degrees(math.atan2(v.z, math.hypot(v.x, v.y)))  # -90 = straight down
    return {"str_min": ratios[0], "str_max": ratios[-1], "str_p05": p95(0.05), "str_p95": p95(0.95),
            "pen_n": pen, "pen_pct": 100.0 * pen / n_out, "pen_depth": depth, "flips": flips,
            "root_drift": drift, "behind": behind, "tip_elev": elev,
            "pen_rings": sorted(pen_rings), "behind_facing": behind_facing}


def bone_matrix_rest_root_inv(rig):
    """World +Y (the unit's back) in root's local frame at bind pose."""
    return (rig.matrix_world @ rig.data.bones["root"].matrix_local).to_3x3().inverted()


def body_tree(body, depsgraph):
    co, tris = evaluated(body, depsgraph)
    return BVHTree.FromPolygons(co, tris)


# --------------------------------------------------------------------------- render

VIEWS = {"front": (0, 1, 0), "side": (-1, 0, 0), "rear": (0, -1, 0)}
MAP_VIEWS = {"front_quarter": (-0.45, 0.55, -0.7), "rear_quarter": (0.45, -0.55, -0.7)}  # from above (UNVERIFIED map angle)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = False
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "OBJECT"
    sh.show_backface_culling = False
    sh.show_cavity = False
    world = scene.world or bpy.data.worlds.new("spike")
    scene.world = world
    world.color = (0.82, 0.84, 0.86)
    cam = bpy.data.objects.new("spike_cam", bpy.data.cameras.new("spike_cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    scene.camera = cam
    return scene, cam


def frame_cam(cam, direction, lo, hi):
    centre, d = (lo + hi) / 2, Vector(direction).normalized()
    extent = (hi - lo).length
    cam.location = centre - d * (extent * 2 + 10)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = extent * 4 + 40
    right = d.cross(Vector((0, 0, 1)))
    right = right.normalized() if right.length > 1e-6 else Vector((1, 0, 0))
    up = right.cross(d).normalized()
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    w = max(abs((c - centre).dot(right)) for c in corners) * 2
    h = max(abs((c - centre).dot(up)) for c in corners) * 2
    cam.data.ortho_scale = max(w, h) * 1.06


def render(scene, path, size):
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


# --------------------------------------------------------------------------- main

def run_anim(a, ie, units, anim):
    _pdx.clear_scene()
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)
    scene, cam = setup_render()
    rig, body = import_body(ie, units)
    body.color = (0.38, 0.38, 0.4, 1)
    prof = back_profile(body)
    objs, wsum = {}, {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for part in a.get("parts", ["tail", "ponytail"]):
        pts, radii = centreline(part, prof)
        for var in PARTS[part]["variants"]:
            o = tube(f"{part}_{var}", pts, radii)
            o.color = PARTS[part]["colour"]
            info = weight(o, rig, part, var)
            objs[(part, var)] = o
            wsum[f"{part}_{var}"] = {**info, "groups": weight_summary(o), "tris": len(o.data.polygons)}
    # rest (bind pose) before the animation: reference shape and a penetration check at rest
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rests = {k: Rest(o, rig, k[0], depsgraph) for k, o in objs.items()}
    tree = body_tree(body, depsgraph)
    rest_metrics = {f"{p}_{v}": measure(o, rig, p, rests[(p, v)], depsgraph, tree) for (p, v), o in objs.items()}
    if a.get("save_rest_render") and anim == a["anims"][0]:
        lo, hi = bounds([body] + list(objs.values()))
        for view, d in VIEWS.items():
            frame_cam(cam, d, lo, hi)
            render(scene, os.path.join(a["out_dir"], f"rest_{view}.png"), a["worst_size"])
    # animation
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    import_anim(ie, rig, os.path.join(units, "animation", anim + ".anim"))
    f0, f1 = scene.frame_start, scene.frame_end
    per_frame = {f"{p}_{v}": [] for (p, v) in objs}
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        tree = body_tree(body, depsgraph)
        for (p, v), o in objs.items():
            m = measure(o, rig, p, rests[(p, v)], depsgraph, tree)
            m["frame"] = f
            per_frame[f"{p}_{v}"].append(m)
    renders = {"contact": {}, "worst": {}, "map": {}}
    if not a.get("only_metrics"):
        # common framing over the whole animation so frames line up
        lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
        for f in range(f0, f1 + 1, max(1, (f1 - f0) // 12)):
            scene.frame_set(f)
            l2, h2 = bounds([body] + list(objs.values()), evaluated_pts=True)
            lo = Vector(map(min, lo, l2))
            hi = Vector(map(max, hi, h2))
        n = a.get("contact_frames", 8)
        frames = sorted({f0 + round(i * (f1 - f0) / max(1, n - 1)) for i in range(n)})
        for var in sorted({v for _, v in objs}):
            for (p, v), o in objs.items():
                o.hide_render = v != var
            shots = []
            for f in ([] if a.get("map_only") else frames):
                scene.frame_set(f)
                for view, d in VIEWS.items():
                    frame_cam(cam, d, lo, hi)
                    shots.append(render(scene, os.path.join(a["out_dir"], "frames", f"{anim}_{var}_f{f:03d}_{view}.png"),
                                        a["size"]))
            renders["contact"][var] = {"frames": frames, "views": list(VIEWS), "files": shots}
            # map scale: middle frame, from above
            scene.frame_set(frames[len(frames) // 2])
            renders["map"][var] = {}
            for mv, d in MAP_VIEWS.items():
                frame_cam(cam, d, lo, hi)
                renders["map"][var][mv] = render(scene, os.path.join(a["out_dir"], "map", f"{anim}_{var}_{mv}.png"),
                                                 a["map_px"])
        # worst frame per part and variant: deepest penetration, then the largest stretch deviation
        for (p, v), o in ([] if a.get("map_only") else objs.items()):
            rows = per_frame[f"{p}_{v}"]
            worst = max(rows, key=lambda m: (m["pen_depth"], max(m["str_max"] - 1, 1 - m["str_min"])))
            for (p2, v2), o2 in objs.items():
                o2.hide_render = (p2, v2) != (p, v)
            scene.frame_set(worst["frame"])
            files = []
            for view, d in VIEWS.items():
                frame_cam(cam, d, lo, hi)
                files.append(render(scene, os.path.join(a["out_dir"], "worst", f"{anim}_{p}_{v}_{view}.png"),
                                    a["worst_size"]))
            renders["worst"][f"{p}_{v}"] = {"frame": worst["frame"], "files": files}
    return {"frames": [f0, f1], "fps": scene.render.fps, "weights": wsum, "rest": rest_metrics,
            "per_frame": per_frame, "renders": renders}


def import_anim(ie, rig, path):
    """io_pdx_mesh refuses an .anim that names bones the mesh lacks; the game ignores those
    tracks. Add unweighted dummies under root for exactly the bones it names, then retry."""
    import re
    try:
        ie.import_animfile(path, frame_start=1)
    except RuntimeError as exc:
        missing = re.findall(r"'([^']+)'", str(exc)) if "Missing bones" in str(exc) else None
        if not missing:
            raise
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for n in missing:
            b = rig.data.edit_bones.new(n)
            b.head, b.tail = (0, 0, 0), (0, 0, 0.5)
            b.parent = rig.data.edit_bones["root"]
        bpy.ops.object.mode_set(mode="OBJECT")
        for n in missing:
            rig.data.bones[n].use_deform = False
        rig.select_set(True)
        ie.import_animfile(path, frame_start=1)


def bounds(objs, evaluated_pts=False):
    pts = []
    if evaluated_pts:
        dg = bpy.context.evaluated_depsgraph_get()
        for o in objs:
            pts += evaluated(o, dg)[0]
    else:
        pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def main():
    a = _pdx.args()
    ie = _pdx.io_module()
    units = os.path.join(a["eu4"], "gfx", "models", "Units")
    a["out_dir"] = a.get("out_dir_override") or a["out_dir"]
    for sub in ("frames", "worst", "map"):
        os.makedirs(os.path.join(a["out_dir"], sub), exist_ok=True)
    out = {"anims": {}}
    for anim in a["anims"]:
        out["anims"][anim] = run_anim(a, ie, units, anim)
    _pdx.write_result(a, out)


if __name__ == "__main__":
    main()
