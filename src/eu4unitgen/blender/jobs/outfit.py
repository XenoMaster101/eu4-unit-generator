"""Blender job (Track W2): the outfit over the base body. The soldier's fitted model (``fresh_rider``'s
``textured_out``: placed, joint-fitted, its arms on the game's hands, its weapons split off) gives the shape; the base
body (``base_body``) gives the topology, weights and UVs, which are kept: only its points move (SPEC Track W, O).

  1. each face of the soldier's model labelled with the base body's tube nearest it (torso, an arm, a leg with its
     foot);
  2. each base-body point pushed along its normal onto the soldier's surface: the hits along the line from
     ``push_in`` inside to ``push`` outside, only those on its own tube's label (an arm never lands on the torso's
     cloth: no web), the nearest; none -> the nearest of its label's surface points, along the normal, within the
     same limits;
  3. "shrink it" (the horse's lesson): an outward push kept at ``1 - shrink`` of itself; within ``joint_reach`` of
     an elbow, knee or wrist at most ``joint_push`` out (bends keep their volume);
  4. the pushes smoothed over the surface (``smooth`` passes), points at the same place moved alike (UV seams);
     normals recomputed;
  5. what of the soldier lies farther than ``push`` from the new body is reported by region (above the neck: a
     helmet's brim; below the hips: a skirt; behind the back: a cloak): the rigid and cloth layers' business.

args: {"body": the base body's geometry JSON, "target": the soldier's textured.obj, "bones", "geometry_out",
       "preview_out", "result_path", "cfg": overrides of OUTFIT}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils.kdtree import KDTree  # noqa: E402

import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
import fresh_horse as fh  # noqa: E402
import fresh_rider as fr  # noqa: E402

OUTFIT = {"push": 0.6, "push_in": 0.35, "shrink": 0.15, "joint_reach": 0.45, "joint_push": 0.3, "smooth": 2,
          "left_over": 0.25}
JOINTS = ("L_elbow", "R_elbow", "L_knee", "R_knee", "L_hand", "R_hand", "L_foot", "R_foot")
# the base body's tubes in order (base_body.build_tubes) and the limb each belongs to
GROUP = {0: "torso", 1: "arm_L", 2: "leg_L", 3: "leg_L", 4: "arm_R", 5: "leg_R", 6: "leg_R"}


def load_target(path):
    """The soldier's model from ``textured_out`` (preview axes) back in PDX axes."""
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    fh.activate(objs[0])
    for o in objs:
        o.select_set(True)
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.transform(Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
    obj.data.update()
    return obj


def vertex_normals(p, tri):
    n = np.zeros_like(p)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    c = {**OUTFIT, **(a.get("cfg") or {})}
    g = json.load(open(a["body"], encoding="utf-8"))
    p = np.array(g["p"], dtype=np.float64).reshape(-1, 3)
    tri = np.array(g["tri"], dtype=np.int64).reshape(-1, 3)
    # the points at one place (a UV seam splits them) move as one
    keys = {}
    gid = np.array([keys.setdefault(tuple(np.round(x, 5)), len(keys)) for x in p])
    G = len(keys)
    gp = np.zeros((G, 3))
    gp[gid] = p
    gtri = gid[tri]
    gn = vertex_normals(gp, gtri)
    label = np.full(G, -1)
    for t, part in zip(gtri, g.get("part") or [0] * len(tri)):
        for v in t:
            if label[v] < 0:
                label[v] = part
    glabel = np.array([GROUP.get(int(x), "torso") for x in label])
    target = load_target(a["target"])
    tco = hs.verts(target)
    tpoly = [list(f.vertices) for f in target.data.polygons]
    tcent = np.array([tco[f].mean(0) for f in tpoly])
    # 1. the soldier's faces labelled by the base body's nearest point
    kd = KDTree(G)
    for i, x in enumerate(gp):
        kd.insert(Vector(x), i)
    kd.balance()
    flab = np.array([glabel[kd.find(Vector(x))[1]] for x in tcent])
    bvh = BVHTree.FromPolygons([tuple(x) for x in tco], tpoly, all_triangles=False)
    by = {}
    for name in set(flab):
        sel = np.flatnonzero(flab == name)
        k = KDTree(len(sel))
        for j, i in enumerate(sel):
            k.insert(Vector(tcent[i]), j)
        k.balance()
        by[name] = (k, sel)
    # 2-3. the pushes
    B = {k: np.asarray(v) for k, v in a["bones"].items()}
    off = np.zeros(G)
    how = {"ray": 0, "nearest": 0, "kept": 0}
    for i in range(G):
        v, n, lab = gp[i], gn[i], glabel[i]
        origin = Vector(v - c["push_in"] * n)
        d = Vector(n)
        left = c["push_in"] + c["push"]
        best = None
        for _ in range(8):
            loc, _, fi, dist = bvh.ray_cast(origin, d, left)
            if loc is None:
                break
            o = float((np.array(loc) - v) @ n)
            if flab[fi] == lab and (best is None or abs(o) < abs(best)):
                best = o
            origin = loc + d * 1e-4
            left -= dist + 1e-4
            if left <= 0:
                break
        if best is not None:
            how["ray"] += 1
        elif lab in by:
            k, sel = by[lab]
            _, j, _ = k.find(Vector(v))
            o = float((tcent[sel[j]] - v) @ n)
            if -c["push_in"] <= o <= c["push"]:
                best = o
                how["nearest"] += 1
        if best is None:
            how["kept"] += 1
            continue
        if best > 0:
            best *= 1.0 - c["shrink"]
            if min(np.linalg.norm(v - B[j]) for j in JOINTS) < c["joint_reach"]:
                best = min(best, c["joint_push"])
        off[i] = best
    # 4. smoothed over the surface
    nb = [set() for _ in range(G)]
    for t in gtri:
        for x in t:
            nb[x].update(int(y) for y in t if y != x)
    for _ in range(int(c["smooth"])):
        off = np.array([0.5 * off[i] + 0.5 * np.mean([off[j] for j in nb[i]]) if nb[i] else off[i] for i in range(G)])
    gp2 = gp + off[:, None] * gn
    gn2 = vertex_normals(gp2, gtri)
    # 5. what's left over
    kd2 = KDTree(G)
    for i, x in enumerate(gp2):
        kd2.insert(Vector(x), i)
    kd2.balance()
    far = np.array([kd2.find(Vector(x))[2] > c["push"] for x in tcent])
    neck, hips = B["head"][1] - 0.3, B["hip"][1]
    regions = {"above_neck": far & (tcent[:, 1] > neck), "below_hips": far & (tcent[:, 1] < hips),
               "behind": far & (tcent[:, 1] >= hips) & (tcent[:, 1] <= neck) & (tcent[:, 2] > 0.6)}
    res = {"pushed": how, "push": {"mean_out": round(float(off[off > 0].mean()) if (off > 0).any() else 0.0, 3),
                                   "mean_in": round(float(off[off < 0].mean()) if (off < 0).any() else 0.0, 3)},
           "left_over": {k: round(float(v.sum() / len(tcent)), 4) for k, v in regions.items()},
           "left_over_all": round(float(far.mean()), 4)}
    out = dict(g)
    out["p"] = [float(x) for x in gp2[gid].ravel()]
    out["n"] = [float(x) for x in gn2[gid].ravel()]
    with open(a["geometry_out"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    if a.get("preview_out"):
        me = bpy.data.meshes.new("outfit")
        me.from_pydata([tuple(x) for x in gp2], [], [tuple(int(y) for y in t) for t in gtri])
        o = bpy.data.objects.new("outfit", me)
        bpy.context.scene.collection.objects.link(o)
        fh.write_obj(o, a["preview_out"])
    res["gaps"] = fr.gaps(gp2, a["bones"])
    _pdx.write_result(a, res)


if __name__ == "__main__":
    main()
