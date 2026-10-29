"""Blender job (Track W1): the base body. TRELLIS.2's model of a plain person in an A-pose gives the shape; the
topology and weights are our own, built on the game's bones (SPEC Track W, B). PDX mesh space (Y up, facing -Z, the
figure's left at +X), as fresh_rider.

  1. the raw model imported, scaled so its top is at ``top``, its feet on the ground, centred; the joint fit
     (``fresh_rider.joint_fit``: the spine inside the torso, the legs at the game's stance); each empty arm swung to
     the game's hand (``free_hand``, ``swing_arm``);
  2. the template: closed tubes of rings along the game's bones: the torso and head (the spine, from the crotch to
     the top of the head), each arm (from inside the shoulder to past the hand node), each leg (from inside the
     hip to under the ankle), each foot (heel to toe, along the ground). A tube's rings every ``SPACING`` and three
     round each joint (``JOINT_RINGS``), so bends happen on rings; the limbs are their own tubes, so nothing webs
     an arm to the torso or the legs together; they start inside the torso (the overlap: no seam to bridge);
  3. each ring's radius in each of its directions measured on the raw model's cross-section in the ring's plane
     (``ring_radii``: its surface points within ``slab`` of the plane, their median the ring's centre, at most
     ``recentre`` off the bone; per direction the ``reach`` percentile of their distance), within ``clamp`` of
     the ring's median (a fused armpit or crotch reads as far), smoothed round the ring and along the tube (rays
     from the bone were tried first: where the bone runs outside the limb they meet its near side: pinched arms);
  4. weights by rule, as the game's body has them (measured from western_generic_01_type1, 2026-10-29): each bone
     owns its stretch of its chain and blends with the next across ``BLEND`` either side of the joint; the torso
     by height (hip, lower_spine from 8.7, top_spine from 11.0, head from 13.95; mid_spine and the wrists carry
     nothing, as on the game's body), the shoulders blended into the top of the torso, the thighs into the hips;
  5. UVs: one box per tube (the torso and head half the texture);
  6. written as the game's geometry (``geometry_out``), the fitted raw model with its UVs as the bake's source
     (``textured_out``), and measures (rings, triangles, joints inside the body, outline gaps).

args: {"vanilla": base arrays (p, tri, weights, bones), "source": {"path", "merge"}, "top": 16.2,
       "geometry_out", "preview_out", "textured_out", "result_path", "cfg": {overrides of SHAPE}}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils.kdtree import KDTree  # noqa: E402

import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
import fresh_horse as fh  # noqa: E402
import fresh_rider as fr  # noqa: E402

SHAPE = {"top": 16.2, "ring_torso": 16, "ring_limb": 12, "ring_foot": 10, "spacing_torso": 0.55, "spacing_limb": 0.5,
         "spacing_foot": 0.35, "joint_rings": [-0.35, 0.0, 0.35], "blend": 0.4,
         "knee_blend": 1.1, "elbow_blend": 0.55, "hip_blend": 0.8, "neck_blend": 0.3, "recentre": 0.3, "slab": 0.06, "reach": 90,
         "clamp": [0.55, 1.4], "max_ray": {"torso": 3.0, "limb": 1.3, "foot": 1.0}, "crotch": 7.0,
         "torso_joints": [8.7, 11.0, 13.95], "shoulder_mix": 0.4, "shoulder_reach": 1.0, "thigh_mix": 0.25,
         "arm_start": 0.6, "leg_start": 0.6, "hand_tip": 0.45, "ankle_below": 0.35, "smooth_along": 3, "spacing_head": 0.3}
BOXES = {"torso": (0.0, 0.0, 0.5, 1.0), "arm_L": (0.5, 0.0, 0.625, 0.62), "arm_R": (0.625, 0.0, 0.75, 0.62),
         "leg_L": (0.75, 0.0, 0.875, 0.62), "leg_R": (0.875, 0.0, 1.0, 0.62), "foot_L": (0.5, 0.64, 0.75, 0.82),
         "foot_R": (0.75, 0.64, 1.0, 0.82)}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Chain:
    """A polyline of points with a length parameter."""

    def __init__(self, pts):
        self.p = [np.asarray(x, dtype=np.float64) for x in pts]
        self.s = [0.0]
        for a, b in zip(self.p, self.p[1:]):
            self.s.append(self.s[-1] + float(np.linalg.norm(b - a)))

    @property
    def length(self):
        return self.s[-1]

    def at(self, t):
        t = min(max(t, 0.0), self.length)
        for i in range(len(self.p) - 1):
            if t <= self.s[i + 1] or i == len(self.p) - 2:
                f = (t - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
                return self.p[i] + f * (self.p[i + 1] - self.p[i])
        return self.p[-1]

    def tangent(self, t, h=0.35):
        return unit(self.at(t + h) - self.at(t - h))


def stations(chain, joints, spacing, joint_rings, start=0.0, end=None):
    """Where the rings go: every ``spacing`` and ``joint_rings`` round each joint, closer ones merged."""
    end = chain.length if end is None else end
    n = max(2, int(math.ceil((end - start) / spacing)) + 1)
    ts = list(np.linspace(start, end, n))
    for j in joints:
        ts += [j + d for d in joint_rings if start < j + d < end]
    ts = sorted(ts)
    out = [ts[0]]
    for t in ts[1:]:
        if t - out[-1] > 0.15:
            out.append(t)
        elif abs(t - end) < 1e-6:
            out[-1] = t
    return out


def frames(chain, ts, ref):
    """A frame per station: the tangent and two axes across it, carried along without twist from ``ref``."""
    out = []
    e1 = None
    for t in ts:
        tg = chain.tangent(t)
        if e1 is None:
            e1 = unit(np.asarray(ref) - (np.asarray(ref) @ tg) * tg)
        else:
            e1 = unit(e1 - (e1 @ tg) * tg)
        out.append((chain.at(t), tg, e1, np.cross(tg, e1)))
    return out


def ring_radii(src, centre, tg, e1, e2, n, max_r, cfg):
    """A ring's radii from the raw model's cross-section in the ring's plane (module doc, 3): its points within
    ``slab`` of the plane and ``max_r`` of the centre; the centre moved onto their median (at most ``recentre``);
    in each of ``n`` directions the ``reach`` percentile of their distance from it. -> (centre, radii or None)."""
    kd, co = src
    slab = cfg["slab"]
    idx = [i for _, i, _ in kd.find_range(Vector(centre), math.hypot(max_r, slab))]
    if len(idx) < 12:
        return centre, None
    rel = co[idx] - centre
    pts = rel[np.abs(rel @ tg) < slab]
    xy = np.stack([pts @ e1, pts @ e2], 1)
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < max_r]
    if len(xy) < 12:
        return centre, None
    mid = np.zeros(2)
    for _ in range(2):  # onto the limb's own middle (another limb's or the torso's points pull the mean)
        near = xy[np.hypot(*(xy - mid).T) < max_r]
        if len(near) < 12:
            break
        m = np.median(near, 0)
        ln = np.hypot(*m)
        mid = m * min(1.0, cfg["recentre"] / ln) if ln > 1e-9 else m
    centre = centre + mid[0] * e1 + mid[1] * e2
    d = xy - mid
    dist, ang = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0]) % (2 * math.pi)
    r = []
    for k in range(n):
        a = 2 * math.pi * k / n
        diff = np.abs((ang - a + math.pi) % (2 * math.pi) - math.pi)
        sel = dist[diff < math.pi / n * 1.5]
        r.append(float(np.percentile(sel, cfg["reach"])) if len(sel) >= 3 else None)
    good = [x for x in r if x is not None]
    if not good:
        return centre, None
    med = float(np.median(good))
    lo, hi = cfg["clamp"]
    r = np.array([min(max(x if x is not None else med, lo * med), hi * med) for x in r])
    return centre, 0.25 * np.roll(r, 1) + 0.5 * r + 0.25 * np.roll(r, -1)


def blend_owner(t, owners, blend):
    """Weights at ``t`` along a chain: ``owners`` [(start, bone) or (start, bone, its blend)] in order; each joint
    (an owner's start) blends linearly across its blend (else ``blend``) either side, half and half at the joint."""
    cur = owners[0][1]
    for o in owners:
        if t >= o[0]:
            cur = o[1]
    for i in range(1, len(owners)):
        s, bone = owners[i][0], owners[i][1]
        bl = owners[i][2] if len(owners[i]) > 2 else blend
        if abs(t - s) < bl:
            f = 0.5 + 0.5 * (t - s) / bl
            return {owners[i - 1][1]: 1.0 - f, bone: f}
    return {cur: 1.0}


def mix(w, bone, amount):
    if amount <= 0:
        return w
    out = {k: v * (1.0 - amount) for k, v in w.items()}
    out[bone] = out.get(bone, 0.0) + amount
    return out


def top4(w):
    items = sorted(w.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(v for _, v in items) or 1.0
    return {k: v / tot for k, v in items if v > 1e-4}


def place_body(obj, top):
    """One scale: the top at ``top``, the feet on the ground; x on the median, z on the torso's median."""
    co = hs.verts(obj)
    y0, y1 = co[:, 1].min(), co[:, 1].max()
    s = top / (y1 - y0)
    x0 = float(np.median(co[:, 0]))
    H = y1 - y0
    torso = co[(co[:, 1] > y0 + 0.5 * H) & (co[:, 1] < y0 + 0.8 * H) & (np.abs(co[:, 0] - x0) < 0.12 * H)]
    off = np.array([-x0 * s, -y0 * s, -float(np.median(torso[:, 2])) * s])
    hs.set_verts(obj, co * s + off)
    return s, off


def build_tubes(bones, src, c):
    """The tubes (module doc, 2-4): [{"name", "rings": [(centre, e1, e2, radii)], "weights": [per ring or per
    vertex], "cap": (start, end)}]."""
    B = {k: np.asarray(v, dtype=np.float64) for k, v in bones.items()}
    tubes = []
    jr = c["joint_rings"]
    # the torso and head along the spine
    hip = B["hip"]
    spine = Chain([[0.0, c["crotch"], hip[2]], hip, B["lower_spine"], B["mid_spine"], B["top_spine"], B["head"],
                   [0.0, c["top"], B["head"][2]]])
    joints_t = []
    for y in c["torso_joints"]:  # the chain's parameter at each height
        ts = np.linspace(0, spine.length, 400)
        joints_t.append(float(ts[np.argmin([abs(spine.at(t)[1] - y) for t in ts])]))
    head = list(np.arange(joints_t[2] + 0.3, spine.length - 0.12, c["spacing_head"]))  # the skull: rounder
    ts = stations(spine, joints_t + head, c["spacing_torso"], jr, 0.0, spine.length - 0.12)
    tubes.append(ring_tube("torso", spine, ts, [0.0, 0.0, -1.0], src, c["ring_torso"], c["max_ray"]["torso"], c,
                           lambda t, p: torso_weights(p, B, c)))
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        node = B[f"{'Left' if side == 'L' else 'Right'}_hand_node"]
        tip = node + c["hand_tip"] * unit(node - B[f"{side}_hand"])
        arm = Chain([B[f"{side}_shoulder"], B[f"{side}_arm"], B[f"{side}_elbow"], B[f"{side}_wrist"], B[f"{side}_hand"],
                     node, tip])
        sj = arm.s  # shoulder 0, arm s[1], elbow s[2], wrist s[3], hand s[4]
        owners = [(0.0, f"{side}_shoulder"), (sj[1], f"{side}_arm"), (sj[2], f"{side}_elbow", c["elbow_blend"]),
                  (sj[4], f"{side}_hand")]
        start = c["arm_start"] * sj[1]
        ts = stations(arm, [sj[1], sj[2], sj[4]], c["spacing_limb"], jr, start, arm.length)

        def arm_w(t, p, owners=owners, sj=sj):
            w = blend_owner(t, owners, c["blend"])
            if t < sj[1]:  # the top of the arm, inside the shoulder: the torso's too (the game's: 0.43)
                w = mix(w, "top_spine", 0.4 * (1.0 - t / sj[1]))
            return w
        tubes.append(ring_tube(f"arm_{side}", arm, ts, [0.0, 0.0, -1.0], src, c["ring_limb"], c["max_ray"]["limb"], c,
                               arm_w, inside_until=sj[1]))
        thigh = B[f"{side}_thigh"]
        ankle = B[f"{side}_foot"]
        bottom = ankle + c["ankle_below"] * unit(ankle - B[f"{side}_knee"])
        leg = Chain([thigh + np.array([0.0, c["leg_start"], 0.0]), thigh, B[f"{side}_knee"], ankle, bottom])
        lj = leg.s
        owners = [(0.0, "hip"), (lj[1] + 0.35, f"{side}_thigh"), (lj[2], f"{side}_knee", c["knee_blend"]),
                  (lj[3], f"{side}_foot")]
        ts = stations(leg, [lj[1], lj[2], lj[3]], c["spacing_limb"], jr, 0.0, leg.length)

        def leg_w(t, p, owners=owners):
            w = blend_owner(t, owners, c["hip_blend"])  # the hip's blend is wider (the game's: 0.7 each way)
            return w
        tubes.append(ring_tube(f"leg_{side}", leg, ts, [0.0, 0.0, -1.0], src, c["ring_limb"], c["max_ray"]["limb"], c,
                               leg_w, inside_until=lj[1]))
        heel = np.array([ankle[0], 0.42, ankle[2] + 0.55])
        toe = B[f"{side}_toe"]
        tip = np.array([toe[0], 0.3, toe[2] - 0.4])
        foot = Chain([heel, np.array([ankle[0], 0.42, ankle[2]]), np.array([toe[0], 0.36, toe[2]]), tip])
        fj = foot.s
        owners = [(0.0, f"{side}_foot"), (fj[2] + 0.1, f"{side}_toe")]
        ts = stations(foot, [fj[2]], c["spacing_foot"], [-0.15, 0.0, 0.15], 0.0, foot.length)
        tubes.append(ring_tube(f"foot_{side}", foot, ts, [0.0, 1.0, 0.0], src, c["ring_foot"], c["max_ray"]["foot"], c,
                               lambda t, p, owners=owners: blend_owner(t, owners, 0.25)))
    return tubes


def torso_weights(p, B, c):
    """A torso vertex's weights: by height along the spine, the shoulders and thighs mixed in near their joints."""
    y = p[1]
    j1, j2, j3 = c["torso_joints"]
    owners = [(-1e9, "hip"), (j1, "lower_spine"), (j2, "top_spine"), (j3, "head")]
    w = None
    for i in range(1, 4):
        s, bone = owners[i]
        bl = c["neck_blend"] if bone == "head" else c["blend"]
        if abs(y - s) < bl:
            f = 0.5 + 0.5 * (y - s) / bl
            w = {owners[i - 1][1]: 1.0 - f, bone: f}
            break
    if w is None:
        w = {next(b for s, b in reversed(owners) if y >= s): 1.0}
    for side in ("L", "R"):
        d = np.linalg.norm(p - B[f"{side}_shoulder"])
        w = mix(w, f"{side}_shoulder", c["shoulder_mix"] * max(0.0, 1.0 - d / c["shoulder_reach"]))
        if y < B["hip"][1]:
            d = np.linalg.norm(p - B[f"{side}_thigh"])
            w = mix(w, f"{side}_thigh", c["thigh_mix"] * max(0.0, 1.0 - d / 1.2))
    return w


def ring_tube(name, chain, ts, ref, src, n, max_ray, c, weight_fn, inside_until=None):
    """One tube's rings, weights and caps. ``inside_until``: a limb's rings before it (inside the shoulder or hip)
    take the first ring's after it (measured there, they'd take the torso's cross-section)."""
    rings = []
    for (centre, tg, e1, e2), t in zip(frames(chain, ts, ref), ts):
        centre2, r = ring_radii(src, centre, tg, e1, e2, n, max_ray, c)
        rings.append([t, centre2, e1, e2, r])
    if inside_until is not None:
        first = next((x for x in rings if x[0] >= inside_until and x[4] is not None), None)
        if first is not None:
            for x in rings:
                if x[0] < inside_until:
                    x[1] = chain.at(x[0]) + (first[1] - chain.at(first[0]))
                    x[4] = first[4].copy()
    known = [x[4] for x in rings if x[4] is not None]  # a ring with nothing measured: its neighbours'
    for i, x in enumerate(rings):
        if x[4] is None:
            near = [rings[j][4] for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(rings) and rings[j][4] is not None]
            x[4] = near[0] if near else (known[0] if known else np.full(n, 0.3))
    for _ in range(int(c["smooth_along"])):  # along the tube: each ring's radii towards its neighbours'
        rr = [x[4].copy() for x in rings]
        for i in range(1, len(rings) - 1):
            rings[i][4] = 0.25 * rr[i - 1] + 0.5 * rr[i] + 0.25 * rr[i + 1]
    verts, weights = [], []
    for t, centre, e1, e2, r in rings:
        row = []
        for k in range(n):
            a = 2 * math.pi * k / n
            p = centre + r[k] * (math.cos(a) * e1 + math.sin(a) * e2)
            row.append(p)
            weights.append(top4(weight_fn(t, p)))
        verts.append(row)
    # the caps: a point past each end ring, along the tube, half its mean radius out
    t0, t1 = rings[0], rings[-1]
    cap0 = t0[1] - 0.5 * float(np.mean(t0[4])) * chain.tangent(t0[0])
    cap1 = t1[1] + 0.5 * float(np.mean(t1[4])) * chain.tangent(t1[0])
    return {"name": name, "rings": verts, "weights": weights, "caps": (cap0, cap1),
            "cap_weights": (top4(weight_fn(t0[0], cap0)), top4(weight_fn(t1[0], cap1))), "n": n}


def make_mesh(tubes):
    """One object from the tubes: quads between rings, fans at the ends, UVs per box, vertex groups."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UV")
    part = bm.faces.layers.int.new("part")  # the tube each face is on (the outfit pushes a limb onto its own limb)
    allw = []
    for ti, tb in enumerate(tubes):
        n = tb["n"]
        u0, v0, u1, v1 = BOXES[tb["name"]]
        m = 0.004  # a margin inside the box
        u0, v0, u1, v1 = u0 + m, v0 + m, u1 - m, v1 - m
        rows = []
        for row in tb["rings"]:
            rows.append([bm.verts.new(Vector(p)) for p in row])
        allw += tb["weights"]
        caps = [bm.verts.new(Vector(p)) for p in tb["caps"]]
        allw += list(tb["cap_weights"])
        R = len(rows)

        def uvat(i, k):
            return (u0 + (u1 - u0) * k / n, v0 + (v1 - v0) * (i + 1) / (R + 1))
        for i in range(R - 1):
            for k in range(n):
                q = [rows[i][k], rows[i][(k + 1) % n], rows[i + 1][(k + 1) % n], rows[i + 1][k]]
                f = bm.faces.new(q)
                f[part] = ti
                for lp, (ii, kk) in zip(f.loops, ((i, k), (i, k + 1), (i + 1, k + 1), (i + 1, k))):
                    lp[uv].uv = uvat(ii, kk)
        for end, i, cv in ((0, 0, caps[0]), (1, R - 1, caps[1])):
            for k in range(n):
                tri = [rows[i][k], rows[i][(k + 1) % n], cv] if end else [rows[i][(k + 1) % n], rows[i][k], cv]
                f = bm.faces.new(tri)
                f[part] = ti
                cvv = (u0 + (u1 - u0) * (k + 0.5) / n, v0 if end == 0 else v1)
                spots = ((i, k), (i, k + 1), None) if end else ((i, k + 1), (i, k), None)
                for lp, sp in zip(f.loops, spots):
                    lp[uv].uv = uvat(*sp) if sp else cvv
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("base_body")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("base_body", me)
    bpy.context.scene.collection.objects.link(obj)
    groups = {}
    for vi, w in enumerate(allw):
        for bone, x in w.items():
            g = groups.get(bone) or obj.vertex_groups.new(name=bone)
            groups[bone] = g
            g.add([vi], x, "REPLACE")
    me.update()
    return obj


def hidden_faces(obj):
    """Each face whose middle lies inside another tube (the limbs' roots inside the torso, the leg's end inside the
    foot): it never shows, so its folds don't count (``check_mesh_deform``'s ``skip``). Inside: an odd number of
    crossings of that tube along a ray."""
    me = obj.data
    attr = me.attributes["part"]
    part = [0] * len(me.polygons)
    attr.data.foreach_get("value", part)
    co = [v.co.copy() for v in me.vertices]
    tubes = {}
    for p in me.polygons:
        tubes.setdefault(part[p.index], []).append(tuple(p.vertices))
    trees = {t: BVHTree.FromPolygons(co, polys) for t, polys in tubes.items()}
    d = Vector((0.577, 0.583, 0.571)).normalized()  # off the axes: no ray along an edge
    out = []
    for p in me.polygons:
        c = p.center
        inside = False
        for t, tree in trees.items():
            if t == part[p.index]:
                continue
            n, o, left = 0, c.copy(), 40.0
            while left > 0:
                loc, _, _, dist = tree.ray_cast(o, d, left)
                if loc is None:
                    break
                n += 1
                o, left = loc + d * 1e-4, left - dist - 1e-4
            if n % 2:
                inside = True
                break
        out.append(inside)
    return out


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    c = {**SHAPE, **(a.get("cfg") or {})}
    if a.get("top"):
        c["top"] = float(a["top"])
    van = a["vanilla"]
    vp = van["p"]
    vanilla = fh.mesh_object("vanilla", [vp[i:i + 3] for i in range(0, len(vp), 3)],
                             [van["tri"][i:i + 3] for i in range(0, len(van["tri"]), 3)])
    bones = van["bones"]
    res = {}
    obj = fh.import_model(a["source"]["path"])
    fh.merge_seams(obj, float(a["source"].get("merge", 1e-5)))
    res["imported"] = {"tris": fh.tri_count(obj), "pieces": fh.count_pieces(obj)}
    s, off = place_body(obj, c["top"])
    res["place"] = {"scale": float(s), "offset": off.tolist()}
    co = hs.verts(obj)
    keep = np.ones(len(co), dtype=bool)
    van_co = fr.surface_points(vanilla)
    res["joints_before"] = fr.joints_report(co, van_co, bones)
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    _, res["joint_fit"] = fr.joint_fit(co, keep, van_co, bones, legs=True)
    moves = {}
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        grip = fr.free_hand(co, keep, bones, side)
        if grip is not None:
            ang, slide = fr.swing_arm(co, keep, grip, bones[f"{side}_hand"], bones[f"{side}_shoulder"], sgn, edges)
            moves[side] = {"turned_deg": round(ang, 1), "slid": round(float(np.linalg.norm(slide)), 3)}
    res["arms"] = moves
    hs.set_verts(obj, co)
    if a.get("textured_out"):
        fh.write_obj(obj, a["textured_out"], uv=True)
    pts = fr.surface_points(obj, per=2)
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(Vector(p), i)
    kd.balance()
    tubes = build_tubes(bones, (kd, pts), c)
    body = make_mesh(tubes)
    res["tubes"] = {tb["name"]: {"rings": len(tb["rings"]), "around": tb["n"], "part": i} for i, tb in enumerate(tubes)}
    bco = hs.verts(body)
    res["joints"] = fr.joints_report(fr.surface_points(body), van_co, bones)
    res["gaps"] = fr.gaps(bco, bones)
    hidden = hidden_faces(body)
    verts, tris, unweighted = fh.export(body, a["geometry_out"], a.get("preview_out"))
    with open(a["geometry_out"], encoding="utf-8") as f:  # export writes one triangle per (triangulated) face, in order
        geo = json.load(f)
    geo["hidden"] = [i for i, h in enumerate(hidden) if h]
    with open(a["geometry_out"], "w", encoding="utf-8") as f:
        json.dump(geo, f)
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "hidden_tris": len(geo["hidden"])}
    _pdx.write_result(a, res)


if __name__ == "__main__":
    main()
