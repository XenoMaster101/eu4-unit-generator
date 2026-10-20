"""Blender job (runs inside Blender only), Track M spike E-B (SPEC M9, M4 S4): scale plates on the
fitted neck and chest.

Usage: blender --background --python eb_plates.py -- args.json

In: the E-A body (shape, proportions, pose and solved camera of view L), the vanilla skeleton's bone
positions (read from the user's install by E-A, never shipped), view L's labels and inpaint mask, and
the plate settings. Steps:
  1. The body in L's pose (Track M body library, numpy), in the camera's mirrored frame (PDX space
     is left-handed; _ref3d_body mirrors x before projecting, so the scene is built mirrored and the
     outputs are mirrored back).
  2. Coverage per body vertex from L's labels, through the solved camera with visibility: scale
     where seen (evidence "observed", or "inferred" on inpainted pixels); straps and trim over it are
     unknown; unseen vertices take their mirror's value ("symmetry"), the rest the nearest known
     vertex's ("prior"). Only the neck and the barrel in front of the saddle (E-B's region).
  3. Plates in rows round an axis (poll, neck bones, down the breast): rows at a fixed surface pitch
     along the visible side, columns at a fixed spacing round each ring, anchored on the top line
     (aligned columns, as the reference's lamellae). Each plate: a round-bottomed, curved plate with
     real thickness, hung from its top edge, tilted so its lower edge clears the next row. A dark
     backing (the armour's leather) is the body offset under the plates.
  4. Checks: plate-plate and plate-body penetrations (analytic, in each plate's frame / per body
     tube), in mm at the master's scale; body visible between plates (rays down each sample of the
     covered surface).
  5. Renders: view L's camera at the reference's size (Workbench lit, for the plate period; flat ID
     for coverage), and Cycles close-ups of the neck and chest; the evidence close-up (plates
     coloured by evidence).
  6. Saves: the .blend (live Geometry Nodes: plate points -> instances of the plate; the realised
     cache beside it), the plates and backing as OBJ in PDX space, result JSON.

Master scale: withers height = args["withers_m"] (assumed; the reference gives no size).
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Matrix, Vector  # noqa: E402 - Blender module
from mathutils.bvhtree import BVHTree  # noqa: E402 - Blender module
from mathutils.kdtree import KDTree  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "ai" / "jobs"))
import _ref3d_body as RB  # noqa: E402 - numpy only
from _ref3d_scene import (body_in_view, camera, check_camera, decode, in_other_tube, inside, load_rgb,  # noqa: E402
                          look_camera, material, mesh_object, orient_outward, render, set_engine_cycles,
                          set_engine_workbench, smooth_normal, to_pdx, to_scene, vertex_normals, vertex_part,
                          world_sky, write_obj)
EVIDENCE = ("observed", "inferred", "symmetry", "prior")
EV_COLOUR = {"observed": (0.25, 0.75, 0.3), "inferred": (0.95, 0.7, 0.1), "symmetry": (0.25, 0.5, 0.95),
             "prior": (0.9, 0.25, 0.25)}


# ---------------------------------------------------------------- images


# ---------------------------------------------------------------- scene helpers


# ---------------------------------------------------------------- body and coverage


def coverage(a, Vs, Fs, part, Bp, cam, bvh, cam_pos, tubes):
    """Per vertex: (covered 0/1, evidence index, in the region). See the module doc, step 2."""
    classes = a["classes"]
    names = list(classes)
    lab = decode(load_rgb(a["view"]["labels"]), classes)
    inp = load_rgb(a["view"]["inpainted"])[..., 0] > 127
    Hh, Ww = lab.shape
    region = (part == 1) | ((part == 0) & (to_pdx(Vs)[:, 2] < Bp["Saddle_Node"][2]))
    for i in np.nonzero(region)[0]:  # vertices buried in another tube aren't surface
        if in_other_tube(Vs[i], part[i], tubes):
            region[i] = False
    p2, _ = RB.project(to_pdx(Vs), cam)
    scale_i = names.index("scale")
    unknown_cls = {names.index(c) for c in a["occluders"] if c in names}
    n = len(Vs)
    state = np.full(n, -1)  # -1 unknown, 0 not covered, 1 covered
    ev = np.full(n, -1)
    tol = a["label_tol_px"]
    for i in np.nonzero(region)[0]:
        u, v = p2[i]
        if not (0 <= u < Ww and 0 <= v < Hh):
            continue
        d = Vector(Vs[i]) - cam_pos
        L = d.length
        hit = bvh.ray_cast(cam_pos, d.normalized(), L - 1e-3)
        if hit[0] is not None and (hit[0] - Vector(Vs[i])).length > 2e-3:
            continue  # hidden in L
        y0, y1 = int(max(v - tol, 0)), int(min(v + tol + 1, Hh))
        x0, x1 = int(max(u - tol, 0)), int(min(u + tol + 1, Ww))
        win = lab[y0:y1, x0:x1].ravel()
        win = win[win >= 0]
        if not len(win):
            continue  # outside the horse: the fit's outline residual, not evidence
        c = np.bincount(win).argmax()
        if c in unknown_cls:
            continue
        state[i] = 1 if c == scale_i else 0
        ev[i] = 1 if inp[int(v), int(u)] else 0
    seen = state >= 0
    kd = KDTree(int(seen.sum()))
    for k, i in enumerate(np.nonzero(seen)[0]):
        kd.insert(Vector(Vs[i]), int(i))
    kd.balance()
    for i in np.nonzero(region & ~seen)[0]:
        m = Vs[i] * np.array([-1.0, 1.0, 1.0])  # the mirror image across the horse's midplane
        co, j, dist = kd.find(Vector(m))
        if j is not None and dist < a["mirror_tol"]:
            state[i], ev[i] = state[j], (1 if ev[j] == 1 else 2)
    for i in np.nonzero(region & (state < 0))[0]:
        co, j, dist = kd.find(Vector(Vs[i]))
        state[i], ev[i] = state[j], 3
    # majority over the 10 nearest region vertices, twice (single-vertex label noise)
    idx = np.nonzero(region)[0]
    kr = KDTree(len(idx))
    for i in idx:
        kr.insert(Vector(Vs[i]), int(i))
    kr.balance()
    for _ in range(2):
        new = state.copy()
        for i in idx:
            nb = [j for _, j, _ in kr.find_n(Vector(Vs[i]), 10)]
            new[i] = 1 if np.mean(state[nb] == 1) >= 0.5 else 0
        state = new
    state[~region] = 0
    return state, ev, region


# ---------------------------------------------------------------- plate layout

def axis_curve(Bp, Hs, step):
    """The rows' axis in the scene frame: poll, the neck bones, then down the breast."""
    fl, fr = Bp["HorseFrontLeftUp"], Bp["HorseFrontRightUp"]
    sh = 0.5 * (fl + fr)
    nb = Bp["HorseNeckBase"]
    q1 = np.array([0.0, 0.5 * (nb[1] + sh[1]), sh[2] - 0.05 * Hs])
    q2 = np.array([0.0, sh[1] - 0.12 * Hs, sh[2] + 0.15 * Hs])  # E-B v001: nearer the front, rings missed the shoulders
    pts = to_scene(np.array([Bp["HorseHead"], Bp["HorseNeckMiddle"], nb, q1, q2]))
    # Catmull-Rom through the points, resampled by arc length
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        for t in np.linspace(0, 1, 40, endpoint=False):
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    dense.append(pts[-1])
    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1], step)
    C = np.stack([np.interp(s, cum, dense[:, k]) for k in range(3)], 1)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return s, C, T


def ring_frames(T):
    """Per axis point: (anchor direction a, side direction b), both normal to the tangent. a points
    to the top line on the neck and forward on the breast (where the columns are anchored); b to the
    horse's left (the side L sees)."""
    up, fwd = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    A, Bd = [], []
    for t in T:
        w = abs(t @ up)  # 1 where the axis runs vertically (the breast)
        a = (1 - w) * up + w * fwd
        a = a - (a @ t) * t
        a /= np.linalg.norm(a)
        A.append(a)
        Bd.append(np.cross(t, a))
    A, Bd = np.array(A), np.array(Bd)
    left = np.array([-1.0, 0.0, 0.0])  # PDX +x (the horse's left) in the mirrored frame
    if np.mean(Bd @ left) < 0:
        Bd = -Bd
    return A, Bd


def cast_rings(bvh, C, A, Bd, nth, Vs, Fs, vn, fpart, tubes, T_):
    """For each axis point, ``nth`` rays round it: the outermost surface hit (within 1.6x the ring's
    median first-hit distance), with the smooth surface normal there (E-B v001: normals from the
    grid's neighbours jumped where the neck meets the chest and stood plates on edge).
    -> hits (n, nth, 3), normals, valid (n, nth)."""
    n = len(C)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    hits = np.zeros((n, nth, 3))
    nrm = np.zeros((n, nth, 3))
    ok = np.zeros((n, nth), bool)
    for i in range(n):
        dirs = np.cos(th)[:, None] * A[i] + np.sin(th)[:, None] * Bd[i]
        first = []
        rec = []
        for d in dirs:
            o, dv = Vector(C[i]), Vector(d)
            hs = []
            start = o
            for _ in range(8):
                loc, normal, fi, dist = bvh.ray_cast(start, dv, 50.0)
                if loc is None:
                    break
                hs.append((np.array(loc), np.array(normal), (loc - o).length, fi))
                start = loc + dv * 1e-4
            rec.append(hs)
            if hs:
                first.append(hs[0][2])
        if not first:
            continue
        # a vertical axis (the breast) sits inside the barrel: its rings reach back over the shoulders
        rmax = (1.6 + 1.4 * abs(float(T_[i] @ np.array([0.0, 1.0, 0.0])))) * float(np.median(first))
        for j, hs in enumerate(rec):
            hs = [h for h in hs if h[2] <= rmax and h[1] @ dirs[j] > 0.2]  # exits, facing outward
            while hs and in_other_tube(hs[-1][0], fpart[hs[-1][3]], tubes):  # E-B v001: plates inside the barrel
                hs.pop()
            if hs:
                hits[i, j], ok[i, j] = hs[-1][0], True
                nrm[i, j] = smooth_normal(Vs, Fs, vn, hs[-1][0], hs[-1][3])
    return th, hits, nrm, ok


def layout(a, bvh, Bp, Hs, cover_at, mpu, Vs, Fs, vn, fpart, tubes):
    """Plate frames along the rows. -> list of dicts (pos, x, y, z axes of the plate; evidence)."""
    pl = a["plate"]
    pitch, spacing = pl["pitch_m"] / mpu, pl["spacing_m"] / mpu
    s, C, T = axis_curve(Bp, Hs, pitch / 6)
    A, Bd = ring_frames(T)
    nth = 480
    th, hits, nrm, ok = cast_rings(bvh, C, A, Bd, nth, Vs, Fs, vn, fpart, tubes, T)
    cov = np.zeros(ok.shape, bool)
    evi = np.zeros(ok.shape, int)
    for i in range(len(C)):
        for j in np.nonzero(ok[i])[0]:
            c, e = cover_at(hits[i, j])
            cov[i, j], evi[i, j] = c, e
    # rows: surface arc length along the side line (theta = 90 deg: the side L sees)
    js = nth // 4
    side = hits[:, js]
    good = ok[:, js]
    seg = np.where(good[1:] & good[:-1], np.linalg.norm(np.diff(side, axis=0), axis=1), np.diff(s))
    arc = np.r_[0, np.cumsum(seg)]
    rows = np.searchsorted(arc, np.arange(arc[0], arc[-1], pitch))
    plates = []
    for r, i in enumerate(rows):
        if i <= 0 or i >= len(C) - 1:
            continue
        ring = hits[i]
        # closed arc length round the ring, anchored at theta 0 (the top line / the breast's front)
        d = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        d[~(ok[i] & np.roll(ok[i], -1))] = 0.0
        cum = np.r_[0, np.cumsum(d)]
        total = cum[-1]
        if total <= 0:
            continue
        ncol = max(1, int(round(total / spacing)))
        for k in range(ncol):
            target = (k + 0.5 * (r % 2) * pl["stagger"]) * total / ncol
            j = int(np.searchsorted(cum, target, side="right") - 1) % nth
            j2 = (j + 1) % nth
            if not (ok[i, j] and ok[i, j2] and cov[i, j]):
                continue
            f = (target - cum[j]) / max(d[j], 1e-9)
            p = (1 - f) * ring[j] + f * ring[j2]
            # frame: the smooth surface normal; "down" the axis (to the next row), in the tangent plane
            z = (1 - f) * nrm[i, j] + f * nrm[i, j2]
            z /= np.linalg.norm(z)
            dn = hits[i + 1, j] - hits[i - 1, j]
            if not (ok[i + 1, j] and ok[i - 1, j]) or np.linalg.norm(dn) > 6 * (s[1] - s[0]):
                dn = T[i]  # the neighbours jumped (a junction): the axis direction
            dn = dn - (dn @ z) * z
            if np.linalg.norm(dn) < 1e-9:
                continue
            dn /= np.linalg.norm(dn)
            y = -dn  # the plate's top points back up the axis
            x = np.cross(y, z)
            cand = {"pos": p, "x": x, "y": y, "z": z, "ev": int(evi[i, j]), "row": r}
            if not crowded(cand, plates, pitch, spacing, pl["width_frac"] * spacing, pl["height_frac"] * pitch):
                plates.append(cand)
    return plates


def crowded(c, plates, pitch, spacing, w, h):
    """Would plate ``c`` sit on an accepted one? Its top within 0.7 pitch below (or 0.2 h above)
    an accepted plate's top, inside that plate's column (E-B v001: rings perpendicular to the bent
    axis converge on the inside of the neck-chest bend; rows stacked 3-4 deep and crossed). Same-row
    neighbours sit a column apart and pass."""
    if not plates:
        return False
    P = np.array([q["pos"] for q in plates[-400:]])
    near = np.linalg.norm(P - c["pos"], axis=1) < 2 * h
    for q in (plates[-400:][k] for k in np.nonzero(near)[0]):
        d = c["pos"] - q["pos"]
        dx, dd = d @ q["x"], -(d @ q["y"])  # across q's column; down q's length
        if abs(dx) < 0.95 * w and -0.2 * h < dd < 0.7 * pitch:
            return True
    return False


# ---------------------------------------------------------------- the plate

def plate_template(w, h, t, kappa, n_out=28, rings=(1.0, 0.7, 0.4)):
    """A round-bottomed plate: top edge at y = 0, hanging to y = -h; curved across (z = -kappa x^2);
    thickness t (outer face +t). -> (V, F, outline sdf function)."""
    rt, rb = 0.18 * w, 0.45 * w
    cy = -h / 2

    def outline(n):
        pts = []
        corners = [((w / 2 - rt, -rt), rt, 0, 90), ((-w / 2 + rt, -rt), rt, 90, 180),
                   ((-w / 2 + rb, -h + rb), rb, 180, 270), ((w / 2 - rb, -h + rb), rb, 270, 360)]
        for (ox, oy), r, a0, a1 in corners:
            for ang in np.linspace(a0, a1, n // 4, endpoint=False):
                pts.append((ox + r * math.cos(math.radians(ang)), oy + r * math.sin(math.radians(ang))))
        return np.array(pts)

    out = outline(n_out)
    centre = np.array([0.0, cy])
    layers = []
    for k in rings:
        layers.append(centre + (out - centre) * k)
    pts2 = np.vstack(layers + [centre[None]])
    m = len(out)
    V, F = [], []
    for zoff in (0.0, t):
        base = len(V)
        for x, y in pts2:
            V.append((x, y, -kappa * x * x + zoff))
        for li in range(len(rings) - 1):
            for q in range(m):
                a0, a1 = base + li * m + q, base + li * m + (q + 1) % m
                b0, b1 = base + (li + 1) * m + q, base + (li + 1) * m + (q + 1) % m
                F += [(a0, a1, b1), (a0, b1, b0)] if zoff else [(a0, b1, a1), (a0, b0, b1)]
        c = base + len(pts2) - 1
        li = len(rings) - 1
        for q in range(m):
            a0, a1 = base + li * m + q, base + li * m + (q + 1) % m
            F.append((a0, a1, c) if zoff else (a0, c, a1))
    top = len(pts2)
    for q in range(m):  # the rim
        a0, a1 = q, (q + 1) % m
        F += [(a0, a1, top + a1), (a0, top + a1, top + a0)]
    V = np.array(V)

    def sdf(p):
        """Signed distance in the plate's plane to its outline (negative inside); p (k, 2)."""
        r = np.where(p[:, 1] > cy, rt, rb)
        q = np.abs(p - centre) - (np.array([w / 2, h / 2]) - r[:, None])
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - r

    return V, np.array(F), sdf


def plate_matrices(plates, tilt, standoff):
    """4x4 per plate: template -> scene. Tilt about the plate's x lifts its lower edge."""
    c, s = math.cos(-tilt), math.sin(-tilt)
    Rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    Ms = []
    for p in plates:
        R = np.stack([p["x"], p["y"], p["z"]], 1) @ Rt
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p["pos"] + p["z"] * standoff
        Ms.append(M)
    return np.array(Ms)


# ---------------------------------------------------------------- checks

def plate_penetrations(Ms, Vt, t, kappa, sdf, reach):
    """Every pair of plates within ``reach``: the deepest point of one inside the other (template
    space of the other: inside the outline and between its faces). -> (depths per pair > 0, pairs)."""
    n = len(Ms)
    kd = KDTree(n)
    for i, M in enumerate(Ms):
        kd.insert(Vector(M[:3, 3]), i)
    kd.balance()
    P = np.c_[Vt, np.ones(len(Vt))]
    world = np.einsum("nij,kj->nki", Ms, P)[..., :3]  # (n, k, 3)
    inv = np.linalg.inv(Ms)
    depths, pairs = [], []
    for i in range(n):
        for _, j, _ in kd.find_range(Vector(Ms[i][:3, 3]), reach):
            if j == i:
                continue
            q = (np.c_[world[i], np.ones(len(Vt))] @ inv[j].T)[:, :3]
            z0 = -kappa * q[:, 0] ** 2
            dxy = -sdf(q[:, :2])
            over = dxy > 0  # within j's outline
            if not over.any():
                continue
            zr = q[over, 2] - z0[over]  # height over j's inner face
            below, above = zr < 0, zr > t
            if below.any() and above.any():
                # i passes through j: separating them takes the smaller excursion plus j's thickness
                # (E-B v001 measured only the depth inside j's slab, never over t/2 = 1 mm)
                m = float(min(-zr[below].min(), zr[above].max() - t) + t)
            else:
                dz = np.minimum(zr, t - zr)
                m = float(np.minimum(dz, dxy[over]).max())
            if m > 0:
                depths.append(m)
                pairs.append((i, j))
    return np.array(depths), pairs


def body_penetrations(Ms, Vt, tubes):
    """Each plate's deepest point inside any body tube (parity inside test; depth = the distance to
    that tube's surface). tubes: [(bvh, bbox lo, bbox hi)]. -> depth per plate (0 outside)."""
    P = np.c_[Vt, np.ones(len(Vt))]
    out = np.zeros(len(Ms))
    for i, M in enumerate(Ms):
        W = (P @ M.T)[:, :3]
        best = 0.0
        for bvh, lo, hi in tubes:
            for w in W[np.all((W >= lo) & (W <= hi), axis=1)]:
                if inside(bvh, w):
                    best = max(best, bvh.find_nearest(Vector(w))[3])
        out[i] = best
    return out


def gaps(bvh_all, n_body_faces, Vs, normals, covered, rng, n=4000, lift=0.05):
    """Rays straight down onto the covered body surface (random covered vertices): the share whose
    first hit is the body itself (not a plate or the backing)."""
    idx = np.nonzero(covered)[0]
    if not len(idx):
        return None
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    body = 0
    for i in pick:
        o = Vector(Vs[i] + normals[i] * lift)
        loc, normal, fi, dist = bvh_all.ray_cast(o, Vector(-normals[i]), 2 * lift)
        if loc is not None and fi < n_body_faces:
            body += 1
    return body / len(pick)


# ---------------------------------------------------------------- live Geometry Nodes setup

def gn_setup(points_ob, plate_ob):
    """Plate points -> instances of the plate (rotation and scale from the points' attributes);
    the group's inputs scale the plates and add tilt, live."""
    ng = bpy.data.node_groups.new("EB_ScalePlates", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_scale = ng.interface.new_socket("Plate scale", in_out="INPUT", socket_type="NodeSocketFloat")
    s_scale.default_value = 1.0
    s_tilt = ng.interface.new_socket("Extra tilt", in_out="INPUT", socket_type="NodeSocketFloat")
    s_tilt.default_value = 0.0
    s_real = ng.interface.new_socket("Realize", in_out="INPUT", socket_type="NodeSocketBool")
    s_real.default_value = False
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    info = N.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = plate_ob
    rot = N.new("GeometryNodeInputNamedAttribute")
    rot.data_type = "FLOAT_VECTOR"
    rot.inputs["Name"].default_value = "plate_rot"
    e2r = N.new("FunctionNodeEulerToRotation")
    inst = N.new("GeometryNodeInstanceOnPoints")
    tilt = N.new("GeometryNodeRotateInstances")
    comb = N.new("ShaderNodeCombineXYZ")
    e2r_t = N.new("FunctionNodeEulerToRotation")
    sw = N.new("GeometryNodeSwitch")
    sw.input_type = "GEOMETRY"
    real = N.new("GeometryNodeRealizeInstances")
    L = ng.links
    L.new(gi.outputs[0], inst.inputs["Points"])
    L.new(info.outputs["Geometry"], inst.inputs["Instance"])
    L.new(rot.outputs["Attribute"], e2r.inputs[0])
    L.new(e2r.outputs[0], inst.inputs["Rotation"])
    L.new(gi.outputs[1], inst.inputs["Scale"])
    L.new(gi.outputs[2], comb.inputs["X"])
    L.new(inst.outputs["Instances"], tilt.inputs["Instances"])
    L.new(comb.outputs["Vector"], e2r_t.inputs[0])
    L.new(e2r_t.outputs[0], tilt.inputs["Rotation"])
    tilt.inputs["Local Space"].default_value = True
    L.new(tilt.outputs["Instances"], real.inputs["Geometry"])
    L.new(gi.outputs[3], sw.inputs["Switch"])
    L.new(tilt.outputs["Instances"], sw.inputs["False"])
    L.new(real.outputs["Geometry"], sw.inputs["True"])
    L.new(sw.outputs["Output"], go.inputs[0])
    bad = [f"{lk.from_node.name} -> {lk.to_node.name}" for lk in L if not lk.is_valid]
    if bad:
        raise RuntimeError(f"invalid Geometry Nodes links: {bad}")
    mod = points_ob.modifiers.new("EB plates", "NODES")
    mod.node_group = ng
    return mod


# ---------------------------------------------------------------- renders


# ---------------------------------------------------------------- main


def main():
    t0 = time.time()
    a = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.get("seed", 0))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    vn = a["view_name"]
    cam = a["body"]["cameras"][vn]
    W, H = a["view"]["size"]

    Bp, Hs, P, V, F, part = body_in_view(a, sk)
    Vs = to_scene(V)
    Fs = orient_outward(Vs, F[:, ::-1], part)  # the mirror flips the winding
    vpart = vertex_part(F, part, len(V))
    ground = float(V[:, 1].min())
    fz = Bp["HorseFrontLeftUp"][2]
    band = (vpart == 0) & (np.abs(V[:, 2] - (fz + 0.4)) < 0.5)
    withers = float(V[band, 1].max())
    mpu = a["withers_m"] / (withers - ground)  # metres per unit
    res = {"metres_per_unit": mpu, "withers_units": withers - ground}

    bvh = BVHTree.FromPolygons([tuple(v) for v in Vs], [tuple(int(i) for i in f) for f in Fs])
    tubes = {}
    for p_ in np.unique(part):
        s_ = part == p_
        idx = np.unique(Fs[s_])
        remap = -np.ones(len(Vs), int)
        remap[idx] = np.arange(len(idx))
        tubes[int(p_)] = (BVHTree.FromPolygons([tuple(v) for v in Vs[idx]], [tuple(int(i) for i in f) for f in remap[Fs[s_]]]),
                          Vs[idx].min(axis=0), Vs[idx].max(axis=0))
    R, t = RB.camera_matrix(cam)
    cam_pos = Vector(-R.T @ t)
    state, ev, region = coverage(a, Vs, Fs, vpart, Bp, cam, bvh, cam_pos, tubes)
    kd = KDTree(len(Vs))
    for i in np.nonzero(region)[0]:
        kd.insert(Vector(Vs[i]), int(i))
    kd.balance()

    edge = float(np.median(np.linalg.norm(Vs[Fs[:, 0]] - Vs[Fs[:, 1]], axis=1)))

    def cover_at(p):
        """Covered (and the evidence) at a surface point: its nearest region vertex's, when one is
        near (a hit far from the region, e.g. the barrel's rear, isn't covered)."""
        co, j, d = kd.find(Vector(p))
        if j is None or d > 2.5 * edge:
            return False, 3
        return state[j] == 1, int(ev[j])

    pl = a["plate"]
    nrm = vertex_normals(Vs, Fs)
    plates = layout(a, bvh, Bp, Hs, cover_at, mpu, Vs, Fs, nrm, part, tubes)
    pitch, spacing = pl["pitch_m"] / mpu, pl["spacing_m"] / mpu
    w, h = pl["width_frac"] * spacing, pl["height_frac"] * pitch
    th, standoff = pl["thickness_m"] / mpu, pl["standoff_m"] / mpu
    tilt = math.atan(pl["tilt_clear"] * (th + pl["clearance_m"] / mpu) / pitch) + math.radians(pl["tilt_extra_deg"])
    kappa = 1.0 / (2 * pl["curve_radius_m"] / mpu)
    Vt, Ft, sdf = plate_template(w, h, th, kappa)
    Ms = plate_matrices(plates, tilt, standoff)
    res["plates"] = {"count": len(plates), "pitch_units": pitch, "spacing_units": spacing, "width_units": w,
                     "height_units": h, "thickness_units": th, "standoff_units": standoff,
                     "tilt_deg": math.degrees(tilt), "kappa": kappa, "rows": int(max([p["row"] for p in plates] or [0]) + 1),
                     "tris_per_plate": len(Ft), "tris": len(Ft) * len(plates)}
    evc = np.bincount([p["ev"] for p in plates], minlength=4)
    res["evidence_share"] = {EVIDENCE[k]: float(evc[k] / max(len(plates), 1)) for k in range(4)}
    print("plates", res["plates"], res["evidence_share"], flush=True)

    # ---- checks
    lim = a["penetration_limit_m"] / mpu
    t1 = time.time()
    dpp, pairs = plate_penetrations(Ms, Vt, th, kappa, sdf, reach=1.6 * max(h, w))
    dpb = body_penetrations(Ms, Vt[::2], list(tubes.values()))
    # SPEC M6 per plate: lift a plate cutting into the body (more tilt, 5 deg at a time, up to 30)
    extra = np.zeros(len(Ms))
    for i in np.nonzero(dpb > lim)[0]:
        for k in range(1, 7):
            Mi = plate_matrices([plates[i]], tilt + math.radians(5 * k), standoff)
            d = body_penetrations(Mi, Vt[::2], list(tubes.values()))[0]
            if d <= lim:
                Ms[i], dpb[i], extra[i] = Mi[0], d, 5 * k
                break
    res["local_tilt"] = {"plates": int((extra > 0).sum()), "max_extra_deg": float(extra.max()) if len(extra) else 0.0}
    if (extra > 0).any():
        dpp, pairs = plate_penetrations(Ms, Vt, th, kappa, sdf, reach=1.6 * max(h, w))
    # SPEC M6 'regenerate region', at its smallest: a plate still in the body, or the lower plate of a
    # pair still crossing, is removed (counted; its place keeps the backing)
    drop = set(np.nonzero(dpb > lim)[0].tolist())
    for d, (i, j) in zip(dpp, pairs):
        if d > lim and i not in drop and j not in drop:
            drop.add(i if (plates[i]["row"], i) > (plates[j]["row"], j) else j)
    keep = np.array([k not in drop for k in range(len(plates))], bool)
    res["removed"] = {"in_body": int((dpb > lim).sum()), "crossing": int(len(drop) - (dpb > lim).sum()),
                      "of": len(plates)}
    if drop:
        plates = [q for q, k in zip(plates, keep) if k]
        Ms, dpb = Ms[keep], dpb[keep]
        dpp, pairs = plate_penetrations(Ms, Vt, th, kappa, sdf, reach=1.6 * max(h, w))
    res["plates"]["count"] = len(plates)
    res["plates"]["tris"] = len(Ft) * len(plates)
    evc = np.bincount([q["ev"] for q in plates], minlength=4)
    res["evidence_share"] = {EVIDENCE[k]: float(evc[k] / max(len(plates), 1)) for k in range(4)}
    res["penetrations"] = {
        "limit_mm": a["penetration_limit_m"] * 1000,
        "plate_plate_pairs_over": int((dpp > lim).sum()), "plate_plate_max_mm": float(dpp.max() * mpu * 1000) if len(dpp) else 0.0,
        "plate_plate_pairs_touching": int(len(dpp)),
        "plate_body_over": int((dpb > lim).sum()), "plate_body_max_mm": float(dpb.max() * mpu * 1000) if len(dpb) else 0.0,
        "seconds": round(time.time() - t1, 1)}
    print("penetrations", res["penetrations"], flush=True)

    # ---- scene objects
    col = bpy.data.collections.new("EB")
    sc.collection.children.link(col)
    m_coat = material("coat", (0.16, 0.09, 0.05), 0.0, 0.6)
    m_plate = material("plate_steel", (0.78, 0.76, 0.72), 1.0, 0.28)
    m_back = material("backing_leather", (0.05, 0.03, 0.02), 0.0, 0.75)
    zone_f = region[Fs].all(axis=1)
    body_zone = mesh_object("body_zone", Vs, Fs[zone_f], (0, 1, 0), col)
    body_rest = mesh_object("body_other", Vs, Fs[~zone_f], (0, 0, 1), col)
    for ob in (body_zone, body_rest):
        ob.data.materials.append(m_coat)
    cov_f = (state[Fs] == 1).all(axis=1)
    back_V = Vs + nrm * (pl["backing_m"] / mpu)
    backing = mesh_object("backing", back_V, Fs[cov_f], (1, 0, 0), col)
    backing.data.materials.append(m_back)
    plate_ob = mesh_object("plate_template", Vt, Ft, (1, 0, 0), col)
    plate_ob.data.materials.append(m_plate)
    plate_ob.hide_render = True
    plate_ob.hide_viewport = True
    # the realised cache
    P4 = np.c_[Vt, np.ones(len(Vt))]
    allV = np.einsum("nij,kj->nki", Ms, P4)[..., :3].reshape(-1, 3)
    allF = (Ft[None] + (np.arange(len(Ms)) * len(Vt))[:, None, None]).reshape(-1, 3)
    cache = mesh_object("plates_cache", allV, allF, (1, 0, 0), col)
    cache.data.materials.append(m_plate)
    # the live setup: points with each plate's rotation (Euler) and evidence
    pts = mesh_object("plate_points", Ms[:, :3, 3], np.zeros((0, 3), int), (1, 0, 0), col)
    eul = np.array([Matrix(M[:3, :3].tolist()).to_euler("XYZ")[:] for M in Ms]) if len(Ms) else np.zeros((0, 3))
    at = pts.data.attributes.new("plate_rot", "FLOAT_VECTOR", "POINT")
    at.data.foreach_set("vector", eul.astype(np.float32).ravel())
    ae = pts.data.attributes.new("evidence", "INT", "POINT")
    ae.data.foreach_set("value", np.array([p["ev"] for p in plates], np.int32))
    gn_setup(pts, plate_ob)
    pts.data.materials.append(m_plate)
    pts.hide_render = True  # the cache renders; the live points stay editable in the .blend
    dg = bpy.context.evaluated_depsgraph_get()
    n_inst = sum(1 for inst in dg.object_instances if inst.is_instance and inst.parent and inst.parent.name == "plate_points")
    res["gn_instances"] = n_inst

    # ---- gaps: body visible between plates (rays down onto the covered surface)
    allbody = np.vstack([Vs])
    comb_V = np.vstack([allbody, back_V, allV])
    nb, nbk = len(Vs), len(back_V)
    comb_F = np.vstack([Fs, Fs[cov_f] + nb, allF + nb + nbk])
    bvh_all = BVHTree.FromPolygons([tuple(v) for v in comb_V], [tuple(int(i) for i in f) for f in comb_F])
    # the fully covered surface: vertices whose every face carries the backing (E-B v001 sampled the
    # coverage border too, where the backing stops by design)
    cov_v = np.zeros(len(Vs), bool)
    cov_v[np.unique(Fs[cov_f])] = True
    for k in range(3):
        cov_v[Fs[~cov_f][:, k]] = False
    res["gaps"] = {"body_visible": gaps(bvh_all, len(Fs), Vs, nrm, cov_v, rng, lift=8 * standoff)}
    # without the backing: the share of the covered surface not under a plate (slits between columns)
    comb_F2 = np.vstack([Fs, allF + nb + nbk])
    bvh_np = BVHTree.FromPolygons([tuple(v) for v in comb_V], [tuple(int(i) for i in f) for f in comb_F2])
    res["gaps"]["not_under_a_plate"] = gaps(bvh_np, len(Fs), Vs, nrm, cov_v, rng, lift=8 * standoff)
    print("gaps", res["gaps"], flush=True)

    # ---- cameras and renders
    cam_ob = camera("cam_L", R, t, cam["focal"], cam["cx"], cam["cy"], W, H)
    res["camera_check_px"] = check_camera(cam_ob, cam, V[::97], W, H)
    if res["camera_check_px"] > 0.75:
        raise RuntimeError(f"Blender camera doesn't match the solved camera: {res['camera_check_px']:.2f} px")
    files = {}
    set_engine_workbench(sc, "MATERIAL")
    render(sc, cam_ob, out / f"{vn}_lit.png", W, H)
    files["lit"] = str(out / f"{vn}_lit.png")
    set_engine_workbench(sc, "OBJECT", flat=True)
    render(sc, cam_ob, out / f"{vn}_id.png", W, H)
    files["id"] = str(out / f"{vn}_id.png")
    if not a.get("closeups", True):  # a calibration round: L's renders only
        res["files"] = files
        res["seconds"] = round(time.time() - t0, 1)
        with open(a["result_path"], "w", encoding="utf-8") as f:
            json.dump(res, f, default=float)
        return
    # evidence close-up (Workbench, plates coloured by evidence)
    centre = Ms[:, :3, 3].mean(axis=0) if len(Ms) else np.zeros(3)
    rad = float(np.percentile(np.linalg.norm(Ms[:, :3, 3] - centre, axis=1), 95)) if len(Ms) else 1.0
    # the close-up: front-left 3/4, a little above (the body's frame, not L's camera)
    close = look_camera("cam_close", centre, 45, 15, rad / math.tan(math.radians(14)) * 1.15)
    # the detail: an observed plate facing L's camera near the plates' middle, seen obliquely from
    # below its row and to one side, so the lifted lower edges, overlap and thickness show
    fwd_L = R[2]
    score = [(p["ev"] == 0) * 2 + float(-(p["z"] @ fwd_L)) - np.linalg.norm(M[:3, 3] - centre) / max(rad, 1e-9)
             for p, M in zip(plates, Ms)]
    k = int(np.argmax(score)) if len(Ms) else 0
    pk = plates[k] if plates else {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    detail_target = Ms[k, :3, 3] if len(Ms) else centre
    view_dir = 0.75 * pk["z"] + 0.35 * pk["x"] - 0.45 * pk["y"]
    view_dir /= np.linalg.norm(view_dir)
    detail = bpy.data.objects.new("cam_detail", bpy.data.cameras.new("cam_detail"))
    detail.data.lens, detail.data.clip_start = 50.0, 0.005
    sc.collection.objects.link(detail)
    loc = Vector(detail_target + view_dir * 5.0 * h)
    detail.location = loc
    detail.rotation_euler = (Vector(detail_target) - loc).to_track_quat("-Z", "Y").to_euler()
    evm = {}
    ev_obs = []
    for k, name in enumerate(EVIDENCE):
        sel = np.array([p["ev"] == k for p in plates])
        if not sel.any():
            continue
        Vk = np.einsum("nij,kj->nki", Ms[sel], P4)[..., :3].reshape(-1, 3)
        Fk = (Ft[None] + (np.arange(int(sel.sum())) * len(Vt))[:, None, None]).reshape(-1, 3)
        ob = mesh_object(f"plates_{name}", Vk, Fk, EV_COLOUR[name], col)
        ob.hide_render = True
        ev_obs.append(ob)
        evm[name] = int(sel.sum())
    cache.hide_render = True
    for ob in ev_obs:
        ob.hide_render = False
    set_engine_workbench(sc, "OBJECT")
    sc.display.shading.show_cavity = True
    render(sc, close, out / "closeup_evidence.png", 1200, 900)
    files["closeup_evidence"] = str(out / "closeup_evidence.png")
    for ob in ev_obs:
        ob.hide_render = True
    cache.hide_render = False
    # Cycles close-ups
    world_sky(sc)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(-20), math.radians(35))
    sc.collection.objects.link(sun)
    res["cycles_device"] = set_engine_cycles(sc, a.get("samples", 96))
    render(sc, close, out / "closeup.png", 1200, 900)
    render(sc, detail, out / "closeup_detail.png", 1200, 900)
    files["closeup"] = str(out / "closeup.png")
    files["closeup_detail"] = str(out / "closeup_detail.png")
    render(sc, cam_ob, out / f"{vn}_cycles.png", W, H)
    files["cycles_view"] = str(out / f"{vn}_cycles.png")
    res["files"] = files
    res["detail_target"] = list(map(float, to_pdx(detail_target[None])[0]))

    # ---- outputs in PDX space
    write_obj(out / "plates.obj", to_pdx(allV), allF[:, ::-1], "Track M E-B scale plates (generated; PDX mesh space)")
    write_obj(out / "backing.obj", to_pdx(back_V), Fs[cov_f][:, ::-1], "Track M E-B backing (generated; PDX mesh space)")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "eb_master.blend"), compress=True)
    res["blend"] = str(out / "eb_master.blend")
    res["seconds"] = round(time.time() - t0, 1)
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f, default=float)


main()
