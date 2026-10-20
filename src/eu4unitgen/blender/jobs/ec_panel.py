"""Blender job (runs inside Blender only), Track M spike E-C (SPEC M9, M4 S5): one caparison side
panel simulated as cloth from a flat pattern on the fitted body.

Usage: blender --background --python ec_panel.py -- args.json

In: the E-A body (shape, proportions, L's pose; both views' solved cameras), the vanilla skeleton's
bone positions (read by E-A from the user's install), both views' labels, inpaint masks, cleaned
images and hem landmarks, the pattern (drop per control column) and the cloth settings. Steps:
  1. The body in L's pose, in the mirrored scene frame (see _ref3d_scene); the collision proxy is
     the body offset by the armour under the cloth (``armour_m``), in metres for the simulation.
  2. The panel: its top seam where L's cloth labels begin on the near side of the barrel (per
     station along the body: the highest visible cloth vertex), front to rear over the stations
     with cloth. The flat pattern is a grid: columns along the seam's arc length, rows down to the
     column's drop (interpolated from the control columns). It starts draped over the proxy
     (each row moved out to the proxy surface, then straight down below it); the pattern is the
     rest shape (shape key), the top row is pinned.
  3. Cloth simulation (fixed settings, frames run in order: deterministic), metres and gravity -Y.
  4. Checks: the hem (bottom row) projected through each view's solved camera against that view's
     hem landmarks (distance to the projected hem line, fraction of body length); cloth vertices
     inside the proxy (parity) over the limit; stretch against the pattern.
  5. Texture: each texel of the panel's pattern UV, where seen in L (or else R) on a clean pixel,
     takes that view's colour (evidence observed); the rest continue their pattern row's observed
     colours (evidence prior). Saved with the evidence map for the seam metric (AI environment).
  6. Renders: both views' solved cameras (Cycles, the textured panel on the body) and a close-up.

Writes the panel (PDX space OBJ, pattern UV), its vertex array (.npy, for the rerun comparison), the
texture and evidence images, the .blend, and the result JSON.
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy  # noqa: E402 - Blender module
import numpy as np
from mathutils import Vector  # noqa: E402 - Blender module
from mathutils.bvhtree import BVHTree  # noqa: E402 - Blender module

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "ai" / "jobs"))
import _ref3d_body as RB  # noqa: E402 - numpy only
from _ref3d_scene import (body_in_view, camera, check_camera, decode, inside, load_rgb, look_camera,  # noqa: E402
                          material, mesh_object, orient_outward, render, set_engine_cycles, to_pdx, to_scene,
                          vertex_normals, vertex_part, world_sky, write_obj)


def visible(bvh, cam_pos, p, tol):
    d = Vector(p) - cam_pos
    hit = bvh.ray_cast(cam_pos, d.normalized(), d.length - tol)
    return hit[0] is None or (hit[0] - Vector(p)).length <= tol


def seam_front(a, Vs, vpart, cam, bvh, cam_pos, n_st, front_z, below_y=None):
    """The chest panel's seam: across the chest at the height where L's chest cloth begins (the
    highest visible cloth vertex per station across; L sees the left half, the seam is fitted
    symmetric: a quadratic in |x|). -> (stations x (PDX), seam heights y (PDX))."""
    names = list(a["classes"])
    lab = decode(load_rgb(a["views"]["L"]["labels"]), a["classes"])
    Hh, Ww = lab.shape
    cloth = {names.index(c) for c in a["panel_classes"] if c in names}
    X = to_pdx(Vs)
    chest = (vpart == 0) & (X[:, 2] < front_z)
    if below_y is not None:  # below the shoulder joint (master v005: the forelock and plume's red set the seam at the head)
        chest &= X[:, 1] < below_y
    p2, _ = RB.project(X, cam)
    xs, ys = [], []
    for i in np.nonzero(chest)[0]:
        u, v = p2[i]
        if not (0 <= u < Ww and 0 <= v < Hh) or not visible(bvh, cam_pos, Vs[i], 0.05):
            continue
        if lab[int(v), int(u)] in cloth:
            xs.append(abs(X[i, 0]))
            ys.append(X[i, 1])
    if len(xs) < 10:
        raise RuntimeError(f"L shows only {len(xs)} cloth vertices on the chest")
    xs, ys = np.array(xs), np.array(ys)
    xmax = 0.95 * np.abs(X[chest, 0]).max()
    st = np.linspace(-xmax, xmax, n_st)
    top = []
    w = 2 * xmax / (n_st - 1)
    for x in np.abs(st):
        m = np.abs(xs - x) < 1.5 * w
        top.append(np.percentile(ys[m], 95) if m.sum() >= 3 else np.nan)
    top = np.array(top)
    ok = ~np.isnan(top)
    c = np.polyfit(np.abs(st[ok]), top[ok], 2) if ok.sum() >= 3 else np.array([0.0, 0.0, np.nanmax(top)])
    return st, np.polyval(c, np.abs(st))


def surface_z(bvh, x, y, z_out):
    """The proxy's front surface at (x, y): a ray from in front of the horse (PDX -z) backward."""
    o = Vector(to_scene(np.array([[x, y, -z_out]]))[0])
    loc, normal, fi, dist = bvh.ray_cast(o, Vector((0.0, 0.0, 1.0)), 2 * z_out)
    if loc is None:
        return None
    return np.array(loc), np.array(normal)


def seam_and_extent(a, Vs, vpart, cam, bvh, cam_pos, mpu, n_st, front_z=None):
    """The panel's top seam (scene frame) from L's cloth labels on the near side of the barrel.
    -> (stations z (PDX), seam heights y (PDX)), or raises when L shows no cloth on the barrel."""
    names = list(a["classes"])
    vL = a["views"]["L"]
    lab = decode(load_rgb(vL["labels"]), a["classes"])
    Hh, Ww = lab.shape
    cloth = {names.index(c) for c in a["panel_classes"] if c in names}
    X = to_pdx(Vs)
    side = (vpart == 0) & (X[:, 0] > 0.25 * np.abs(X[:, 0]).max())  # the barrel's near (left, +X) side
    p2, _ = RB.project(X, cam)
    zs, ys = [], []
    for i in np.nonzero(side)[0]:
        u, v = p2[i]
        if not (0 <= u < Ww and 0 <= v < Hh) or not visible(bvh, cam_pos, Vs[i], 2e-3):
            continue
        if lab[int(v), int(u)] in cloth:
            zs.append(X[i, 2])
            ys.append(X[i, 1])
    if len(zs) < 20:
        raise RuntimeError(f"L shows only {len(zs)} cloth vertices on the barrel's near side")
    zs, ys = np.array(zs), np.array(ys)
    lo, hi = np.percentile(zs, 3), np.percentile(zs, 97)
    # the panel runs past the labelled cloth's ends (E-C: the side hem landmark sat at the panel's front
    # edge); not at the front in the master: the near foreleg stands in front of the cloth (z: the horse faces -Z)
    span = hi - lo
    lo = lo - a["panel"].get("extend_front", a["panel"].get("extend", 0.0)) * span
    hi = hi + a["panel"].get("extend_rear", a["panel"].get("extend", 0.0)) * span
    if front_z is not None:  # the side panel starts behind the shoulder: the chest's cloth is the chest panel's
        lo = max(lo, front_z)  # (master v004: the near panel covered the shoulder the reference armours)
    st = np.linspace(lo, hi, n_st)
    width = (hi - lo) / (n_st - 1)
    top = []
    for z in st:
        m = np.abs(zs - z) < width
        top.append(np.percentile(ys[m], 95) if m.sum() >= 3 else np.nan)
    top = np.array(top)
    ok = ~np.isnan(top)
    # a smooth seam: a quadratic along the body, refitted without the stations off it by over 2 MAD
    # (E-C on pilot 12's body: the highest cloth label per station jumped round the gold trim, the
    # ornaments and the saddle, and the panel's top came out a sawtooth)
    c = np.polyfit(st[ok], top[ok], 2)
    res = top[ok] - np.polyval(c, st[ok])
    mad = np.median(np.abs(res - np.median(res))) + 1e-9
    keep = np.abs(res) <= 2 * 1.4826 * mad
    if keep.sum() >= 5:
        c = np.polyfit(st[ok][keep], top[ok][keep], 2)
    return st, np.polyval(c, st)


def surface_x(bvh, y, z, x_out, sign=1.0):
    """The proxy's surface at (y, z) (scene frame) on the near side (``sign`` 1: PDX +x, the horse's
    left) or the far side (-1): a ray from outside toward the midplane. -> (point, normal) or None."""
    o = Vector(to_scene(np.array([[sign * x_out, y, z]]))[0])
    d = Vector((sign, 0.0, 0.0))  # scene +x = PDX -x: toward the midplane from the near side
    loc, normal, fi, dist = bvh.ray_cast(o, d, 2 * x_out)
    if loc is None:
        return None
    return np.array(loc), np.array(normal)


def build_panel(a, st, top, proxy_bvh, mpu):
    """The draped start (scene frame) and the flat pattern of the panel grid.
    -> (V start (rows+1, cols, 3), pattern (rows+1, cols, 2) in units, column drops in units)."""
    pn = a["panel"]
    cols, rows = pn["cols"], pn["rows"]
    zc = np.linspace(st[0], st[-1], cols)  # stations: z along the side, or x across the chest (front)
    yc = np.interp(zc, st, top)
    front = pn.get("side") == "front"
    ctrl = np.array(pn["drop_m"]) / mpu
    drops = np.interp(np.linspace(0, 1, cols), np.linspace(0, 1, len(ctrl)), ctrl)
    clear = pn["start_clear_m"] / mpu
    x_out = 50.0
    sign = -1.0 if pn.get("side") == "right" else 1.0  # the far side: the near side's seam, mirrored
    V = np.zeros((rows + 1, cols, 3))
    for c in range(cols):
        last = None
        for r in range(rows + 1):
            y = yc[c] - drops[c] * r / rows
            if front:
                hit = surface_z(proxy_bvh, zc[c], y, x_out)
                facing = hit is not None and hit[1][2] < -0.2  # facing forward (-Z)
            else:
                hit = surface_x(proxy_bvh, y, zc[c], x_out, sign)
                facing = hit is not None and sign * hit[1][0] < -0.2  # facing out (scene -x is PDX +x)
            if facing:
                p = hit[0] + hit[1] * clear
                last = p
            elif last is not None:
                p = np.array([last[0], to_scene(np.array([[0, y, 0]]))[0][1], last[2]])
            elif front:  # in front of the chest, never inside the collider (master v005: it started inside and exploded)
                p = to_scene(np.array([[zc[c], y, pn["_front_z"] - clear]]))[0]
            else:
                p = to_scene(np.array([[sign * 1.0, y, zc[c]]]))[0]
            V[r, c] = p
    # the flat pattern, flared: each row as wide as it starts draped over the body (E-C on pilot 12's
    # body: a rectangle the seam's width stretched 10-25% round the barrel's bulge), centred on the
    # middle column; the drop down
    P = np.zeros((rows + 1, cols, 2))
    mid = cols // 2
    for r in range(rows + 1):
        u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(V[r], axis=0), axis=1))]
        P[r, :, 0] = u - u[mid]
    for c in range(cols):
        P[:, c, 1] = -drops[c] * np.arange(rows + 1) / rows
    return V, P, drops


def grid_faces(rows, cols):
    F = []
    for r in range(rows):
        for c in range(cols - 1):
            a0, a1 = r * cols + c, r * cols + c + 1
            b0, b1 = (r + 1) * cols + c, (r + 1) * cols + c + 1
            F += [(a0, b0, b1), (a0, b1, a1)]
    return np.array(F)


def simulate(V0, P, F, cols, proxy_ob, a, mpu, sc):
    """Cloth on the panel grid, in metres. -> final vertices (scene frame, units)."""
    cs = a["cloth"]
    S = mpu  # units -> metres
    ob = mesh_object("panel_sim", V0.reshape(-1, 3) * S, F)
    # the flat pattern as the rest shape (z = 0 plane; only edge lengths matter)
    ob.shape_key_add(name="Basis")
    rest = ob.shape_key_add(name="pattern")
    flat = np.c_[P.reshape(-1, 2) * S, np.zeros(len(V0.reshape(-1, 3)))]
    rest.data.foreach_set("co", flat.astype(np.float32).ravel())
    pin = ob.vertex_groups.new(name="pin")
    pin.add(list(range(cols)), 1.0, "REPLACE")
    mod = ob.modifiers.new("cloth", "CLOTH")
    st = mod.settings
    st.quality = cs["quality"]
    st.mass = cs["panel_mass_kg"] / len(V0.reshape(-1, 3))  # Blender's cloth mass is per vertex (E-C v001: 360 kg)
    st.air_damping = cs["air_damping"]
    st.tension_stiffness = st.compression_stiffness = cs["stiffness"]
    st.shear_stiffness = cs["shear"]
    st.bending_stiffness = cs["bending"]
    st.tension_damping = st.compression_damping = st.shear_damping = cs["damping"]
    st.vertex_group_mass = "pin"
    st.rest_shape_key = rest
    st.use_dynamic_mesh = False
    cl = mod.collision_settings
    cl.collision_quality = cs["collision_quality"]
    cl.distance_min = cs["distance_m"]
    cl.use_self_collision = False
    mod.point_cache.frame_start, mod.point_cache.frame_end = 1, cs["frames"]
    sc.frame_start, sc.frame_end = 1, cs["frames"]
    for f in range(1, cs["frames"] + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    out = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", out)
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    return out.reshape(-1, 3) / S


def hem_errors(a, Vf, rows, cols, vn):
    """Each of the view's hem landmarks: its distance to the projected hem line (the bottom row,
    densified), as a fraction of body length."""
    v = a["views"][vn]
    hem = to_pdx(Vf.reshape(rows + 1, cols, 3)[-1])
    t = np.linspace(0, cols - 1, 8 * cols)
    dense = np.stack([np.interp(t, np.arange(cols), hem[:, k]) for k in range(3)], 1)
    p2, _ = RB.project(dense, v["camera"])
    out = {}
    for name, xy in v["hem_landmarks"].items():
        d = np.linalg.norm(p2 - np.array(xy), axis=1)
        k = int(np.argmin(d))
        out[name] = {"error": float(d[k] / v["body_length"]), "dv_px": float(xy[1] - p2[k][1]), "column": float(t[k])}
    return out


def texture(a, Vf, rows, cols, F, body_bvh_parts, mpu, res_px):
    """Pattern-UV texture: observed texels from L, then R (visible, clean, a panel class there); the
    rest continue their row's observed colours. -> (rgb (H, W, 3) uint8, evidence (H, W): 0 L,
    1 R, 2 prior)."""
    names = list(a["classes"])
    G = Vf.reshape(rows + 1, cols, 3)
    W = res_px
    H = max(8, int(round(res_px * rows / max(cols - 1, 1))))
    uu = np.linspace(0, cols - 1, W)
    vv = np.linspace(0, rows, H)
    rgb = np.zeros((H, W, 3), np.float32)
    ev = np.full((H, W), 2, np.int8)
    cls_t = np.full((H, W), names.index("cloth"), np.int16)  # the texels' classes: the reference's labels
    allV = np.vstack([body_bvh_parts[0], Vf])
    allF = np.vstack([body_bvh_parts[1], F + len(body_bvh_parts[0])])
    bvh = BVHTree.FromPolygons([tuple(p) for p in allV], [tuple(int(i) for i in f) for f in allF])
    for k, vn in enumerate(("L", "R")):
        v = a["views"][vn]
        img = load_rgb(v["image"]).astype(np.float32)
        lab = decode(load_rgb(v["labels"]), a["classes"])
        inp = load_rgb(v["inpainted"])[..., 0] > 127
        ok_cls = {names.index(c) for c in a["panel_classes"] if c in names}
        Hh, Ww = lab.shape
        R, t = RB.camera_matrix(v["camera"])
        cam_pos = Vector(-R.T @ t)
        for yi, r in enumerate(vv):
            r0 = min(int(r), rows - 1)
            fr = r - r0
            for xi, c in enumerate(uu):
                if ev[yi, xi] != 2:
                    continue
                c0 = min(int(c), cols - 2)
                fc = c - c0
                p = ((1 - fr) * ((1 - fc) * G[r0, c0] + fc * G[r0, c0 + 1])
                     + fr * ((1 - fc) * G[r0 + 1, c0] + fc * G[r0 + 1, c0 + 1]))
                (u, w), _ = RB.project(to_pdx(p[None])[0], v["camera"])
                if not (0 <= u < Ww - 1 and 0 <= w < Hh - 1):
                    continue
                if inp[int(w), int(u)] or lab[int(w), int(u)] not in ok_cls:
                    continue
                if not visible(bvh, cam_pos, p, 0.02):
                    continue
                rgb[yi, xi] = img[int(round(w)), int(round(u))]
                cls_t[yi, xi] = lab[int(round(w)), int(round(u))]
                ev[yi, xi] = k
    seen = ev < 2
    base = push_pull(rgb, seen)
    # detail: within each row (the caparison's design runs in bands), the observed texture mirrored
    # across the nearest border, so colour and weave both continue there (E-C: detail from random
    # places along the row put gold next to plain red, seam p95 8.5); beyond the mirrored width, the
    # smooth continuation
    fill = base.copy()
    for yi in range(H):
        obs = np.nonzero(seen[yi])[0]
        if len(obs) < 4:
            continue
        for xi in np.nonzero(~seen[yi])[0]:
            left, right = obs[obs < xi], obs[obs > xi]
            cands = []
            if len(left):
                b = left[-1]
                cands.append((xi - b, b - (xi - b)))
            if len(right):
                b = right[0]
                cands.append((b - xi, b + (b - xi)))
            d, src = min(cands)
            if 0 <= src < W and seen[yi, src]:
                fill[yi, xi] = rgb[yi, src]
                cls_t[yi, xi] = cls_t[yi, src]
    # feather the border: observed texels within ``feather`` of the unseen part blend toward the
    # continuation (E-C: the seam metric sat at the gate, 8.2)
    fpx = 4
    dist = np.full(seen.shape, fpx + 1.0)
    edge = ~seen
    for k in range(1, fpx + 1):
        grown = _maxfilter_bool(edge, k)
        dist[seen & grown & (dist > k)] = k
    wgt = np.clip(dist / (fpx + 1.0), 0.0, 1.0)[..., None]
    rgb = np.where(seen[..., None], wgt * rgb + (1 - wgt) * fill, fill)
    return np.clip(rgb, 0, 255).astype(np.uint8), ev, np.where(cls_t >= 0, cls_t, names.index("cloth"))


def _maxfilter_bool(m, r):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    k = 2 * r + 1
    x = sw(np.pad(m, ((r, r), (0, 0))), k, axis=0).max(axis=-1)
    return sw(np.pad(x, ((0, 0), (r, r))), k, axis=1).max(axis=-1)


def blur(img, w, r):
    """Box blur (radius r) of ``img`` weighted by mask ``w``."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    k = 2 * r + 1
    wi = np.pad(img * w[..., None], ((r, r), (r, r), (0, 0)), mode="edge")
    ww = np.pad(w.astype(np.float32), r, mode="edge")
    num = sw(wi, (k, k), axis=(0, 1)).sum(axis=(-1, -2))
    den = sw(ww, (k, k)).sum(axis=(-1, -2))
    out = num / np.maximum(den, 1e-6)[..., None]
    return np.where(den[..., None] > 0, out, img)


def push_pull(img, w):
    """Fill where ``w`` is False by the push-pull pyramid: a smooth continuation of the known texels
    (no seam at the border by construction)."""
    if w.all() or not w.any():
        return img.copy()
    H, W = w.shape
    h2, w2 = (H + 1) // 2, (W + 1) // 2
    pi = np.zeros((h2 * 2, w2 * 2, 3), np.float64)
    pw = np.zeros((h2 * 2, w2 * 2))
    pi[:H, :W] = img * w[..., None]
    pw[:H, :W] = w
    si = pi.reshape(h2, 2, w2, 2, 3).sum(axis=(1, 3))
    sw_ = pw.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
    low = si / np.maximum(sw_, 1e-9)[..., None]
    low = push_pull(low, sw_ > 0)
    up = np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)[:H, :W]
    return np.where(w[..., None], img, up)


def save_png(path, arr):
    H, W = arr.shape[:2]
    img = bpy.data.images.new(path.stem, W, H, alpha=True)
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :3] = (arr[..., :3] if arr.ndim == 3 else arr[..., None]) / 255.0
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    return img


def main():
    t0 = time.time()
    a = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    a["view_name"] = "L"
    Bp, Hs, P, V, F, part = body_in_view(a, sk)
    Vs = to_scene(V)
    Fs = orient_outward(Vs, F[:, ::-1], part)
    vpart = vertex_part(F, part, len(V))
    ground = float(V[:, 1].min())
    band = (vpart == 0) & (np.abs(V[:, 2] - (Bp["HorseFrontLeftUp"][2] + 0.4)) < 0.5)
    mpu = a["withers_m"] / (float(V[band, 1].max()) - ground)
    res = {"metres_per_unit": mpu}
    bvh = BVHTree.FromPolygons([tuple(v) for v in Vs], [tuple(int(i) for i in f) for f in Fs])
    camL = a["views"]["L"]["camera"]
    R, t = RB.camera_matrix(camL)
    front_z = 0.5 * (Bp["HorseFrontLeftUp"][2] + Bp["HorseFrontRightUp"][2]) + 0.15 * Hs  # the armour's patch line
    if a["panel"].get("side") == "front":
        st, top = seam_front(a, Vs, vpart, camL, bvh, Vector(-R.T @ t), 24, front_z,
                             0.5 * (Bp["HorseFrontLeftUp"][1] + Bp["HorseFrontRightUp"][1]))
        a["_chest_front_z"] = float(V[vpart == 0, 2].min())
    else:
        st, top = seam_and_extent(a, Vs, vpart, camL, bvh, Vector(-R.T @ t), mpu, 24,
                                  front_z if a["panel"].get("behind_shoulder", True) else None)
    res["seam"] = {"z": st.tolist(), "y": top.tolist()}

    # the collision proxy: the body offset by the armour under the cloth
    nrm = vertex_normals(Vs, Fs)
    PV = Vs + nrm * (a["armour_m"] / mpu)
    proxy_bvh = BVHTree.FromPolygons([tuple(v) for v in PV], [tuple(int(i) for i in f) for f in Fs])
    proxy = mesh_object("proxy", PV * mpu, Fs)
    cm = proxy.modifiers.new("collision", "COLLISION")
    proxy.collision.thickness_outer = a["cloth"]["distance_m"]
    proxy.collision.cloth_friction = a["cloth"]["friction"]
    sc.gravity = (0.0, -9.81, 0.0)
    sc.use_gravity = True

    pn = a["panel"]
    rows, cols = pn["rows"], pn["cols"]
    a["panel"]["_front_z"] = a.get("_chest_front_z", 0.0) - a["armour_m"] / mpu
    V0, Pat, drops = build_panel(a, st, top, proxy_bvh, mpu)
    Fg = grid_faces(rows, cols)  # rows: the intervals; (rows + 1) x cols vertices
    Vf = simulate(V0, Pat, Fg, cols, proxy, a, mpu, sc)
    np.save(out / "panel_vertices.npy", Vf)
    (out / "panel_vertices.json").write_text(json.dumps(Vf.round(7).tolist()), encoding="utf-8")  # the app has no numpy
    res["panel"] = {"rows": rows, "cols": cols, "drops_m": (drops * mpu).tolist(), "verts": len(Vf), "tris": len(Fg),
                    "width_m": float((Pat[0, -1, 0] - Pat[0, 0, 0]) * mpu)}

    # ---- checks
    lim = a["penetration_limit_m"] / mpu
    proxy_tri = BVHTree.FromPolygons([tuple(v) for v in PV], [tuple(int(i) for i in f) for f in Fs])
    lo, hi = PV.min(axis=0), PV.max(axis=0)
    depths = []
    for p in Vf:
        if np.all(p >= lo) and np.all(p <= hi) and inside(proxy_tri, p):
            depths.append(proxy_tri.find_nearest(Vector(p))[3])
    depths = np.array(depths) if depths else np.zeros(0)
    res["intersections"] = {"over_limit": int((depths > lim).sum()), "max_mm": float(depths.max() * mpu * 1000) if len(depths) else 0.0,
                            "limit_mm": a["penetration_limit_m"] * 1000}
    G = Vf.reshape(rows + 1, cols, 3)
    Pg = Pat
    e_now = np.r_[np.linalg.norm(np.diff(G, axis=0), axis=2).ravel(), np.linalg.norm(np.diff(G, axis=1), axis=2).ravel()]
    e_rest = np.r_[np.linalg.norm(np.diff(Pg, axis=0), axis=2).ravel(), np.linalg.norm(np.diff(Pg, axis=1), axis=2).ravel()]
    ratio = e_now / np.maximum(e_rest, 1e-9)
    nv = rows * cols  # where the over-stretched edges are (vertical edges first, then horizontal)
    rv = ratio[:nv].reshape(rows, cols)
    rh = ratio[nv:].reshape(rows + 1, cols - 1)
    res["stretch_where"] = {"vertical_by_row": (rv > 1.25).sum(axis=1).tolist(), "vertical_by_col": (rv > 1.25).sum(axis=0).tolist(),
                            "horizontal_by_row": (rh > 1.25).sum(axis=1).tolist(), "horizontal_by_col": (rh > 1.25).sum(axis=0).tolist()}
    res["stretch"] = {"max": float(ratio.max()), "p99": float(np.percentile(ratio, 99)), "p95": float(np.percentile(ratio, 95)),
                      "min": float(ratio.min())}
    res["hem"] = {vn: hem_errors(a, Vf, rows, cols, vn) for vn in a["views"]}
    print("panel", res["panel"], res["intersections"], res["stretch"], flush=True)
    print("hem", json.dumps(res["hem"]), flush=True)

    if not a.get("finish", True):
        res["seconds"] = round(time.time() - t0, 1)
        with open(a["result_path"], "w", encoding="utf-8") as f:
            json.dump(res, f, default=float)
        return

    # ---- texture with evidence
    if a.get("texture_from"):  # the far panel: the near panel's texture, by symmetry
        tex = load_rgb(a["texture_from"])
        ev = np.full(tex.shape[:2], 3, np.int8)
    else:
        tex, ev, cls_t = texture(a, Vf, rows, cols, Fg, (Vs, Fs), mpu, a["texture_px"])
        lab_rgb = np.array([a["classes"][c] for c in a["classes"]], np.uint8)[cls_t]
        save_png(out / "panel_labels.png", lab_rgb)  # the master's class map of the panel
    save_png(out / "panel_texture.png", tex)
    evc = np.zeros(ev.shape + (3,), np.uint8)
    evc[ev == 0] = (60, 200, 80)
    evc[ev == 1] = (70, 120, 240)
    evc[ev == 2] = (230, 60, 60)
    evc[ev == 3] = (150, 90, 230)  # symmetry (the far panel)
    save_png(out / "panel_evidence.png", evc)
    res["texture"] = {"path": str(out / "panel_texture.png"), "evidence": str(out / "panel_evidence.png"),
                      "share": {k: float((ev == i).mean()) for i, k in enumerate(("observed_L", "observed_R", "prior", "symmetry"))}}

    # ---- scene for renders (units, not metres)
    bpy.data.objects.remove(proxy)
    body = mesh_object("body", Vs, Fs)
    body.data.materials.append(material("coat", (0.16, 0.09, 0.05), 0.0, 0.6))
    armour = mesh_object("armour_proxy", PV, Fs)
    armour.data.materials.append(material("armour", (0.55, 0.55, 0.58), 0.9, 0.35))
    panel = mesh_object("panel", Vf, Fg)
    uv = panel.data.uv_layers.new(name="pattern")
    uvs = np.c_[np.tile(np.linspace(0, 1, cols), rows + 1), 1 - np.repeat(np.linspace(0, 1, rows + 1), cols)]
    loop_uv = uvs[np.array([lp.vertex_index for lp in panel.data.loops])]
    uv.data.foreach_set("uv", loop_uv.astype(np.float32).ravel())
    m = material("caparison", (0.5, 0.1, 0.1), 0.0, 0.8)
    nt = m.node_tree
    texn = nt.nodes.new("ShaderNodeTexImage")
    texn.image = bpy.data.images.load(str(out / "panel_texture.png"))
    nt.links.new(texn.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    m.use_backface_culling = False
    panel.data.materials.append(m)
    for poly in panel.data.polygons:
        poly.use_smooth = True
    sol = panel.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness = 0.004 / mpu
    world_sky(sc)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(-20), math.radians(35))
    sc.collection.objects.link(sun)
    res["cycles_device"] = set_engine_cycles(sc, a.get("samples", 64))
    files = {}
    for vn, v in a["views"].items():
        W, H = v["size"]
        R, t = RB.camera_matrix(v["camera"])
        cob = camera(f"cam_{vn}", R, t, v["camera"]["focal"], v["camera"]["cx"], v["camera"]["cy"], W, H)
        res[f"camera_check_px_{vn}"] = check_camera(cob, v["camera"], V[::97], W, H)
        render(sc, cob, out / f"{vn}_cycles.png", W, H)
        files[f"{vn}_cycles"] = str(out / f"{vn}_cycles.png")
    centre = Vf.mean(axis=0)
    rad = float(np.percentile(np.linalg.norm(Vf - centre, axis=1), 95))
    close = look_camera("cam_close", centre, -20, 12, rad / math.tan(math.radians(14)) * 1.1)
    render(sc, close, out / "closeup.png", 1200, 900)
    files["closeup"] = str(out / "closeup.png")
    res["files"] = files
    write_obj(out / "panel.obj", to_pdx(Vf), Fg[:, ::-1], "Track M E-C caparison side panel (simulated; PDX mesh space)")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "ec_panel.blend"), compress=True)
    res["seconds"] = round(time.time() - t0, 1)
    with open(a["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f, default=float)


if __name__ == "__main__":  # the master build imports this module
    main()
