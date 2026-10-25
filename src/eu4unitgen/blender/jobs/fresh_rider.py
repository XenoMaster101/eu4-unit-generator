"""Blender job (M23): a TRELLIS.2 rider to the game: a body on the vanilla infantry skeleton (the rider wears the
infantry pdxmesh and plays its horse_* animations, SPEC §7.2) and its lance and shield as rigid parts on the hand
nodes. PDX mesh space (Y up, facing -Z, the figure's right hand at -X), as fresh_horse; most steps are
fresh_horse's ``direct`` path.

  1. import, the seams welded (all pieces kept: the weapons are joined to the hands);
  2. the split, in the model's own units:
       lance: the column standing far to one side over the top of the body; its axis fitted to it, its shaft
       radius measured below the hand; the lance is what lies within 1.6 shaft radii of the axis (1.15 at the
       grip, so the fingers stay; 0.035 of the height round the blade and its tassel);
       shield: on the other side, the outermost points (sure shield) give its plane; the shield is what is
       connected to them within the plate's thickness of that plane;
  3. scale and place: the helmet's top (the highest point near the centre line that isn't the crest's red) at
     16.4, the feet on the ground, centred;
  4. each arm turned about its shoulder so the lance's grip points at Right_hand_node and the shield's at
     Left_hand_node (the rest of the gap slid), blended in over the arm's first quarter; each weapon carried with
     its hand;
  5. the crest (red, over the helmet) smoothed on the closed solid into one plume, like the horse's mane;
  6. the body: a closed solid (flood fill), quadric collapse to ``target_tris``, packed UVs, weights transferred
     from vanilla's body (nearest surface, smoothed), the cloth that hangs behind the legs on the hips (see
     ``cloth_weights``);
  7. each weapon: its own closed solid and collapse, packed UVs, exported twice: in the model's frame (its bake
     target) and in its node's frame (the part: a weapon along -Z from the grip, the rider's "down", levelled
     forward in game; the shield's face outwards along +X, its top along +Y).

args: {"vanilla": base arrays of the infantry body (p, tri, weights, bones), "source": {"path", "merge"},
       "target_tris", "part_tris": {"lance", "shield"}, "close_voxel", "smooth", "influences",
       "geometry_out", "preview_out", "fitted_out", "textured_out", "parts_dir", "result_path"}
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _colour as cc  # noqa: E402
import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
import fresh_horse as fh  # noqa: E402

NECK = 14.1  # vanilla's neck: between top_spine (13.82) and head (14.5); the crest is looked for above it


def pca_axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt


def split(obj):
    """The lance and shield masks (module doc, 2), in the model's own units. -> (lance, shield, info)."""
    co = hs.verts(obj)
    e = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    y0, y1 = co[:, 1].min(), co[:, 1].max()
    H = y1 - y0
    x0 = float(np.median(co[:, 0]))
    info = {"height_raw": round(float(H), 4)}
    lance = np.zeros(len(co), dtype=bool)
    top = co[co[:, 1] > y0 + 0.93 * H]
    far = top[np.abs(top[:, 0] - x0) > 0.15 * H]
    side = 0.0
    if len(far) > 20:
        xl, zl = np.median(far[:, 0]), np.median(far[:, 2])
        col = co[np.hypot(co[:, 0] - xl, co[:, 2] - zl) < 0.04 * H]
        c, vt = pca_axis(col)
        a = vt[0] if vt[0][1] > 0 else -vt[0]
        rel = co - c
        t = rel @ a
        r = np.linalg.norm(rel - t[:, None] * a, axis=1)
        tc = col @ a - c @ a
        tmin, tmax = float(tc.min()), float(tc.max())
        span = tmax - tmin
        low = (r < 0.04 * H) & (t > tmin) & (t < tmin + 0.35 * span)
        rs = float(np.median(r[low])) if low.any() else 0.01 * H
        hand = (r > 2.5 * rs) & (r < 0.06 * H) & (t > tmin + 0.2 * span) & (t < tmin + 0.8 * span)
        tg = float(np.median(t[hand])) if hand.any() else tmin + 0.5 * span
        lim = np.where(np.abs(t - tg) < 0.05 * H, 1.15 * rs, 1.6 * rs)
        lim = np.where(t > tg + 0.12 * H, np.maximum(lim, 0.035 * H), lim)
        lance = (r < lim) & (t >= tmin - 0.01 * H) & (t <= tmax + 0.01 * H)
        side = float(np.sign(xl - x0))
        info["lance"] = {"axis": a.tolist(), "centre": c.tolist(), "shaft_radius": rs, "grip_t": tg,
                         "grip": (c + tg * a).tolist(), "t": [tmin, tmax], "vertices": int(lance.sum())}
    shield = np.zeros(len(co), dtype=bool)
    sgn = -side if side else 1.0
    out = (co[:, 0] - x0) * sgn
    sure = out > 0.75 * out.max()
    if sure.sum() > 50:
        sc, vt = pca_axis(co[sure])
        n = vt[2]
        if (sc - co.mean(0)) @ n < 0:
            n = -n
        d = (co - sc) @ n
        thick = float(np.percentile(np.abs(d[sure]), 95)) * 1.5 + 0.004 * H
        plane = co - d[:, None] * n
        ok = (np.abs(d) < thick) & (np.linalg.norm(plane - sc, axis=1) < 0.35 * H) & ~lance
        shield = fh.flood(ok, sure & ok, e)
        # the hand behind it: body points just behind the plate near its middle
        behind = ~shield & ~lance & (d < 0) & (d > -0.08 * H) & (np.linalg.norm(plane - sc, axis=1) < 0.12 * H)
        grip = co[behind].mean(0) if behind.any() else sc - 0.03 * H * n
        info["shield"] = {"normal": n.tolist(), "centre": sc.tolist(), "thickness": thick, "grip": grip.tolist(),
                          "vertices": int(shield.sum())}
    return lance, shield, info


def place(obj, bones, keep, top=16.4):
    """Scale (the helmet's top at ``top``: vanilla's body is 16.65 with its hat), feet on the ground, centred (x on
    the body's median, z on its torso's). The helmet's top: the highest point near the centre line that isn't the
    crest's red (the narrowest-slice neck took the crest for the head: the figure came out 15.7 tall).
    -> (scale, offset, helmet top before)."""
    co = hs.verts(obj)
    rgb, _ = fh.vertex_colours(obj)
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (rgb[:, 1] - rgb[:, 2]), 2 * rgb[:, 0] - rgb[:, 1] - rgb[:, 2])) + 360) % 360
    chroma = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
    body = co[keep]
    y0 = body[:, 1].min()
    H = body[:, 1].max() - y0
    is_crest, _ = cc.crest_colour(co, rgb, keep, y0 + 0.85 * H)
    red = is_crest(hue, chroma)
    x0 = float(np.median(body[:, 0]))
    head = keep & ~red & (np.abs(co[:, 0] - x0) < 0.06 * H) & (co[:, 1] > y0 + 0.75 * H)
    helmet = float(np.percentile(co[head, 1], 99.5)) if head.sum() > 20 else body[:, 1].max()
    s = top / (helmet - y0)
    torso = body[(body[:, 1] > y0 + 0.5 * H) & (body[:, 1] < y0 + 0.8 * (helmet - y0)) & (np.abs(body[:, 0] - x0) < 0.12 * H)]
    off = np.array([-x0 * s, -y0 * s, -float(np.median(torso[:, 2])) * s])
    hs.set_verts(obj, co * s + off)
    return s, off, helmet


def torso_half(co, bands=((8.6, 9.6), (9.6, 10.6), (10.6, 11.6)), reach=3.0):
    """The body's half-width (x) and half-depth (z) at the waist and chest: the 95th percentile of |x| and of
    |z - median z| over points within ``reach`` of the centre line (the arms are farther out at those heights)."""
    xs, zs = [], []
    for y0, y1 in bands:
        sl = co[(co[:, 1] >= y0) & (co[:, 1] < y1) & (np.abs(co[:, 0]) < reach)]
        if len(sl) > 20:
            xs.append(np.percentile(np.abs(sl[:, 0]), 95))
            zs.append(np.percentile(np.abs(sl[:, 2] - np.median(sl[:, 2])), 95))
    return (float(np.median(xs)) if xs else None), (float(np.median(zs)) if zs else None)


def slim_body(co, keep, vanilla_co, low=0.7, edge=2.0):
    """M23 (the user, test 29: bulkier than vanilla's rider): x and z scaled about the centre line towards vanilla's
    rider body's waist and chest (half-width and half-depth: ``torso_half``), at least ``low`` (the armour, cloak and
    skirt need room); before the arms are turned, so the hands still reach their nodes. -> (kx, kz, measures)."""
    vx, vz = torso_half(vanilla_co)
    ox, oz = torso_half(co[keep])
    kx = float(np.clip(vx / ox, low, 1.0)) if vx and ox else 1.0
    kz = float(np.clip(vz / oz, low, 1.0)) if vz and oz else 1.0
    zc = float(np.median(co[keep & (co[:, 1] > 8.6) & (co[:, 1] < 11.6), 2]))
    SLIM_ZC["zc"] = zc
    SLIM_ZC["edge"] = edge
    # across the torso (|x| under ``edge``) x scales; beyond, the arms shift in whole (their shape and length kept:
    # scaled, the hands fell short of their nodes and the slide stretched the arms)
    ax = np.abs(co[:, 0])
    co[:, 0] = np.sign(co[:, 0]) * np.where(ax < edge, ax * kx, ax - edge * (1 - kx))
    co[:, 2] = zc + (co[:, 2] - zc) * kz
    return kx, kz, {"vanilla": [vx, vz], "ours": [ox, oz]}


def swing_arm(co, movable, grip, target, shoulder, side):
    """``movable`` points on one arm turned about the shoulder so the grip points where the target is (arm length
    kept: the hands were 4.6 and 5.9 from vanilla's, holding the lance in front and the shield at the hip; sliding
    them stretched the arms), blended in over the arm's first quarter; then the rest of the gap (length) slid,
    faded in the same way. -> (angle in degrees, the grip's end position)."""
    sh = np.asarray(shoulder, dtype=np.float64)
    g, tg = np.asarray(grip) - sh, np.asarray(target) - sh
    t = np.clip(((co - sh) @ g) / max(g @ g, 1e-9), 0, 1.2)
    near = np.linalg.norm(co - (sh + np.clip(t, 0, 1)[:, None] * g), axis=1) < 1.3
    m = movable & (co[:, 0] * side > 0) & near & (t > 0.0)
    k = np.cross(g, tg)
    ang = math.atan2(np.linalg.norm(k), g @ tg)
    k = k / max(np.linalg.norm(k), 1e-9)
    w = fh.smoothstep(t, 0.0, 0.25)
    rel = co[m] - sh
    th = (ang * w[m])[:, None]
    kx = np.cross(k[None], rel)
    rot = rel * np.cos(th) + kx * np.sin(th) + k[None] * (rel @ k)[:, None] * (1 - np.cos(th))
    end = sh + (g * np.cos(ang) + np.cross(k, g) * np.sin(ang) + k * (g @ k) * (1 - np.cos(ang)))
    slide = np.asarray(target) - end
    co[m] = sh + rot + w[m, None] * slide
    return math.degrees(ang), slide


SLIM_ZC = {}


def slimmed(p, kx, kz, co):
    """A point measured before ``slim_body``, moved as it moved the body."""
    zc, edge = SLIM_ZC.get("zc", 0.0), SLIM_ZC.get("edge", 2.0)
    ax = abs(p[0])
    x = np.sign(p[0]) * (ax * kx if ax < edge else ax - edge * (1 - kx))
    return np.array([x, p[1], zc + (p[2] - zc) * kz])


def carry(pts, grip, target, shoulder):
    """A weapon carried with its hand: the arm's full turn about the shoulder, then onto the target."""
    sh = np.asarray(shoulder, dtype=np.float64)
    g, tg = np.asarray(grip) - sh, np.asarray(target) - sh
    k = np.cross(g, tg)
    ang = math.atan2(np.linalg.norm(k), g @ tg)
    k = k / max(np.linalg.norm(k), 1e-9)
    rel = pts - sh
    rot = rel * math.cos(ang) + np.cross(k[None], rel) * math.sin(ang) + k[None] * (rel @ k)[:, None] * (1 - math.cos(ang))
    end = sh + (g * math.cos(ang) + np.cross(k, g) * math.sin(ang) + k * (g @ k) * (1 - math.cos(ang)))
    return sh + rot + (np.asarray(target) - end)


def frame_rotation(a_from, b_from, a_to, b_to):
    """The rotation taking the orthonormal pair (a_from, b_from) onto (a_to, b_to)."""
    def basis(a, b):
        a = a / np.linalg.norm(a)
        b = b - (b @ a) * a
        b /= np.linalg.norm(b)
        return np.stack([a, b, np.cross(a, b)], 1)
    return basis(np.asarray(a_to, float), np.asarray(b_to, float)) @ basis(np.asarray(a_from, float), np.asarray(b_from, float)).T


def part_object(src, mask, name):
    """A copy of ``src`` with only ``mask``'s vertices."""
    o = src.copy()
    o.data = src.data.copy()
    o.name = name
    bpy.context.scene.collection.objects.link(o)
    fh.delete_vertices(o, np.flatnonzero(~mask).tolist())
    return o


def plume_line(crest_pts, helmet_pts, helmet_top, head, fin, n=12):
    """M23: the raw plume's centre line, from the finial on the helmet's top: the crest's points in shells by their
    distance from the finial's top, each shell's centroid and side-to-side spread (its half width). Stops where the
    line would come down to the helmet (within its radius + 0.35, under its top + 0.3) or near the cloak (under the
    neck + 1.5): the gap the user asked for. In the head bone's frame (the mesh's axes, the bone at the origin)."""
    hc = np.median(helmet_pts[helmet_pts[:, 1] > helmet_top - 0.3][:, [0, 2]], axis=0)
    hr = float(np.percentile(np.hypot(helmet_pts[:, 0] - hc[0], helmet_pts[:, 2] - hc[1]), 90))
    base = np.array([hc[0], helmet_top - 0.1, hc[1]])
    tip = base + np.array([0.0, fin, 0.0])
    d = np.linalg.norm(crest_pts - tip, axis=1)
    edges = np.linspace(0.0, float(np.percentile(d, 97)), n + 1)
    line, width = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 20:
            continue
        c = crest_pts[m].mean(0)
        near_helmet = np.hypot(c[0] - hc[0], c[2] - hc[1]) < hr + 0.35 and c[1] < helmet_top + 0.3
        if near_helmet or c[1] < NECK + 1.5:
            break
        line.append(c)
        width.append(float(np.percentile(np.abs(crest_pts[m, 0] - c[0]), 80)))
    return {"finial_base": (base - head).round(4).tolist(), "finial_height": fin, "helmet_radius": round(hr, 3),
            "line": (np.array(line) - head).round(4).tolist() if line else [], "half_width": [round(w, 3) for w in width]}


def export_part(o, path_model, path_node, R, origin):
    """The part as bake_colour's low geometry (model frame) and as the mesh part's geometry (node frame)."""
    fh.export(o, path_model, None)
    g = json.loads(open(path_model, encoding="utf-8").read())
    p = (np.array(g["p"]).reshape(-1, 3) - origin) @ R.T
    with open(path_node, "w", encoding="utf-8") as f:
        json.dump({"p": p.ravel().round(5).tolist(), "tri": g["tri"], "uv": g["u0"]}, f)
    return {"tris": len(g["tri"]) // 3, "bounds": [p.min(0).round(3).tolist(), p.max(0).round(3).tolist()]}


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    vp = van["p"]
    vanilla = fh.mesh_object("vanilla", [vp[i:i + 3] for i in range(0, len(vp), 3)],
                             [van["tri"][i:i + 3] for i in range(0, len(van["tri"]), 3)])
    for vi, row in enumerate(van["weights"]):
        for bone, w in row.items():
            g = vanilla.vertex_groups.get(bone) or vanilla.vertex_groups.new(name=bone)
            g.add([vi], w, "REPLACE")
    bones = van["bones"]
    res = {}
    obj = fh.import_model(a["source"]["path"])
    fh.merge_seams(obj, float(a["source"].get("merge", 1e-5)))
    res["imported"] = {"tris": fh.tri_count(obj), "pieces": fh.count_pieces(obj)}
    lance, shield, res["split"] = split(obj)
    s, off, _ = place(obj, bones, ~lance & ~shield)
    res["place"] = {"scale": float(s), "offset": off.tolist()}
    if a.get("fitted_out"):
        fh.write_obj(obj, a["fitted_out"], uv=True)
    co = hs.verts(obj)
    fitted = co.copy()  # placed, not yet slimmed or posed: as the reference shows it (Track P1.2's part crops)
    kx, kz, measured = slim_body(co, ~lance & ~shield, hs.verts(vanilla), float(a.get("slim_low", 0.7)))
    res["slim"] = {"x": round(kx, 3), "z": round(kz, 3), **measured}
    moves, dirs = {}, {}
    li, si = res["split"].get("lance"), res["split"].get("shield")
    if li:
        grip = slimmed(np.array(li["grip"]) * s + off, kx, kz, co)
        side = -1.0 if grip[0] < 0 else 1.0
        node = "Right_hand_node" if side < 0 else "Left_hand_node"
        sh = bones["R_shoulder" if side < 0 else "L_shoulder"]
        before = co.copy()
        ang, slide = swing_arm(co, ~lance & ~shield, grip, bones[node], sh, side)
        co[lance] = carry(before[lance], grip, bones[node], sh)
        ax = carry(np.array([grip, grip + np.array(li["axis"])]), grip, bones[node], sh)
        dirs["lance"] = (ax[1] - ax[0], None)
        moves["lance"] = {"node": node, "grip": grip.round(3).tolist(), "turned_deg": round(ang, 1),
                          "slid": round(float(np.linalg.norm(slide)), 3)}
    if si:
        grip = slimmed(np.array(si["grip"]) * s + off, kx, kz, co)
        side = -1.0 if grip[0] < 0 else 1.0
        node = "Right_hand_node" if side < 0 else "Left_hand_node"
        sh = bones["R_shoulder" if side < 0 else "L_shoulder"]
        before = co.copy()
        ang, slide = swing_arm(co, ~lance & ~shield, grip, bones[node], sh, side)
        co[shield] = carry(before[shield], grip, bones[node], sh)
        ax = carry(np.array([grip, grip + np.array(si["normal"]), grip + np.array([0.0, 1.0, 0.0])]), grip, bones[node], sh)
        dirs["shield"] = (ax[1] - ax[0], ax[2] - ax[0])
        moves["shield"] = {"node": node, "grip": grip.round(3).tolist(), "turned_deg": round(ang, 1),
                           "slid": round(float(np.linalg.norm(slide)), 3)}
    hs.set_verts(obj, co)
    res["arms"] = moves
    # the crest: the crest's own colour over the helmet (the cataphract's is red)
    rgb, lum = fh.vertex_colours(obj)
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (rgb[:, 1] - rgb[:, 2]), 2 * rgb[:, 0] - rgb[:, 1] - rgb[:, 2])) + 360) % 360
    chroma = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1)
    is_crest, crest_h = cc.crest_colour(co, rgb, ~lance & ~shield, NECK + 1.2)
    crest = ~lance & ~shield & (co[:, 1] > NECK + 1.2) & is_crest(hue, chroma)
    res["crest_vertices"] = int(crest.sum())
    res["crest_hue"] = None if crest_h is None else round(crest_h, 1)
    # the helmet's steel: not the crest's colour, above the neck, near the centre line (its axis and radius for the
    # collar cut)
    steel = ~lance & ~shield & ~is_crest(hue, chroma) & (co[:, 1] > NECK) & (np.abs(co[:, 0]) < 1.5) \
        & (co[:, 1] < NECK + 2.6)
    helmet_pts = co[steel] if steel.any() else None
    crest_pts = co[crest]
    # Track P1.2: where each part lies in the placed model, to find it in the reference (ref3d.part_gen.locate)
    def sample(mask, n):
        idx = np.flatnonzero(mask)
        return fitted[idx[:: max(1, len(idx) // n)]].round(4).tolist() if len(idx) else []
    with open(os.path.join(a["parts_dir"], "fitted_points.json"), "w", encoding="utf-8") as f:
        json.dump({"all": sample(np.ones(len(fitted), dtype=bool), 30000), "lance": sample(lance, 3000),
                   "shield": sample(shield, 3000), "crest": sample(crest, 3000),
                   "helmet": sample(steel, 3000)}, f)
    # the parts, with their own raw copies (bake sources)
    parts = {}
    helmet_top = float(np.percentile(helmet_pts[:, 1], 99)) if helmet_pts is not None and len(helmet_pts) > 20 else NECK + 2.0
    if a.get("plume") == "part" and len(crest_pts) > 200 and helmet_pts is not None:
        # the plume as a rigid part on the head (the user, test 31: test 29's arc was right, but smoothed into the body
        # it merged into the helmet, mail and cloak; cut out as it was, its rise went with the gap): its centre line
        # for a clean tube (trellis.plume_part) out of a finial on the helmet's top
        res["plume_line"] = plume_line(crest_pts, helmet_pts, helmet_top, np.array(bones["head"]),
                                       float(a.get("finial_height", 0.6)))
    for key, mask in (("lance", lance), ("shield", shield)):
        if mask.sum() > 50:
            p = part_object(obj, mask, key)
            fh.write_obj(p, os.path.join(a["parts_dir"], f"{key}_textured.obj"), uv=True)
            parts[key] = p
    # the body: the weapons off; the bake source as it is
    fh.delete_vertices(obj, np.flatnonzero(lance | shield).tolist())
    if a.get("textured_out"):
        fh.write_obj(obj, a["textured_out"], uv=True)
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    fh.SOLID.clear()
    # the slot between the legs, from just under the crotch down: closed round both legs, the skirt was one block
    # that tore when the seated legs spread and swung forward; split, each half moves with its leg
    crotch = min(bones["L_thigh"][1], bones["R_thigh"][1]) - 0.5
    gap = float(a.get("leg_gap", 0.3))
    # and the cloth hanging under each arm, outside the torso: draped over the shield arm, it hung from it in one
    # sheet beside the leg and tore when the arm moved (the arms' flips: 64 against vanilla's 2)
    side_x = float(a.get("side_cut", 2.1))
    slit_x, slit = float(a.get("slit_x", 1.85)), float(a.get("slit_half", 0.0))  # slits off: no gain (54 against 57 flips)
    under = min(bones["R_hand"][1], bones["L_hand"][1]) - 1.0
    fh.SOLID.update({"closing": 0.2, "opening": 0.0, "relax": 3,
                     "carve": [[[-gap, -1.0, -20.0], [gap, crotch, 20.0]],
                               # side slits: the skirt's front panels follow the thighs, its back ones the hips
                               *([[[slit_x, -1.0, -slit], [20.0, crotch, slit]], [[-20.0, -1.0, -slit], [-slit_x, crotch, slit]]]
                                 if slit > 0 else []),
                               [[side_x, -1.0, -20.0], [20.0, under, 20.0]], [[-20.0, -1.0, -20.0], [-side_x, under, 20.0]]]})
    res["leg_slot"] = {"half_width": gap, "top": round(crotch, 2)}
    # the collar round the helmet (the user, test 30: the head read as a hooded blob): above the neck, anything farther
    # out than the helmet (its steel near the centre line: the median axis, the 90th percentile radius, + 0.1) is cut
    if a.get("collar_cut", True) and helmet_pts is not None and len(helmet_pts) > 50:
        hc = np.median(helmet_pts[:, [0, 2]], axis=0)
        hr = float(np.percentile(np.hypot(helmet_pts[:, 0] - hc[0], helmet_pts[:, 2] - hc[1]), 90)) + 0.1
        fh.SOLID["carve_out"] = [{"centre": hc.tolist(), "radius": hr, "y0": NECK - 0.2}]
        res["collar_cut"] = {"centre": hc.round(3).tolist(), "radius": round(hr, 3), "above": NECK - 0.2}
    res["side_cut"] = {"beyond_x": side_x, "below_y": round(under, 2)}
    vs = float(a.get("close_voxel", 0.1))
    fh.collapse_to(obj, 10 ** 9, vs)
    fh.keep_largest(obj)
    if len(crest_pts) and a.get("plume", "cut") in ("cut", "part"):  # (a part: its own tube replaces it)
        # the plume cut off the closed solid (smoothed, it was a heavy red arc over the head: the user, test 29); a
        # small crest part (a tuft on the head bone) takes its place in the loadout
        from mathutils.kdtree import KDTree
        tree = KDTree(len(crest_pts))
        for i, q in enumerate(crest_pts):
            tree.insert(q, i)
        tree.balance()
        cut = [i for i, q in enumerate(hs.verts(obj)) if q[1] > NECK + 1.2 and tree.find(q)[2] < 0.4]
        fh.delete_vertices(obj, cut)
        fh.keep_largest(obj)
        res["plume_voxels_cut"] = len(cut)
        res["plume_holes_filled"] = fh.fill_open(obj)
        crest_pts = crest_pts[:0]  # nothing left to keep rigid
    res["body_tris"] = fh.collapse_to(obj, int(a["target_tris"]), None)
    if res.get("plume_line"):
        # the finial on the low body's crown (the raw helmet's top points lean forward: it stood on the dome's front
        # edge); the line moved with it, keeping its shape
        v = hs.verts(obj)
        top = v[v[:, 1] > v[:, 1].max() - 0.2]
        crown = np.array([top[:, 0].mean(), v[:, 1].max() - 0.08, top[:, 2].mean()]) - np.array(bones["head"])
        pl = res["plume_line"]
        shift = crown - np.array(pl["finial_base"])
        pl["finial_base"] = crown.round(4).tolist()
        pl["line"] = (np.array(pl["line"]) + shift).round(4).tolist() if pl["line"] else []
        pl["crown_shift"] = shift.round(3).tolist()
    if a.get("even", 0):  # the collapse's slivers evened out (a few smoothing passes on the low mesh; their flips)
        fh.activate(obj)
        mod = obj.modifiers.new("even", "SMOOTH")
        mod.factor, mod.iterations = 0.4, int(a["even"])
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.data.uv_layers.new(name="UV")
    fh.packed_uv(obj, float(a.get("uv_margin", 0.002)))
    res["uv_area"] = round(fh.uv_area(obj), 4)
    wstats = {}
    fh.transfer_weights(obj, vanilla, a.get("smooth", 12), a.get("influences", 4), None, wstats)
    res["regions"] = rider_weights(obj, bones, a.get("influences", 4), int(a.get("weight_smooth", 12)), plume=crest_pts)
    verts, tris, unweighted = fh.export(obj, a["geometry_out"], a.get("preview_out"))
    res["final"] = {"vertices": verts, "tris": tris, "unweighted_vertices": unweighted, "pieces": fh.count_pieces(obj)}
    # the parts in their nodes' frames
    res["parts"] = {}
    for key, p in parts.items():
        while p.data.uv_layers:
            p.data.uv_layers.remove(p.data.uv_layers[0])
        fh.SOLID.clear()
        fh.SOLID.update({"closing": 0.1, "opening": 0.0, "relax": 1})
        fh.collapse_to(p, int(a.get("part_tris", {}).get(key, 300)), 0.06)
        fh.keep_largest(p)
        p.data.uv_layers.new(name="UV")
        fh.packed_uv(p, 0.004)
        node = moves[key]["node"]
        origin = np.array(bones[node])
        a_dir, b_dir = dirs[key]  # after the arm's turn
        place_at = None
        if key == "lance":  # the axis (tip) to -Z (the rider's "down"), the figure's forward (-Z) to +Y
            R = frame_rotation(a_dir, np.array([0.0, 0.0, -1.0]), [0, 0, -1], [0, 1, 0])
        elif a.get("shield_frame"):
            # strapped to the forearm (the user, test 29: it sat on the rein hand over the horse's neck): its face
            # along the rider's outward side and its top up, in the hand node's frame as the seated pose has it
            # (rider_frames), centred half way to the elbow and out from the forearm
            f = a["shield_frame"]
            R = frame_rotation(a_dir, b_dir, f["out"], f["up"])
            # and lowered (centred on the raised forearm, the kite's top stood over the helmet: its rim showed as a
            # ribbon over the head from behind), a little smaller
            place_at = 0.5 * np.array(f["elbow"]) + float(a.get("shield_out", 0.6)) * np.array(f["out"])                 - float(a.get("shield_down", 1.5)) * np.array(f["up"])
            k = float(a.get("shield_scale", 0.85))
            R = R * k
        else:  # the face outwards to +X, the shield's top to +Y (the hoplon's convention: on the rein hand)
            R = frame_rotation(a_dir, b_dir, [1, 0, 0], [0, 1, 0])
        if place_at is not None:
            centre = hs.verts(p).mean(0)
            origin = centre - np.linalg.solve(R, place_at)  # (centre - origin) @ R.T == place_at (R may carry a scale)
        res["parts"][key] = {"node": node, "R": np.asarray(R).round(6).tolist(), "origin": np.asarray(origin).round(6).tolist(),
                             **({"strapped_at": place_at.round(3).tolist()} if place_at is not None else {}),
                             **export_part(p, os.path.join(a["parts_dir"], f"{key}_low.json"),
                                           os.path.join(a["parts_dir"], f"{key}_part.json"), R, origin)}
    _pdx.write_result(a, res)


ARMS = {"R": ["R_shoulder", "R_arm", "R_elbow", "R_wrist", "R_hand", "Right_hand_node"],
        "L": ["L_shoulder", "L_arm", "L_elbow", "L_wrist", "L_hand", "Left_hand_node"]}
LEGS = {"R": ["R_thigh", "R_knee", "R_foot", "R_toe"], "L": ["L_thigh", "L_knee", "L_foot", "L_toe"]}
SPINE = ["hip", "lower_spine", "mid_spine", "top_spine"]


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def chain_rows(co, bones, chain):
    """Weights along a bone chain: each segment owned by its first bone, inverse distance^4, two nearest
    segments blended (vanilla's joints are about half and half)."""
    pts = [np.array(bones[b]) for b in chain]
    d = np.stack([_seg_dist(co, pts[i], pts[i + 1])[0] for i in range(len(pts) - 1)], 1)
    w = 1.0 / np.maximum(d, 0.05) ** 4
    idx = np.argsort(-w, axis=1)[:, :2]
    rows = []
    for i in range(len(co)):
        a_, b_ = idx[i]
        tot = w[i, a_] + w[i, b_]
        rows.append({chain[a_]: w[i, a_] / tot, chain[b_]: w[i, b_] / tot})
    return rows, d.min(1)


def rider_weights(obj, bones, influences=4, smooth=12, back=0.8, plume=None):
    """M23: region rules over the nearest-surface weights (which only hold where the new mesh hugs vanilla's body;
    the rider's plume, cloak and armour don't: the plume stretched on the shoulders' weights, the cloak mixed the
    spine, hips and thighs, ~100 flips seated against vanilla's 12):
      head: above the neck all on ``head`` (the helmet and plume rigid), faded over 13.9-14.4;
      arms: beyond the shoulders and near an arm's chain, that chain (shoulder to hand), faded in over 1 unit;
      the cloak: behind the back (z over ``back``) and under the shoulders, the spine by height (hip, lower,
      mid and top spine), never a leg or arm;
      legs and the skirt's front halves: under the hips, in front of the back plane, near a leg's chain, that
      chain, faded in under the hips;
    then ``smooth`` passes of neighbour averaging, at most ``influences`` bones. -> counts per region."""
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    co = hs.verts(obj)
    rows = [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in me.vertices]
    counts = {"head": 0, "arms": 0, "cloak": 0, "legs": 0}
    neck0, neck1 = 13.9, 14.4
    hip = np.array(bones["hip"])
    arm_rows = {s_: chain_rows(co, bones, ch) for s_, ch in ARMS.items()}
    leg_rows = {s_: chain_rows(co, bones, ch) for s_, ch in LEGS.items()}
    spine_y = [bones[b][1] for b in SPINE]

    def mix(row, new, f):
        out = {k: w * (1 - f) for k, w in row.items()}
        for k, w in new.items():
            out[k] = out.get(k, 0.0) + w * f
        return out
    if plume is not None and len(plume):  # the plume: all of it rigid with the head (it hangs below the neck)
        from mathutils.kdtree import KDTree
        tree = KDTree(len(plume))
        for i, q in enumerate(plume):
            tree.insert(q, i)
        tree.balance()
        on_plume = np.array([tree.find(q)[2] < 0.6 for q in co])
    else:
        on_plume = np.zeros(len(co), dtype=bool)
    counts["plume"] = int(on_plume.sum())
    for i, p in enumerate(co):
        x, y, z = p
        if on_plume[i]:
            rows[i] = {"head": 1.0}
            continue
        if y > neck0:
            rows[i] = mix(rows[i], {"head": 1.0}, float(fh.smoothstep(np.array(y), neck0, neck1)))
            counts["head"] += 1
            continue
        side = "L" if x > 0 else "R"
        sh = abs(bones[f"{side}_shoulder"][0])
        arow, adist = arm_rows[side]
        if abs(x) > sh and adist[i] < 1.2 and y > hip[1] and z <= back:
            rows[i] = mix(rows[i], arow[i], float(fh.smoothstep(np.array(abs(x)), sh, sh + 1.0)))
            counts["arms"] += 1
            continue
        if z > back and y < bones["top_spine"][1]:  # the cloak behind: the spine by height
            k = int(np.clip(np.searchsorted(spine_y, y) - 1, 0, len(SPINE) - 2))
            f = float(np.clip((y - spine_y[k]) / (spine_y[k + 1] - spine_y[k]), 0, 1))
            rows[i] = {SPINE[k]: 1 - f, SPINE[k + 1]: f}
            counts["cloak"] += 1
            continue
        lrow, ldist = leg_rows[side]
        if y < hip[1] and ldist[i] < 2.2:
            rows[i] = mix(rows[i], lrow[i], float(fh.smoothstep(np.array(hip[1] - y), 0.0, 1.0)))
            counts["legs"] += 1
    nb = [[] for _ in rows]
    for e in me.edges:
        a_, b_ = e.vertices
        nb[a_].append(b_)
        nb[b_].append(a_)
    for _ in range(smooth):
        new = []
        for i, row in enumerate(rows):
            if not nb[i]:
                new.append(row)
                continue
            m = {}
            for j in nb[i]:
                for k, w in rows[j].items():
                    m[k] = m.get(k, 0.0) + w / len(nb[i])
            new.append({k: 0.5 * row.get(k, 0.0) + 0.5 * m.get(k, 0.0) for k in set(row) | set(m)})
        rows = new
    for v, row in zip(me.vertices, rows):
        top = sorted(((k, w) for k, w in row.items() if w >= 0.01), key=lambda kv: -kv[1])[:influences]
        total = sum(w for _, w in top) or 1.0
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        for k, w in top:
            (obj.vertex_groups.get(k) or obj.vertex_groups.new(name=k)).add([v.index], w / total, "REPLACE")
    return counts


def cloth_weights(obj, bones, behind=0.6, influences=4):
    """The cloth hanging below the hips and behind the legs (the cloak's back and the skirt's back: farther than
    ``behind`` behind the nearest leg bone) takes the hips' weights, not a leg's: seated, the legs swing forward
    and the cloak dragged on them tore (the leg weights also pulled it into the horse). -> vertices changed."""
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    legs = {"L_thigh", "R_thigh", "L_knee", "R_knee", "L_foot", "R_foot", "L_toe", "R_toe"}
    hip = np.array(bones["hip"])
    n = 0
    for v in me.vertices:
        row = {names[g.group]: g.weight for g in v.groups if g.weight > 0}
        leg_w = sum(w for k, w in row.items() if k in legs)
        if leg_w <= 0 or v.co.y > hip[1]:
            continue
        side = "L" if v.co.x > 0 else "R"
        knee = np.array(bones[f"{side}_knee"])
        if v.co.z - knee[2] < behind:  # in front of or round the leg: it moves with it
            continue
        keep = {k: w for k, w in row.items() if k not in legs}
        f = float(np.clip((v.co.z - knee[2] - behind) / 0.8, 0.0, 1.0))  # blended in over 0.8 behind
        new = {k: w for k, w in row.items() if k in legs}
        new = {k: w * (1 - f) for k, w in new.items()}
        new.update({k: keep.get(k, 0.0) for k in keep})
        new["hip"] = new.get("hip", 0.0) + leg_w * f
        new = dict(sorted(new.items(), key=lambda kv: -kv[1])[:influences])  # the format's limit
        total = sum(new.values()) or 1.0
        for g in list(v.groups):
            obj.vertex_groups[g.group].remove([v.index])
        for k, w in new.items():
            if w > 1e-3:
                (obj.vertex_groups.get(k) or obj.vertex_groups.new(name=k)).add([v.index], w / total, "REPLACE")
        n += 1
    return {"behind_legs_on_hips": n}


main()
