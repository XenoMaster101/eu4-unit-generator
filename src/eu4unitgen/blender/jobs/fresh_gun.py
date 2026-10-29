"""Blender job (Track A): a TRELLIS.2 gun to the game: a rigid model on the base game's gun skeleton
(``artillery.mesh``: Root -> Cannon -> wheels), so the game's own artillery animations move it (the barrel's recoil
on Cannon, the wheels rolling on wheels). PDX mesh space (Y up), as fresh_horse.

  1. import, seams welded, the small loose pieces dropped (the wheels are often pieces of their own: kept);
  2. the frame: the ground at the lowest point; the long axis from the points seen from above; the muzzle the
     end whose outer points stand higher (the barrel rides over the trail, which ends on the ground);
  3. the wheels: the widest points sideways (beyond 0.55 of the half-width) and under the barrel's height; their
     axle at the middle of their extent along the gun and up, their radius half their height;
  4. one uniform scale so the wheels' radius is the base game's (the wheels bone's height: they touch the ground),
     the axle on the wheels bone, the barrel towards -x as the base game's (its cannon_muzzle locator is at the
     low-x end);
  5. the bake source written as it now stands (``textured_out``), then a voxel remesh (``voxel``, every piece but
     specks kept) and quadric collapse to ``target_tris``,
     packed UVs; every vertex on one bone (rigid, as the base game's): within the wheels' radius of the axle and
     outside their inner face -> wheels, the rest -> Cannon; ``geometry_out``.

args: {"source": .glb, "vanilla": {"bones": {name: [x, y, z]}, "muzzle": [x, y, z], "extent": [[lo], [hi]]},
       "target_tris", "uv_margin", "geometry_out", "preview_out", "textured_out", "result_path"}
Result: {"frame", "wheels", "scale", "muzzle": {"ours", "vanilla", "off"}, "final": {"tris", "bounds"}, "shares"}.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402 - Blender module
import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
import fresh_horse as fh  # noqa: E402

PIECE_MIN = 0.004  # loose pieces with fewer than this share of the vertices are dropped (specks, not wheels)
WHEEL_SIDE = 0.55  # the wheels: beyond this share of the half-width sideways


def drop_specks(obj, share=PIECE_MIN):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    seen, pieces = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, piece = [v], []
        seen.add(v.index)
        while stack:
            x = stack.pop()
            piece.append(x)
            for e in x.link_edges:
                o = e.other_vert(x)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        pieces.append(piece)
    n = len(bm.verts)
    drop = [v for p in pieces if len(p) < share * n for v in p]
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {"pieces": len(pieces), "dropped": sum(1 for p in pieces if len(p) < share * n)}


def frame(co):
    """(origin on the ground, the long axis towards the muzzle, the side axis), all horizontal; and what told."""
    y0 = float(co[:, 1].min())
    xz = co[:, [0, 2]]
    c = xz.mean(0)
    _, _, vt = np.linalg.svd(xz - c, full_matrices=False)
    u = vt[0]
    t = (xz - c) @ u
    lo, hi = np.percentile(t, 12), np.percentile(t, 88)
    h_lo, h_hi = float(np.median(co[t <= lo, 1])), float(np.median(co[t >= hi, 1]))
    if h_lo > h_hi:  # the muzzle end stands higher than the trail's end
        u = -u
    U = np.array([u[0], 0.0, u[1]])
    Y = np.array([0.0, 1.0, 0.0])
    S = np.cross(-U, Y)  # right-handed with (x = -U, y = Y): x × y = z
    return np.array([c[0], y0, c[1]]), U, S, {"end_heights": [round(h_lo - y0, 4), round(h_hi - y0, 4)]}


def wheels(co, o, U, S):
    rel = co - o
    t, y, s = rel @ U, rel[:, 1], rel @ S
    half = float(np.abs(s).max())
    top = float(y.max())
    w = (np.abs(s) > WHEEL_SIDE * half) & (y < 0.8 * top)
    if w.sum() < 50:
        raise SystemExit("no wheels found (nothing wide at the sides, under the barrel)")
    tc = 0.5 * (np.percentile(t[w], 2) + np.percentile(t[w], 98))
    r = 0.5 * (np.percentile(y[w], 98) - max(0.0, float(np.percentile(y[w], 2))))
    yc = max(r, 0.5 * (np.percentile(y[w], 2) + np.percentile(y[w], 98)))
    inner = float(np.percentile(np.abs(s[w]), 5))
    return {"t": float(tc), "y": float(yc), "radius": float(r), "inner": inner, "half_width": half,
            "points": int(w.sum())}


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    van = a["vanilla"]
    bones = van["bones"]
    axle = np.array(bones["wheels"], dtype=np.float64)
    res = {}
    obj = fh.import_model(a["source"])
    fh.merge_seams(obj, 1e-5)
    res["imported"] = {"tris": fh.tri_count(obj), **drop_specks(obj)}
    co = hs.verts(obj)
    o, U, S, res["frame"] = frame(co)
    wh = wheels(co, o, U, S)
    k = float(axle[1]) / wh["radius"]  # the wheels' radius to the base game's: they touch the ground
    rel = co - o
    t, y, s = rel @ U, rel[:, 1], rel @ S
    # the barrel along -x (the base game's muzzle at its low-x end), the axle on the wheels bone
    new = np.stack([axle[0] - k * (t - wh["t"]), axle[1] + k * (y - wh["y"]), k * s], -1)
    # no wider than the base game's gun (the bombard's track was 1.85 times it: two crews' worth at map zoom); only
    # sideways, so the wheels stay round (they turn in x-y)
    half = float(np.abs(van["extent"][0][2])) if van.get("extent") else None
    sz = min(1.0, half / float(np.abs(new[:, 2]).max())) if half else 1.0
    new[:, 2] *= sz
    res["width_squeeze"] = round(sz, 3)
    hs.set_verts(obj, new)
    res["scale"] = round(k, 4)
    res["wheels"] = {**{kk: round(v, 4) if isinstance(v, float) else v for kk, v in wh.items()},
                     "radius_game": round(k * wh["radius"], 4)}
    tip = new[np.argmin(new[:, 0])]
    muzzle = np.array(van["muzzle"])
    res["muzzle"] = {"ours": tip.round(3).tolist(), "vanilla": muzzle.round(3).tolist(),
                     "off": round(float(np.linalg.norm(tip - muzzle)), 3)}
    res["length"] = {"ours": round(float(new[:, 0].max() - new[:, 0].min()), 3),
                     "vanilla": round(float(van["extent"][1][0] - van["extent"][0][0]), 3)}
    if a.get("textured_out"):
        fh.write_obj(obj, a["textured_out"], uv=True)
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    # TRELLIS.2's surface is open and non-manifold: the collapse stalls on it (18,012 for 1,500 on the bombard).
    # A voxel remesh first, as for horses, but every piece kept (the wheels may stand apart from the carriage) and
    # small voxels with no closing, so the spokes stay open
    fh.activate(obj)
    mod = obj.modifiers.new("close", "REMESH")
    mod.mode, mod.voxel_size, mod.adaptivity = "VOXEL", float(a.get("voxel", 0.03)), 0.0  # the trail's planks: 0.045 tore them
    bpy.ops.object.modifier_apply(modifier=mod.name)
    res["remeshed"] = {"tris": fh.tri_count(obj), **drop_specks(obj, 0.01)}
    mod = obj.modifiers.new("relax", "SMOOTH")
    mod.factor, mod.iterations = 0.5, 2
    bpy.ops.object.modifier_apply(modifier=mod.name)
    res["tris"] = fh.collapse_to(obj, int(a.get("target_tris", 1500)), None)
    obj.data.uv_layers.new(name="UV")
    fh.packed_uv(obj, float(a.get("uv_margin", 0.003)))
    co = hs.verts(obj)
    rad = k * wh["radius"] * 1.06
    inner = k * sz * wh["inner"] * 0.92
    on_wheel = (np.hypot(co[:, 0] - axle[0], co[:, 1] - axle[1]) < rad) & (np.abs(co[:, 2]) > inner)
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    gw, gc = obj.vertex_groups.new(name="wheels"), obj.vertex_groups.new(name="Cannon")
    gw.add(np.flatnonzero(on_wheel).tolist(), 1.0, "REPLACE")
    gc.add(np.flatnonzero(~on_wheel).tolist(), 1.0, "REPLACE")
    res["shares"] = {"wheels": round(float(on_wheel.mean()), 3), "Cannon": round(float(1 - on_wheel.mean()), 3)}
    nv, nt, unweighted = fh.export(obj, a["geometry_out"], a.get("preview_out"))
    co = hs.verts(obj)
    res["final"] = {"vertices": nv, "tris": nt, "unweighted": unweighted,
                    "bounds": [co.min(0).round(3).tolist(), co.max(0).round(3).tolist()]}
    _pdx.write_result(a, res)


main()
