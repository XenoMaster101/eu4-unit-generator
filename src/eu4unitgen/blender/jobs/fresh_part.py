"""Blender job (Track P1.2): a part's own TRELLIS.2 model (made from its crop of the reference) into a game part.

The rule (the user, 2026-10-24): parts are never placeholder primitives with flat colour; each matches the
reference's shape and carries its texture and normal map. TRELLIS.2 picks its own facing, so the model is first
turned so that it looks from the front as the crop shows it:
  1. import, seams welded, the largest piece; centred, scaled so its largest extent is ``size``;
  2. the front frame (x the crop's right, y its up, z towards the viewer):
     a flat part (a shield: its thinnest principal extent under 0.35 of its longest) faces along its thinnest axis,
     the side whose colours are nearer the crop's (the painted face, not the back), its longest axis up, the end
     that is wider where the crop is wider (a kite's top);
     anything else: every axis direction and the principal axes' (both signs), each with in-plane turns every 15
     degrees, the vertices splatted orthographically (the nearest per pixel keeps its colour) and scored against
     the crop: silhouette IoU (both framed by their boxes, aspect kept) less the mean colour distance (0-1) where
     both show the part;
  3. the bake source written as it now stands (``textured_out``, preview axes, its own UVs);
  4. a closed solid and quadric collapse to ``target_tris``, the largest piece, packed UVs; ``geometry_out``.

args: {"source": .glb, "crop": the RGBA crop .png (its alpha the part), "size", "target_tris", "close_voxel",
       "geometry_out", "textured_out", "result_path"}
Result: {"view": {"axis", "turn_deg", "iou", "colour_dist"}, "tris", "bounds", "candidates"}.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402 - Blender module
import numpy as np  # noqa: E402 - Blender's numpy

import _horse_shape as hs  # noqa: E402
import _pdx  # noqa: E402
from _part_view import flat_view  # noqa: E402
import fresh_horse as fh  # noqa: E402

RES = 96


def crop_target(path):
    """The crop's mask and colours framed by the part's box in a RES square (aspect kept, centred)."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[::-1]  # rows top down
    m = px[..., 3] > 0.5
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    out_m = np.zeros((RES, RES), dtype=bool)
    out_c = np.zeros((RES, RES, 3))
    gy, gx = np.mgrid[0:RES, 0:RES]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx = np.clip((cx + (gx + 0.5 - RES / 2) / RES * side).astype(int), 0, w - 1)
    sy = np.clip((cy + (gy + 0.5 - RES / 2) / RES * side).astype(int), 0, h - 1)
    out_m = m[sy, sx]
    out_c = px[sy, sx, :3] * 255.0
    return out_m, out_c


def splat(co, rgb, R):
    """``co`` turned by ``R`` (rows: the view's right, up, towards the viewer), splatted in a RES square framed by
    its box (aspect kept): the mask and, per pixel, the colour of the nearest vertex."""
    q = co @ R.T
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    side = max(hi - lo)
    c = (lo + hi) / 2
    ix = np.clip(((q[:, 0] - c[0]) / side * RES + RES / 2).astype(int), 0, RES - 1)
    iy = np.clip(((c[1] - q[:, 1]) / side * RES + RES / 2).astype(int), 0, RES - 1)
    order = np.argsort(q[:, 2])  # far first: the nearest written last
    m = np.zeros((RES, RES), dtype=bool)
    col = np.zeros((RES, RES, 3))
    m[iy, ix] = True
    col[iy[order], ix[order]] = rgb[order]
    # close the splat's pinholes (one pixel)
    grown = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return grown, col


def rotation(fwd, up):
    """Rows: right, up, towards the viewer (= -fwd, the view direction's opposite)."""
    z = -np.asarray(fwd, dtype=np.float64)
    z /= np.linalg.norm(z)
    u = np.asarray(up, dtype=np.float64) - (np.asarray(up) @ z) * z
    if np.linalg.norm(u) < 1e-6:
        return None
    u /= np.linalg.norm(u)
    return np.stack([np.cross(u, z), u, z])


def best_view(co, rgb, mask, colours):
    _, _, vt = np.linalg.svd(co - co.mean(0), full_matrices=False)
    dirs = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)] + [vt[2], -vt[2]]
    cands = []
    for k, d in enumerate(dirs):
        a = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        for turn in range(0, 360, 15):
            t = math.radians(turn)
            R = rotation(d, a * math.cos(t) + b * math.sin(t))
            if R is None:
                continue
            m, c = splat(co, rgb, R)
            inter = m & mask
            iou = inter.sum() / max((m | mask).sum(), 1)
            cd = float(np.abs(c[inter] - colours[inter]).mean()) if inter.any() else 255.0
            cands.append((iou - cd / 255.0, iou, cd, k, turn, R))
    cands.sort(key=lambda x: -x[0])
    return cands


def main():
    a = _pdx.args()
    _pdx.clear_scene()
    obj = fh.import_model(a["source"])
    fh.merge_seams(obj, 1e-5)
    fh.keep_largest(obj)
    co = hs.verts(obj)
    lo, hi = co.min(0), co.max(0)
    co = (co - (lo + hi) / 2) * (float(a["size"]) / max(hi - lo))
    rgb, _ = fh.vertex_colours(obj)
    mask, colours = crop_target(a["crop"])
    _, sv, vt = np.linalg.svd(co - co.mean(0), full_matrices=False)
    if sv[2] / sv[0] < float(a.get("flat_below", 0.35)):
        R, res_view = flat_view(co, rgb, mask, colours, vt, sv)
        res = {"view": res_view}
    else:
        cands = best_view(co, rgb, mask, colours)
        score, iou, cd, k, turn, R = cands[0]
        res = {"view": {"flat": False, "axis": int(k), "turn_deg": int(turn), "iou": round(float(iou), 3),
                        "colour_dist": round(cd, 1)},
               "candidates": [[round(float(c[1]), 3), round(c[2], 1), c[3], c[4]] for c in cands[:5]]}
    hs.set_verts(obj, co @ R.T)
    fh.write_obj(obj, a["textured_out"], uv=True)
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    fh.SOLID.clear()
    fh.SOLID.update({"closing": float(a.get("closing", 0.1)), "opening": 0.0, "relax": 1})
    fh.collapse_to(obj, int(a.get("target_tris", 400)), float(a.get("close_voxel", 0.06)))
    fh.keep_largest(obj)
    obj.data.uv_layers.new(name="UV")
    fh.packed_uv(obj, 0.004)
    _, tris, _ = fh.export(obj, a["geometry_out"], None)
    v = hs.verts(obj)
    res.update({"tris": tris, "bounds": [v.min(0).round(3).tolist(), v.max(0).round(3).tolist()]})
    _pdx.write_result(a, res)


main()
