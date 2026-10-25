"""A flat part's front frame (Track P1.2; numpy only: fresh_part and tests/blender_rules_check.py import it)."""
import numpy as np


def crop_profile(mask):
    """Is the crop's part wider in its top half than its bottom half? (rows' widths, framed)"""
    rows = mask.sum(1)
    ys = np.flatnonzero(rows)
    mid = (ys.min() + ys.max()) // 2
    return float(rows[ys.min():mid].mean()) > float(rows[mid:ys.max() + 1].mean())


def flat_view(co, rgb, mask, colours, vt, sv):
    """The front frame of a flat part (module doc, 2): (R rows right/up/towards, a record)."""
    n, up = vt[2], vt[0]
    d = (co - co.mean(0)) @ n
    target = colours[mask].mean(0)
    sides = {s: float(np.abs(rgb[(d * s) > np.percentile(d * s, 70)].mean(0) - target).mean()) for s in (1.0, -1.0)}
    s = min(sides, key=sides.get)
    z = n * s
    t = (co - co.mean(0)) @ up
    across = np.cross(up, z)
    w = np.abs((co - co.mean(0)) @ across)
    top_wider = np.percentile(w[t > 0], 90) > np.percentile(w[t < 0], 90)
    if top_wider != crop_profile(mask):
        up = -up
    R = np.stack([np.cross(up, z), up, z])
    return R, {"flat": True, "front_colour_dist": round(sides[s], 1), "back_colour_dist": round(sides[-s], 1),
               "thin_over_long": round(float(sv[2] / sv[0]), 3)}
