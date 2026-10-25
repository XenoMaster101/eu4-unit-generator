"""Run inside Blender (tests/test_blender_rules.py starts it): the Blender jobs' numpy rule helpers on synthetic data.
Prints one line, ``RESULT <json>``: each check's name and whether it held (with what was measured)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "eu4unitgen", "blender", "jobs"))

import numpy as np  # noqa: E402

import _colour as cc  # noqa: E402
import _reins  # noqa: E402
from _part_view import crop_profile, flat_view  # noqa: E402

out = {}


def check(name, ok, **seen):
    out[name] = {"ok": bool(ok), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in seen.items()}}


def img(colour, n=4000):
    return np.tile(np.asarray(colour, dtype=np.float64) / 255.0, (n, 1))


# the cloth is the subject's dominant saturated non-gold colour, whatever its hue
for name, cloth in (("red", (150, 30, 28)), ("blue", (30, 50, 160)), ("green", (40, 120, 40))):
    c = np.vstack([img(cloth, 3000), img((200, 160, 70), 2000), img((120, 120, 122), 3000)])
    h, share = cc.dominant_hue(c, np.ones(len(c), dtype=bool))
    want, _, _ = cc.hcl(np.asarray(cloth) / 255.0)
    check(f"cloth_hue_{name}", h is not None and cc.near_hue(h, float(want), 6.0), hue=h, want=float(want))
# gold never counts as cloth
c = np.vstack([img((200, 160, 70), 5000), img((120, 120, 122), 3000)])
check("gold_is_never_cloth", cc.dominant_hue(c, np.ones(len(c), dtype=bool))[0] is None)
# the old fixed window is the default with no cloth hue
win = cc.cloth_window(None)
check("default_window_is_the_old_red", bool(win(np.array([340.0]))[0]) and bool(win(np.array([15.0]))[0])
      and not bool(win(np.array([40.0]))[0]))
# a horse whose head and neck are bay: the coat; the cloth is the other colour (a blue pad)
H = 64
pos = np.zeros((H, H, 3))
pos[..., 2] = np.linspace(-10, 10, H)[None, :]  # z: the head (front) at the left
pos[..., 1] = 12.0
px = np.zeros((H, H, 3))
px[:, : H // 3] = np.array([150, 80, 40]) / 255.0  # bay coat on the head and neck
px[:, H // 3:] = np.array([150, 80, 40]) / 255.0
px[: H // 2, H // 2:] = np.array([30, 50, 160]) / 255.0  # a blue pad on the body
r = cc.cloth_and_coat(px, np.ones((H, H), dtype=bool), pos, {"HorseNeckBase": [0.0, 11.0, -3.0]})
check("coat_kept_apart_from_cloth", r["coat"] is not None and r["cloth"] is not None and cc.near_hue(r["cloth"], 230.0, 20.0),
      seen=r)
# the crest's colour: whatever saturated colour sits over the helmet
co = np.zeros((3000, 3))
co[:, 1] = np.linspace(10, 18, 3000)
rgb = np.where((co[:, 1] > 16.0)[:, None], np.array([40, 60, 170]), np.array([130, 130, 135])).astype(np.float64)
is_crest, h = cc.crest_colour(co, rgb, np.ones(3000, dtype=bool), 15.0)
hue, chroma, _ = cc.hcl(rgb / 255.0)
m = is_crest(hue, chroma)
check("crest_any_colour", h is not None and m[co[:, 1] > 16.2].all() and not m[co[:, 1] < 15.8].any(), hue=h)
# a flat part faces along its thinnest axis, the side matching the crop's colours, its wider end up as in the crop
rng = np.random.default_rng(0)
u = rng.uniform(-1, 1, 6000)
v = rng.uniform(-2, 2, 6000)
keep = np.abs(u) < (0.4 + 0.6 * (v + 2) / 4)  # wider towards +v
u, v = u[keep], v[keep]
side = rng.choice([-1.0, 1.0], len(u))
pts = np.stack([u, side * 0.05, v], 1)  # the plate in x-z, thin along y; its painted face on +y
cols = np.where((side > 0)[:, None], np.array([150, 30, 28.0]), np.array([90, 70, 50.0]))
mask = np.zeros((96, 96), dtype=bool)
for y in range(10, 86):
    w = int(8 + 30 * (85 - y) / 75)  # wider at the top of the image
    mask[y, 48 - w:48 + w] = True
colours = np.zeros((96, 96, 3))
colours[mask] = [150, 30, 28]
_, sv, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
R, rec = flat_view(pts, cols, mask, colours, vt, sv)
check("flat_part_faces_its_painted_side", R[2] @ np.array([0, 1.0, 0]) > 0.99, z=R[2], rec=rec)
check("flat_part_wide_end_up", R[1] @ np.array([0, 0, 1.0]) > 0.99, up=R[1])
check("crop_profile_wider_top", crop_profile(mask))
# the reins: from the bit's sides to Rem, weighted head -> neck -> Rem, built only when the raw model shows them
head = np.array([[x, y, z] for x in np.linspace(-0.4, 0.4, 5) for y in np.linspace(12.5, 14.0, 6) for z in np.linspace(-9.2, -12.0, 12)])
b = _reins.build(head, [0.0, 13.9, -9.2], [0.0, -0.2, -0.98], [0.0, 14.12, -2.36], segs=5)
check("reins_two_straps_within_budget", len(b["f"]) <= 70 and b["v"][:, 0].max() > 0.3 and b["v"][:, 0].min() < -0.3,
      tris=len(b["f"]))
w = [_reins.weights(t) for t in (0.0, 0.3, 0.7, 1.0)]
check("reins_weights_head_to_rem", w[0] == {"HorseHead": 1.0} and w[3] == {"Rem": 1.0} and "HorseNeckMiddle" in w[1]
      and all(abs(sum(x.values()) - 1) < 1e-6 for x in w))
bl, br, _ = _reins.bit_points(head, [0.0, 13.9, -9.2], [0.0, -0.2, -0.98])
rem = np.array([0.0, 14.12, -2.36])
pts = np.array([bl + (rem + [0.3, 0, 0] - bl) * t for t in np.linspace(0.15, 0.85, 60)])
ok, col, n = _reins.found(pts, np.tile([60, 40, 30], (60, 1)), np.full(60, 0.5), bl, br, rem)
check("reins_found_when_the_raw_shows_them", ok and col[0] == [60.0, 40.0, 30.0], n=n)
ok2, _, _ = _reins.found(pts, np.tile([60, 40, 30], (60, 1)), np.full(60, 0.05), bl, br, rem)
check("reins_not_built_when_the_low_kept_them", not ok2)
print("RESULT " + json.dumps(out))
