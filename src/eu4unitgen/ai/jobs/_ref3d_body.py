"""Track M body fitting (spike E-A, stage S2/S3): the app's anatomy (blender/jobs/_horse_anatomy,
numpy only) on a proportioned and posed copy of the vanilla skeleton (bone positions read from the
user's install, never shipped), seen through a solved camera per reference view. Runs ONLY in the
AI environment; imported by ref3d_ea_fit.

PDX mesh space: Y up, the horse faces -Z, its left side at +X: a left-handed frame, so x is
mirrored before projecting (else every view comes out mirrored). Image space: crop pixels, y down.

  skeleton(bones, seat, k):  proportions: front and hind leg length (k["leg_f"], k["leg_b"]), trunk
      length (k["body"]), neck length (k["neck"]); the trunk rides on the legs.
  pose(bones, seat, P, a):   sagittal joint angles (degrees, about X): per leg at Up, Down, Knee and
      Heel; the neck at NeckBase and NeckMiddle. Adds each hoof's top and sole (the anatomy hangs
      the hoof on them).
  landmarks(...):            the template's body landmarks on the built shape.
  Camera: orbit about a target: azimuth / elevation / roll (degrees), distance, focal (px), and the
      image position of the target.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "blender" / "jobs"))
import _horse_anatomy as ha  # noqa: E402

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")
JOINTS = ("Up", "Down", "Knee", "Heel")
TARGET = np.array([0.0, 7.0, 0.0])


# ---------------------------------------------------------------- skeleton and pose

def skeleton(bones, seat, k):
    """Proportioned copy of the skeleton -> (bones, seat)."""
    B = {n: np.array(v, dtype=np.float64) for n, v in bones.items()}
    out = dict(B)
    zf, zb = B["HorseFrontLeftUp"][2], B["HorseBackLeftUp"][2]
    dyf = (k.get("leg_f", 1.0) - 1) * B["HorseFrontLeftUp"][1]
    dyb = (k.get("leg_b", 1.0) - 1) * B["HorseBackLeftUp"][1]
    sb = k.get("body", 1.0)

    def trunk(p):
        w = np.clip((p[2] - zf) / (zb - zf), 0, 1)
        return np.array([p[0], p[1] + dyf * (1 - w) + dyb * w, p[2] * sb])

    for n, p in B.items():
        if n.startswith("Horse") and any(n.startswith(f"Horse{l}") for l in LEGS) or n == "Root":
            continue
        out[n] = trunk(p)
    for leg in LEGS:
        s = k.get("leg_f" if leg.startswith("Front") else "leg_b", 1.0)
        up = B[f"Horse{leg}Up"]
        new_up = np.array([up[0], up[1] * s, up[2] * sb])
        for j in JOINTS:
            out[f"Horse{leg}{j}"] = new_up + (B[f"Horse{leg}{j}"] - up) * s
    sn = k.get("neck", 1.0)
    nb = out["HorseNeckBase"]
    nm = nb + (B["HorseNeckMiddle"] - B["HorseNeckBase"]) * sn
    out["HorseNeckMiddle"], out["HorseHead"] = nm, nm + (B["HorseHead"] - B["HorseNeckMiddle"]) * sn
    sz = B["Saddle_Node"][2]
    w = np.clip((sz - zf) / (zb - zf), 0, 1)
    return out, float(seat + dyf * (1 - w) + dyb * w)


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    """About Y; positive turns a point ahead (-Z) toward +X (the horse's left)."""
    a = np.radians(-deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def pose(bones, seat, P, a):
    """Posed bones (with HoofTop / Sole per leg). ``a``: {"FrontLeft": [up, down, knee, heel], ...,
    "neck": [base, middle], "neck_yaw": [base, middle]} in degrees; on a leg positive swings the part
    below the joint forwards (-Z); on the neck positive raises it; neck_yaw (optional, Track M: the
    reference's neck turns to the viewer) bends it sideways, positive to the horse's left."""
    B = {n: np.array(v, dtype=np.float64) for n, v in bones.items()}
    hoof_h = P["hoof_height"] * seat
    ahead = P.get("pastern_ahead", 0.0) * seat  # anatomy style 2: the pastern slopes forward
    for leg in LEGS:
        heel = B[f"Horse{leg}Heel"]
        pts = [B[f"Horse{leg}{j}"] for j in JOINTS] + [np.array([heel[0], hoof_h, heel[2] - ahead]),
                                                         np.array([heel[0], 0.0, heel[2] - ahead])]
        ang = list(a.get(leg, [0, 0, 0, 0]))
        for i, deg in enumerate(ang):  # joint i turns everything after it
            if not deg:
                continue
            R, c = _rot_x(deg), pts[i]
            for j in range(i + 1, len(pts)):
                pts[j] = c + R @ (pts[j] - c)
        for j, n in enumerate(JOINTS):
            B[f"Horse{leg}{n}"] = pts[j]
        B[f"Horse{leg}HoofTop"], B[f"Horse{leg}Sole"] = pts[4], pts[5]
    nb_, nm_ = a.get("neck", [0, 0])
    yb, ym = a.get("neck_yaw", [0, 0])
    for i, (joint, deg, yaw) in enumerate((("HorseNeckBase", nb_, yb), ("HorseNeckMiddle", nm_, ym))):
        if deg or yaw:
            R, c = _rot_y(yaw) @ _rot_x(deg), B[joint]
            for n in ("HorseNeckMiddle", "HorseHead")[i:]:
                B[n] = c + R @ (B[n] - c)
    return B


# ---------------------------------------------------------------- shape and landmarks

PART_NAMES = ["barrel", "neck", "head", "ear_l", "ear_r", "FrontLeft", "FrontRight", "BackLeft", "BackRight"]


def build(bones, seat, params):
    """-> (vertices, triangles, part index per triangle) of the anatomy on these bones."""
    r = ha.build({k: v.tolist() if hasattr(v, "tolist") else v for k, v in bones.items()}, seat, params)
    vs, fs, part, off = [], [], [], 0
    for i, (v, f) in enumerate(r["tubes"]):
        vs.append(v)
        fs.append(f + off)
        part.append(np.full(len(f), i))
        off += len(v)
    return np.vstack(vs), np.vstack(fs), np.concatenate(part), r["params"]


def _head_frame(P):
    hd = ha.head_direction(P)
    _, u, v = ha._frame(hd)
    return hd, u, v


def _head_point(B, H, P, t, side, up):
    """A point on the head's surface: ``t`` along it (0 poll, 1 muzzle), ``side`` across (+1 the
    left, +X), ``up`` from the centre line (+1 the top, -1 the jaw)."""
    hd, u, v = _head_frame(P)
    L = P["head_length"] * H
    depth = np.interp(t, [-0.08, 0.2, 0.45, 1.0], [0.8 * P["jowl_depth"], P["jowl_depth"],
                                                  0.5 * (P["jowl_depth"] + P["muzzle_depth"]), P["muzzle_depth"]]) * H
    half = np.interp(t, [-0.08, 0.3, 1.0], [0.85 * P["forehead_width"], P["forehead_width"], P["muzzle_width"]]) * H
    dm, wm, off = ha.head_profile_at(P, t)  # the free head shape, as the anatomy builds it
    depth, half = depth * dm, half * wm
    c = B["HorseHead"] + hd * (t * L) - v * depth * 0.35 + v * off * H
    return c + u * side * half + v * up * depth / 2


def landmarks(B, H, P):
    """The template's body landmarks (names as in templates/horse.json; near = the horse's left)."""
    if P.get("anatomy", 1) == 2:
        return _landmarks2(B, H, P)
    hd, u, v = _head_frame(P)
    L = P["head_length"] * H
    out = {"poll": B["HorseHead"] + v * 0.03 * H,
           "eye": _head_point(B, H, P, 0.3, 0.85, 0.35),
           "muzzle_front": _head_point(B, H, P, 1.0, 0.0, 0.0) + hd * 0.02 * L,
           "nostril": _head_point(B, H, P, 0.93, 0.75, 0.1),
           "bit_ring": _head_point(B, H, P, 0.8, 0.9, -0.45),
           "chin": _head_point(B, H, P, 0.86, 0.0, -1.0),
           "throat": _head_point(B, H, P, 0.05, 0.0, -1.0)}
    ears = []
    for side in (1.0, -1.0):
        base = B["HorseHead"] + u * side * 0.5 * P["forehead_width"] * H + v * 0.05 * H
        ears.append(base + (v * 0.9 - hd * 0.45) * P["ear_length"] * H)
    out["ear_tip_a"], out["ear_tip_b"] = ears  # matched as an unordered pair
    for leg, name in (("FrontLeft", "fore_near"), ("FrontRight", "fore_far"), ("BackLeft", "hind_near"), ("BackRight", "hind_far")):
        out[f"{name}_fetlock"] = B[f"Horse{leg}Heel"]
        out[f"{name}_hoof"] = B[f"Horse{leg}Sole"]
    return out


def _landmarks2(B, H, P):
    """Style 2's landmarks: on its head's surface (ha.head2_point) and its ears."""
    poll = B["HorseHead"]
    L = P["head_length"] * H

    def hp(t, side, q):
        return ha.head2_point(P, poll, H, t, side, q)

    out = {"poll": hp(0.0, 0.0, 1.0), "eye": hp(0.30, 0.95, 0.5), "nostril": hp(0.935, 0.9, 0.3),
           "muzzle_front": hp(1.0, 0.0, 0.2) + ha.head_direction(P) * 0.012 * L,
           "bit_ring": hp(0.84, 1.15, -0.4), "chin": hp(0.90, 0.0, -1.0), "throat": hp(0.05, 0.0, -1.0)}
    (_, _, ta), (_, _, tb) = ha.ear2_tips(P, poll, H)
    out["ear_tip_a"], out["ear_tip_b"] = ta, tb
    for leg, name in (("FrontLeft", "fore_near"), ("FrontRight", "fore_far"), ("BackLeft", "hind_near"), ("BackRight", "hind_far")):
        out[f"{name}_fetlock"] = B[f"Horse{leg}Heel"]
        out[f"{name}_hoof"] = B[f"Horse{leg}Sole"]
    return out


UNORDERED = (("ear_tip_a", "ear_tip_b"),)
BODY_LANDMARKS = ("poll", "eye", "muzzle_front", "nostril", "bit_ring", "chin", "throat", "ear_tip_a", "ear_tip_b",
                  "fore_near_fetlock", "fore_near_hoof", "fore_far_fetlock", "fore_far_hoof", "hind_near_fetlock",
                  "hind_near_hoof", "hind_far_fetlock", "hind_far_hoof")


# ---------------------------------------------------------------- camera

def camera_matrix(c):
    """c: {"az", "el", "roll" (deg), "dist", "focal", "cx", "cy"} -> (R, t): Xc = R (X - TARGET) + t.
    In the mirrored (right-handed) frame: az 0 looks at the horse's left side, +90 from its front
    (-Z), -90 from behind."""
    az, el, roll = np.radians([c["az"], c["el"], c["roll"]])
    # camera position direction from the target (the horse's left is -x once mirrored)
    d = np.array([-np.cos(el) * np.cos(az), np.sin(el), -np.cos(el) * np.sin(az)])
    fwd = -d
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = np.cos(roll), np.sin(roll)
    right, up = cr * right + sr * up, -sr * right + cr * up
    R = np.stack([right, up, fwd])  # camera x right, y up, z forward
    return R, np.array([0.0, 0.0, c["dist"]])


def project(X, c):
    R, t = camera_matrix(c)
    Xc = ((np.asarray(X) - TARGET) * np.array([-1.0, 1.0, 1.0])) @ R.T + t
    z = np.maximum(Xc[..., 2], 1e-3)
    return np.stack([c["cx"] + c["focal"] * Xc[..., 0] / z, c["cy"] - c["focal"] * Xc[..., 1] / z], axis=-1), Xc[..., 2]


# ---------------------------------------------------------------- raster

def silhouette(V2, F, shape, sel=None):
    """Union of the projected triangles (optionally a subset) -> bool mask."""
    import cv2
    m = np.zeros(shape, np.uint8)
    tri = V2[F if sel is None else F[sel]]
    tri = np.round(tri * 16).astype(np.int32)  # 4 bits of sub-pixel precision
    # one call per triangle: fillPoly with many polygons fills by the even-odd rule, so overlapping
    # triangles cancel (E-A pilots 1-6: holes wherever tubes overlap, legs inside the barrel)
    for t in tri:
        cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=4)
    return m > 0


def shaded(V, F, c, shape, colour=(170, 150, 130), light=(0.4, 0.8, -0.45), bg=None):
    """Flat-shaded painter's render of a mesh through camera ``c`` (for the proof sheets)."""
    import cv2
    V2, z = project(V, c)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    R, _ = camera_matrix(c)
    l = np.array(light) / np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.abs(n @ l)  # two-sided
    order = np.argsort(-z[F].mean(axis=1))
    img = np.zeros((*shape, 3), np.uint8) if bg is None else bg.copy()
    pts = np.round(V2[F] * 16).astype(np.int32)
    col = np.array(colour, np.float32)
    for i in order:
        cv2.fillPoly(img, [pts[i]], tuple(int(x) for x in col * shade[i]), lineType=cv2.LINE_AA, shift=4)
    return img


def decimate(V, F, cell):
    """Vertex clustering on a grid of ``cell`` (pymeshlab's plugins are blocked on this machine)."""
    q = np.floor(V / cell).astype(np.int64)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vn = np.zeros((len(keys), 3))
    np.add.at(Vn, inv, V)
    Vn /= np.bincount(inv)[:, None]
    Fn = inv[F]
    ok = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = np.unique(np.sort(Fn[ok], axis=1), axis=0)
    return Vn, Fn
