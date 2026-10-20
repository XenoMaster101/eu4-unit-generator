"""Track M spike E-A: fit and score the body candidates in both reference cameras (SPEC M9, M4 S2/S3).
Runs ONLY in the AI environment (numpy, SciPy, OpenCV, trimesh).

Candidates:
  b2:          round 5's B (its fitted shape, frozen); only the camera and pose per view are solved;
  procedural:  the anatomy's shape (33 parameters) and skeleton proportions (4) fitted to both views,
               with the cameras and poses;
  shells:      TripoSG meshes (seeds on view L), each aligned (similarity) through the procedural
               fit's cameras to the horse masks of both views;
  hybrid:      as procedural, plus the best shell's barrel half-widths (shoulder, ribs, hip; minus
               an armour allowance) as a prior: TripoSG as the depth prior no single view gives.
Each fitted candidate runs ``restarts`` times from seeded perturbed starts; the best is kept and the
spread reported. Every finished stage (and a stage's best point, each minute) is checkpointed in
``<out_dir>/checkpoints``: rerun into the same out_dir, the job resumes there (_resilience). If the
AI drive drops out, the job saves and exits with EXIT_DRIVE_LOST. Residuals (fractions of body length): body landmarks; the outline chamfer both ways
inside each view's zones (legs; the head in L), where pixels of an occluding class (fringe,
tassels, plume, straps) or inpainted ones are unknown, not scored; containment (the bare body
inside the horse mask: armour and cloth lie over it); weak priors.

args: {"skeleton": json path, "views": {name: {"image", "labels", "fg", "inpainted", "never_use",
       "landmarks", "body_length", "zones": {"legs": [poly], "head": [poly]}}}, "classes": {name: rgb},
       "occluders": {zone: [class]}, "body": {zone: [class]}, "candidates": [...], "shells": [glb],
       "restarts", "seed", "label_scale", "gates", "armour_allowance", "cameras": {view: {param: [lo, hi]}},
       "landmark_weights": {name: w}, "evals": {stage: n}, "out_dir", "result_path"}
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _resilience as R  # noqa: E402

WATCH = None  # R.DriveWatch, set by main: every evaluation checks the AI drive is still there
CKPT_EVERY = 60.0  # s: a Powell stage saves its best point this often
SHAPE = ["barrel_depth", "withers_rise", "croup_rise", "rump_drop", "breast_ahead", "rump_behind", "breast_width",
         "shoulder_width", "rib_width", "hip_width", "buttock_width", "neck_base_depth", "neck_throat_depth",
         "neck_width", "crest", "head_length", "jowl_depth", "forehead_width", "muzzle_depth", "muzzle_width",
         "ear_length", "upper_arm", "forearm", "knee", "stifle", "gaskin", "hock", "cannon", "fetlock", "pastern",
         "hoof_top", "hoof_bottom", "hoof_height"]
SIGNED = {"withers_rise", "croup_rise"}  # can cross zero: absolute scale and bounds
SKEL = ["leg_f", "leg_b", "body", "neck"]
# Solved camera parameters: "scale" is px per unit at the target (focal / dist), "persp" the
# perspective strength (10 / dist). E-A pilot 7 solved focal and dist directly: L slid to the dist
# bound with focal following, since one coordinate step changed both size and perspective.
CAM = ["az", "el", "roll", "persp", "scale", "cx", "cy"]
CAM_SCALE = {"az": 5, "el": 5, "roll": 3, "persp": 0.03, "scale": 3, "cx": 20, "cy": 20}
# roll: the illustrations stand on level ground (pilot 7 pinned L's roll at -14); az is narrowed
# per view by args["cameras"] (pilot 7: leg-only landmarks let R fit the mirrored front view)
CAM_BOUNDS = {"az": (-180, 180), "el": (-25, 45), "roll": (-8, 8), "persp": (0.005, 0.5), "scale": (5, 300),
              "cx": (-2000, 4000), "cy": (-2000, 4000)}
CAM_LIMITS = {}  # view -> {param: (lo, hi)}, from args["cameras"]
EVALS = {"landmarks": 400, "outlines: camera+pose": 4000, "outlines: shape": 3000, "outlines: joint": 5000}


def cam_internal(c):
    """A camera as solved (CAM) from an orbit camera {az, el, roll, dist, focal, cx, cy}."""
    if "persp" in c:
        return dict(c)
    return {**{k: c[k] for k in ("az", "el", "roll", "cx", "cy")}, "persp": 10.0 / c["dist"], "scale": c["focal"] / c["dist"]}


def cam_orbit(c):
    """The orbit camera _ref3d_body projects with, from a solved one."""
    dist = 10.0 / c["persp"]
    return {**{k: c[k] for k in ("az", "el", "roll", "cx", "cy")}, "dist": dist, "focal": c["scale"] * dist}


def cam_bounds(vn, c):
    return CAM_LIMITS.get(vn, {}).get(c, CAM_BOUNDS[c])
LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")
POSE_BOUNDS = {"Front": [(-45, 60), (-60, 60), (-110, 20), (-60, 60)],
               "Back": [(-45, 45), (-60, 60), (-30, 90), (-60, 60)]}
N_ZONE, N_CONT = 120, 160


def poly_mask(shape, polys):
    import cv2
    m = np.zeros(shape, np.uint8)
    for p in polys or []:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m > 0


def boundary(m):
    import cv2
    m8 = m.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0


def dist_to(mask):
    """Distance (px) to the nearest True pixel of ``mask``."""
    import cv2
    if not mask.any():
        return np.full(mask.shape, 1e4, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def sample(points, n):
    """``n`` points from an (k, 2) array, evenly by index (deterministic); None when empty."""
    if len(points) == 0:
        return None
    idx = np.linspace(0, len(points) - 1, n).round().astype(int)
    return points[idx]


class View:
    """A reference view's targets."""

    def __init__(self, name, v, classes, a):
        import cv2
        from PIL import Image

        import ref3d_ed2d as E
        self.name = name
        self.image = np.array(Image.open(v["image"]).convert("RGB"))
        cls = E.decode(np.array(Image.open(v["labels"]).convert("RGB")), classes)
        H, W = cls.shape
        self.shape = (H, W)
        fg = np.array(Image.open(v["fg"]).convert("L")) > 127
        inp = np.array(Image.open(v["inpainted"]).convert("L")) > 127
        self.never = poly_mask(self.shape, v.get("never_use"))
        self.fg = fg & ~self.never
        self.bl = v["body_length"]
        names = list(classes)
        self.lm = {k: np.array(p, float) for k, p in v["landmarks"].items()}
        self.lm = {k: p for k, p in self.lm.items() if not self.never[int(min(max(p[1], 0), H - 1)), int(min(max(p[0], 0), W - 1))]}
        tol = max(1, int(round(a["label_scale"] * self.bl)))
        self.tol = tol
        self.zones = {}
        for z, polys in (v.get("zones") or {}).items():
            zm = poly_mask(self.shape, polys) & ~self.never
            if not zm.any():
                continue
            body = np.isin(cls, [names.index(c) for c in a["body"][z]]) & fg
            occ = (np.isin(cls, [names.index(c) for c in a["occluders"][z]]) & fg) | (inp & fg)
            known = zm & ~cv2.dilate(occ.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            target = body & known
            # target edges that are real body edges: inside the known zone, away from its border
            inner = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            tb = boundary(target) & inner
            ys, xs = np.nonzero(tb)
            self.zones[z] = {"zone": zm, "known": known, "inner": inner, "target": target,
                             "target_cells": cells(target), "cell_sel": cells(known) > 0,
                             "known_ss": up(known), "inner_ss": up(inner), "box": zone_box(known),
                             "dt_target_edge": dist_to(tb), "edge_samples": sample(np.stack([xs, ys], 1), N_ZONE)}
        self.dt_out = dist_to(cv2.dilate(self.fg.astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8)).astype(bool))
        self.outside_ss = up((self.dt_out > 0) & ~self.never)


def lm_residuals(view, lm2):
    import _ref3d_body as RB
    r, order = [], []
    names = [k for k in RB.BODY_LANDMARKS if k in view.lm and k in lm2]
    pairs = {a: b for a, b in RB.UNORDERED}
    done = set()
    for k in names:
        if k in done:
            continue
        if k in pairs and pairs[k] in names:
            a_, b_ = k, pairs[k]
            d1 = np.r_[lm2[a_] - view.lm[a_], lm2[b_] - view.lm[b_]]
            d2 = np.r_[lm2[b_] - view.lm[a_], lm2[a_] - view.lm[b_]]
            r.append(d1 if np.sum(d1 ** 2) <= np.sum(d2 ** 2) else d2)
            order.append(k)
            done |= {a_, b_}
        else:
            r.append(lm2[k] - view.lm[k])
            order.append(k)
            done.add(k)
    if not r:
        return np.zeros(0)
    w = np.concatenate([np.full(len(d), LM_WEIGHT.get(k, 1.0)) for d, k in zip(r, order)])
    return W_LM * w * np.concatenate(r) / view.bl


CELL = 12          # px: region fill and containment are compared per cell
SS = 3             # the fit renders outlines at 3x: at 1x an edge's place is known to +-0.5 px, which
                   # swamped the finite differences of 1-5 px moves (E-A pilot, Jacobian check)
W_LM, W_AREA, W_EDGE, W_CONT = 10.0, 0.5, 1.0, 0.5
HEAD_TURN = False  # args["head_turn"]: neck and head yaw per view (E-A round 9)
HEAD_FIT = {}  # args["head_fit"]: {"view", "stations", "evals"}: the free head shape and its own stage (round 10)
HEAD_WIDTHS = {"forehead_width", "muzzle_width"}
HEAD_PARAMS = ["head_length", "jowl_depth", "forehead_width", "muzzle_depth", "muzzle_width", "ear_length"]
HEAD_LM = {"poll", "eye", "muzzle_front", "nostril", "bit_ring", "chin", "throat", "ear_tip_a", "ear_tip_b"}
RUMP = ["croup_rise", "rump_drop", "rump_behind", "hip_width", "buttock_width"]  # R's part of the master (M10)
LM_WEIGHT = {}  # landmark -> weight, from args["landmark_weights"] (pilot 7: the head's landmarks lost to the outlines)


def cells(m, c=CELL):
    """Mean of a mask over c x c cells (padded) -> (H/c, W/c) floats."""
    H, W = m.shape
    ph, pw = -H % c, -W % c
    p = np.pad(m.astype(np.float32), ((0, ph), (0, pw)))
    return p.reshape(p.shape[0] // c, c, p.shape[1] // c, c).mean(axis=(1, 3))


def zone_box(known, margin=2):
    """The zone's bounding box in supersampled pixels, aligned to the cells (whole cells, a margin
    of cells round it): the zone terms are computed inside it only (E-A: the full-image distance
    transform at 3x was most of each evaluation)."""
    ys, xs = np.nonzero(known)
    c = CELL
    y0, x0 = max(0, (ys.min() // c - margin) * c), max(0, (xs.min() // c - margin) * c)
    y1 = min(known.shape[0] // c * c, (ys.max() // c + margin + 1) * c)
    x1 = min(known.shape[1] // c * c, (xs.max() // c + margin + 1) * c)
    return y0 * SS, y1 * SS, x0 * SS, x1 * SS


def up(m, s=SS):
    return np.repeat(np.repeat(m, s, axis=0), s, axis=1)


def render_terms(view, V2, F):
    """Residuals with fixed slots (pilot fit v001: boundary points sampled from the render changed
    slot with every small change, so the Jacobian was noise): per zone, the fill per cell (render
    minus target, known pixels only) and the distance from fixed target edge points to the
    rendered edge; containment per cell (render outside the horse mask)."""
    import _ref3d_body as RB
    H, W = view.shape
    sil = RB.silhouette(V2 * SS, F, (H * SS, W * SS))
    r = []
    for z, Z in view.zones.items():
        y0, y1, x0, x1 = Z["box"]
        rm = sil[y0:y1, x0:x1] & Z["known_ss"][y0:y1, x0:x1]
        tc = Z["target_cells"][y0 // (CELL * SS):y1 // (CELL * SS), x0 // (CELL * SS):x1 // (CELL * SS)]
        cs = Z["cell_sel"][y0 // (CELL * SS):y1 // (CELL * SS), x0 // (CELL * SS):x1 // (CELL * SS)]
        r.append(W_AREA * (cells(rm, CELL * SS)[:tc.shape[0], :tc.shape[1]] - tc)[cs])
        if Z["edge_samples"] is not None:
            d = dist_to(boundary(rm) & Z["inner_ss"][y0:y1, x0:x1])
            e = Z["edge_samples"] * SS + SS // 2 - np.array([x0, y0])
            ok = (e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] < x1 - x0) & (e[:, 1] < y1 - y0)
            dd = np.full(len(e), 0.3 * view.bl * SS)
            dd[ok] = d[e[ok, 1], e[ok, 0]]
            r.append(W_EDGE * np.minimum(dd / SS, 0.3 * view.bl) / view.bl)
    r.append(W_CONT * cells(sil & view.outside_ss, CELL * SS).ravel())
    return np.concatenate(r)


class Model:
    """Parameter vector <-> (shape, skeleton, cameras, poses) for one candidate."""

    def __init__(self, sk, views, free_shape, base_shape, priors, init):
        self.sk, self.views, self.free_shape = sk, views, free_shape
        self.base = dict(base_shape)
        self.priors = priors
        self.names, self.x0, self.scale, self.lo, self.hi = [], [], [], [], []

        def add(name, x0, scale, lo, hi):
            self.names.append(name)
            self.x0.append(x0)
            self.scale.append(scale)
            self.lo.append(lo)
            self.hi.append(hi)

        if free_shape:
            for p in SHAPE:
                v0 = init["shape"].get(p, base_shape[p])
                if p in SIGNED:
                    add(f"shape.{p}", v0, 0.02, -0.15, 0.15)
                elif p in HEAD_WIDTHS:  # anatomy sanity: widths unseen in a side or 3/4 view
                    add(f"shape.{p}", v0, 0.05 * abs(base_shape[p]), 0.8 * base_shape[p], 1.25 * base_shape[p])
                else:
                    add(f"shape.{p}", v0, 0.15 * abs(base_shape[p]), 0.5 * base_shape[p], 2.0 * base_shape[p])
            for k in SKEL:
                add(f"skel.{k}", init["skel"].get(k, 1.0), 0.05, 0.75, 1.35)
            K = HEAD_FIT.get("stations", 0)
            hp0 = list(init.get("hp") or ([1.0] * K + [1.0] * K + [0.0] * K))
            # anatomy sanity (SPEC S2; E-A pilot 11: L's 3/4 view can't tell the head's width from its
            # depth, and the fit widened the head into a bulb that only matched L's outline)
            for i in range(3 * K):  # depth multipliers, width multipliers, vertical offsets
                lo, hi, sc_ = ((0.7, 1.5, 0.08) if i < K else (0.85, 1.2, 0.05) if i < 2 * K else (-0.06, 0.06, 0.01))
                add(f"hp.{i}", hp0[i], sc_, lo, hi)
        for vn in views:
            ci = cam_internal(init["cam"][vn])
            for c in CAM:
                add(f"{vn}.cam.{c}", ci[c], CAM_SCALE[c], *cam_bounds(vn, c))
            for leg in LEGS:
                for j in range(4):
                    lo, hi = POSE_BOUNDS["Front" if leg.startswith("Front") else "Back"][j]
                    add(f"{vn}.pose.{leg}.{j}", init["pose"][vn].get(leg, [0, 0, 0, 0])[j], 8, lo, hi)
            for j in range(2):
                add(f"{vn}.pose.neck.{j}", init["pose"][vn].get("neck", [0, 0])[j], 8, -45, 45)
            add(f"{vn}.pose.head_pitch", init["pose"][vn].get("head_pitch", base_shape["head_pitch"]), 8, 0, 95)
            if HEAD_TURN:  # E-A round 9: the reference's head turns to the viewer; sagittal joints can't
                for j in range(2):
                    add(f"{vn}.pose.neck_yaw.{j}", init["pose"][vn].get("neck_yaw", [0, 0])[j], 5, -30, 30)
                add(f"{vn}.pose.head_yaw", init["pose"][vn].get("head_yaw", 0.0), 5, -40, 40)
        self.x0, self.scale = np.array(self.x0, float), np.array(self.scale, float)
        self.lo, self.hi = np.array(self.lo, float), np.array(self.hi, float)
        self.x0 = np.clip(self.x0, self.lo + 1e-9, self.hi - 1e-9)

    def unpack(self, x):
        d = dict(zip(self.names, x))
        shape = dict(self.base)
        skel = {}
        hp = []
        for k, v in d.items():
            if k.startswith("shape."):
                shape[k[6:]] = v
            elif k.startswith("skel."):
                skel[k[5:]] = v
            elif k.startswith("hp."):
                hp.append(v)
        if hp:
            shape["head_profile"] = hp
        cams, poses = {}, {}
        for vn in self.views:
            cams[vn] = cam_orbit({c: d[f"{vn}.cam.{c}"] for c in CAM})
            poses[vn] = {leg: [d[f"{vn}.pose.{leg}.{j}"] for j in range(4)] for leg in LEGS}
            poses[vn]["neck"] = [d[f"{vn}.pose.neck.{j}"] for j in range(2)]
            poses[vn]["head_pitch"] = d[f"{vn}.pose.head_pitch"]
            if HEAD_TURN:
                poses[vn]["neck_yaw"] = [d[f"{vn}.pose.neck_yaw.{j}"] for j in range(2)]
                poses[vn]["head_yaw"] = d[f"{vn}.pose.head_yaw"]
        return shape, skel, cams, poses

    def geometry(self, x, vn, mesh=True):
        import _ref3d_body as RB
        shape, skel, cams, poses = self.unpack(x)
        B, H = RB.skeleton(self.sk["bones"], self.sk["seat"], skel)
        P = {**shape, "head_pitch": poses[vn]["head_pitch"], "head_yaw": poses[vn].get("head_yaw", 0.0)}
        Bp = RB.pose(B, H, P, poses[vn])
        if not mesh:  # landmarks need only the bones and parameters
            return None, None, None, RB.landmarks(Bp, H, {**RB.ha.DEFAULTS, **P}), cams[vn]
        V, F, part, P = RB.build(Bp, H, P)
        return V, F, part, RB.landmarks(Bp, H, P), cams[vn]

    def blocks(self, x):
        """-> (smooth residuals: landmarks and priors, outline residuals)."""
        import _ref3d_body as RB
        if WATCH:
            WATCH.check()
        smooth, rend = [], []
        for vn, view in self.views.items():
            V, F, part, lm3, cam = self.geometry(x, vn)
            names = list(lm3)
            p2, _ = RB.project(np.array([lm3[k] for k in names]), cam)
            smooth.append(lm_residuals(view, dict(zip(names, p2))))
            V2, _ = RB.project(V, cam)
            rend.append(render_terms(view, V2, F))
        smooth.append(self.prior_terms(x))
        return np.concatenate(smooth), np.concatenate(rend)

    def residuals(self, x, which=("lm", "render", "prior")):
        import _ref3d_body as RB
        if WATCH:
            WATCH.check()
        r = []
        for vn, view in self.views.items():
            V, F, part, lm3, cam = self.geometry(x, vn, mesh="render" in which)
            if "lm" in which:
                names = list(lm3)
                p2, _ = RB.project(np.array([lm3[k] for k in names]), cam)
                r.append(lm_residuals(view, dict(zip(names, p2))))
            if "render" in which:
                V2, _ = RB.project(V, cam)
                r.append(render_terms(view, V2, F))
        if "prior" in which:
            r.append(self.prior_terms(x))
        return np.concatenate(r)

    def prior_terms(self, x):
        out = []
        for name, v, x0 in zip(self.names, x, self.x0):
            kind = name.split(".")[0] if not name.startswith(tuple(self.views)) else name.split(".")[1]
            if kind == "shape":
                p = name[6:]
                mu, sig, w = self.priors.get(p, (self.base[p], 0.5 * max(abs(self.base[p]), 0.02), 0.02))
                out.append(w * (v - mu) / sig)
            elif kind == "skel":
                out.append(0.02 * (v - 1.0) / 0.3)
            elif kind == "hp":  # a weak pull to the parametric head (multipliers 1, offsets 0)
                k = int(name[3:])
                ref = 0.0 if k >= 2 * HEAD_FIT.get("stations", 0) else 1.0
                out.append(0.02 * (v - ref) / (0.05 if ref == 0.0 else 0.3))
            elif kind == "pose":
                out.append(0.004 * v / 30.0 if not name.endswith("head_pitch") else 0.0)  # yaws too: a weak pull to straight
        return np.array(out)


def solve(model, which, mask=None, max_nfev=60, seed=0, diff_step=0.15, loss="linear"):
    """least_squares over the parameters selected by ``mask`` (others fixed), in normalised
    variables (offset / the parameter's scale) with a fixed finite-difference step of 15% of a
    scale: the objective comes from rasterised outlines, so smaller steps (scipy's default is
    relative to the value: ~0.01 deg for a joint near zero) move edges by under a pixel and give a
    noisy Jacobian."""
    from scipy.optimize import least_squares
    x = model.x0.copy()
    sel = np.ones(len(x), bool) if mask is None else mask
    x0, sc = x[sel].copy(), model.scale[sel]
    lo, hi = (model.lo[sel] - x0) / sc, (model.hi[sel] - x0) / sc

    def f(z):
        xx = x.copy()
        xx[sel] = x0 + z * sc
        return model.residuals(xx, which)

    res = least_squares(f, np.zeros(len(x0)), bounds=(lo, hi), diff_step=diff_step, max_nfev=max_nfev, method="trf",
                        loss=loss, f_scale=0.1 if loss != "linear" else 1.0)
    x[sel] = x0 + res.x * sc
    model.x0 = x
    return res


class _Res:  # least_squares-like record for the history
    def __init__(self, cost, nfev, message):
        self.cost, self.nfev, self.njev, self.status, self.message = cost, nfev, 0, 0, message


def powell(model, which, mask=None, maxfev=1500, f_scale=0.1, on_best=None):
    """Derivative-free polish of the outline terms (E-A pilot 3: least_squares stopped after a few
    evaluations on them, every step rejected on raster jitter). Powell's line searches, in the
    normalised variables. Landmarks and priors stay squared; only the outline terms get the soft-L1
    loss (outliers: occluders the labels missed). Pilot 5 put the landmarks under it too, and their
    large residuals were discounted as outliers: B's worst landmark went from 9% to 24%.
    ``on_best(x, nfev)`` gets the best point so far every CKPT_EVERY s, and when the drive drops out."""
    from scipy.optimize import minimize
    x = model.x0.copy()
    s = np.ones(len(x), bool) if mask is None else mask
    x0, sc = x[s].copy(), model.scale[s]
    lo, hi = (model.lo[s] - x0) / sc, (model.hi[s] - x0) / sc
    best = {"f": np.inf, "x": x.copy(), "n": 0, "saved": time.time()}

    def full(z):
        xx = x.copy()
        xx[s] = x0 + np.clip(z, lo, hi) * sc
        return xx

    def cost(z):
        xx = full(z)
        smooth, rend = model.blocks(xx)
        r = rend / f_scale
        f = float(0.5 * smooth @ smooth + f_scale ** 2 * np.sum(np.sqrt(1 + r * r) - 1))
        best["n"] += 1
        if f < best["f"]:
            best["f"], best["x"] = f, xx
        if on_best and time.time() - best["saved"] > CKPT_EVERY:
            on_best(best["x"], best["n"])
            best["saved"] = time.time()
        return f

    try:
        res = minimize(cost, np.zeros(len(x0)), method="Powell", bounds=list(zip(lo, hi)),
                       options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-5})
    except R.DriveLost:
        if on_best and best["n"]:
            on_best(best["x"], best["n"])
        raise
    model.x0 = full(res.x)
    return _Res(float(res.fun), int(res.nfev), str(res.message))


def sel(model, prefixes):
    return np.array([any(n.startswith(p) for p in prefixes) for n in model.names])


def init_cameras(sk, views, shape, seed):
    """Per view: the camera from the landmarks alone (rest pose, the given shape), best of three
    azimuth starts."""
    import _ref3d_body as RB
    rng = np.random.default_rng(seed)
    out = {}
    B, H = RB.skeleton(sk["bones"], sk["seat"], {})
    for vn, view in views.items():
        lo, hi = cam_bounds(vn, "az")
        lo, hi = max(lo, -170), min(hi, 170)
        best = None
        for az in np.linspace(lo, hi, 5)[1:-1]:  # three starts inside the view's window
            ys, xs = np.nonzero(view.fg)
            c0 = {"az": float(np.clip(az + rng.normal(0, 3), lo + 1, hi - 1)), "el": 8 + rng.normal(0, 3), "roll": 0.0, "dist": 60.0,
                  "focal": 60.0 * (ys.max() - ys.min()) / 14.0, "cx": float(xs.mean()), "cy": float(ys.mean())}
            m = Model(sk, {vn: view}, False, shape, {}, {"cam": {vn: c0}, "pose": {vn: {}}, "shape": {}, "skel": {}})
            r = solve(m, ("lm",), sel(m, [f"{vn}.cam."]), max_nfev=EVALS["landmarks"])
            if best is None or r.cost < best[0]:
                best = (r.cost, m.unpack(m.x0)[2][vn])
        out[vn] = best[1]
    return out


def _note(hist, log, stage, r):
    hist.append({"stage": stage, "cost": float(r.cost), "nfev": int(r.nfev), "njev": int(r.njev or 0),
                 "status": int(r.status), "message": r.message})
    log(f"  {stage} cost {r.cost:.4f} nfev {r.nfev} njev {r.njev} {r.message}")


HEAD_STARTS = [(n, h) for n in (0.0, 25.0, 50.0) for h in (40.0, 65.0, 85.0)]


def restore(sk, views, free_shape, base_shape, priors, state):
    """A Model at a checkpointed parameter vector."""
    x = dict(zip(state["names"], state["x"]))
    m = Model(sk, views, free_shape, base_shape, priors,
              {"cam": {vn: {c: x[f"{vn}.cam.{c}"] for c in CAM} for vn in views}, "pose": {vn: {} for vn in views},
               "shape": {}, "skel": {}, "hp": [x[n] for n in state["names"] if n.startswith("hp.")]})
    if m.names != state["names"]:
        raise ValueError("checkpoint doesn't match this candidate's parameters")
    m.x0 = np.clip(np.array(state["x"], float), m.lo + 1e-9, m.hi - 1e-9)  # bounds may have narrowed since
    return m


def fit_candidate(sk, views, base_shape, free_shape, priors, seed, log, ck=None, key=None):
    """1. cameras from the landmarks (rest pose); 2. landmarks only (smooth: E-A pilot's Jacobian
    check), camera + pose (+ proportions and shape when free), from 9 neck / head-pitch starts:
    the reference carries its head far higher than vanilla's rest; 3. outlines added: camera + pose,
    then shape, then everything, by Powell (derivative-free; rasterised outlines stay jittery at the
    pixel). With ``ck``, each finished stage is saved under ``key`` and a rerun resumes after the last
    one (inside a Powell stage: from its saved best point, with the evaluations left). -> (model, history)."""
    st = (ck.load(key) if ck else None) or {}
    hist = st.get("hist", [])

    def save(m, partial=None):
        if ck:
            ck.save(key, {"names": m.names, "x": m.x0, "hist": hist, "partial": partial, "done": False})

    if "x" in st:
        m = restore(sk, views, free_shape, base_shape, priors, st)
        log(f"  resumed after: {hist[-1]['stage']}")
    else:
        rng = np.random.default_rng(seed)
        cams = init_cameras(sk, views, base_shape, seed)
        for vn, c in cams.items():  # restarts differ: perturbed starts, inside the bounds
            lo, hi = cam_bounds(vn, "az")
            c["az"] = float(np.clip(c["az"] + rng.normal(0, 4), lo + 0.5, hi - 0.5))
            lo, hi = cam_bounds(vn, "el")
            c["el"] = float(np.clip(c["el"] + rng.normal(0, 3), lo + 0.5, hi - 0.5))
        best = None
        for neck, pitch in HEAD_STARTS:
            poses = {vn: {**{leg: list(rng.normal(0, 4, 4)) for leg in LEGS}, "neck": [neck, 0.0], "head_pitch": pitch}
                     for vn in views}
            m = Model(sk, views, free_shape, base_shape, priors, {"cam": cams, "pose": poses, "shape": {}, "skel": {}})
            keys = [f"{vn}.cam." for vn in views] + [f"{vn}.pose." for vn in views] + (["shape.", "skel."] if free_shape else [])
            r = solve(m, ("lm", "prior"), sel(m, keys), max_nfev=EVALS["landmarks"], diff_step=0.01)
            if best is None or r.cost < best[0].cost:
                best = (r, m, (neck, pitch))
        r, m, start = best
        _note(hist, log, f"landmarks (start neck {start[0]:.0f}, pitch {start[1]:.0f})", r)
        save(m)
    vp = [f"{vn}.cam." for vn in views] + [f"{vn}.pose." for vn in views]
    stages = [("outlines: camera+pose", sel(m, vp), EVALS["outlines: camera+pose"])]
    if free_shape:
        stages += [("outlines: shape", sel(m, ["shape.", "skel."]), EVALS["outlines: shape"]),
                   ("outlines: joint", None, EVALS["outlines: joint"])]
    hv = HEAD_FIT.get("view")
    if free_shape and HEAD_FIT and hv in views:  # E-A round 10: the head (and chanfron) as its own shape
        stages += [("outlines: head", sel(m, ["hp."] + [f"shape.{p}" for p in HEAD_PARAMS]
                                         + [f"{hv}.pose.neck", f"{hv}.pose.head_pitch", f"{hv}.pose.head_yaw"]),
                    HEAD_FIT.get("evals", 3000))]
    done = {h["stage"].removesuffix(" (resumed)") for h in hist}
    for stage, mask, fev in stages:
        if stage in done:
            continue
        part = st.get("partial") if (st.get("partial") or {}).get("stage") == stage else None
        used = 0
        if part:
            m.x0, used = np.array(part["x"], float), part["nfev"]
            log(f"  {stage}: resumed at its best point after {used} evaluations")

        def on_best(x, n, stage=stage, used=used, start=m.x0.copy()):
            if ck:
                ck.save(key, {"names": m.names, "x": start, "hist": hist, "done": False,
                              "partial": {"stage": stage, "x": x, "nfev": used + n}})

        r = powell(m, ("lm", "render", "prior"), mask, maxfev=max(fev - used, 50), on_best=on_best)
        _note(hist, log, stage + (" (resumed)" if part else ""), r)
        save(m)
    return m, hist


# ---------------------------------------------------------------- scoring

def score_view(view, sil, lm2, gates):
    import _ref3d_metrics as M
    out = {"regions": {}, "landmarks": {}}
    for z, Z in view.zones.items():
        out["regions"][z] = M.tol_iou(sil & Z["known"], Z["target"], view.tol)
    if lm2 is not None:
        for k in view.lm:
            if k in lm2:
                out["landmarks"][k] = float(np.hypot(*(lm2[k] - view.lm[k])) / view.bl)
        for a_, b_ in (("ear_tip_a", "ear_tip_b"),):
            if a_ in out["landmarks"] and b_ in out["landmarks"]:
                sw = (np.hypot(*(lm2[b_] - view.lm[a_])) + np.hypot(*(lm2[a_] - view.lm[b_]))) / view.bl
                if sw < out["landmarks"][a_] + out["landmarks"][b_]:
                    out["landmarks"][a_] = float(np.hypot(*(lm2[b_] - view.lm[a_])) / view.bl)
                    out["landmarks"][b_] = float(np.hypot(*(lm2[a_] - view.lm[b_])) / view.bl)
    area = max(int(sil.sum()), 1)
    out["outside_share"] = float((sil & (view.dt_out > 0) & ~view.never).sum() / area)
    out["horse_iou"] = M.tol_iou(sil & ~view.never, view.fg, view.tol)
    reg = [v for v in out["regions"].values() if v is not None]
    head_max = gates.get("landmark_max_head", gates["landmark_max"])  # the head may get its own limit (painted reference)
    lms = list(out["landmarks"].values())
    lm_ok = all(v <= (head_max if k in HEAD_LM else gates["landmark_max"]) for k, v in out["landmarks"].items())
    out["gates"] = {"region_iou": {"pass": bool(reg) and min(reg) >= gates["region_iou_min"], "worst": min(reg) if reg else None},
                    "landmarks": {"pass": bool(lms) and lm_ok, "worst": max(lms) if lms else None,
                                  "worst_body": max([v for k, v in out["landmarks"].items() if k not in HEAD_LM], default=None),
                                  "worst_head": max([v for k, v in out["landmarks"].items() if k in HEAD_LM], default=None)}}
    out["landmark_rms"] = float(np.sqrt(np.mean(np.square(lms)))) if lms else None
    return out


def vanilla_distance(V, sk):
    """Median distance from the candidate's surface (rest pose) to vanilla's, and the share within
    0.15 (SPEC round 5's evidence the geometry is new)."""
    from scipy.spatial import cKDTree
    vp = np.array(sk["vanilla_p"]).reshape(-1, 3)
    vt = np.array(sk["vanilla_tri"]).reshape(-1, 3)
    rng = np.random.default_rng(0)
    tri = vp[vt]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(vt), 60000, p=area / area.sum())
    u, w = rng.random((2, 60000))
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    pts = tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + w[:, None] * (tri[idx, 2] - tri[idx, 0])
    d, _ = cKDTree(pts).query(V)
    return {"median": float(np.median(d)), "within_0.15": float((d < 0.15).mean())}


# ---------------------------------------------------------------- shells

def load_shell(path, cell_frac=0.006):
    import _ref3d_body as RB
    import trimesh
    m = trimesh.load(path, force="mesh")
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    return RB.decimate(V, F, cell_frac * diag)


def shell_canonical(Vs):
    """Rotation putting a shell in the body's frame (E-A: TripoSG returns each shell in its own
    canonical orientation, not the input camera's: seed 11 came out side-on, seed 44 turned 90 deg).
    Up stays y (legs point down in every shell); the main horizontal axis becomes z; the end whose
    top is higher (the neck rises) is the head, sent to -Z. -> 3x3 matrix."""
    c = Vs - Vs.mean(axis=0)
    xz = c[:, [0, 2]]
    w, v = np.linalg.eigh(np.cov(xz.T))
    ax = v[:, np.argmax(w)]  # long axis in (x, z)
    th = np.arctan2(ax[0], ax[1])  # angle from +z
    R = np.array([[np.cos(-th), 0, np.sin(-th)], [0, 1, 0], [-np.sin(-th), 0, np.cos(-th)]])
    z = (c @ R.T)[:, 2]
    y = c[:, 1]
    lo, hi = np.percentile(z, 25), np.percentile(z, 75)
    if np.percentile(y[z > hi], 95) > np.percentile(y[z < lo], 95):  # the head is at +z: turn round
        R = np.diag([-1.0, 1.0, -1.0]) @ R
    return R


def shell_place(Vs, x, R0):
    """Shell -> PDX space: the canonical rotation ``R0``, then x: [rx, ry, rz (deg), log scale,
    tx, ty, tz] about the body's target point. A shell is near-symmetric, so its handedness is not
    solved."""
    from scipy.spatial.transform import Rotation

    import _ref3d_body as RB
    extra = Rotation.from_euler("xyz", x[:3], degrees=True).as_matrix()
    c = Vs.mean(axis=0)
    return RB.TARGET + np.exp(x[3]) * ((Vs - c) @ (extra @ R0).T) + x[4:7]


def fit_shell(Vs, Fs, views, cams, height):
    """Similarity placement of a shell against the horse masks of both views (outline chamfer)."""
    import _ref3d_body as RB
    from scipy.optimize import least_squares
    R0 = shell_canonical(Vs)
    span = float(np.ptp(Vs[:, 1]))
    x0 = np.array([0, 0, 0, np.log(height / max(span, 1e-6)), 0, 0, 0], float)
    edges = {}
    for vn, view in views.items():
        fb = boundary(view.fg) & ~view.never
        ys, xs = np.nonzero(fb)
        edges[vn] = (dist_to(fb), sample(np.stack([xs, ys], 1), 200))

    def f(x):
        X = shell_place(Vs, x, R0)
        r = []
        for vn, view in views.items():
            V2, _ = RB.project(X, cams[vn])
            sil = RB.silhouette(V2, Fs, view.shape) & ~view.never
            sb = boundary(sil)
            ys, xs = np.nonzero(sb)
            pts = sample(np.stack([xs, ys], 1), 200)
            dt, e = edges[vn]
            r.append(np.full(200, 0.3) if pts is None else np.minimum(dt[pts[:, 1], pts[:, 0]], 0.3 * view.bl) / view.bl)
            r.append(np.minimum(dist_to(sb)[e[:, 1], e[:, 0]], 0.3 * view.bl) / view.bl)
        return np.concatenate(r)

    res = least_squares(f, x0, x_scale=np.array([5, 5, 5, 0.05, 0.3, 0.3, 0.3]), diff_step=0.02, max_nfev=80)
    return res.x, float(res.cost), R0


def shell_widths(X, B, H, allowance):
    """Barrel half-widths (fractions of H) of a placed shell at the shoulder, ribs and hip, minus an
    armour and cloth allowance: the prior TripoSG gives the hybrid."""
    out = {}
    stations = {"shoulder_width": B["HorseFrontLeftUp"][2] + 0.4, "rib_width": B["Saddle_Node"][2],
                "hip_width": B["HorseBackLeftUp"][2]}
    ylo = min(B["HorseFrontLeftUp"][1], B["HorseBackLeftUp"][1]) - 0.5
    for k, z in stations.items():
        s = X[(np.abs(X[:, 2] - z) < 0.3) & (X[:, 1] > ylo)]
        if len(s) > 20:
            half = 0.5 * float(np.percentile(s[:, 0], 98) - np.percentile(s[:, 0], 2))
            out[k] = max(0.05, half / H - allowance)
    return out


# ---------------------------------------------------------------- main

def preload(shells):
    """Import every library the run will need now: one loaded lazily mid-run comes off the AI drive,
    which may be gone by then."""
    import cv2  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.spatial  # noqa: F401
    import scipy.spatial.transform  # noqa: F401
    from PIL import Image  # noqa: F401

    import _ref3d_body  # noqa: F401
    import _ref3d_metrics  # noqa: F401
    import ref3d_ed2d  # noqa: F401
    if shells:
        import trimesh  # noqa: F401


def main(args_path):
    global WATCH
    import cv2
    from PIL import Image

    import _ref3d_body as RB

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    preload(a.get("shells"))
    WATCH = R.DriveWatch()
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ck = R.Checkpoints(out)
    logf = open(out / "fit.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    CAM_LIMITS.update({vn: {k: tuple(v) for k, v in lim.items()} for vn, lim in (a.get("cameras") or {}).items()})
    LM_WEIGHT.update(a.get("landmark_weights") or {})
    global HEAD_TURN
    HEAD_TURN = bool(a.get("head_turn"))
    HEAD_FIT.update(a.get("head_fit") or {})
    EVALS.update(a.get("evals") or {})
    views = {n: View(n, v, a["classes"], a) for n, v in a["views"].items()}
    defaults = {**RB.ha.DEFAULTS, "rider_clear": 99.0, "rider_clear_top": 99.0}  # the master isn't fitted round a game rider
    res = {"candidates": {}}
    t0 = time.time()

    def finish(name, m, hist, extra=None):
        shape, skel, cams, poses = m.unpack(m.x0)
        rec = {"history": hist, "shape": shape, "skeleton": skel, "cameras": cams, "poses": poses, "views": {}}
        for vn, view in views.items():
            if vn not in m.views:  # an independent fit of one view
                continue
            V, F, part, lm3, cam = m.geometry(m.x0, vn)
            V2, _ = RB.project(V, cam)
            sil = RB.silhouette(V2, F, view.shape)
            names = list(lm3)
            p2, _ = RB.project(np.array([lm3[k] for k in names]), cam)
            rec["views"][vn] = score_view(view, sil, dict(zip(names, p2)), a["gates"])
            img = RB.shaded(V, F, cam, view.shape, bg=(view.image * 0.45).astype(np.uint8))
            for k, p in zip(names, p2):
                if k in view.lm:
                    cv2.line(img, tuple(int(q) for q in p), tuple(int(q) for q in view.lm[k]), (255, 60, 60), 1)
                    cv2.circle(img, tuple(int(q) for q in view.lm[k]), 3, (255, 255, 0), -1)
            Image.fromarray(img).save(out / f"{name}_{vn}.png")
            rec["views"][vn]["render"] = str(out / f"{name}_{vn}.png")
            ov = view.image.copy()
            ov[boundary(cv2.dilate(sil.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0)] = (0, 255, 255)
            Image.fromarray(ov).save(out / f"{name}_{vn}_outline.png")
            rec["views"][vn]["outline"] = str(out / f"{name}_{vn}_outline.png")
        B, H = RB.skeleton(sk["bones"], sk["seat"], skel)
        Vr, Fr, _, _ = RB.build(RB.pose(B, H, shape, {}), H, shape)
        rec["vanilla_distance"] = vanilla_distance(Vr, sk)
        _write_obj(out / f"{name}_rest.obj", Vr, Fr)
        rec["mesh_rest"] = str(out / f"{name}_rest.obj")
        if extra:
            rec.update(extra)
        res["candidates"][name] = rec
        ck.save(f"cand_{name}", rec)
        log(f"{name}: " + json.dumps({vn: {"regions": r["regions"], "lm_worst": r["gates"]["landmarks"]["worst"],
                                           "outside": r["outside_share"]} for vn, r in rec["views"].items()}))

    def fitted(key, base_shape, free, priors, seed, vs=None):
        """One fit (of ``vs``, default every view), or its finished checkpoint. -> (cost, model, history)."""
        vs = vs or views
        st = ck.load(key)
        if st and st.get("done"):
            log(f"{key}: done earlier (checkpoint)")
            return st["cost"], restore(sk, vs, free, base_shape, priors, st), st["hist"]
        m, hist = fit_candidate(sk, vs, base_shape, free, priors, seed, log, ck, key)
        cost = float(m.residuals(m.x0).dot(m.residuals(m.x0)) / 2)
        ck.save(key, {"names": m.names, "x": m.x0, "hist": hist, "cost": cost, "done": True})
        return cost, m, hist

    def candidate_done(name):
        rec = ck.load(f"cand_{name}")
        if rec:
            res["candidates"][name] = rec
            log(f"{name}: scored earlier (checkpoint)")
        return rec is not None

    def run_independent(name, base_shape, priors):
        """SPEC S3's fallback (E-A round 10): each view fitted on its own (shape, camera, pose); the
        master body is L's, with R's rump (M10: L for head, neck, chest and forelegs, R for the
        hindquarters), scored in both views with each view's own camera and pose."""
        per = {}
        for vn in views:
            runs = []
            for k in range(a["restarts"]):
                log(f"{name} view {vn} restart {k}")
                runs.append(fitted(f"{name}_{vn}_r{k}", base_shape, True, priors, a["seed"] + 101 * k, {vn: views[vn]}))
            per[vn] = runs
            best = min(runs, key=lambda r: r[0])
            if not candidate_done(f"{name}_{vn}"):
                finish(f"{name}_{vn}", best[1], best[2], {"restarts": [{"cost": r[0], "history": r[2]} for r in runs]})
        bests = {vn: min(r, key=lambda q: q[0]) for vn, r in per.items()}
        mL = bests["L"][1]
        shape, skel, cams, poses = mL.unpack(mL.x0)
        if "R" in bests:
            sR, _, cR, pR = bests["R"][1].unpack(bests["R"][1].x0)
            shape = {**shape, **{k: sR[k] for k in RUMP}}
            cams, poses = {**cams, "R": cR["R"]}, {**poses, "R": pR["R"]}
        m = Model(sk, views, True, base_shape, priors,
                  {"cam": cams, "pose": poses, "shape": shape, "skel": skel, "hp": shape.get("head_profile")})
        if candidate_done(name):
            return m
        hist_c = []
        if "R" in bests:
            # R's camera and pose were solved for R's own body: re-solve them for the master body (E-A
            # pilot 10: reused as they were, R's legs scored 0.14)
            key = f"{name}_combined"
            st = ck.load(key)
            if st and st.get("done"):
                m.x0, hist_c = np.array(st["x"], float), st["hist"]
                log(f"{key}: done earlier (checkpoint)")
            else:
                r = powell(m, ("lm", "render", "prior"), sel(m, ["R.cam.", "R.pose."]), maxfev=EVALS["outlines: camera+pose"])
                _note(hist_c, log, "combined: R camera+pose for the master body", r)
                ck.save(key, {"names": m.names, "x": m.x0, "hist": hist_c, "done": True})
        n = min(len(r) for r in per.values())
        costs = [sum(per[vn][k][0] for vn in per) for k in range(n)]
        finish(name, m, bests["L"][2] + (bests["R"][2] if "R" in bests else []) + hist_c,
               {"restarts": [{"cost": c} for c in costs], "independent": True,
                "parts": {"L": "everything but the rump", "R": RUMP}})
        return m

    def run_fitted(name, base_shape, free, priors):
        if a.get("independent") and free:
            return run_independent(name, base_shape, priors)
        runs = []
        for k in range(a["restarts"]):
            log(f"{name} restart {k}")
            runs.append(fitted(f"{name}_r{k}", base_shape, free, priors, a["seed"] + 101 * k))
        runs.sort(key=lambda r: r[0])
        best = runs[0][1]
        if candidate_done(name):
            return best
        X = np.array([r[1].x0 for r in runs])
        spread = {n: float(np.ptp(X[:, i]) / max(s, 1e-9)) for i, (n, s) in enumerate(zip(best.names, best.scale))}
        finish(name, best, runs[0][2], {"restarts": [{"cost": r[0], "history": r[2]} for r in runs],
                                         "spread_in_scales": {k: round(v, 2) for k, v in sorted(spread.items(), key=lambda kv: -kv[1])[:15]}})
        return best

    b2 = {**defaults, **sk["b2_params"]}
    if "b2" in a["candidates"] and not candidate_done("b2"):
        log("b2 (round 5's B, shape frozen)")
        _, m, hist = fitted("b2_r0", b2, False, {}, a["seed"])
        finish("b2", m, hist)
    proc = None
    if "procedural" in a["candidates"]:
        proc = run_fitted("procedural", defaults, True, {})
    priors = {}
    if a.get("shells") and proc is not None:
        shape, skel, cams, poses = proc.unpack(proc.x0)
        B, H = RB.skeleton(sk["bones"], sk["seat"], skel)
        BL = RB.pose(B, H, shape, poses["L"])
        height = float(BL["HorseHead"][1] + 1.5)
        shells = {}
        for p in a["shells"]:
            nm = Path(p).stem
            rec = ck.load(f"shell_{nm}")
            if rec:
                shells[nm] = rec
                log(f"shell {nm}: placed earlier (checkpoint)")
                continue
            WATCH.check()
            Vs, Fs = load_shell(p)
            x, cost, R0 = fit_shell(Vs, Fs, views, cams, height)
            X = shell_place(Vs, x, R0)
            rec = {"path": p, "placement": x.tolist(), "cost": cost, "tris": int(len(Fs)), "views": {}}
            for vn, view in views.items():
                V2, _ = RB.project(X, cams[vn])
                sil = RB.silhouette(V2, Fs, view.shape)
                rec["views"][vn] = score_view(view, sil, None, a["gates"])
                img = RB.shaded(X, Fs, cams[vn], view.shape, colour=(150, 170, 190), bg=(view.image * 0.45).astype(np.uint8))
                Image.fromarray(img).save(out / f"shell_{nm}_{vn}.png")
                rec["views"][vn]["render"] = str(out / f"shell_{nm}_{vn}.png")
            rec["widths"] = shell_widths(X, BL, H, a["armour_allowance"])
            shells[nm] = rec
            ck.save(f"shell_{nm}", rec)
            log(f"shell {nm}: cost {cost:.4f} " + json.dumps({vn: {"horse": r["horse_iou"], "regions": r["regions"]} for vn, r in rec["views"].items()}))
        res["shells"] = shells
        best = min(shells, key=lambda n: shells[n]["cost"])
        res["best_shell"] = best
        priors = {k: (v, 0.1 * v, 0.05) for k, v in shells[best]["widths"].items()}
        res["hybrid_priors"] = {k: v[0] for k, v in priors.items()}
    if "hybrid" in a["candidates"] and priors:
        run_fitted("hybrid", defaults, True, priors)
    res["seconds"] = round(time.time() - t0, 1)  # this session's time only, when resumed
    text = json.dumps(res, default=float)
    (out / "result.json").write_text(text, encoding="utf-8")  # the project drive: kept if the AI drive goes now
    Path(a["result_path"]).write_text(text, encoding="utf-8")


def _write_obj(path, V, F):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Track M E-A candidate (our own anatomy; PDX mesh space)\n")
        f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as exc:
        # the drive can go between two checks: any failure while it's gone counts as the drive's
        if isinstance(exc, R.DriveLost) or not R.drive_ok():
            print(f"STOPPED: the AI drive dropped out ({exc!r}). Progress is checkpointed; rerun into the "
                  "same out_dir to resume.", flush=True)
            sys.exit(R.EXIT_DRIVE_LOST)
        raise
