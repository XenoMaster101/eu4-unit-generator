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
spread reported. Residuals (fractions of body length): body landmarks; the outline chamfer both ways
inside each view's zones (legs; the head in L), where pixels of an occluding class (fringe,
tassels, plume, straps) or inpainted ones are unknown, not scored; containment (the bare body
inside the horse mask: armour and cloth lie over it); weak priors.

args: {"skeleton": json path, "views": {name: {"image", "labels", "fg", "inpainted", "never_use",
       "landmarks", "body_length", "zones": {"legs": [poly], "head": [poly]}}}, "classes": {name: rgb},
       "occluders": {zone: [class]}, "body": {zone: [class]}, "candidates": [...], "shells": [glb],
       "restarts", "seed", "label_scale", "gates", "armour_allowance", "out_dir", "result_path"}
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SHAPE = ["barrel_depth", "withers_rise", "croup_rise", "rump_drop", "breast_ahead", "rump_behind", "breast_width",
         "shoulder_width", "rib_width", "hip_width", "buttock_width", "neck_base_depth", "neck_throat_depth",
         "neck_width", "crest", "head_length", "jowl_depth", "forehead_width", "muzzle_depth", "muzzle_width",
         "ear_length", "upper_arm", "forearm", "knee", "stifle", "gaskin", "hock", "cannon", "fetlock", "pastern",
         "hoof_top", "hoof_bottom", "hoof_height"]
SIGNED = {"withers_rise", "croup_rise"}  # can cross zero: absolute scale and bounds
SKEL = ["leg_f", "leg_b", "body", "neck"]
CAM = ["az", "el", "roll", "dist", "focal", "cx", "cy"]
CAM_SCALE = {"az": 5, "el": 5, "roll": 3, "dist": 10, "focal": 300, "cx": 20, "cy": 20}
CAM_BOUNDS = {"az": (-180, 180), "el": (-25, 45), "roll": (-15, 15), "dist": (20, 400), "focal": (200, 20000),
              "cx": (-2000, 4000), "cy": (-2000, 4000)}
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
                             "known_ss": up(known), "inner_ss": up(inner),
                             "dt_target_edge": dist_to(tb), "edge_samples": sample(np.stack([xs, ys], 1), N_ZONE)}
        self.dt_out = dist_to(cv2.dilate(self.fg.astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8)).astype(bool))
        self.outside_ss = up((self.dt_out > 0) & ~self.never)


def lm_residuals(view, lm2):
    import _ref3d_body as RB
    r = []
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
            done |= {a_, b_}
        else:
            r.append(lm2[k] - view.lm[k])
            done.add(k)
    return W_LM * np.concatenate(r) / view.bl if r else np.zeros(0)


CELL = 12          # px: region fill and containment are compared per cell
SS = 3             # the fit renders outlines at 3x: at 1x an edge's place is known to +-0.5 px, which
                   # swamped the finite differences of 1-5 px moves (E-A pilot, Jacobian check)
W_LM, W_AREA, W_EDGE, W_CONT = 5.0, 0.5, 1.0, 0.5


def cells(m, c=CELL):
    """Mean of a mask over c x c cells (padded) -> (H/c, W/c) floats."""
    H, W = m.shape
    ph, pw = -H % c, -W % c
    p = np.pad(m.astype(np.float32), ((0, ph), (0, pw)))
    return p.reshape(p.shape[0] // c, c, p.shape[1] // c, c).mean(axis=(1, 3))


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
        rm = sil & Z["known_ss"]
        r.append(W_AREA * (cells(rm, CELL * SS) - Z["target_cells"])[Z["cell_sel"]])
        if Z["edge_samples"] is not None:
            d = dist_to(boundary(rm) & Z["inner_ss"])
            e = Z["edge_samples"] * SS + SS // 2
            r.append(W_EDGE * np.minimum(d[e[:, 1], e[:, 0]] / SS, 0.3 * view.bl) / view.bl)
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
                else:
                    add(f"shape.{p}", v0, 0.15 * abs(base_shape[p]), 0.5 * base_shape[p], 2.0 * base_shape[p])
            for k in SKEL:
                add(f"skel.{k}", init["skel"].get(k, 1.0), 0.05, 0.75, 1.35)
        for vn in views:
            for c in CAM:
                add(f"{vn}.cam.{c}", init["cam"][vn][c], CAM_SCALE[c], *CAM_BOUNDS[c])
            for leg in LEGS:
                for j in range(4):
                    lo, hi = POSE_BOUNDS["Front" if leg.startswith("Front") else "Back"][j]
                    add(f"{vn}.pose.{leg}.{j}", init["pose"][vn].get(leg, [0, 0, 0, 0])[j], 8, lo, hi)
            for j in range(2):
                add(f"{vn}.pose.neck.{j}", init["pose"][vn].get("neck", [0, 0])[j], 8, -45, 45)
            add(f"{vn}.pose.head_pitch", init["pose"][vn].get("head_pitch", base_shape["head_pitch"]), 8, 0, 95)
        self.x0, self.scale = np.array(self.x0, float), np.array(self.scale, float)
        self.lo, self.hi = np.array(self.lo, float), np.array(self.hi, float)
        self.x0 = np.clip(self.x0, self.lo + 1e-9, self.hi - 1e-9)

    def unpack(self, x):
        d = dict(zip(self.names, x))
        shape = dict(self.base)
        skel = {}
        for k, v in d.items():
            if k.startswith("shape."):
                shape[k[6:]] = v
            elif k.startswith("skel."):
                skel[k[5:]] = v
        cams, poses = {}, {}
        for vn in self.views:
            cams[vn] = {c: d[f"{vn}.cam.{c}"] for c in CAM}
            poses[vn] = {leg: [d[f"{vn}.pose.{leg}.{j}"] for j in range(4)] for leg in LEGS}
            poses[vn]["neck"] = [d[f"{vn}.pose.neck.{j}"] for j in range(2)]
            poses[vn]["head_pitch"] = d[f"{vn}.pose.head_pitch"]
        return shape, skel, cams, poses

    def geometry(self, x, vn):
        import _ref3d_body as RB
        shape, skel, cams, poses = self.unpack(x)
        B, H = RB.skeleton(self.sk["bones"], self.sk["seat"], skel)
        P = {**shape, "head_pitch": poses[vn]["head_pitch"]}
        Bp = RB.pose(B, H, P, poses[vn])
        V, F, part, P = RB.build(Bp, H, P)
        return V, F, part, RB.landmarks(Bp, H, P), cams[vn]

    def residuals(self, x, which=("lm", "render", "prior")):
        import _ref3d_body as RB
        r = []
        for vn, view in self.views.items():
            V, F, part, lm3, cam = self.geometry(x, vn)
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
            elif kind == "pose":
                out.append(0.004 * v / 30.0 if not name.endswith("head_pitch") else 0.0)
        return np.array(out)


def solve(model, which, mask=None, max_nfev=60, seed=0):
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

    res = least_squares(f, np.zeros(len(x0)), bounds=(lo, hi), diff_step=0.15, max_nfev=max_nfev, method="trf")
    x[sel] = x0 + res.x * sc
    model.x0 = x
    return res


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
        starts = (20, 35, 50) if vn == "L" else (-35, -50, -65)
        best = None
        for az in starts:
            ys, xs = np.nonzero(view.fg)
            c0 = {"az": az + rng.normal(0, 3), "el": 8 + rng.normal(0, 3), "roll": 0.0, "dist": 60.0,
                  "focal": 60.0 * (ys.max() - ys.min()) / 14.0, "cx": float(xs.mean()), "cy": float(ys.mean())}
            m = Model(sk, {vn: view}, False, shape, {}, {"cam": {vn: c0}, "pose": {vn: {}}, "shape": {}, "skel": {}})
            r = solve(m, ("lm",), sel(m, [f"{vn}.cam."]), max_nfev=200)
            if best is None or r.cost < best[0]:
                best = (r.cost, m.unpack(m.x0)[2][vn])
        out[vn] = best[1]
    return out


def fit_candidate(sk, views, base_shape, free_shape, priors, seed, log):
    """Cameras, then camera + pose, then (free shape) everything. -> (model, history)."""
    rng = np.random.default_rng(seed)
    cams = init_cameras(sk, views, base_shape, seed)
    for c in cams.values():  # restarts differ: perturbed starts
        c["az"] += rng.normal(0, 4)
        c["el"] += rng.normal(0, 3)
    poses = {vn: {leg: list(rng.normal(0, 4, 4)) for leg in LEGS} for vn in views}
    m = Model(sk, views, free_shape, base_shape, priors, {"cam": cams, "pose": poses, "shape": {}, "skel": {}})
    hist = []
    vp = [f"{vn}.cam." for vn in views] + [f"{vn}.pose." for vn in views]
    r = solve(m, ("lm", "render", "prior"), sel(m, vp), max_nfev=40)
    hist.append({"stage": "camera+pose", "cost": float(r.cost), "nfev": int(r.nfev), "njev": int(r.njev or 0), "status": int(r.status), "message": r.message})
    log(f"  camera+pose cost {r.cost:.4f} nfev {r.nfev} njev {r.njev} {r.message}")
    if free_shape:
        r = solve(m, ("lm", "render", "prior"), sel(m, ["shape.", "skel."]), max_nfev=40)
        hist.append({"stage": "shape", "cost": float(r.cost), "nfev": int(r.nfev), "njev": int(r.njev or 0), "status": int(r.status), "message": r.message})
        log(f"  shape cost {r.cost:.4f} nfev {r.nfev} njev {r.njev} {r.message}")
        r = solve(m, ("lm", "render", "prior"), None, max_nfev=30)
        hist.append({"stage": "joint", "cost": float(r.cost), "nfev": int(r.nfev), "njev": int(r.njev or 0), "status": int(r.status), "message": r.message})
        log(f"  joint cost {r.cost:.4f} nfev {r.nfev} njev {r.njev} {r.message}")
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
    lms = list(out["landmarks"].values())
    out["gates"] = {"region_iou": {"pass": bool(reg) and min(reg) >= gates["region_iou_min"], "worst": min(reg) if reg else None},
                    "landmarks": {"pass": bool(lms) and max(lms) <= gates["landmark_max"], "worst": max(lms) if lms else None}}
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


def shell_place(Vs, x, cam_L):
    """Shell (TripoSG frame: x right, y up, z towards the viewer of the input view) -> PDX space.
    x: [rx, ry, rz (deg, small extra turn), log scale, tx, ty, tz]; the start puts the shell's frame
    on camera L's axes (the shell was made from view L)."""
    import _ref3d_body as RB
    from scipy.spatial.transform import Rotation
    R, _ = RB.camera_matrix(cam_L)
    base = R.T @ np.diag([1.0, 1.0, -1.0])  # shell axes -> camera L axes (right-handed frame)
    extra = Rotation.from_euler("xyz", x[:3], degrees=True).as_matrix()
    c = Vs.mean(axis=0)
    Xrh = RB.TARGET * np.array([-1.0, 1.0, 1.0]) + np.exp(x[3]) * ((Vs - c) @ (base @ extra).T) + x[4:7]
    return Xrh * np.array([-1.0, 1.0, 1.0])  # back to PDX (mirrored x)


def fit_shell(Vs, Fs, views, cams, height):
    """Similarity placement of a shell against the horse masks of both views (outline chamfer)."""
    import _ref3d_body as RB
    from scipy.optimize import least_squares
    span = float(np.ptp(Vs[:, 1]))
    x0 = np.array([0, 0, 0, np.log(height / max(span, 1e-6)), 0, 0, 0], float)
    edges = {}
    for vn, view in views.items():
        fb = boundary(view.fg) & ~view.never
        ys, xs = np.nonzero(fb)
        edges[vn] = (dist_to(fb), sample(np.stack([xs, ys], 1), 200))

    def f(x):
        X = shell_place(Vs, x, cams["L"])
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

    res = least_squares(f, x0, x_scale=np.array([5, 5, 5, 0.05, 0.3, 0.3, 0.3]), diff_step=1e-3, max_nfev=80)
    return res.x, float(res.cost)


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

def main(args_path):
    import cv2
    from PIL import Image

    import _ref3d_body as RB

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    sk = json.loads(Path(a["skeleton"]).read_text(encoding="utf-8"))
    views = {n: View(n, v, a["classes"], a) for n, v in a["views"].items()}
    defaults = {**RB.ha.DEFAULTS, "rider_clear": 99.0, "rider_clear_top": 99.0}  # the master isn't fitted round a game rider
    res = {"candidates": {}}
    t0 = time.time()

    def finish(name, m, hist, extra=None):
        shape, skel, cams, poses = m.unpack(m.x0)
        rec = {"history": hist, "shape": shape, "skeleton": skel, "cameras": cams, "poses": poses, "views": {}}
        for vn, view in views.items():
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
        log(f"{name}: " + json.dumps({vn: {"regions": r["regions"], "lm_worst": r["gates"]["landmarks"]["worst"],
                                           "outside": r["outside_share"]} for vn, r in rec["views"].items()}))

    def run_fitted(name, base_shape, free, priors):
        runs = []
        for k in range(a["restarts"]):
            log(f"{name} restart {k}")
            m, hist = fit_candidate(sk, views, base_shape, free, priors, a["seed"] + 101 * k, log)
            runs.append((float(m.residuals(m.x0).dot(m.residuals(m.x0)) / 2), m, hist))
        runs.sort(key=lambda r: r[0])
        best = runs[0][1]
        X = np.array([r[1].x0 for r in runs])
        spread = {n: float(np.ptp(X[:, i]) / max(s, 1e-9)) for i, (n, s) in enumerate(zip(best.names, best.scale))}
        finish(name, best, runs[0][2], {"restarts": [{"cost": r[0], "history": r[2]} for r in runs],
                                         "spread_in_scales": {k: round(v, 2) for k, v in sorted(spread.items(), key=lambda kv: -kv[1])[:15]}})
        return best

    b2 = {**defaults, **sk["b2_params"]}
    if "b2" in a["candidates"]:
        log("b2 (round 5's B, shape frozen)")
        m, hist = fit_candidate(sk, views, b2, False, {}, a["seed"], log)
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
            Vs, Fs = load_shell(p)
            x, cost = fit_shell(Vs, Fs, views, cams, height)
            X = shell_place(Vs, x, cams["L"])
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
            log(f"shell {nm}: cost {cost:.4f} " + json.dumps({vn: {"horse": r["horse_iou"], "regions": r["regions"]} for vn, r in rec["views"].items()}))
        res["shells"] = shells
        best = min(shells, key=lambda n: shells[n]["cost"])
        res["best_shell"] = best
        priors = {k: (v, 0.1 * v, 0.05) for k, v in shells[best]["widths"].items()}
        res["hybrid_priors"] = {k: v[0] for k, v in priors.items()}
    if "hybrid" in a["candidates"] and priors:
        run_fitted("hybrid", defaults, True, priors)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a["result_path"]).write_text(json.dumps(res, default=float), encoding="utf-8")


def _write_obj(path, V, F):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Track M E-A candidate (our own anatomy; PDX mesh space)\n")
        f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)


if __name__ == "__main__":
    main(sys.argv[1])
