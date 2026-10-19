"""Reshaping a generated horse onto the vanilla horse's rest pose (Track C, test 15; Blender jobs
only: numpy and mathutils). PDX mesh space: Y up, the horse facing -Z, its left side at +X.

The mesh is moved to fit the bones, never the bones to the mesh:
  * legs: each leg's centreline, tracked up from the hoof slice by slice inside its own cross
    section (skirts further out are ignored), maps by distance from the ground onto the vanilla
    leg chain (ground under the heel -> Heel -> Knee -> Down -> Up): bent or striding legs come
    out straight, at their own length;
  * head and neck: the same feature finder on both meshes (the head's principal axis, its back
    third's centre = the poll region, the neck centreline): our head is turned and moved onto
    vanilla's, the neck maps by fraction of its length;
  * torso: anchors that stay put.
The landmark pairs drive a 3D biharmonic spline (phi(r) = r plus an affine part), applied to every
vertex. Diagnostics (landmarks, per-leg straightening) come back for the job's result.
"""
import math

import numpy as np

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")


def verts(obj) -> np.ndarray:
    a = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def set_verts(obj, co: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
    obj.data.update()


def chain(bones: dict, names: list[str]) -> np.ndarray:
    return np.array([bones[n] for n in names], dtype=np.float64)


def leg_chain(bones: dict, leg: str) -> np.ndarray:
    """Vanilla leg from the ground under the heel up to the leg's top joint."""
    heel = np.array(bones[f"Horse{leg}Heel"], dtype=np.float64)
    ground = np.array([heel[0], 0.0, heel[2]])
    return np.vstack([ground, chain(bones, [f"Horse{leg}{p}" for p in ("Heel", "Knee", "Down", "Up")])])


def along(poly: np.ndarray, s: float) -> np.ndarray:
    """The point at arc length ``s`` along ``poly`` (clamped; past the end it continues straight)."""
    seg = np.diff(poly, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    if s <= 0:
        return poly[0].copy()
    for p, d, l in zip(poly[:-1], seg, lens):
        if s <= l:
            return p + d * (s / l)
        s -= l
    return poly[-1] + seg[-1] / max(lens[-1], 1e-9) * s


def arc(poly: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def track_leg(co: np.ndarray, leg: str, mid_z: float, top_y: float, step: float = 0.35, radius: float = 0.8):
    """The leg's centreline from its hoof up to ``top_y``: per slice of height ``step``, the
    centre of the vertices within ``radius`` (in x, z) of the slice below's centre. Stops where
    the cross-section gets wider than a leg (a skirt or the body). -> (points, why it stopped)."""
    left, front = leg.endswith("Left"), leg.startswith("Front")
    side = co[((co[:, 0] > 0) == left) & ((co[:, 2] < mid_z) == front)]
    if not len(side):
        return np.zeros((0, 3)), "no vertices"
    floor = side[:, 1].min()
    low = side[side[:, 1] < floor + 0.5]
    centre = low[:, [0, 2]].mean(axis=0)
    pts = [np.array([centre[0], floor, centre[1]])]
    y = floor + step
    while y < top_y:
        sl = side[(side[:, 1] >= y - step / 2) & (side[:, 1] < y + step / 2)]
        near = sl[np.linalg.norm(sl[:, [0, 2]] - centre, axis=1) < radius]
        if len(near) < 4:
            return np.array(pts), f"lost at y {y:.2f}"
        spread = np.linalg.norm(near[:, [0, 2]] - near[:, [0, 2]].mean(axis=0), axis=1).max()
        if spread > radius * 0.98 and y > floor + 2.0:
            return np.array(pts), f"widens at y {y:.2f}"
        centre = near[:, [0, 2]].mean(axis=0)
        pts.append(np.array([centre[0], y, centre[1]]))
        y += step
    return np.array(pts), "reached the top"


def head_features(co: np.ndarray, neck_base: np.ndarray, depth: float = 3.5):
    """(head centre, head axis pointing to the muzzle, poll centre, neck centreline) of a horse
    mesh: the head = the front ``depth`` units above y 8; the axis = its principal direction,
    turned to point forward; the poll = the centre of the head's back third along that axis; the
    neck = per 0.5 along z from the neck base to the poll, the centre of the upper cross-section."""
    up = co[co[:, 1] > 8.0]
    front = up[:, 2].min()
    head = up[up[:, 2] < front + depth]
    centre = head.mean(axis=0)
    _, _, vt = np.linalg.svd(head - centre, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    if axis[2] > 0:
        axis = -axis
    t = (head - centre) @ axis
    poll = head[t < t.min() + (t.max() - t.min()) / 3].mean(axis=0)
    neck = []
    for z in np.arange(neck_base[2], poll[2], -0.5):
        sl = co[(np.abs(co[:, 2] - z) < 0.25) & (co[:, 1] > neck_base[1] - 2.5) & (np.abs(co[:, 0]) < 1.6)]
        if len(sl):
            top = sl[sl[:, 1] > sl[:, 1].max() - 2.0]
            neck.append(top.mean(axis=0))
    return centre, axis, poll, np.array(neck)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 3x3 rotation turning unit vector ``a`` onto unit vector ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def spline(src: np.ndarray, dst: np.ndarray):
    """3D biharmonic spline (phi(r) = r) with an affine part, mapping ``src`` onto ``dst``."""
    n = len(src)
    k = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    p = np.hstack([np.ones((n, 1)), src])
    a = np.zeros((n + 4, n + 4))
    a[:n, :n], a[:n, n:], a[n:, :n] = k + np.eye(n) * 1e-6, p, p.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = dst
    sol = np.linalg.solve(a, rhs)
    w, aff = sol[:n], sol[n:]

    def apply(x: np.ndarray, chunk: int = 40000) -> np.ndarray:
        out = np.empty_like(x)
        for i in range(0, len(x), chunk):
            c = x[i:i + chunk]
            out[i:i + chunk] = np.linalg.norm(c[:, None, :] - src[None, :, :], axis=2) @ w + \
                np.hstack([np.ones((len(c), 1)), c]) @ aff
        return out
    return apply


def reshape(obj, vanilla, bones: dict, leg_top: float = 6.0, anchors: int = 60, seed: int = 0) -> dict:
    """Warp ``obj`` onto the vanilla rest pose (module doc). -> diagnostics."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    co, vco = verts(obj), verts(vanilla)
    mid_z = (B["HorseFrontLeftUp"][2] + B["HorseBackLeftUp"][2]) / 2
    src, dst, log = [], [], {"legs": {}}
    for leg in LEGS:
        pts, why = track_leg(co, leg, mid_z, leg_top)
        target = leg_chain(B, leg)
        if len(pts) < 3:
            log["legs"][leg] = {"tracked": int(len(pts)), "stopped": why}
            continue
        s = arc(pts)
        for i in range(0, len(pts), 2):
            src.append(pts[i])
            dst.append(along(target, s[i]))
        tip = pts[0]
        log["legs"][leg] = {"tracked": int(len(pts)), "stopped": why, "hoof": tip.round(2).tolist(),
                            "hoof_to": along(target, 0).round(2).tolist(), "top_y": float(pts[-1][1]),
                            "max_move": float(max(np.linalg.norm(along(target, s[i]) - pts[i]) for i in range(len(pts))))}
    oc, oaxis, opoll, oneck = head_features(co, B["HorseNeckBase"])
    vc, vaxis, vpoll, vneck = head_features(vco, B["HorseNeckBase"])
    r = rotation_between(oaxis, vaxis)
    head = co[(co[:, 1] > 8.0) & (co[:, 2] < co[co[:, 1] > 8.0][:, 2].min() + 3.5)]
    rng = np.random.default_rng(seed)
    for p in head[rng.choice(len(head), size=min(12, len(head)), replace=False)]:
        src.append(p)
        dst.append(vpoll + r @ (p - opoll))
    log["head"] = {"turn_deg": float(math.degrees(math.acos(max(-1, min(1, float(oaxis @ vaxis)))))),
                   "poll": opoll.round(2).tolist(), "poll_to": vpoll.round(2).tolist()}
    if len(oneck) >= 2 and len(vneck) >= 2:
        os_, vs_ = arc(oneck), arc(vneck)
        for f in (0.25, 0.5, 0.75):
            src.append(along(oneck, f * os_[-1]))
            dst.append(along(vneck, f * vs_[-1]))
    body = co[(co[:, 2] > B["HorseNeckBase"][2] + 0.5) & (co[:, 2] < B["HorseSpineEnd"][2] + 2.0) & (co[:, 1] > leg_top + 1.5)]
    for p in body[rng.choice(len(body), size=min(anchors, len(body)), replace=False)]:
        src.append(p)
        dst.append(p)
    warp = spline(np.array(src), np.array(dst))
    new = warp(co)
    set_verts(obj, new)
    log["landmarks"] = len(src)
    log["max_vertex_move"] = float(np.linalg.norm(new - co, axis=1).max())
    log["head_after"] = {"poll": head_features(new, B["HorseNeckBase"])[2].round(2).tolist()}
    return log


def flatten_lump(obj, z0: float, z1: float, y0: float, y1: float, margin: float = 0.3) -> int:
    """Vertices in the box on either flank sticking out more than ``margin`` past the other
    flank's width at the same (y, z) (a rider's leg on one side) are pushed in to that width.
    No vertex is deleted, so no hole opens. -> how many moved."""
    co = verts(obj)
    box = np.nonzero((co[:, 2] >= z0) & (co[:, 2] <= z1) & (co[:, 1] >= y0) & (co[:, 1] <= y1))[0]
    keys = np.stack([np.floor(co[:, 1] / 0.5), np.floor(co[:, 2] / 0.5)], axis=1).astype(int)
    width = {}
    for i in box:
        k = (keys[i, 0], keys[i, 1], bool(co[i, 0] > 0))
        width[k] = max(width.get(k, 0.0), abs(co[i, 0]))
    moved = 0
    for i in box:
        other = width.get((keys[i, 0], keys[i, 1], not bool(co[i, 0] > 0)))
        if other is not None and abs(co[i, 0]) > other + margin:
            co[i, 0] = math.copysign(other, co[i, 0])
            moved += 1
    set_verts(obj, co)
    return moved


def nearest_on_chains(co: np.ndarray, chains: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(distance, nearest point) of each vertex to the polylines."""
    best = np.full(len(co), np.inf)
    point = np.zeros_like(co)
    for poly in chains:
        for a, b in zip(poly[:-1], poly[1:]):
            d = b - a
            t = np.clip(((co - a) @ d) / max(d @ d, 1e-12), 0, 1)
            q = a + t[:, None] * d
            dist = np.linalg.norm(co - q, axis=1)
            closer = dist < best
            best[closer], point[closer] = dist[closer], q[closer]
    return best, point


def near_chains(co: np.ndarray, chains: list[np.ndarray], radius: float) -> np.ndarray:
    """Mask: vertices within ``radius`` of any of the polylines."""
    return nearest_on_chains(co, chains)[0] < radius


def bone_segments(bones: dict) -> dict:
    """Each deforming bone's segment (from the bone to its child; the chain ends extended: the
    hoof to the ground, the head to the muzzle, the tail end past its tip)."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = {}
    for leg in LEGS:
        names = [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")]
        for a, b in zip(names[:-1], names[1:]):
            seg[a] = (B[a], B[b])
        heel = B[names[-1]]
        seg[names[-1]] = (heel, np.array([heel[0], 0.0, heel[2]]))
    neck = ["HorseNeckBase", "HorseNeckMiddle", "HorseHead"]
    for a, b in zip(neck[:-1], neck[1:]):
        seg[a] = (B[a], B[b])
    head = B["HorseHead"]
    seg["HorseHead"] = (head, head + np.array([0.0, -1.5, -3.3]))  # towards vanilla's muzzle
    tail = ["HorseSpineEnd", "HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd"]
    for a, b in zip(tail[1:-1], tail[2:]):
        seg[a] = (B[a], B[b])
    end, tip = B["HorseTailEnd"], B["HorseTailTip"]
    seg["HorseTailEnd"] = (end, end + (end - tip) * 0.5)
    return seg


def segment_distance(co: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((co - a) @ d) / max(d @ d, 1e-12), 0, 1)
    return np.linalg.norm(co - (a + t[:, None] * d), axis=1)


def chain_weights(co: np.ndarray, bones: dict, power: float = 4.0) -> tuple[list, dict]:
    """Per vertex: weights by distance to the bone chains (head and neck, each leg, the tail) in
    their regions, else None (the torso keeps nearest-surface weights). -> (rows, region counts).
    Each region includes its spine parent along the segment to the chain's first joint, so the
    weights blend into the torso instead of stepping at the region's border. Regions: head/neck
    in front of 0.8 behind the neck base, above y 9; a leg below y 9.3 within 0.9 of its chain
    (1.4 above y 7.5);
    the tail on the centre line (|x| < 0.7) behind the rump within 1.0 of its chain."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = bone_segments(bones)
    regions = {"head": ("HorseSpineTop", "HorseNeckBase", ["HorseNeckBase", "HorseNeckMiddle", "HorseHead"]),
               "tail": ("HorseSpineEnd", "HorseTailTop", ["HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd"])}
    for leg in LEGS:
        parent = "HorseSpineTop" if leg.startswith("Front") else "HorseSpineEnd"
        regions[leg] = (parent, f"Horse{leg}Up", [f"Horse{leg}{p}" for p in ("Up", "Down", "Knee", "Heel")])
    rows, counts = [None] * len(co), {}
    for region, (parent, first, names) in regions.items():
        dist = {b: segment_distance(co, *seg[b]) for b in names}
        dist[parent] = segment_distance(co, B[parent], B[first])
        chain_min = np.min([dist[b] for b in names], axis=0)
        if region == "head":
            mask = (co[:, 2] < B["HorseNeckBase"][2] + 0.8) & (co[:, 1] > 9.0)
        elif region == "tail":
            mask = (np.abs(co[:, 0]) < 0.7) & (co[:, 2] > B["HorseSpineEnd"][2] + 1.5) & (chain_min < 1.0)
        else:  # the leg itself below the elbow or stifle, the shoulder or hip above it
            mask = (co[:, 1] < 9.3) & (chain_min < np.where(co[:, 1] < 7.5, 0.9, 1.4))
        idx = np.nonzero(mask)[0]
        counts[region] = int(len(idx))
        if not len(idx):
            continue
        order = [parent] + names
        d = np.stack([dist[b][idx] for b in order], axis=1)
        w = 1.0 / np.maximum(d, 0.05) ** power
        for row, i in zip(w, idx):
            if rows[i] is None:  # first region wins
                top = np.argsort(-row)[:3]
                total = row[top].sum()
                rows[i] = {order[k]: float(row[k] / total) for k in top}
    return rows, counts


TAIL = ("HorseTailTop", "HorseTailMiddle", "HorseTailTip", "HorseTailEnd")


def clean_surface_rows(co: np.ndarray, rows: list, chain_rows: list, bones: dict, far: float = 1.0) -> int:
    """Nearest-surface weights of the torso (vertices without chain weights) cleaned of bones
    that only fit vanilla's own shape: the tail (our rump sits where vanilla's tail is: the
    caparison follows the hindquarters), the reins (``Rem``) and a leg's lower bones on vertices
    farther than ``far`` from that leg's chain (a skirt doesn't swing with the leg). Emptied
    rows fall back to the nearest spine bone. -> how many rows changed. In place."""
    B = {k: np.array(v, dtype=np.float64) for k, v in bones.items()}
    seg = bone_segments(bones)
    lower = {}
    for leg in LEGS:
        d = np.min([segment_distance(co, *seg[f"Horse{leg}{p}"]) for p in ("Down", "Knee", "Heel")], axis=0)
        for p in ("Down", "Knee", "Heel"):
            lower[f"Horse{leg}{p}"] = d > far
    spine = ("HorseSpineTop", "HorseSpineEnd")
    changed = 0
    for i, row in enumerate(rows):
        if chain_rows[i] is not None or not row:
            continue
        keep = {b: w for b, w in row.items()
                if b not in TAIL and b != "Rem" and not (b in lower and lower[b][i])}
        if len(keep) != len(row):
            changed += 1
            if not keep:
                near = min(spine, key=lambda b: float(np.linalg.norm(co[i] - B[b])))
                keep = {near: 1.0}
            total = sum(keep.values())
            rows[i] = {b: w / total for b, w in keep.items()}
    return changed
