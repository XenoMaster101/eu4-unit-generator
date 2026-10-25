"""The reins, rebuilt (Track P1: thin parts kept or rebuilt instead of lost to the triangle budget; numpy only:
fresh_horse and tests/blender_rules_check.py import it).

TRELLIS.2 models the reins (horse 2's reference shows a bridle with reins), but a strap a few hundredths thick
doesn't survive the closed solid and ~2,900 triangles: they were lost (M22's known limits). Rebuilt: a flat strap
from each side of the bit to vanilla's ``Rem`` bone (Swedish for rein: the horse's skeleton carries one at the
withers, where the rider's hands hold them), sagging a little, a thin three-sided prism (``segs`` segments), coloured
from the raw model's own reins (bake_colour ``part_colours``), weighted along head -> neck -> Rem
(``weights``). Built only when the raw model shows reins (``found``: its points along that path that the low mesh
lost).
"""
import numpy as np

REINS = 7  # the barding pieces' part id (after _horse_barding's TAIL = 6)


def bit_points(head_pts: np.ndarray, poll, axis, frac: float = 0.85, radial: float = 1.3):
    """The bit's two ends: at ``frac`` of the head's length from the poll along its axis, the sides of the head's
    own cross-section there (points within ``radial`` of the axis: the bridle's straps and the chest stay out; the
    90th percentile of x either side) at the mouth's height (its lower third). -> (left, right, the head's length)."""
    poll, axis = np.asarray(poll, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    rel = head_pts - poll
    t = rel @ axis
    near = np.linalg.norm(rel - t[:, None] * axis, axis=1) < radial
    pts, t = head_pts[near], t[near]
    L = float(np.percentile(t, 99))
    st = pts[np.abs(t - frac * L) < max(0.08 * L, 0.2)]
    if len(st) < 8:
        st = pts[np.argsort(np.abs(t - frac * L))[:20]]
    y = float(np.percentile(st[:, 1], 33))
    c = st.mean(0)
    xl = float(np.percentile(st[st[:, 0] >= c[0], 0], 90)) if (st[:, 0] >= c[0]).any() else c[0] + 0.3
    xr = float(np.percentile(st[st[:, 0] <= c[0], 0], 10)) if (st[:, 0] <= c[0]).any() else c[0] - 0.3
    return np.array([xl + 0.03, y, c[2]]), np.array([xr - 0.03, y, c[2]]), L


def path(bit, rem, sag: float, segs: int) -> np.ndarray:
    """Points from the bit to the hand end (``rem``), sagging ``sag`` at the middle (a parabola)."""
    bit, rem = np.asarray(bit, dtype=np.float64), np.asarray(rem, dtype=np.float64)
    t = np.linspace(0.0, 1.0, segs + 1)
    pts = bit[None] + (rem - bit)[None] * t[:, None]
    pts[:, 1] -= sag * 4 * t * (1 - t)
    return pts


def strap(pts: np.ndarray, width: float = 0.11, thick: float = 0.035):
    """A three-sided prism along ``pts`` (flat side out, an edge in), capped. -> (vertices, faces, t along 0-1)."""
    n = len(pts)
    v, f, tt = [], [], []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        d /= max(np.linalg.norm(d), 1e-9)
        side = np.cross(d, [0.0, 1.0, 0.0])
        side /= max(np.linalg.norm(side), 1e-9)
        up = np.cross(side, d)
        for o in (side * width / 2, -side * width / 2, -up * thick):
            v.append(pts[i] + o)
            tt.append(i / (n - 1))
    for i in range(n - 1):
        a, b = 3 * i, 3 * (i + 1)
        for j in range(3):
            k = (j + 1) % 3
            f.append([a + j, b + j, b + k])
            f.append([a + j, b + k, a + k])
    f.append([0, 2, 1])
    e = 3 * (n - 1)
    f.append([e, e + 1, e + 2])
    return np.array(v), np.array(f), np.array(tt)


def build(head_pts, poll, axis, rem, sag=0.35, half_gap=0.3, segs=6):
    """Both reins as one barding piece: {"v", "f", "part", "hem", "anchor", "chain" (t, 0 at the bit), "log"}."""
    bl, br, L = bit_points(head_pts, poll, axis)
    rem = np.asarray(rem, dtype=np.float64)
    vs, fs, ts = [], [], []
    for bit, sx in ((bl, 1.0), (br, -1.0)):
        end = rem + np.array([sx * half_gap, 0.0, 0.0])
        v, f, t = strap(path(bit, end, sag, segs))
        fs.append(f + sum(len(x) for x in vs))
        vs.append(v)
        ts.append(t)
    v = np.vstack(vs)
    n = len(np.vstack(fs))
    return {"v": v, "f": np.vstack(fs), "part": [REINS] * n, "hem": [0.0] * n, "anchor": v.copy(),
            "chain": np.concatenate(ts), "log": {"bit_left": bl.round(3).tolist(), "bit_right": br.round(3).tolist(),
                                                 "head_length": round(L, 3), "tris": n}}


def weights(t: float) -> dict:
    """Bone weights along a rein (``t`` 0 at the bit, 1 at the hands): the bit end on the head, then blending
    through the neck's middle onto Rem."""
    keys = [(0.0, "HorseHead"), (0.15, "HorseHead"), (0.5, "HorseNeckMiddle"), (0.85, "Rem"), (1.0, "Rem")]
    for (t0, b0), (t1, b1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            out = {b0: 1.0 - f}
            out[b1] = out.get(b1, 0.0) + f
            return {k: w for k, w in out.items() if w > 1e-3}
    return {"Rem": 1.0}


def found(raw_pts, raw_rgb, low_distance, bit_l, bit_r, rem, half_gap=0.3, reach=0.45, lost=0.15, min_points=30):
    """Does the raw model show reins? Its points within ``reach`` of either bit -> hand path that the low mesh
    lost (``low_distance``: each raw point's distance to the low surface, over ``lost``), spread over at least 40%
    of the path's length. -> (yes, their colour's 25th and 75th percentiles (0-255), the count)."""
    rem = np.asarray(rem, dtype=np.float64)
    hits, ts = np.zeros(len(raw_pts), dtype=bool), []
    for bit, sx in ((bit_l, 1.0), (bit_r, -1.0)):
        end = rem + np.array([sx * half_gap, 0.0, 0.0])
        d = end - bit
        t = np.clip(((raw_pts - bit) @ d) / max(d @ d, 1e-9), 0, 1)
        dist = np.linalg.norm(raw_pts - (bit + t[:, None] * d), axis=1)
        m = (dist < reach) & (low_distance > lost) & (t > 0.1) & (t < 0.9)
        hits |= m
        ts.append(t[m])
    t_all = np.concatenate(ts) if ts else np.zeros(0)
    spread = float(np.ptp(t_all)) if len(t_all) else 0.0
    ok = bool(hits.sum() >= min_points and spread >= 0.4)
    col = None
    if hits.any():
        c = np.asarray(raw_rgb)[hits]
        col = [np.percentile(c, 25, axis=0).round().tolist(), np.percentile(c, 75, axis=0).round().tolist()]
    return ok, col, int(hits.sum())
