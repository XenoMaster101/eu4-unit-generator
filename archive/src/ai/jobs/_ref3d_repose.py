"""Track M finish line, step 2 (the game version): the master moved from its fitted skeleton and pose (view L's)
onto the vanilla horse skeleton at rest, where the game's animations play it. Numpy only; used by the Blender
job game_master.

Each point moves with what it belongs to:
  trunk  the barrel and what lies on it (plates, panels, the chest drapes, straps, tail, tassels): the exact
         inverse of _ref3d_body.skeleton's trunk transform (the trunk isn't posed, only proportioned: its
         length scaled by ``body``, raised by the legs' factors);
  neck   along the neck's bones (NeckBase, NeckMiddle, Head): each point's place along its nearest segment
         and its offset from it, turned from the posed segment onto the rest one; blended into the trunk's
         move near the neck's base;
  head   rigidly with the head (the poll and the head's axis: the master's pitch and turn onto the default
         pitch, no turn);
  leg    along its leg's chain (Up, Down, Knee, Heel, hoof top, sole), blended into the trunk's move near the
         leg's top (inside the barrel).
Next to a joint a point blends its two nearest segments (inverse distance^4), so joints don't tear.
"""
import numpy as np

import _ref3d_body as RB

LEGS = ("FrontLeft", "FrontRight", "BackLeft", "BackRight")
CHAIN = ("Up", "Down", "Knee", "Heel", "HoofTop", "Sole")


class Repose:
    def __init__(self, sk, body, view="L", keep_head_pitch=False, neck_keep=0.0):
        """``sk``: the skeleton JSON (bones, seat); ``body``: an E-A candidate's shape, skeleton and poses.
        M20 (the user: the game head droops, long and thin): ``keep_head_pitch`` keeps the fitted head's
        pitch (else the anatomy's default, 40 deg: horse 2's 25 deg head turned 15 deg nose-down);
        ``neck_keep`` keeps that share of the fitted neck's raise (0: vanilla's rest neck; horse 2's neck
        was raised 28 deg at its middle). The geometry then sits off the vanilla neck bones by that share
        of the raise, and still moves rigidly with them."""
        ha = RB.ha
        self.k = dict(body["skeleton"])
        shape = {**ha.DEFAULTS, **body["shape"]}
        pose = body["poses"][view]
        B, H = RB.skeleton(sk["bones"], sk["seat"], self.k)
        self.P = {**shape, "head_pitch": pose["head_pitch"], "head_yaw": pose.get("head_yaw", 0.0)}
        self.Bp = RB.pose(B, H, self.P, pose)
        self.H = H
        B0 = {n: np.array(v, dtype=np.float64) for n, v in sk["bones"].items()}
        self.B0raw = B0
        self.H0 = float(sk["seat"])
        self.P0 = {**shape, "head_pitch": pose["head_pitch"] if keep_head_pitch else ha.DEFAULTS["head_pitch"],
                   "head_yaw": 0.0}
        nb, nm = pose.get("neck", [0.0, 0.0])
        self.Br = RB.pose(B0, self.H0, self.P0, {"neck": [nb * neck_keep, nm * neck_keep]} if neck_keep else {})
        self.zf, self.zb = B0["HorseFrontLeftUp"][2], B0["HorseBackLeftUp"][2]
        self.dyf = (self.k.get("leg_f", 1.0) - 1) * B0["HorseFrontLeftUp"][1]
        self.dyb = (self.k.get("leg_b", 1.0) - 1) * B0["HorseBackLeftUp"][1]
        self.sb = self.k.get("body", 1.0)

    # ---- the moves
    def trunk(self, X):
        z = X[:, 2] / self.sb
        w = np.clip((z - self.zf) / (self.zb - self.zf), 0, 1)
        return np.c_[X[:, 0], X[:, 1] - self.dyf * (1 - w) - self.dyb * w, z]

    def head(self, X):
        ha = RB.ha
        hp, hr = ha.head_direction(self.P), ha.head_direction(self.P0)
        _, up, vp = ha._frame(hp)
        _, ur, vr = ha._frame(hr)
        Rp, Rr = np.stack([hp, up, vp], 1), np.stack([hr, ur, vr], 1)
        return self.Br["HorseHead"] + (X - self.Bp["HorseHead"]) @ Rp @ Rr.T

    @staticmethod
    def _rot(a, b):
        """The rotation turning unit ``a`` onto unit ``b`` (the shortest arc)."""
        v, c = np.cross(a, b), float(a @ b)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K / (1 + c)

    def chain(self, X, names, base_blend=0.3):
        """Along a bone chain (module doc); the first segment's top ``base_blend`` blends into the trunk."""
        Pp = np.array([self.Bp[n] for n in names])
        Pr = np.array([self.Br[n] for n in names])
        segs = []
        for j in range(len(names) - 1):
            dp, dr = Pp[j + 1] - Pp[j], Pr[j + 1] - Pr[j]
            lp = max(np.linalg.norm(dp), 1e-9)
            segs.append((Pp[j], dp, lp, Pr[j], dr, self._rot(dp / lp, dr / max(np.linalg.norm(dr), 1e-9))))
        n = len(X)
        dist = np.zeros((n, len(segs)))
        moved = np.zeros((n, len(segs), 3))
        tt = np.zeros((n, len(segs)))
        for j, (a, dp, lp, ar, dr, R) in enumerate(segs):
            t = np.clip(((X - a) @ dp) / lp ** 2, 0, 1)
            foot = a + t[:, None] * dp
            o = X - foot
            dist[:, j] = np.linalg.norm(o, axis=1)
            moved[:, j] = ar + t[:, None] * dr + o @ R.T
            tt[:, j] = t
        order = np.argsort(dist, axis=1)[:, :2]
        rows = np.arange(n)
        w = 1.0 / np.maximum(dist[rows[:, None], order], 1e-4) ** 4
        w /= w.sum(axis=1, keepdims=True)
        out = (moved[rows[:, None], order] * w[..., None]).sum(axis=1)
        # the chain's root lies inside the trunk: blend into the trunk's move there
        first = (order[:, 0] == 0)
        k = np.where(first, np.clip(tt[:, 0] / base_blend, 0, 1), 1.0)
        k = k * k * (3 - 2 * k)
        return k[:, None] * out + (1 - k[:, None]) * self.trunk(X)

    def neck(self, X):
        return self.chain(X, ("HorseNeckBase", "HorseNeckMiddle", "HorseHead"))

    def leg(self, X, leg):
        return self.chain(X, [f"Horse{leg}{j}" for j in CHAIN], base_blend=0.35)

    def apply(self, X, mode):
        """``X`` (n, 3) PDX points; ``mode``: per point "trunk", "neck", "head" or a leg name."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        mode = np.asarray(mode)
        for m in np.unique(mode):
            s = mode == m
            out[s] = (self.trunk(X[s]) if m == "trunk" else self.neck(X[s]) if m == "neck"
                      else self.head(X[s]) if m == "head" else self.leg(X[s], m))
        return out


# the anatomy's parts (_ref3d_body.PART_NAMES) -> modes
PART_MODE = {0: "trunk", 1: "neck", 2: "head", 3: "head", 4: "head", 5: "FrontLeft", 6: "FrontRight",
             7: "BackLeft", 8: "BackRight"}
