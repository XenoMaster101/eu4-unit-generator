"""The base game's infantry animation sets and how each holds its weapons (the user, after T38: the animation review).

Read from the user's install at run time (nothing copied): every base-game ``*_INFANTRY_<n>`` entity's pdxmesh gives
its set (the prefix of its idle animation, e.g. ``polearm_twohanded``) and its attaches give what each hand holds.
As read from 1.37 (2026-10-29):

  rifleman            62 bodies   a musket or rifle in Left_hand_node
  polearm_twohanded   19          western_halbard, asian_spear_01 in Right_hand_node
  polearm_onehanded    9          a spear alone in Right_hand_node
  spear_shield         8          a spear in Right_hand_node and a shield in Left_hand_node
  musketeer_02, eastern_generic_02_special   a musket in Left_hand_node

There is no sword set: a shield means spear_shield, a long weapon alone polearm_twohanded (``core.loadout``'s auto
stance and ``trellis.foot_stance``, checked against this catalogue). Each set's weapon convention (``convention``):
the most common weapon in its hand, measured in its hand node's frame: its long axis, length and where along it
the node (the hand) grips it, from the butt (western_halbard 0.44, asian_spear_01 0.50, muslim_spear 0.47).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

INFANTRY = re.compile(r"_INFANTRY(_\d+)?$")
HANDS = ("Right_hand_node", "Left_hand_node")
# the hand each set's weapon is in (the rest: the right)
WEAPON_HAND = {"rifleman": "Left_hand_node", "musketeer_02": "Left_hand_node",
               "eastern_generic_02_special": "Left_hand_node"}


def _index(eu4: Path):
    from eu4unitgen.library.scanner import GameFiles, build_index
    gf = GameFiles(Path(eu4), include_dlc=False)
    return gf, build_index(gf)


@lru_cache(maxsize=2)
def catalogue(eu4: str) -> dict:
    """{set: {"bodies": [pdxmesh], "units": n, "Right_hand_node": {entity: n}, "Left_hand_node": {...}}}."""
    from eu4unitgen.library.scanner import _attach_targets, resolved
    _, idx = _index(Path(eu4))
    out: dict = {}
    for name in idx.entities:
        if not INFANTRY.search(name) or not idx.entity(name, True):
            continue
        pm = resolved(idx, name, "pdxmesh")
        rec = (idx.pdxmeshes.get(pm) or [{}])[0] if pm else {}
        idle = (rec.get("animations") or {}).get("idle") or ""
        if not idle.endswith("_idle_animation"):
            continue
        s = out.setdefault(idle[: -len("_idle_animation")], {"bodies": set(), "units": 0, "by_body": {}, "entities": [],
                                                               **{h: Counter() for h in HANDS}})
        s["entities"].append((name, pm, _attach_targets(resolved(idx, name, "attach") or [])))
        s["bodies"].add(pm)
        s["units"] += 1
        for node, target in _attach_targets(resolved(idx, name, "attach") or []).items():
            if node in HANDS:
                s[node][target] += 1
                s["by_body"].setdefault(pm, {}).setdefault(node, Counter())[target] += 1
    return {k: {"bodies": sorted(v["bodies"]), "units": v["units"], **{h: dict(v[h].most_common()) for h in HANDS},
                "by_body": {b: {n: dict(c.most_common()) for n, c in d.items()} for b, d in v["by_body"].items()},
                "entities": sorted(v["entities"])}
            for k, v in out.items()}


def long_axis(pts: list) -> list:
    """The points' principal axis (power iteration on the covariance), its largest component positive."""
    n = len(pts)
    m = [sum(p[k] for p in pts) / n for k in range(3)]
    c = [[sum((p[i] - m[i]) * (p[j] - m[j]) for p in pts) for j in range(3)] for i in range(3)]
    v = [1.0, 0.9, 1.1]
    for _ in range(60):
        w = [sum(c[i][j] * v[j] for j in range(3)) for i in range(3)]
        s = math.sqrt(sum(x * x for x in w)) or 1.0
        v = [x / s for x in w]
    big = max(range(3), key=lambda k: abs(v[k]))
    return [-x for x in v] if v[big] < 0 else v


def grip_measure(pts: list, axis: list | None = None) -> dict:
    """A weapon's length along its axis and where the origin (its hand node) grips it: 0 the butt, 1 the head.
    The head is the end farther from the grip (a halberd's blade, a spear's point)."""
    ax = axis or long_axis(pts)
    t = [sum(p[k] * ax[k] for k in range(3)) for p in pts]
    lo, hi = min(t), max(t)
    if abs(lo) > abs(hi):  # the head is the longer side from the hand: make it +axis
        ax, lo, hi = [-x for x in ax], -hi, -lo
    return {"axis": [round(x, 4) for x in ax], "length": round(hi - lo, 3),
            "grip": round(-lo / (hi - lo), 3) if hi > lo else 0.5}


@lru_cache(maxsize=8)
def convention(eu4: str, stance: str, body: str | None = "western_generic_01_type1") -> dict | None:
    """The set's weapon convention: the weapon ``body``'s own units hold in that set's weapon hand (our soldiers are
    built on western_generic_01_type1: western_halbard), else the set's most common, measured in mesh units (as our
    parts are: both drawn at pdxmesh scale 0.5). -> {"stance", "node", "weapon", "axis", "length", "grip"} or None."""
    from eu4unitgen.library.scanner import _walk, parse_pdx_mesh, resolved
    sets = catalogue(eu4)
    node = WEAPON_HAND.get(stance, "Right_hand_node")
    held = ((sets.get(stance) or {}).get("by_body", {}).get(body) or {}).get(node) or (sets.get(stance) or {}).get(node) or {}
    if not held:
        return None
    gf, idx = _index(Path(eu4))
    for weapon in held:
        pm = resolved(idx, weapon, "pdxmesh")
        rec = (idx.pdxmeshes.get(pm) or [{}])[0] if pm else {}
        data = gf.read(rec.get("file") or "") if rec.get("file") else None
        if not data:
            continue
        pts = []
        for n in _walk(parse_pdx_mesh(data)):
            p = (n.get("props") or {}).get("p")
            if p and "tri" in (n.get("props") or {}):
                pts += [p[i:i + 3] for i in range(0, len(p), 3)]
        if len(pts) >= 8:
            return {"stance": stance, "node": node, "weapon": weapon, **grip_measure(pts)}
    return None


def slide_to_grip(pts: list, grip: float) -> tuple[list, dict]:
    """A weapon's points (in its hand node's frame, along +z as ``trellis.foot_part`` leaves it) slid along z so the
    hand grips it at ``grip`` of its length from the butt (the base game's place for its stance), and its shaft
    centred on the hand (x, y: the median of the middle half of its length, the head and butt left out).
    -> (points, {"grip_before", "grip_after", "slid", "centred"})."""
    zs = [p[2] for p in pts]
    lo, hi = min(zs), max(zs)  # the head at +z (foot_part points it there; the English bill's butt is its longer side)
    before = -lo / (hi - lo)
    dz = -(lo + grip * (hi - lo))
    mid = [p for p in pts if lo + 0.25 * (hi - lo) <= p[2] <= lo + 0.75 * (hi - lo)] or pts
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    cx, cy = med([p[0] for p in mid]), med([p[1] for p in mid])
    out = [[p[0] - cx, p[1] - cy, p[2] + dz] for p in pts]
    return out, {"grip_before": round(before, 3), "grip_after": round(grip, 3), "slid": round(dz, 3),
                 "centred": [round(-cx, 3), round(-cy, 3)]}


def example_unit(eu4: str, stance: str, body: str | None = "western_generic_01_type1") -> str | None:
    """A base-game infantry entity playing ``stance`` and holding its convention's weapon (on ``body`` if one does):
    the review's comparison."""
    conv = convention(eu4, stance, body)
    ents = (catalogue(eu4).get(stance) or {}).get("entities") or []
    if not ents:
        return None
    holds = [e for e in ents if conv and e[2].get(conv["node"]) == conv["weapon"]] or ents
    return next((e[0] for e in holds if e[1] == body), holds[0][0])
