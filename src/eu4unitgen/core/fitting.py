"""Fitting: part sizes and placement measured from the body a part attaches to (SPEC §10 Phase 3).

Measured from the body's ``.mesh`` (read from the user's install at runtime; only numbers are
kept): the body's stature (the head joint's height in the bind pose) and the head's shape, from
the vertices skinned mostly to the ``head`` bone, in that bone's frame. Stature, not the
bounding box: the box includes headgear (western tier 4's shako makes it 18.06 against 16.65),
which mustn't lengthen spears. Every generic infantry body shares one skeleton, so they all
scale 1.0. On 1.37.5 the western tier-1
helmet top is 2.15 above the head joint and the head spans z -1.27..1.11; the in-game-tested
spike crest (test 7: "size and height good") sat at base 1.9, radius 1.4.

Reference body: western_generic_01_type1 (head joint 14.499 up), which every default in
``core.parts.KINDS`` was set on. Other skeletons scale lengths by stature / 14.499.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eu4unitgen.library.scanner import _walk, parse_pdx_mesh, summarize_mesh

REFERENCE_STATURE = 14.499  # head joint height of western_generic_01_type1, the body test 7 was fitted on
CREST_SINK = 0.25  # how far the crest's base sits into the helmet top (test 7: 1.9 on a 2.15 top)
CREST_RADIUS_FACTOR = 1.1  # crest arc radius / the head's larger front-or-back extent (test 7: 1.4 on 1.27)
CREST_RADIUS_RANGE = (0.8, 2.5)
EARS_SINK = 0.45  # horse ears' base below the helmet top (UNVERIFIED in game; test 10)
EARS_SPREAD_FACTOR = 0.63  # ear base x / the head's half-width (0.62 on western type1's 0.988)
HELMET_TOP, HELMET_SIDE, HELMET_BACK = 0.12, 0.13, 0.2  # greek_helmet clearance over the bare head
HELMET_BROW = 0.8  # the brow's z / the head's front reach (the nose reaches further than the forehead)
HEAD_WEIGHT = 0.5  # a vertex belongs to the head when its largest weight is on ``head`` and above this


@dataclass
class BodyFit:
    mesh_file: str
    height: float | None = None  # top of the bounding box, headgear included (shown, not used to scale)
    stature: float | None = None  # the head joint's height in the bind pose
    head: dict | None = None  # head-frame extents of head-skinned vertices: top, xmin, xmax, zmin, zmax, count
    bones: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def scale(self) -> float:
        """Length factor against the reference skeleton; 1 when the stature is unknown."""
        return self.stature / REFERENCE_STATURE if self.stature else 1.0


def rigid(bind: list[float]) -> tuple[list[float], bool]:
    """A bind pose (row-major 3x4) with its rotation's columns scaled back to unit length, and
    whether that changed anything. Some vanilla bones carry a scaled inverse bind matrix
    (1.37.5: ``Right_hand_node`` in western_generic_03_type1 is scaled by ~1e12, so its bind pose
    shrinks anything on it to a point); the engine animates from the .anim, so in game only the
    rotation and position matter."""
    rows = [bind[0:3], bind[4:7], bind[8:11]]
    out = [list(r) for r in rows]
    changed = False
    for j in range(3):
        n = sum(rows[i][j] ** 2 for i in range(3)) ** 0.5
        if n > 0 and abs(n - 1) > 1e-3:
            changed = True
            for i in range(3):
                out[i][j] = rows[i][j] / n
    return [v for i in range(3) for v in (*out[i], bind[4 * i + 3])], changed


def _skin_points(tree: dict, bone_ix: int, bind: list[float]) -> list[list[float]]:
    """Vertices whose largest weight is on bone ``bone_ix`` (> HEAD_WEIGHT), in that bone's frame:
    p_local = R^T (p - t), with bind = [R | t] row-major 3x4. Only vertices a triangle uses count:
    a derived body keeps removed parts' vertices (e.g. type1's helmet) unreferenced."""
    bind, _ = rigid(bind)
    rot = [bind[0:3], bind[4:7], bind[8:11]]
    t = [bind[3], bind[7], bind[11]]
    out = []
    for node in _walk(tree):
        if node["name"] != "mesh":
            continue
        skin = next((c for c in node["children"] if c["name"] == "skin"), None)
        if skin is None:
            continue
        p, ix, w = node["props"].get("p", []), skin["props"].get("ix", []), skin["props"].get("w", [])
        k = (skin["props"].get("bones") or [0])[0]
        if not k or len(ix) < k * (len(p) // 3):
            continue
        used = set(node["props"].get("tri", [])) or set(range(len(p) // 3))
        for i in range(len(p) // 3):
            if i not in used:
                continue
            ws, bs = w[i * k:(i + 1) * k], ix[i * k:(i + 1) * k]
            j = max(range(k), key=lambda q: ws[q])
            if bs[j] == bone_ix and ws[j] > HEAD_WEIGHT:
                d = [p[3 * i + a] - t[a] for a in range(3)]
                out.append([sum(rot[r][c] * d[r] for r in range(3)) for c in range(3)])
    return out


def measure_body(data: bytes, mesh_file: str) -> BodyFit:
    """A body mesh's fitting facts. Missing data (no skeleton, no skin, no head) leaves the
    matching field None and a note; parts then keep their defaults."""
    tree = parse_pdx_mesh(data)
    summary = summarize_mesh(tree)
    fit = BodyFit(mesh_file, bones=list(summary["bones"]))
    box = summary.get("aabb")
    if box and box["max"][1] > box["min"][1]:
        fit.height = round(box["max"][1], 3)
    head = summary["skeleton"].get("head")
    if not head or not head.get("bind"):
        fit.notes.append(f"{mesh_file}: no head bone with a bind pose; lengths not scaled, head parts keep "
                         "their defaults")
        return fit
    if head["bind"][7] > 0:
        fit.stature = round(head["bind"][7], 3)
    skeleton = next((c for n in _walk(tree) for c in n["children"] if c["name"] == "skeleton"), None)
    ixs = {b["name"]: (b["props"].get("ix") or [None])[0] for b in (skeleton or {"children": []})["children"]}
    pts = _skin_points(tree, ixs.get("head"), head["bind"])
    if not pts:
        fit.notes.append(f"{mesh_file}: no vertices skinned to the head; head parts keep their defaults")
        return fit
    xs, ys, zs = zip(*pts)
    fit.head = {"top": round(max(ys), 3), "xmin": round(min(xs), 3), "xmax": round(max(xs), 3),
                "zmin": round(min(zs), 3), "zmax": round(max(zs), 3), "count": len(pts)}
    return fit


def fitted_params(kind: str, body: BodyFit | None) -> dict[str, float]:
    """The params fitting sets for a part kind on ``body`` (the rest keep the kind's defaults).
    ``scaled`` lengths are the defaults times the body's height factor."""
    from eu4unitgen.core.parts import KINDS

    if body is None:
        return {}
    k = KINDS[kind]
    out = {name: round(k.params[name].default * body.scale, 3) for name in k.scaled}
    if kind == "crest" and body.head:
        h = body.head
        lo, hi = CREST_RADIUS_RANGE
        out["base"] = round(h["top"] - CREST_SINK, 3)
        out["radius"] = round(min(hi, max(lo, CREST_RADIUS_FACTOR * max(-h["zmin"], h["zmax"]))), 3)
    if kind == "greek_helmet" and body.head:
        h = body.head
        out["top"] = round(h["top"] + HELMET_TOP, 3)
        out["half_width"] = round(max(-h["xmin"], h["xmax"]) + HELMET_SIDE, 3)
        out["front"] = round(h["zmin"] * HELMET_BROW - 0.1, 3)
        out["back"] = round(h["zmax"] + HELMET_BACK, 3)
    if kind == "horse_ears" and body.head:
        h = body.head
        out["base"] = round(h["top"] - EARS_SINK, 3)
        out["spread"] = round(EARS_SPREAD_FACTOR * max(-h["xmin"], h["xmax"]), 3)
    return out
