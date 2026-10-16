"""Derived bodies: a base-game body .mesh with skinned additions (tail, hair) appended (SPEC Track R).

This is the standard path for every derived body: io_pdx_mesh's export is not used for skinned
bodies (test 10 pre-check P2: it writes every face corner as its own vertex and turns
``attach_free``'s all-zero bind matrix into identity). Instead the vanilla file is parsed and
every block is kept as it is (skeleton, locators, material, vertices, skin rows, triangles); only
the additions' vertices, skin rows and triangles are appended, and the ``aabb`` grows to hold
them. ``scanner.write_pdx_mesh`` writes the result (byte-identical on every vanilla mesh).

Spaces: everything here is PDX mesh space (Y up, the unit faces -Z), UVs in image convention
(v down), as the .mesh stores them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from eu4unitgen.core.chain_weights import MAX_INFLUENCES, Chain, WeightError, check_weights, weights
from eu4unitgen.library.scanner import parse_pdx_mesh, write_pdx_mesh


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length) if length > 1e-12 else (0.0, 0.0, 0.0)


def tangents(p: list[float], n: list[float], u0: list[float], tri: list[int]) -> list[float]:
    """Per-vertex tangents as the .mesh stores them (``ta``: x, y, z, handedness w = +-1), from
    flattened positions, normals, UVs and triangle indices. Per-triangle UV tangents are summed
    per vertex, made orthogonal to the normal, and w is the sign of (n x t) . bitangent.
    Checked against vanilla's stored tangents (``tangent_agreement``)."""
    count = len(p) // 3
    tsum = [[0.0, 0.0, 0.0] for _ in range(count)]
    bsum = [[0.0, 0.0, 0.0] for _ in range(count)]
    for k in range(0, len(tri), 3):
        i0, i1, i2 = tri[k], tri[k + 1], tri[k + 2]
        p0, p1, p2 = (p[i * 3:i * 3 + 3] for i in (i0, i1, i2))
        (s0, t0), (s1, t1), (s2, t2) = (u0[i * 2:i * 2 + 2] for i in (i0, i1, i2))
        e1, e2 = _sub(p1, p0), _sub(p2, p0)
        du1, dv1, du2, dv2 = s1 - s0, t1 - t0, s2 - s0, t2 - t0
        det = du1 * dv2 - du2 * dv1
        if abs(det) < 1e-12:
            continue
        r = 1.0 / det
        t = tuple((e1[j] * dv2 - e2[j] * dv1) * r for j in range(3))
        b = tuple((e2[j] * du1 - e1[j] * du2) * r for j in range(3))
        for i in (i0, i1, i2):
            for j in range(3):
                tsum[i][j] += t[j]
                bsum[i][j] += b[j]
    out = []
    for i in range(count):
        nv = tuple(n[i * 3:i * 3 + 3])
        t = _norm(_sub(tuple(tsum[i]), tuple(x * _dot(nv, tsum[i]) for x in nv)))
        if t == (0.0, 0.0, 0.0):  # degenerate UVs: any unit vector orthogonal to the normal
            t = _norm(_cross(nv, (0.0, 1.0, 0.0) if abs(nv[1]) < 0.9 else (1.0, 0.0, 0.0)))
        w = -1.0 if _dot(_cross(nv, t), tuple(bsum[i])) < 0 else 1.0
        out += [t[0], t[1], t[2], w]
    return out


def tangent_agreement(stored: list[float], computed: list[float]) -> dict:
    """How well ``computed`` tangents match ``stored`` ones (both ``ta`` arrays): the share of
    vertices within 10 degrees, the median angle, and the share whose handedness agrees."""
    angles, same_w = [], 0
    for i in range(0, len(stored), 4):
        a, b = stored[i:i + 3], computed[i:i + 3]
        c = max(-1.0, min(1.0, _dot(_norm(a), _norm(b))))
        angles.append(math.degrees(math.acos(c)))
        same_w += (stored[i + 3] < 0) == (computed[i + 3] < 0)
    count = len(angles) or 1
    ordered = sorted(angles)
    return {"within_10deg": sum(1 for x in angles if x <= 10) / count, "median_deg": ordered[len(ordered) // 2],
            "w_agrees": same_w / count}


# --------------------------------------------------------------------------- the writer

VERTEX_ARRAYS = (("p", 3), ("n", 3), ("ta", 4), ("u0", 2))


class DerivedBodyError(ValueError):
    pass


@dataclass
class Addition:
    """One skinned addition in PDX mesh space: flattened ``p``/``n`` (x, y, z), ``u0`` (u, v,
    image convention), triangles over its own vertices, and per-vertex weights (bone -> w)."""
    name: str
    p: list[float]
    n: list[float]
    u0: list[float]
    tri: list[int]
    weights: list[dict[str, float]]
    chain: tuple[str, ...] = ()  # the bones it may use (validator: nothing outside)
    rings: tuple[int, ...] = ()  # each vertex's ring (0 = root), for the deformation check

    @property
    def count(self) -> int:
        return len(self.p) // 3


def _only_mesh(tree: dict) -> tuple[dict, dict]:
    """The single shape's single ``mesh`` node and the shape's ``skeleton`` node (bodies with more
    shapes or materials are refused: no base-game unit mesh has them, 1.37.5)."""
    objects = [c for c in tree["children"] if c["name"] == "object"]
    shapes = [s for o in objects for s in o["children"]]
    if len(shapes) != 1:
        raise DerivedBodyError(f"a derived body needs a one-shape base mesh, not {len(shapes)} shapes")
    meshes = [c for c in shapes[0]["children"] if c["name"] == "mesh"]
    skeletons = [c for c in shapes[0]["children"] if c["name"] == "skeleton"]
    if len(meshes) != 1 or len(skeletons) != 1:
        raise DerivedBodyError("a derived body needs a base mesh with one mesh block and a skeleton")
    return meshes[0], skeletons[0]


def winding_sign(p: list[float], n: list[float], tri: list[int]) -> int:
    """+1 if most triangles' (v1 - v0) x (v2 - v0) points along their vertices' normals, else -1:
    the base mesh's winding convention, which additions are made to follow."""
    score = 0
    for k in range(0, len(tri), 3):
        i0, i1, i2 = tri[k:k + 3]
        g = _cross(_sub(p[i1 * 3:i1 * 3 + 3], p[i0 * 3:i0 * 3 + 3]), _sub(p[i2 * 3:i2 * 3 + 3], p[i0 * 3:i0 * 3 + 3]))
        avg = tuple(n[i0 * 3 + j] + n[i1 * 3 + j] + n[i2 * 3 + j] for j in range(3))
        score += 1 if _dot(g, avg) > 0 else -1
    return 1 if score >= 0 else -1


def oriented(p: list[float], n: list[float], tri: list[int], sign: int) -> list[int]:
    """``tri`` with each triangle wound so its face normal agrees with its vertex normals per ``sign``."""
    out = []
    for k in range(0, len(tri), 3):
        i0, i1, i2 = tri[k:k + 3]
        g = _cross(_sub(p[i1 * 3:i1 * 3 + 3], p[i0 * 3:i0 * 3 + 3]), _sub(p[i2 * 3:i2 * 3 + 3], p[i0 * 3:i0 * 3 + 3]))
        avg = tuple(n[i0 * 3 + j] + n[i1 * 3 + j] + n[i2 * 3 + j] for j in range(3))
        out += [i0, i1, i2] if (_dot(g, avg) > 0) == (sign > 0) else [i0, i2, i1]
    return out


def _child(node: dict, name: str) -> dict | None:
    return next((c for c in node["children"] if c["name"] == name), None)


def base_vertex_count(derived: bytes, adds: list["Addition"]) -> int:
    """How many vertices of ``derived`` are the base body's (all but the appended additions')."""
    mesh, _ = _only_mesh(parse_pdx_mesh(derived))
    return len(mesh["props"]["p"]) // 3 - sum(a.count for a in adds)


def bone_indices(skeleton: dict) -> dict[str, int]:
    return {b["name"]: b["props"]["ix"][0] for b in skeleton["children"]}


def append_additions(base: bytes, additions: list[Addition], remove: list[int] | tuple[int, ...] = ()) -> bytes:
    """The base body's .mesh bytes with the base triangles numbered in ``remove`` dropped (their
    vertices stay, unreferenced, so every vertex array keeps the base's values) and ``additions``
    appended (see the module doc)."""
    tree = parse_pdx_mesh(base)
    mesh, skeleton = _only_mesh(tree)
    skin = _child(mesh, "skin")
    if skin is None or skin["props"].get("bones") != [MAX_INFLUENCES]:
        raise DerivedBodyError(f"the base mesh isn't skinned with {MAX_INFLUENCES} influences per vertex")
    if "boundingsphere" in mesh["props"]:
        raise DerivedBodyError("the base mesh has a boundingsphere (not a base-game body)")
    index = bone_indices(skeleton)
    props = mesh["props"]
    sign = winding_sign(props["p"], props["n"], props["tri"])
    if remove:
        count = len(props["tri"]) // 3
        if any(not 0 <= t < count for t in remove):
            raise DerivedBodyError(f"a removed triangle is outside the base's {count} triangles")
        props["tri"] = kept_triangles(props["tri"], remove)
    for add in additions:
        if not (len(add.p) == len(add.n) == add.count * 3 and len(add.u0) == add.count * 2
                and len(add.weights) == add.count and len(add.tri) % 3 == 0):
            raise DerivedBodyError(f"{add.name}: array lengths don't match its {add.count} vertices")
        if any(not 0 <= i < add.count for i in add.tri):
            raise DerivedBodyError(f"{add.name}: a triangle points outside its vertices")
        problems = check_weights(add.weights, list(index), set(add.chain) if add.chain else None)
        if problems:
            raise DerivedBodyError(f"{add.name}: " + "; ".join(problems[:5]))
        first = len(props["p"]) // 3
        tri = oriented(add.p, add.n, add.tri, sign)
        props["p"] += [float(x) for x in add.p]
        props["n"] += [float(x) for x in add.n]
        props["ta"] += tangents(add.p, add.n, add.u0, tri)
        props["u0"] += [float(x) for x in add.u0]
        props["tri"] += [first + i for i in tri]
        for row in add.weights:
            items = sorted(row.items(), key=lambda kv: -kv[1])
            skin["props"]["ix"] += [index[b] for b, _ in items] + [-1] * (MAX_INFLUENCES - len(items))
            skin["props"]["w"] += [float(w) for _, w in items] + [0.0] * (MAX_INFLUENCES - len(items))
    box = _child(mesh, "aabb")
    if box is not None:
        pts = props["p"]
        box["props"]["min"] = [min(box["props"]["min"][j], *pts[j::3]) for j in range(3)]
        box["props"]["max"] = [max(box["props"]["max"][j], *pts[j::3]) for j in range(3)]
    return write_pdx_mesh(tree)


def kept_triangles(tri: list[int], remove) -> list[int]:
    """``tri`` without the triangles numbered in ``remove`` (order kept)."""
    gone = set(remove)
    return [i for k in range(0, len(tri), 3) if k // 3 not in gone for i in tri[k:k + 3]]


def check_derived(base: bytes, derived: bytes, remove: list[int] | tuple[int, ...] = ()) -> tuple[list[str], dict]:
    """Problems with ``derived`` as a derived body of ``base``, and counts. Every vanilla block must
    be unchanged (skeleton, locators, material, every vertex array, skin rows, triangles) except
    that exactly the base triangles numbered in ``remove`` are gone; only appended vertices, skin
    rows and triangles may differ, and the aabb may only grow to hold them."""
    problems: list[str] = []
    try:
        bt, dt = parse_pdx_mesh(base), parse_pdx_mesh(derived)
        bm, bs = _only_mesh(bt)
        dm, ds = _only_mesh(dt)
    except (ValueError, DerivedBodyError) as exc:
        return [f"can't read: {exc}"], {}
    if ds != bs:
        problems.append("skeleton differs from the base body's")
    if [c for c in dt["children"] if c["name"] != "object"] != [c for c in bt["children"] if c["name"] != "object"]:
        problems.append("locators differ from the base body's")
    if _child(dm, "material") != _child(bm, "material"):
        problems.append("material differs from the base body's")
    if sorted(dm["props"]) != sorted(bm["props"]):
        problems.append(f"mesh properties {sorted(dm['props'])} differ from the base's {sorted(bm['props'])}")
    nb, nd = len(bm["props"]["p"]) // 3, len(dm["props"]["p"]) // 3
    for key, width in VERTEX_ARRAYS:
        b, d = bm["props"].get(key, []), dm["props"].get(key, [])
        if d[:len(b)] != b:
            problems.append(f"the base's {key} values changed")
        if len(d) != nd * width:
            problems.append(f"{key} holds {len(d) // width} vertices, not {nd}")
    base_tri = kept_triangles(bm["props"]["tri"], remove)
    if dm["props"]["tri"][:len(base_tri)] != base_tri:
        problems.append("the base's triangles changed" + (f" (beyond the {len(set(remove))} removed)" if remove else ""))
    if any(not 0 <= i < nd for i in dm["props"]["tri"]):
        problems.append("a triangle points outside the vertices")
    bskin, dskin = _child(bm, "skin"), _child(dm, "skin")
    if dskin is None or dskin["props"].get("bones") != bskin["props"].get("bones"):
        problems.append("skin influence count differs from the base's")
    else:
        for key in ("ix", "w"):
            b, d = bskin["props"][key], dskin["props"][key]
            if d[:len(b)] != b:
                problems.append(f"the base's skin {key} changed")
            if len(d) != nd * MAX_INFLUENCES:
                problems.append(f"skin {key} covers {len(d) // MAX_INFLUENCES} vertices, not {nd}")
        ix, w = dskin["props"]["ix"], dskin["props"]["w"]
        nbones = len(ds["children"])
        for v in range(nb, nd):
            row = [(ix[v * 4 + k], w[v * 4 + k]) for k in range(MAX_INFLUENCES) if ix[v * 4 + k] >= 0]
            if any(not 0 <= i < nbones for i, _ in row) or abs(sum(x for _, x in row) - 1) > 1e-4:
                problems.append(f"added vertex {v}: bad skin row {row}")
                break
    pts = dm["props"]["p"]
    box = _child(dm, "aabb")
    if box is None or any(min(pts[j::3]) < box["props"]["min"][j] - 1e-5 or max(pts[j::3]) > box["props"]["max"][j] + 1e-5
                          for j in range(3)):
        problems.append("aabb doesn't hold every vertex")
    if "boundingsphere" in dm["props"]:
        problems.append("has a boundingsphere (none of the base-game meshes has one)")
    return problems, {"base_vertices": nb, "added_vertices": nd - nb,
                      "base_triangles": len(bm["props"]["tri"]) // 3, "removed_triangles": len(set(remove)),
                      "added_triangles": (len(dm["props"]["tri"]) - len(base_tri)) // 3}


# --------------------------------------------------------------------------- addition kinds and building
# Shapes: blender/jobs/build_additions.py (plan after test 10). Chains: spike R0 / pre-check P1 (SPEC Track R).
ADDITION_KINDS: dict[str, dict] = {
    # exempt: rings sunk or flared into the body by design (deformation check C2); behind: the joint
    # it must stay behind in the unit's facing frame (C5), or None.
    "tail": {"shape": "tail", "params": {}, "chain": Chain("length", ("hip", "lower_spine"), {"lower_spine": 0.5}),
             "exempt": [0, 1], "behind": "hip"},
    "tuft": {"shape": "tuft", "params": {"length": 1.5, "angle": 45}, "chain": Chain("rigid", ("head",)),
             "exempt": [0], "behind": "top_spine"},
    "ears": {"shape": "ears", "params": {}, "chain": Chain("rigid", ("head",)), "exempt": [0], "behind": None},
}
HAIR_PATCH = (105, 30, 115, 40)  # western_generic_01_type1 diffuse px: the face's dark hair, no country mask
PATCH_TEX = 512


@dataclass(frozen=True)
class Removal:
    """Base triangles to drop: any triangle with a vertex mostly on ``bone`` whose UV centroid lies
    in ``uv_px`` (x0, y0, x1, y1, image pixels of a ``tex``-px texture; x1, y1 exclusive)."""
    bone: str
    uv_px: tuple[float, float, float, float]
    tex: int = 512


# Per base body (its .mesh path): named parts that can be removed. type1's helmet (H1, 2026-10-16):
# 369 head triangles with UVs in px 128-240 x 0-200 plus one boundary triangle at (128, 99); a
# closed, textured hair cap is underneath.
REMOVALS: dict[str, dict[str, Removal]] = {
    "gfx/models/units/western_generic_01_type1.mesh": {"helmet": Removal("head", (127.5, 0, 240, 200))},
}


def removable(base_file: str) -> list[str]:
    return sorted(REMOVALS.get(base_file.lower(), {}))


def removed_triangles(base: bytes, base_file: str, names: list[str]) -> list[int]:
    """Indices of the base triangles the named removals drop (sorted, unique)."""
    table = REMOVALS.get(base_file.lower(), {})
    tree = parse_pdx_mesh(base)
    mesh, skeleton = _only_mesh(tree)
    props, skin = mesh["props"], _child(mesh, "skin")["props"]
    index = bone_indices(skeleton)
    out: set[int] = set()
    for name in names:
        r = table.get(name)
        if r is None:
            raise DerivedBodyError(f"{base_file} has no removable {name!r} (known: {', '.join(removable(base_file)) or 'none'})")
        bone = index[r.bone]

        def mostly(v):
            ws = skin["w"][v * 4:v * 4 + 4]
            return skin["ix"][v * 4 + max(range(4), key=lambda k: ws[k])] == bone

        x0, y0, x1, y1 = r.uv_px
        tri, u0 = props["tri"], props["u0"]
        for k in range(0, len(tri), 3):
            vs = tri[k:k + 3]
            u = sum(u0[v * 2] for v in vs) / 3 * r.tex
            w = sum(u0[v * 2 + 1] for v in vs) / 3 * r.tex
            if x0 <= u < x1 and y0 <= w < y1 and any(mostly(v) for v in vs):
                out.add(k // 3)
    return sorted(out)


def build_derived(runner, base: bytes, kinds: list[str], patch: tuple[int, int, int, int] = HAIR_PATCH,
                  tex: int = PATCH_TEX, remove: list[int] | tuple[int, ...] = ()) -> tuple[bytes, list[Addition]]:
    """The derived body: ``base`` (.mesh bytes of a base-game body) without the triangles in
    ``remove`` and with the ``kinds`` additions built in background Blender (``build_additions``,
    which measures the body without the removed parts), weighted by their chains, appended."""
    import tempfile
    from pathlib import Path

    tree = parse_pdx_mesh(base)
    _, skeleton = _only_mesh(tree)
    names = list(bone_indices(skeleton))
    for k in kinds:
        if k not in ADDITION_KINDS:
            raise DerivedBodyError(f"unknown addition {k!r}; known: {', '.join(ADDITION_KINDS)}")
        try:
            ADDITION_KINDS[k]["chain"].check(names)
        except WeightError as exc:
            raise DerivedBodyError(f"{k}: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="eu4unitgen_derived_") as tmp:
        body = Path(tmp) / "base.mesh"
        body.write_bytes(append_additions(base, [], remove) if remove else base)
        res = runner.run_job("build_additions", {
            "body": str(body), "patch_px": list(patch), "tex": tex,
            "additions": [{"name": k, "shape": ADDITION_KINDS[k]["shape"], "params": ADDITION_KINDS[k]["params"]}
                          for k in kinds]}, timeout=300).result
    adds = []
    for k, raw in zip(kinds, res["additions"]):
        chain = ADDITION_KINDS[k]["chain"]
        adds.append(Addition(k, raw["p"], raw["n"], raw["u0"], raw["tri"], weights(chain, raw["rings"]), chain.bones,
                             tuple(raw["rings"])))
    return append_additions(base, adds, remove), adds
