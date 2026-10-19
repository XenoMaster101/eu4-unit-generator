"""Fresh mesh on a vanilla skeleton (Track C, test 15): new geometry and weights written into a
copy of a base-game skinned .mesh, so the game plays the base mesh's own animations on it.

What is kept from the base mesh, byte for byte: the file header, the object and shape names, the
``skeleton`` (bone names, ``ix``, ``pa``, ``tx``: what the .anim tracks bind to), the ``locator``
block and the skin's influence count (``bones``: 3 on the base-game horse, 4 on infantry bodies;
the ``ix``/``w`` arrays always hold 4 slots per vertex). The ``material`` is kept unless new
texture names are given. Replaced: ``p``, ``n``, ``ta`` (computed from the UVs), ``u0``, ``tri``,
the skin's ``ix``/``w`` and the ``aabb``.

Base meshes are base game only, and the result is a derived mesh (CLAUDE.md rule 2): generated
mod output only, never in the repo or the app, never from DLC.
"""
from __future__ import annotations

from dataclasses import dataclass

from eu4unitgen.core.chain_weights import check_weights
from eu4unitgen.core.derived_body import (DerivedBodyError, TEXTURE_SLOTS, _child, _only_mesh, bone_indices,
                                          oriented, tangents, winding_sign)
from eu4unitgen.library.scanner import parse_pdx_mesh, write_pdx_mesh

SLOTS = 4  # ix/w entries per vertex in every base-game skinned mesh


class FreshMeshError(ValueError):
    pass


@dataclass
class FreshGeometry:
    """New geometry in the base mesh's space (PDX: Y up, the unit facing -Z): flattened ``p``/``n``,
    ``u0`` (image convention), triangles, and per-vertex weights (bone name -> w)."""
    p: list[float]
    n: list[float]
    u0: list[float]
    tri: list[int]
    weights: list[dict[str, float]]

    @property
    def count(self) -> int:
        return len(self.p) // 3


def influences(base: bytes) -> int:
    """The base mesh's skin influence count (``skin.bones``)."""
    mesh, _ = _only_mesh(parse_pdx_mesh(base))
    skin = _child(mesh, "skin")
    if skin is None:
        raise FreshMeshError("the base mesh isn't skinned")
    return skin["props"]["bones"][0]


def write_fresh(base: bytes, geo: FreshGeometry, textures: dict[str, str] | None = None) -> bytes:
    """``base`` (a base-game skinned .mesh) with its geometry replaced by ``geo`` (module doc).
    ``textures``: new material texture names (diffuse, normal, specular: bare file names)."""
    try:
        tree = parse_pdx_mesh(base)
        mesh, skeleton = _only_mesh(tree)
    except (ValueError, DerivedBodyError) as exc:
        raise FreshMeshError(f"base mesh: {exc}") from exc
    skin = _child(mesh, "skin")
    if skin is None:
        raise FreshMeshError("the base mesh isn't skinned")
    if "boundingsphere" in mesh["props"]:
        raise FreshMeshError("the base mesh has a boundingsphere (not a base-game mesh)")
    limit = skin["props"]["bones"][0]
    index = bone_indices(skeleton)
    n = geo.count
    if not (len(geo.p) == len(geo.n) == n * 3 and len(geo.u0) == n * 2 and len(geo.weights) == n
            and len(geo.tri) % 3 == 0 and geo.tri):
        raise FreshMeshError(f"array lengths don't match the {n} vertices")
    if any(not 0 <= i < n for i in geo.tri):
        raise FreshMeshError("a triangle points outside the vertices")
    problems = check_weights(geo.weights, list(index))
    problems += [f"vertex {i}: {len(w)} influences (the base mesh allows {limit})"
                 for i, w in enumerate(geo.weights) if len(w) > limit]
    if problems:
        raise FreshMeshError("; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    props = mesh["props"]
    sign = winding_sign(props["p"], props["n"], props["tri"])  # follow the base mesh's winding
    tri = oriented(geo.p, geo.n, geo.tri, sign)
    props["p"] = [float(x) for x in geo.p]
    props["n"] = [float(x) for x in geo.n]
    props["ta"] = tangents(geo.p, geo.n, geo.u0, tri)
    props["u0"] = [float(x) for x in geo.u0]
    props["tri"] = list(tri)
    ix, w = [], []
    for row in geo.weights:
        items = sorted(row.items(), key=lambda kv: -kv[1])
        ix += [index[b] for b, _ in items] + [-1] * (SLOTS - len(items))
        w += [float(v) for _, v in items] + [0.0] * (SLOTS - len(items))
    skin["props"]["ix"], skin["props"]["w"] = ix, w
    box = _child(mesh, "aabb")
    if box is not None:
        box["props"]["min"] = [min(geo.p[j::3]) for j in range(3)]
        box["props"]["max"] = [max(geo.p[j::3]) for j in range(3)]
    if textures:
        material = _child(mesh, "material")
        for key, kind in TEXTURE_SLOTS:
            if textures.get(kind):
                material["props"][key] = [textures[kind]]
    return write_pdx_mesh(tree)


def check_fresh(base: bytes, fresh: bytes) -> list[str]:
    """Re-import checks: problems with ``fresh`` as a fresh mesh on ``base``'s skeleton (empty = fine)."""
    problems = []
    try:
        bt, ft = parse_pdx_mesh(base), parse_pdx_mesh(fresh)
        bmesh, bskel = _only_mesh(bt)
        fmesh, fskel = _only_mesh(ft)
    except (ValueError, DerivedBodyError) as exc:
        return [f"doesn't parse as a one-shape skinned mesh: {exc}"]
    if write_pdx_mesh({"name": "root", "props": {}, "children": [bskel]}) != \
            write_pdx_mesh({"name": "root", "props": {}, "children": [fskel]}):
        problems.append("the skeleton differs from the base mesh's (bone names, ix, pa or tx)")
    if [c["name"] for c in bt["children"]] != [c["name"] for c in ft["children"]]:
        problems.append("top-level blocks differ from the base mesh's")
    bloc = next((c for c in bt["children"] if c["name"] == "locator"), None)
    floc = next((c for c in ft["children"] if c["name"] == "locator"), None)
    if (bloc is None) != (floc is None) or (bloc is not None and bloc != floc):
        problems.append("the locator block differs from the base mesh's")
    props = fmesh["props"]
    count = len(props.get("p", [])) // 3
    for key, width in (("n", 3), ("ta", 4), ("u0", 2)):
        if len(props.get(key, [])) != count * width:
            problems.append(f"{key} has {len(props.get(key, []))} values for {count} vertices")
    if any(not 0 <= i < count for i in props.get("tri", [])):
        problems.append("a triangle points outside the vertices")
    if "boundingsphere" in props:
        problems.append("a boundingsphere (base-game meshes have none)")
    skin = _child(fmesh, "skin")
    bskin = _child(bmesh, "skin")
    if skin is None or bskin is None:
        return problems + ["no skin"]
    if skin["props"]["bones"] != bskin["props"]["bones"]:
        problems.append(f"skin influences {skin['props']['bones']} differ from the base's {bskin['props']['bones']}")
    limit = skin["props"]["bones"][0]
    ix, w = skin["props"]["ix"], skin["props"]["w"]
    if len(ix) != count * SLOTS or len(w) != count * SLOTS:
        problems.append(f"skin arrays hold {len(ix)}/{len(w)} values for {count} vertices x {SLOTS}")
        return problems
    bones = set(bone_indices(fskel).values())
    for v in range(count):
        row = [(ix[v * SLOTS + k], w[v * SLOTS + k]) for k in range(SLOTS)]
        used = [(i, x) for i, x in row if i >= 0]
        if len(used) > limit or any(i not in bones for i, _ in used) or abs(sum(x for _, x in used) - 1) > 1e-3:
            problems.append(f"vertex {v}: skin {row} (max {limit} bones from the skeleton, weights summing to 1)")
            break
    box = _child(fmesh, "aabb")
    if box is not None and count:
        p = props["p"]
        if any(abs(box["props"]["min"][j] - min(p[j::3])) > 1e-4 or abs(box["props"]["max"][j] - max(p[j::3])) > 1e-4
               for j in range(3)):
            problems.append("the aabb doesn't bound the vertices")
    return problems
