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


def base_arrays(base: bytes) -> dict:
    """The base mesh as the ``fresh_horse`` job's ``vanilla`` argument: positions, triangles,
    UVs (``u0``, image convention), per-vertex weights by bone name, and each bone's bind position
    (PDX mesh space)."""
    from eu4unitgen.library.scanner import summarize_mesh

    tree = parse_pdx_mesh(base)
    mesh, skeleton = _only_mesh(tree)
    names = {i: b for b, i in bone_indices(skeleton).items()}
    skin = _child(mesh, "skin")["props"]
    weights = []
    for v in range(len(mesh["props"]["p"]) // 3):
        row = {}
        for k in range(SLOTS):
            i, w = skin["ix"][v * SLOTS + k], skin["w"][v * SLOTS + k]
            if i >= 0 and w > 0:
                row[names[i]] = row.get(names[i], 0.0) + w
        weights.append(row)
    bones = {}
    for name, rec in summarize_mesh(tree)["skeleton"].items():
        if rec.get("bind"):
            b = rec["bind"]
            bones[name] = [b[3], b[7], b[11]]
    return {"p": list(mesh["props"]["p"]), "tri": list(mesh["props"]["tri"]), "u0": list(mesh["props"].get("u0", [])),
            "weights": weights, "bones": bones}


def geometry_from_job(path) -> FreshGeometry:
    """The ``fresh_horse`` job's geometry JSON -> FreshGeometry."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreshGeometry(d["p"], d["n"], d["u0"], d["tri"], d["weights"])


@dataclass
class FreshHorse:
    """What the validator and the checklist need to know about one fresh horse."""
    tier: int
    entity: str  # the cavalry (horse) entity that wears it
    pdxmesh: str
    file: str
    base_pdxmesh: str
    base_file: str
    textures: dict  # kind -> generated texture path
    triangles: int
    rights: str
    source: str = ""


def _own_cannon(naming, owner: str, tier: int, entities: dict, idx, new_entities: list | None) -> dict:
    """The tier's own cannon entity (``<owner>_ARTILLERY_<tier>_cannon``): a full copy of the one the tier's artillery
    entity attaches at ``cannon_node`` (its pdxmesh, states and their events, attaches: the clone chain resolved),
    added to ``entities`` and ``new_entities``; the artillery entity now attaches it."""
    from eu4unitgen.core.generator import BaseChain, GenerationError
    art = entities.get(naming.unit_entity(owner, "artillery", tier))
    if art is None:
        raise GenerationError(f"tier {tier}: no artillery entity to put a fresh gun on")
    gunman = next((a for a in art["attaches"] if any(n == "cannon_node" for n, _ in a["nodes"])), None)
    if gunman is None:
        raise GenerationError(f"tier {tier}: {art['name']} attaches no cannon (no cannon_node)")
    vanilla = dict(gunman["nodes"])["cannon_node"]
    name = naming.cannon_entity(owner, tier)
    if name not in entities:
        flat = BaseChain(idx, "westerngfx").flat(vanilla)
        own = {"name": name, "pdxmesh": flat["pdxmesh"], "version": art["version"], "states": flat["states"],
               "attaches": flat["attaches"]}
        entities[name] = own
        if new_entities is not None:
            new_entities.append(own)
    gunman["nodes"] = [(n, name if n == "cannon_node" else e) for n, e in gunman["nodes"]]
    return entities[name]


HORSE_TRIANGLES = 3000  # Track C, test 15: the budget for a fresh horse (the base-game horse has 1,874)
GUN_TRIANGLES = 1500  # Track A: the budget for a fresh gun (the base-game gun has 682)


def apply_horses(naming, set_name: str, owner: str, entities: dict, records: list, by_name: dict, loadout,
                 game, idx, kind: str = "horse", new_entities: list | None = None) -> tuple[list, dict, list[str]]:
    """Put each tier's fresh horse (``loadout.horses``) on its cavalry entity: the .mesh (the base
    horse's skeleton, locators and skin count with the new geometry), its textures (from the PNGs,
    procedural DDS in the base textures' formats, full mips) beside it in the set's model folder,
    and a pdxmesh with the base horse's animations, scale and keys. With ``kind`` "rider" (M23): each
    tier's fresh rider (``loadout.riders``) on its cavalry rider entity, on the base infantry body's
    skeleton (the rider plays its horse_* ids). With ``kind`` "soldier": each tier's fresh soldier
    (``loadout.soldiers``) on its infantry entity, on the same skeleton; its pdxmesh plays the base body's
    animation ids with the infantry's own on top (its stance, and artillery_attack: the infantry is the gun
    crew). With ``kind`` "gun" (Track A): each tier's fresh gun (``loadout.guns``) on the base game's gun skeleton,
    worn by the tier's own cannon entity (``naming.cannon_entity``: written in full from the one the artillery entity
    attached, its states and muzzle event kept; never ``clone =``, SPEC §3), which the artillery entity now attaches
    at ``cannon_node``; the new entities go into ``new_entities``. -> (files, records, notes)."""
    from eu4unitgen.core.generator import OWNED, GeneratedFile, GenerationError
    from eu4unitgen.core.loadout import LoadoutError
    from eu4unitgen.library.scanner import summarize_mesh
    from eu4unitgen.textures.dds import DDSFormat, full_mip_count, read_header, write_dds

    files, out, notes = [], {}, []
    rider, soldier, gun = kind == "rider", kind == "soldier", kind == "gun"
    body = rider or soldier
    source = (loadout.riders if rider else getattr(loadout, "soldiers", {}) if soldier else getattr(loadout, "guns", {})
              if gun else loadout.horses)
    for tier, h in sorted(source.items()):
        if gun:
            entity = _own_cannon(naming, owner, tier, entities, idx, new_entities)
        else:
            entity = entities[naming.rider_entity(owner, tier) if rider else naming.unit_entity(owner, "infantry", tier)
                              if soldier else naming.unit_entity(owner, "cavalry", tier)]
        if not body and not gun and entity["pdxmesh"] == naming.pdxmesh(set_name, "horse", tier):
            raise GenerationError(f"tier {tier}: a texture recipe's horse slot and a fresh horse; pick one")
        rec = idx.pdxmesh(h["base"], base_only=True)
        if rec is None or not game.in_base_game(rec.get("file") or ""):
            raise GenerationError(f"tier {tier}: fresh horse base {h['base']!r} isn't a base-game pdxmesh")
        base = game.read(rec["file"])
        try:
            geo = geometry_from_job(loadout.path(h["geometry"]))
        except (OSError, ValueError, KeyError) as exc:
            raise GenerationError(f"tier {tier}: can't read the fresh horse geometry {h['geometry']}: {exc}") from exc
        shape = summarize_mesh(parse_pdx_mesh(base))["shapes"][0]
        folder = rec["file"].rsplit("/", 1)[0]
        names, paths = {}, {}
        for tex in ("diffuse", "specular", "normal"):
            try:
                img = loadout.image(h["textures"][tex]).convert("RGBA")
            except LoadoutError as exc:
                raise GenerationError(f"tier {tier}: {exc}") from exc
            w, hh = img.size
            if w & (w - 1) or hh & (hh - 1):
                raise GenerationError(f"tier {tier}: the fresh {kind}'s {tex} is {w}x{hh}; sizes must be powers of two")
            fmt = read_header(game.read(f"{folder}/{shape[tex]}"))["format"]
            names[tex] = (naming.fresh_rider_texture if rider else naming.fresh_soldier_texture if soldier
                          else naming.fresh_gun_texture if gun else naming.fresh_horse_texture)(set_name, tier, tex)
            paths[tex] = f"{naming.part_dir(set_name)}/{names[tex]}"
            files.append(GeneratedFile(paths[tex], write_dds(img, DDSFormat(fmt, w, hh, full_mip_count(w, hh))), OWNED,
                                       procedural=True))
        try:
            data = write_fresh(base, geo, names)
        except FreshMeshError as exc:
            raise GenerationError(f"tier {tier}: fresh {kind}: {exc}") from exc
        file, name = ((naming.fresh_rider_file(set_name, tier), naming.fresh_rider_pdxmesh(set_name, tier)) if rider
                      else (naming.fresh_soldier_file(set_name, tier), naming.fresh_soldier_pdxmesh(set_name, tier))
                      if soldier else (naming.fresh_gun_file(set_name, tier), naming.fresh_gun_pdxmesh(set_name, tier))
                      if gun else (naming.fresh_horse_file(set_name, tier), naming.fresh_horse_pdxmesh(set_name, tier)))
        files.append(GeneratedFile(file, data, OWNED, derived_from=rec["file"]))
        anims = dict(rec.get("animations") or {})
        if soldier:  # the infantry's own ids (its stance pdxmesh's, or the culture's) over the body's
            worn = by_name.get(entity["pdxmesh"]) or idx.pdxmesh(entity["pdxmesh"]) or {}
            anims.update(worn.get("animations") or {})
        pdx = {"name": name, "file": file, "base": h["base"], "animations": anims,
               "scale": rec.get("scale"), "extra": dict(rec.get("extra") or {}), "settings": []}
        records.append(pdx)
        by_name[name] = pdx
        entity["pdxmesh"] = name
        out[tier] = FreshHorse(tier, entity["name"], name, file, h["base"], rec["file"], paths, len(geo.tri) // 3,
                               h["rights"], h.get("source", ""))
        notes.append(f"tier {tier}: {entity['name']} is a fresh {kind} ({file}:{len(geo.tri) // 3} triangles on "
                     f"{h['base']}'s skeleton and animations); reference {h.get('source') or '(none given)'}, "
                     f"rights: {h['rights']}")
    return files, out, notes
