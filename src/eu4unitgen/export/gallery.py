"""Whole-unit previews for the Result screen's gallery (SPEC Track F, F9).

A unit on the map is an entity tree: the entity's pdxmesh plays its state's animation, and each
``attach`` puts another entity on a bone of it (the rider on the horse's ``Saddle_Node``, the gun
crew on ``infantry_node`` and the cannon on ``cannon_node`` of the artillery base, weapons,
shields and helmets on hand and head bones). ``entity_tree`` reads that tree from a World (the
generated set over the mod and the game, as the validator sees it), so the gallery shows the
entities the game will get. ``stage`` writes the tree's meshes and their textures (diffuse tinted
with the unit colours, as PdxMeshColor does) into a temporary folder for the ``render_unit``
Blender job. Everything staged is base-game-derived: temporary files and user output only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

MAX_DEPTH = 4  # horse -> rider -> weapon; base -> crew -> shield


@dataclass
class Node:
    entity: str
    mesh_file: str  # game path of the .mesh
    scale: float = 1.0
    anim_file: str | None = None  # game path of the .anim the state plays (None: rigid)
    attach: list[tuple[str, "Node"]] = field(default_factory=list)  # (bone on this mesh, child)


class GalleryError(ValueError):
    pass


def entity_tree(world, entity: str, state: str = "idle", depth: int = 0) -> Node:
    """``entity`` as the game assembles it in ``state``: mesh, animation, attached entities."""
    from eu4unitgen.library.scanner import resolved

    if depth > MAX_DEPTH:
        raise GalleryError(f"{entity}: attaches nest deeper than {MAX_DEPTH}")
    if world.idx.entity(entity) is None:  # the winning definition; base game, mod and set (never DLC)
        raise GalleryError(f"entity {entity!r} isn't defined")
    pdxmesh = resolved(world.idx, entity, "pdxmesh")
    pm = world.idx.pdxmesh(pdxmesh or "")
    if pm is None:
        raise GalleryError(f"{entity}: pdxmesh {pdxmesh!r} isn't defined")
    anim = None
    for s in resolved(world.idx, entity, "states") or []:
        if s.get("name") == state and s.get("animation"):
            atype = world.idx.animation_type((pm.get("animations") or {}).get(s["animation"]) or "")
            anim = atype["file"] if atype else None
    node = Node(entity, pm["file"], float(pm.get("scale") or 1.0), anim)
    for a in resolved(world.idx, entity, "attach") or []:
        for bone, child in a.items():
            if bone != "name" and isinstance(child, str):
                node.attach.append((bone, entity_tree(world, child, state, depth + 1)))
    return node


def parts_of(node: Node) -> dict[str, str]:
    """{bone: entity} for the rigid parts anywhere in the tree (a rider's or a crew's too): what
    the Parts cards show."""
    out: dict[str, str] = {}
    for bone, child in node.attach:
        if child.anim_file is None:
            out[bone] = child.entity
        out.update(parts_of(child))
    return out


def fingerprint(node: Node, world, colours) -> str:
    """A hash of everything ``stage`` would hand the renderer for ``node``'s tree: each mesh, the
    textures its shapes name, the animation, scales, attach bones, and the unit colours. Two equal
    fingerprints render the same tile, so an edit re-renders only the units whose print changed."""
    import hashlib

    from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh

    h = hashlib.sha256(repr([list(c) for c in colours]).encode("utf-8"))

    def walk(n: Node) -> None:
        h.update(f"|{n.entity}|{n.mesh_file}|{n.scale}|{n.anim_file}|".encode("utf-8"))
        data = world.read(n.mesh_file)
        h.update(data or b"<missing>")
        if data is not None:
            base = str(PurePosixPath(n.mesh_file).parent)
            for shape in summarize_mesh(parse_pdx_mesh(data))["shapes"]:
                for k in ("diffuse", "specular", "normal"):
                    name = shape.get(k)
                    h.update(f"|{k}={name}|".encode("utf-8"))
                    h.update((world.read(f"{base}/{name}") if name else None) or b"<none>")
        if n.anim_file:
            h.update(world.read(n.anim_file) or b"<missing>")
        for bone, child in n.attach:
            h.update(f"|attach {bone}|".encode("utf-8"))
            walk(child)
    walk(node)
    return h.hexdigest()


def stage(node: Node, world, colours, folder: Path, n: list | None = None) -> dict:
    """Write ``node``'s tree into ``folder`` (one subfolder per mesh, its textures beside it as its
    materials name them) -> the Blender job's node: {"mesh", "anim", "scale", "attach": [{"bone",
    "bind", "node"}]}."""
    from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh
    from eu4unitgen.textures.dds import DDSFormat, full_mip_count, read_dds, write_dds
    from eu4unitgen.textures.recolour import shade

    n = n if n is not None else [0]
    n[0] += 1
    here = folder / f"{n[0]:02d}_{node.entity}"
    here.mkdir(parents=True, exist_ok=True)
    data = world.read(node.mesh_file)
    if data is None:
        raise GalleryError(f"{node.entity}: {node.mesh_file} isn't in the game, the mod or the set")
    mesh_path = here / PurePosixPath(node.mesh_file).name
    mesh_path.write_bytes(data)
    summary = summarize_mesh(parse_pdx_mesh(data))
    base = str(PurePosixPath(node.mesh_file).parent)
    textured = False
    for shape in summary["shapes"]:
        names = {k: shape.get(k) for k in ("diffuse", "specular", "normal")}
        imgs = {}
        for k, name in names.items():
            raw = world.read(f"{base}/{name}") if name else None
            if raw is not None:
                try:
                    imgs[k] = read_dds(raw)
                except (OSError, ValueError):
                    pass
        if names["diffuse"] and "diffuse" in imgs:
            alpha = imgs["diffuse"].convert("RGBA").getchannel("A").getextrema()
            textured = textured or alpha[1] > 0
            d = shade(imgs["diffuse"], imgs.get("specular"), colours, shape.get("shader") or "")
            w, h = d.size
            (here / names["diffuse"]).write_bytes(write_dds(d.convert("RGBA"), DDSFormat("DXT1", w, h,
                                                                                     full_mip_count(w, h))))
    # a mesh whose diffuse is missing or fully transparent is a helper the game doesn't show
    # (artillery_base is one triangle under the gun, textured with nospec.dds: 4x4, alpha 0): hidden
    out = {"name": node.entity, "mesh": str(mesh_path), "scale": node.scale, "anim": None, "attach": [],
           "hidden": bool(summary["shapes"]) and not textured}
    if node.anim_file:
        raw = world.read(node.anim_file)
        if raw is not None:
            anim_path = here / PurePosixPath(node.anim_file).name
            anim_path.write_bytes(raw)
            out["anim"] = str(anim_path)
    for bone, child in node.attach:
        bind = (summary["skeleton"].get(bone) or {}).get("bind")
        out["attach"].append({"bone": bone, "bind": list(bind) if bind else None,
                              "node": stage(child, world, colours, folder, n)})
    return out
