"""Retextured meshes without copying a mesh (SPEC §10 Phase 2).

A retextured slot gets its own pdxmesh whose ``file`` is the vanilla ``.mesh`` and whose
``meshsettings`` block names new textures for the mesh's shape. The syntax is vanilla
(``interface/assets/dharma_mapitems.gfx``), but there it only restates the mesh's own
textures, so whether it overrides a unit mesh's textures is UNVERIFIED until in-game test 4.

Two ways to name the new textures, both under test:
  * ``BESIDE_MESH``: bare file names, like vanilla; the DDS goes into the mod at the vanilla
    mesh's own folder (where the mesh's bare texture names resolve, SPEC §4).
  * ``OWN_DIR``: full paths into the set's texture folder (``naming.texture_dir``). UNVERIFIED
    whether texture names in meshsettings may be paths at all.

Bases are base-game pdxmeshes only; derived textures are made from base-game files only and
go to generated mod output only (CLAUDE.md rule 2, SPEC §9).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from PIL import Image

from eu4unitgen.core.generator import OWNED, BaseChain, GeneratedFile, GenerationError
from eu4unitgen.core.naming import MESH_ROLES, Naming
from eu4unitgen.library.scanner import GameFiles, Index, parse_pdx_mesh, summarize_mesh
from eu4unitgen.textures.dds import DDSFormat, read_dds, read_header, write_dds

BESIDE_MESH, OWN_DIR = "beside_mesh", "own_dir"
PATH_MODES = (BESIDE_MESH, OWN_DIR)
SHADER_FILE = "gfx/FX/pdxmesh.shader"  # as in every vanilla meshsettings block


@dataclass
class Retexture:
    role: str  # MESH_ROLES
    tier: int  # the sprite tier whose entities use the new pdxmesh
    base_pdxmesh: str  # vanilla, base game
    path_mode: str  # PATH_MODES
    edit: Callable[[Image.Image], Image.Image]  # applied to the base diffuse

    def __post_init__(self) -> None:
        if self.role not in MESH_ROLES:
            raise GenerationError(f"retexture role {self.role!r} must be one of {MESH_ROLES}")
        if self.path_mode not in PATH_MODES:
            raise GenerationError(f"path mode {self.path_mode!r} must be one of {PATH_MODES}")


def retextured_mesh(naming: Naming, set_name: str, r: Retexture, game: GameFiles,
                    idx: Index) -> tuple[dict, list[GeneratedFile]]:
    """The template record of the new pdxmesh, and the derived DDS file(s) it names."""
    rec = idx.pdxmesh(r.base_pdxmesh, base_only=True)
    if rec is None:
        raise GenerationError(f"{r.base_pdxmesh!r} isn't a base-game pdxmesh (DLC meshes are never a base, SPEC §9)")
    if rec.get("meshsettings"):
        raise GenerationError(f"{r.base_pdxmesh!r} already has meshsettings; retexturing it isn't supported")
    file = rec.get("file")
    if not file or not game.in_base_game(file):
        raise GenerationError(f"{r.base_pdxmesh!r}: mesh {file!r} isn't in the base game")
    shapes = summarize_mesh(parse_pdx_mesh(game.read(file)))["shapes"]
    if len(shapes) != 1:
        raise GenerationError(f"{file} has {len(shapes)} shapes; only one-shape meshes can be retextured yet")
    shape = shapes[0]
    if not all(shape.get(k) for k in ("diffuse", "normal", "specular", "shader", "shape")):
        raise GenerationError(f"{file}: shape {shape} doesn't name all three textures and a shader")
    folder = str(PurePosixPath(file).parent)
    base_diffuse = f"{folder}/{shape['diffuse']}"
    if not game.in_base_game(base_diffuse):
        raise GenerationError(f"{base_diffuse} isn't in the base game")
    data = game.read(base_diffuse)
    new = write_dds(r.edit(read_dds(data)), DDSFormat.of(read_header(data)))

    name = naming.texture(set_name, r.role, r.tier, "diffuse")
    if r.path_mode == BESIDE_MESH:
        path, diffuse = f"{folder}/{name}", name
        normal, specular = shape["normal"], shape["specular"]
    else:
        path = diffuse = f"{naming.texture_dir(set_name)}/{name}"
        normal, specular = f"{folder}/{shape['normal']}", f"{folder}/{shape['specular']}"
    mesh = {
        "name": naming.pdxmesh(set_name, r.role, r.tier), "file": file, "base": r.base_pdxmesh,
        "animations": dict(rec.get("animations") or {}), "scale": rec.get("scale"), "extra": dict(rec.get("extra") or {}),
        "settings": [{"shape": shape["shape"], "index": 0, "diffuse": diffuse, "normal": normal,
                      "specular": specular, "shader": shape["shader"], "shader_file": SHADER_FILE}],
    }
    return mesh, [GeneratedFile(path, new, OWNED, derived_from=base_diffuse)]


def apply_retextures(entities: list[dict], owner: str, tier: int, meshes: dict[str, str], chain: BaseChain,
                     naming: Naming, version: int) -> list[dict]:
    """Point one tier of ``owner``'s entities (from ``unit_entities``) at retextured pdxmeshes
    (role -> pdxmesh name). The rider uses the infantry's. A retextured cannon gets the owner's
    own cannon entity (a full copy of the vanilla one), attached on the gunman's cannon_node."""
    by_name = {e["name"]: e for e in entities}
    if "infantry" in meshes:
        for name in (naming.unit_entity(owner, "infantry", tier), naming.rider_entity(owner, tier)):
            by_name[name]["pdxmesh"] = meshes["infantry"]
    if "horse" in meshes:
        by_name[naming.unit_entity(owner, "cavalry", tier)]["pdxmesh"] = meshes["horse"]
    if "cannon" in meshes:
        base = chain.flat(chain.slot("artillery", tier)["cannon_entity"])
        cannon = naming.cannon_entity(owner, tier)
        entities.append({"name": cannon, "pdxmesh": meshes["cannon"], "version": version,
                         "states": base["states"], "attaches": base["attaches"]})
        art = by_name[naming.unit_entity(owner, "artillery", tier)]
        for a in art["attaches"]:
            if a["name"] == "gunman":
                a["nodes"] = [(node, cannon if node == "cannon_node" else target) for node, target in a["nodes"]]
    return entities
