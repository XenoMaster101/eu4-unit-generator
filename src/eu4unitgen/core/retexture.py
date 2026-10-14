"""Retextured meshes without copying a mesh (SPEC §4, §10 Phase 2).

A retextured slot gets its own pdxmesh whose ``file`` is the vanilla ``.mesh`` and whose
``meshsettings`` block names new textures for the mesh's shape (syntax from vanilla
``interface/assets/dharma_mapitems.gfx``).

Confirmed in-game (test 4, 2026-10-14): the override works on unit meshes, and the engine
**prefixes the mesh's folder to every meshsettings texture name**, so names must be bare file
names and the DDS goes into the mod's copy of the vanilla mesh's folder
(``gfx/models/Units/hr_mac_infantry_1_diffuse.dds``). Textures we don't change keep their
vanilla bare names. A generated name must never equal a vanilla/DLC texture's name in that
folder: the mod's file would replace it for every nation (rule ``texture_paths``).

Bases are base-game pdxmeshes only; derived textures are made from base-game files only and
go to generated mod output only (CLAUDE.md rule 2, SPEC §9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable

from PIL import Image

from eu4unitgen.core.generator import OWNED, BaseChain, GeneratedFile, GenerationError
from eu4unitgen.core.naming import MESH_ROLES, Naming
from eu4unitgen.library.scanner import GameFiles, Index, parse_pdx_mesh, summarize_mesh
from eu4unitgen.textures.dds import DDSFormat, read_dds, read_header, write_dds

SHADER_FILE = "gfx/FX/pdxmesh.shader"  # as in every vanilla meshsettings block


@dataclass
class BaseTextures:
    """The base mesh's shape and its decoded textures (base game only)."""
    pdxmesh: str
    mesh_file: str
    folder: str
    shape: dict  # summarize_mesh shape: shape, shader, diffuse, normal, specular (bare names)
    game: GameFiles
    _cache: dict = field(default_factory=dict, repr=False)

    def path(self, kind: str) -> str:
        return f"{self.folder}/{self.shape[kind]}"

    def raw(self, kind: str) -> bytes:
        rel = self.path(kind)
        if not self.game.in_base_game(rel):
            raise GenerationError(f"{rel} isn't in the base game")
        return self.game.read(rel)

    def image(self, kind: str) -> Image.Image:
        if kind not in self._cache:
            self._cache[kind] = read_dds(self.raw(kind))
        return self._cache[kind]

    @property
    def diffuse(self) -> Image.Image:
        return self.image("diffuse")

    @property
    def specular(self) -> Image.Image:
        return self.image("specular")

    @property
    def uses_country_colour(self) -> bool:
        return self.shape.get("shader") == "PdxMeshColor"


@dataclass
class Edited:
    """New images for a slot; None keeps the vanilla texture."""
    diffuse: Image.Image | None = None
    specular: Image.Image | None = None


@dataclass
class Retexture:
    role: str  # MESH_ROLES
    tier: int  # the sprite tier whose entities use the new pdxmesh
    base_pdxmesh: str  # vanilla, base game
    edit: Callable[[BaseTextures], Edited]

    def __post_init__(self) -> None:
        if self.role not in MESH_ROLES:
            raise GenerationError(f"retexture role {self.role!r} must be one of {MESH_ROLES}")


def base_textures(pdxmesh: str, game: GameFiles, idx: Index) -> BaseTextures:
    rec = idx.pdxmesh(pdxmesh, base_only=True)
    if rec is None:
        raise GenerationError(f"{pdxmesh!r} isn't a base-game pdxmesh (DLC meshes are never a base, SPEC §9)")
    if rec.get("meshsettings"):
        raise GenerationError(f"{pdxmesh!r} already has meshsettings; retexturing it isn't supported")
    file = rec.get("file")
    if not file or not game.in_base_game(file):
        raise GenerationError(f"{pdxmesh!r}: mesh {file!r} isn't in the base game")
    shapes = summarize_mesh(parse_pdx_mesh(game.read(file)))["shapes"]
    if len(shapes) != 1:
        raise GenerationError(f"{file} has {len(shapes)} shapes; only one-shape meshes can be retextured yet")
    shape = shapes[0]
    if not all(shape.get(k) for k in ("diffuse", "normal", "specular", "shader", "shape")):
        raise GenerationError(f"{file}: shape {shape} doesn't name all three textures and a shader")
    return BaseTextures(pdxmesh, file, str(PurePosixPath(file).parent), shape, game)


def retextured_mesh(naming: Naming, set_name: str, r: Retexture, game: GameFiles,
                    idx: Index) -> tuple[dict, list[GeneratedFile]]:
    """The template record of the new pdxmesh, and the derived DDS files it names (written
    into the mod at the vanilla mesh's folder, named by bare file name)."""
    base = base_textures(r.base_pdxmesh, game, idx)
    rec = idx.pdxmesh(r.base_pdxmesh, base_only=True)
    edited = r.edit(base)
    names = {"diffuse": base.shape["diffuse"], "normal": base.shape["normal"], "specular": base.shape["specular"]}
    files = []
    for kind, img in (("diffuse", edited.diffuse), ("specular", edited.specular)):
        if img is None:
            continue
        raw = base.raw(kind)
        name = naming.texture(set_name, r.role, r.tier, kind)
        files.append(GeneratedFile(f"{base.folder}/{name}", write_dds(img, DDSFormat.of(read_header(raw))), OWNED,
                                   derived_from=base.path(kind)))
        names[kind] = name
    if not files:
        raise GenerationError(f"{r.role} tier {r.tier}: the edit changed no texture")
    mesh = {
        "name": naming.pdxmesh(set_name, r.role, r.tier), "file": base.mesh_file, "base": r.base_pdxmesh,
        "animations": dict(rec.get("animations") or {}), "scale": rec.get("scale"), "extra": dict(rec.get("extra") or {}),
        "settings": [{"shape": base.shape["shape"], "index": 0, "diffuse": names["diffuse"], "normal": names["normal"],
                      "specular": names["specular"], "shader": base.shape["shader"], "shader_file": SHADER_FILE}],
    }
    return mesh, files


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


def default_base(chain: BaseChain, role: str, tier: int) -> str:
    """The base culture's own pdxmesh for a slot (base game only; SPEC §4 recolour bases)."""
    if role == "infantry":
        return chain.slot("infantry", tier)["pdxmesh"]
    if role == "horse":
        return chain.slot("cavalry", tier)["horse_pdxmesh"]
    return chain.slot("artillery", tier)["cannon_pdxmesh"]


def retexture_set(naming: Naming, set_name: str, owner: str, entities: list[dict], recipe, chain: BaseChain,
                  game: GameFiles, idx: Index, version: int) -> tuple[list[dict], list[GeneratedFile], list[str]]:
    """Apply a TextureRecipe to ``owner``'s entities: one retextured pdxmesh per active slot,
    its derived DDS files, and notes. ``entities`` is changed in place."""
    from eu4unitgen.core.recipe import RecipeError, apply_slot

    meshes, files, notes = [], [], []
    by_tier: dict[int, dict[str, str]] = {}
    for (tier, role), slot in recipe.active().items():
        base = slot.base or default_base(chain, role, tier)
        holder: dict = {}

        def edit(b: BaseTextures, slot=slot, role=role, holder=holder) -> Edited:
            result = apply_slot(recipe, slot, b.diffuse, b.specular, b.shape["shader"], role)
            holder["notes"] = result.notes
            return Edited(result.diffuse, result.specular)
        try:
            mesh, made = retextured_mesh(naming, set_name, Retexture(role, tier, base, edit), game, idx)
        except RecipeError as exc:
            raise GenerationError(f"texture recipe slot {tier}/{role}: {exc}") from exc
        meshes.append(mesh)
        files += made
        by_tier.setdefault(tier, {})[role] = mesh["name"]
        notes += [f"textures {tier}/{role} (base {base}): {n}" for n in holder.get("notes", [])]
    for tier, roles in sorted(by_tier.items()):
        apply_retextures(entities, owner, tier, roles, chain, naming, version)
    return meshes, files, notes
