"""Procedural rigid parts: weapons, shields and helmet crests attached to a bone (SPEC §10 Phase 3).

A part is our own geometry (built by the Blender job ``build_parts``) and our own textures
(``textures.procedural``): nothing is derived from game files. Each part becomes a static
.mesh in the set's model folder, a pdxmesh and an entity that unit entities attach to a bone:

    attach = { Right_hand_node = "hr_mac_dory_1"  head = "hr_mac_crest_1" }

Vanilla conventions (measured, base_models.json): hand weapons run along +Z with the grip at
the origin, pdxmesh scale 0.5, PdxMeshStandard, no bones or locators. Attaching to ``head``
works (vanilla never does it; in-game test 7).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core.generator import OWNED, GeneratedFile, GenerationError
from eu4unitgen.core.naming import Naming
from eu4unitgen.library.scanner import strip_mesh_props
from eu4unitgen.textures.procedural import part_textures

PART_SCALE = 0.5  # vanilla weapons' pdxmesh scale
# (diffuse colour, gloss %) per colour zone; the Blender job maps each zone's faces to its swatch
PART_ZONES = {
    "dory": [((110, 75, 40), 3), ((180, 130, 60), 13)],  # wood shaft, bronze head + sauroter
    "crest": [((170, 20, 25), 3)],  # dyed horsehair
}
ATTACH_BONE = {"dory": "Right_hand_node", "crest": "head"}
BUILT = tuple(PART_ZONES)  # kinds the Blender job can build so far (spike: dory, crest)


@dataclass
class PartSpec:
    kind: str
    tier: int
    params: dict = field(default_factory=dict)


@dataclass
class BuiltParts:
    meshes: list[dict]  # pdxmesh template records
    entities: list[dict]  # entity template records
    files: list[GeneratedFile]
    attach: dict[tuple[int, str], str]  # (tier, bone) -> part entity
    log: str = ""


def build_parts(runner, naming: Naming, set_name: str, specs: list[PartSpec], version: int) -> BuiltParts:
    """Build ``specs`` in background Blender (``runner``: blender.runner.BlenderRunner)."""
    for s in specs:
        if s.kind not in BUILT:
            raise GenerationError(f"part kind {s.kind!r} can't be built yet (built: {', '.join(BUILT)})")
    out = BuiltParts([], [], [], {})
    with tempfile.TemporaryDirectory(prefix="eu4unitgen_parts_") as tmp:
        job = []
        for s in specs:
            name = naming.part(set_name, s.kind, s.tier)
            job.append({"kind": s.kind, "shape": f"{name}Shape", "out_mesh": str(Path(tmp) / f"{name}.mesh"),
                        "shader": "PdxMeshStandard", "zones": len(PART_ZONES[s.kind]), "params": s.params,
                        "textures": {"diff": naming.part_texture(set_name, s.kind, s.tier, "diffuse"),
                                     "n": naming.part_texture(set_name, s.kind, s.tier, "normal"),
                                     "spec": naming.part_texture(set_name, s.kind, s.tier, "specular")}})
        res = runner.run_job("build_parts", {"parts": job}, timeout=300)
        out.log = res.log
        for s, built in zip(specs, res.result["parts"]):
            data = Path(built["out_mesh"]).read_bytes()
            mesh_rel = naming.part_mesh_file(set_name, s.kind, s.tier)
            out.files.append(GeneratedFile(mesh_rel, strip_mesh_props(data), OWNED, procedural=True))
            for kind, dds in part_textures(PART_ZONES[s.kind]).items():
                rel = f"{naming.part_dir(set_name)}/{naming.part_texture(set_name, s.kind, s.tier, kind)}"
                out.files.append(GeneratedFile(rel, dds, OWNED, procedural=True))
            pdx = naming.pdxmesh(set_name, s.kind, s.tier)
            out.meshes.append({"name": pdx, "file": mesh_rel, "animations": {}, "scale": PART_SCALE, "extra": {},
                               "settings": []})
            entity = naming.part(set_name, s.kind, s.tier)
            out.entities.append({"name": entity, "pdxmesh": pdx, "version": version, "states": [], "attaches": []})
            out.attach[(s.tier, ATTACH_BONE[s.kind])] = entity
    return out


def attach_parts(entities: list[dict], owner: str, attach: dict[tuple[int, str], str], naming: Naming) -> None:
    """Put parts on ``owner``'s infantry entities (which also dress riders' and gun crews' bodies
    only when those use the infantry entity: artillery crews do; riders are separate entities).
    A part on a bone replaces whatever that infantry already attaches there (e.g. the halberd)."""
    by_name = {e["name"]: e for e in entities}
    for (tier, bone), part in sorted(attach.items()):
        ent = by_name[naming.unit_entity(owner, "infantry", tier)]
        kept = []
        for a in ent["attaches"]:
            nodes = [(n, t) for n, t in a["nodes"] if n != bone]
            if nodes:
                kept.append({**a, "nodes": nodes})
        unnamed = next((a for a in kept if a["name"] is None), None)
        if unnamed is None:
            unnamed = {"name": None, "nodes": []}
            kept.append(unnamed)
        unnamed["nodes"].append((bone, part))
        ent["attaches"] = kept
