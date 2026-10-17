"""Validator rules. Each rule checks the generated set in the context of the World
(vanilla + DLC + existing mod + generated) and returns Issues.

Severity: ``error`` blocks export; ``warning`` is shown in the dry run; ``info`` is a note.
SPEC §6 lists the error.log lines these rules pre-empt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from eu4unitgen.core.generator import BINARY_SUFFIXES, SNIPPET, GeneratedSet, set_trigger
from eu4unitgen.core.naming import LOADOUT_ROLES, PART_KINDS, TIERS, UNIT_TYPES, Naming
from eu4unitgen.core.nation import MANEUVER, Nation
from eu4unitgen.core.project import Project
from eu4unitgen.library.tags import TagFacts, apply_history, coverage, culture_groups
from eu4unitgen.pdxscript.parser import Block, Pair, Scalar, parse_bytes
from eu4unitgen.validate.graph import GENERATED, MOD, VANILLA, Graph, Node, World

ERROR, WARNING, INFO = "error", "warning", "info"
LIVE_STATES = {"idle", "walk", "attack", "retreat"}
RIDER_IDS = {"horse_idle", "horse_walk", "horse_attack", "horse_retreat"}


@dataclass(frozen=True)
class Issue:
    severity: str
    rule: str
    message: str
    path: str | None = None
    subject: str | None = None

    def __str__(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        return f"{self.severity.upper():7} {self.rule}: {self.message}{where}"


@dataclass
class Context:
    project: Project
    nation: Nation
    generated: GeneratedSet
    world: World
    graph: Graph

    @property
    def naming(self) -> Naming:
        return self.project.naming

    def gen_entities(self) -> dict[str, dict]:
        return {n: d for n, defs in self.world.idx.entities.items() for d in defs if d["origin"] == GENERATED}

    @property
    def owner(self) -> str:
        """Who the set's entities are named after (its first tag, or the set for a culture-only set)."""
        return self.naming.entity_owner(self.nation.tags, self.nation.set_name)

    def gen_defs(self, kind: str) -> dict[str, dict]:
        return {n: d for n, defs in getattr(self.world.idx, kind).items() for d in defs if d["origin"] == GENERATED}


Rule = Callable[[Context], list[Issue]]
RULES: dict[str, Rule] = {}


def rule(name: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        RULES[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- files

@rule("parse_clean")
def parse_clean(ctx: Context) -> list[Issue]:
    """Generated script must parse with no warnings (no stray or missing braces)."""
    out = []
    for f in ctx.generated.files.values():
        if f.path.endswith((".yml", ".md")) or is_binary(f.path):
            continue
        for w in parse_bytes(f.data).warnings:
            out.append(Issue(ERROR, "parse_clean", w, f.path))
    return out


@rule("file_encoding")
def file_encoding(ctx: Context) -> list[Issue]:
    out = []
    for f in ctx.generated.files.values():
        if f.path.endswith(".yml"):
            if not f.data.startswith(b"\xef\xbb\xbf"):
                out.append(Issue(ERROR, "file_encoding", "localisation needs a UTF-8 BOM", f.path))
            if not re.search(r"_l_[a-z_]+\.yml$", f.path):
                out.append(Issue(ERROR, "file_encoding", "localisation file name must end _l_<language>.yml", f.path))
            first = f.data[3:].split(b"\n", 1)[0].strip()
            lang = re.search(r"_l_([a-z_]+)\.yml$", f.path)
            if lang and first != f"l_{lang.group(1)}:".encode():
                out.append(Issue(ERROR, "file_encoding", f"first line must be l_{lang.group(1)}:", f.path))
        elif f.role != SNIPPET and f.based_on is None and not is_binary(f.path):
            try:
                f.data.decode("utf-8")
            except UnicodeDecodeError:
                out.append(Issue(ERROR, "file_encoding", "generated script is not valid UTF-8", f.path))
    return out


# --------------------------------------------------------------------------- textures

TEXTURE_KEYS = ("texture_diffuse", "texture_normal", "texture_specular")
SHADERS = {"PdxMeshColor", "PdxMeshStandard", "PdxMeshTextureAtlas", "PdxMeshSnow", "PdxMeshAlphaBlend",
           "PdxMeshAlphaBlendNoZWrite", "PdxMesh_NoFoW_NoTI", "Collision"}  # SPEC §4


def is_binary(path: str) -> bool:
    return path.lower().endswith(BINARY_SUFFIXES)


@rule("dds_format")
def dds_format(ctx: Context) -> list[Issue]:
    """Every generated DDS is derived from a base-game texture (never DLC, CLAUDE.md rule 2) and
    matches its format, size and mip count (SPEC §4: a mismatch shows as a black/wrong unit), or
    is a procedural texture (made from scratch) in vanilla's form: DXT1/DXT5, full mips, no DX10."""
    from eu4unitgen.textures.dds import DDSError, DDSFormat, check_matches, full_mip_count, read_header

    out = []
    game = ctx.world.game
    for f in ctx.generated.files.values():
        if not f.path.lower().endswith(".dds"):
            continue
        base = f.derived_from
        if f.procedural and not base:
            try:
                h = read_header(f.data)
                own = DDSFormat(h["format"], h["width"], h["height"], full_mip_count(h["width"], h["height"]))
                problems = check_matches(f.data, own)
            except (DDSError, ValueError) as exc:
                problems = [str(exc)]
            out += [Issue(ERROR, "dds_format", f"procedural texture: {p}", f.path) for p in problems]
            continue
        if not base:
            out.append(Issue(ERROR, "dds_format", "generated texture doesn't record the base texture it came from", f.path))
            continue
        if not game.in_base_game(base):
            what = "a DLC file" if game.exists(base) else "missing from the install"
            out.append(Issue(ERROR, "dds_format", f"base texture {base} is {what}; derived textures come from "
                                                  "base-game files only", f.path))
            continue
        try:
            target = DDSFormat.of(read_header(game.read(base)))
        except (DDSError, ValueError) as exc:
            out.append(Issue(ERROR, "dds_format", f"base texture {base}: {exc}", f.path))
            continue
        for problem in check_matches(f.data, target):
            out.append(Issue(ERROR, "dds_format", f"{problem} (base {base})", f.path))
    return out


@rule("mesh_settings")
def mesh_settings(ctx: Context) -> list[Issue]:
    """Generated pdxmeshes with a meshsettings block: the mesh is base game, the block names
    a shape the mesh has at that index, a known shader (the mesh's own), the vanilla shader
    file and three textures that resolve."""
    from eu4unitgen.validate.graph import meshsettings_texture_path

    out = []
    w = ctx.world
    for name, rec in sorted(ctx.gen_defs("pdxmeshes").items()):
        settings = rec.get("meshsettings") or []
        if not settings:
            continue
        where = rec["defined_in"]
        mesh_file = rec.get("file") or ""
        if not w.game.in_base_game(mesh_file) or w.origin_of_file(mesh_file) != VANILLA:
            out.append(Issue(ERROR, "mesh_settings", f"{name}: mesh {mesh_file} must be a base-game file "
                                                     "(no DLC, no mod copy)", where, name))
            continue
        shapes = (w.mesh_summary(mesh_file) or {}).get("shapes", [])
        for ms in settings:
            shape_name, index = ms.get("name"), ms.get("index")
            try:
                shape = shapes[int(index)]
            except (TypeError, ValueError, IndexError):
                out.append(Issue(ERROR, "mesh_settings", f"{name}: index {index!r} isn't a shape of {mesh_file} "
                                                         f"({len(shapes)} shapes)", where, name))
                continue
            if shape.get("shape") != shape_name:
                out.append(Issue(ERROR, "mesh_settings", f"{name}: shape {index} of {mesh_file} is "
                                                         f"{shape.get('shape')!r}, not {shape_name!r}", where, name))
            if ms.get("shader") not in SHADERS:
                out.append(Issue(ERROR, "mesh_settings", f"{name}: unknown shader {ms.get('shader')!r}", where, name))
            elif ms.get("shader") != shape.get("shader"):
                out.append(Issue(WARNING, "mesh_settings", f"{name}: shader {ms.get('shader')} differs from the "
                                                           f"mesh's own {shape.get('shader')}", where, name))
            if not ms.get("shader_file") or not w.exists(ms["shader_file"]):
                out.append(Issue(ERROR, "mesh_settings", f"{name}: shader_file {ms.get('shader_file')!r} not found",
                                 where, name))
            for key in TEXTURE_KEYS:
                if not ms.get(key):
                    out.append(Issue(ERROR, "mesh_settings", f"{name}: no {key}", where, name))
                    continue
                rel = meshsettings_texture_path(mesh_file, ms[key])
                if not w.exists(rel):
                    out.append(Issue(ERROR, "mesh_settings", f"{name}: {key} {ms[key]!r} resolves to {rel}, "
                                                             "which doesn't exist", where, name))
    return out


@rule("texture_paths")
def texture_paths(ctx: Context) -> list[Issue]:
    """The engine prefixes the mesh's folder to every meshsettings texture name (in-game test 4:
    full paths render black). So (1) meshsettings textures are bare file names; (2) every
    generated DDS sits in the folder of a mesh whose generated pdxmesh names it; (3) a generated
    texture never takes the path of a vanilla, DLC or mod file: in the shared folder, the mod's
    file would replace that texture for every nation."""
    from eu4unitgen.validate.graph import meshsettings_texture_path

    out, named = [], set()
    w = ctx.world
    for name, rec in sorted(ctx.gen_defs("pdxmeshes").items()):
        mesh_file = rec.get("file") or ""
        for ms in rec.get("meshsettings") or []:
            for key in TEXTURE_KEYS:
                tex = ms.get(key)
                if not tex:
                    continue
                if "/" in tex or "\\" in tex:
                    out.append(Issue(ERROR, "texture_paths", f"{name}: {key} {tex!r} must be a bare file name; the "
                                     f"engine prefixes the mesh's folder ({PurePosixPath(mesh_file).parent})",
                                     rec["defined_in"], name))
                    continue
                named.add(meshsettings_texture_path(mesh_file, tex).lower())
    for f in ctx.generated.files.values():  # textures a generated (procedural) .mesh names itself
        if f.path.lower().endswith(".mesh"):
            folder = str(PurePosixPath(f.path).parent)
            for shape in (w.mesh_summary(f.path) or {}).get("shapes", []):
                for key in ("diffuse", "normal", "specular"):
                    if shape.get(key):
                        named.add(f"{folder}/{shape[key]}".lower())
    for f in ctx.generated.files.values():
        if not f.path.lower().endswith(".dds"):
            continue
        if f.path.lower() not in named:
            out.append(Issue(ERROR, "texture_paths", "generated texture isn't in the folder of a mesh that names it "
                                                     "(a generated pdxmesh's meshsettings or a generated .mesh)", f.path))
        if w.game.exists(f.path):
            where = "base-game" if w.game.in_base_game(f.path) else "DLC"
            out.append(Issue(ERROR, "texture_paths", f"same path as a {where} texture: the mod's file would replace "
                                                     "it for every nation", f.path))
        elif w.mod is not None and w.mod.exists(f.path):
            out.append(Issue(ERROR, "texture_paths", "same path as a file already in the mod", f.path))
    return out


PART_TRI_BUDGET = 3000  # vanilla weapons: tens to a few hundred triangles; a body ~2,000


@rule("attach_bones")
def attach_bones(ctx: Context) -> list[Issue]:
    """Every attach node on a generated entity is a bone (or locator) of the entity's own mesh
    (case-insensitive: vanilla writes both Saddle_Node and saddle_node). Any bone works, including
    ``head``, which vanilla never uses (in-game test 7)."""
    w, out = ctx.world, []
    for name, rec in sorted(ctx.gen_entities().items()):
        pdx = w.pdxmesh(w.resolve(name, "pdxmesh") or "")
        summary = w.mesh_summary(pdx["file"]) if pdx and pdx.get("file") else None
        if not summary:
            continue  # missing_nodes reports it
        nodes = {b.lower() for b in summary.get("bones", [])} | {loc.lower() for loc in summary.get("locators", {})}
        for a in w.resolve(name, "attach") or []:
            for bone in (k for k in a if k != "name"):
                if bone.lower() not in nodes:
                    out.append(Issue(ERROR, "attach_bones", f"{name}: attach node {bone!r} isn't a bone or locator of "
                                                            f"{pdx['file']}", rec["defined_in"], name))
    return out


@rule("part_mesh")
def part_mesh(ctx: Context) -> list[Issue]:
    """Generated .mesh files (procedural parts): parse, vanilla static-weapon layout (no
    boundingsphere, no skeleton), a known shader, within the triangle budget, non-degenerate."""
    from eu4unitgen.library.scanner import _walk, parse_pdx_mesh

    out = []
    for f in ctx.generated.files.values():
        if not f.path.lower().endswith(".mesh") or f.derived_from:
            continue  # derived bodies: rule derived_body
        try:
            tree = parse_pdx_mesh(f.data)
        except (ValueError, IndexError) as exc:
            out.append(Issue(ERROR, "part_mesh", f"doesn't parse: {exc}", f.path))
            continue
        tris = 0
        for node in _walk(tree):
            if "boundingsphere" in node["props"]:
                out.append(Issue(ERROR, "part_mesh", "has a boundingsphere (vanilla meshes never do)", f.path))
            if node["name"] == "skeleton":
                out.append(Issue(ERROR, "part_mesh", "has a skeleton; rigid parts are static meshes", f.path))
            tris += len(node["props"].get("tri", [])) // 3
        summary = ctx.world.mesh_summary(f.path) or {}
        for shape in summary.get("shapes", []):
            if shape.get("shader") not in SHADERS:
                out.append(Issue(ERROR, "part_mesh", f"shape {shape.get('shape')}: unknown shader {shape.get('shader')!r}",
                                 f.path))
        if tris == 0 or tris > PART_TRI_BUDGET:
            out.append(Issue(ERROR, "part_mesh", f"{tris} triangles (budget 1-{PART_TRI_BUDGET})", f.path))
        box = summary.get("aabb")
        if not box or any(hi - lo <= 0 for lo, hi in zip(box["min"], box["max"])):
            out.append(Issue(ERROR, "part_mesh", f"degenerate bounding box {box}", f.path))
    return out


@rule("loadout")
def loadout(ctx: Context) -> list[Issue]:
    """A loadout's parts, crews and stances (core.loadout): each part's entity, pdxmesh and .mesh
    are generated and agree (shader included); it hangs on its kind's bone and nowhere else; hand
    weapons have their grip at the origin; a country-coloured part's spec map carries the mask;
    a shield-bearer uses the spear_shield set; a crew entity is the gunman of its artillery; a
    stance is in the pdxmesh."""
    from eu4unitgen.core.loadout import STANCE_TYPES
    from eu4unitgen.core.parts import KINDS
    from eu4unitgen.textures.dds import DDSError, read_dds

    res = ctx.generated.loadout
    if res is None:
        return []
    w, out = ctx.world, []
    ents = ctx.gen_entities()
    hung: dict[str, set[str]] = {}  # part entity -> bones it hangs on
    for name in ents:
        for a in w.resolve(name, "attach") or []:
            for bone, target in a.items():
                if bone != "name":
                    hung.setdefault(target, set()).add(bone)
    for r in res.parts:
        rec = ents.get(r.entity)
        if rec is None:
            out.append(Issue(ERROR, "loadout", f"part entity {r.entity} wasn't generated", subject=r.entity))
            continue
        where = rec["defined_in"]
        pdx_name = w.resolve(r.entity, "pdxmesh") or ""
        pdx = ctx.gen_defs("pdxmeshes").get(pdx_name)
        if pdx is None or (pdx.get("file") or "").lower() != r.mesh_file.lower():
            out.append(Issue(ERROR, "loadout", f"{r.entity}: pdxmesh {pdx_name!r} isn't a generated pdxmesh on "
                                               f"{r.mesh_file}", where, r.entity))
            continue
        mesh = ctx.generated.files.get(r.mesh_file)
        if mesh is None:
            out.append(Issue(ERROR, "loadout", f"{r.entity}: {r.mesh_file} wasn't generated", where, r.entity))
            continue
        summary = w.mesh_summary(r.mesh_file) or {}
        shapes = summary.get("shapes", [])
        if [sh.get("shader") for sh in shapes] != [r.shader]:
            out.append(Issue(ERROR, "loadout", f"{r.entity}: mesh shader {[sh.get('shader') for sh in shapes]} "
                                               f"isn't the part's {r.shader}", r.mesh_file, r.entity))
        bones = hung.get(r.entity, set())
        if not bones:
            out.append(Issue(WARNING, "loadout", f"{r.entity}: no generated entity attaches it", where, r.entity))
        elif {b.lower() for b in bones} != {r.bone.lower()}:
            out.append(Issue(ERROR, "loadout", f"{r.entity}: a {r.kind} belongs on {r.bone}, but hangs on "
                                               f"{sorted(bones)}", where, r.entity))
        box = summary.get("aabb")
        if KINDS[r.kind].bone == "Right_hand_node" and box and not box["min"][2] < 0 < box["max"][2]:
            out.append(Issue(ERROR, "loadout", f"{r.entity}: the grip (z = 0) isn't inside the weapon "
                                               f"(z {box['min'][2]:.2f}..{box['max'][2]:.2f})", r.mesh_file, r.entity))
        if r.shader == "PdxMeshColor" and shapes and shapes[0].get("specular"):
            spec_path = f"{PurePosixPath(r.mesh_file).parent}/{shapes[0]['specular']}"
            spec = ctx.generated.files.get(spec_path)
            try:
                rgb_max = max(hi for _, hi in read_dds(spec.data).convert("RGB").getextrema()) if spec else 0
            except (DDSError, OSError, ValueError):
                rgb_max = 0
            if rgb_max == 0:
                out.append(Issue(ERROR, "loadout", f"{r.entity}: PdxMeshColor, but its spec map has no country-colour "
                                                   "mask (RGB all 0)", spec_path, r.entity))
    for crew, art in sorted(res.crews.items()):
        gunmen = [a.get("infantry_node") for a in w.resolve(art, "attach") or [] if a.get("name") == "gunman"]
        if gunmen != [crew]:
            out.append(Issue(ERROR, "loadout", f"{crew}: {art}'s gunman attaches {gunmen}, not this crew", subject=crew))
    for entity, (stance, pdx_name) in sorted(res.stances.items()):
        anims = (w.pdxmesh(w.resolve(entity, "pdxmesh") or "") or {}).get("animations") or {}
        if w.resolve(entity, "pdxmesh") != pdx_name or any(anims.get(i) != t for i, t in STANCE_TYPES[stance].items()):
            out.append(Issue(ERROR, "loadout", f"{entity}: stance {stance} isn't in its pdxmesh "
                                               f"{w.resolve(entity, 'pdxmesh')!r}", subject=entity))
    weapons = {r.entity for r in res.parts if r.bone == "Right_hand_node"}
    ours = {r.entity for r in res.parts}
    for name in sorted(ents):
        nodes = {b.lower(): t for a in w.resolve(name, "attach") or [] for b, t in a.items() if b != "name"}
        if nodes.get("right_hand_node") in weapons and nodes.get("left_hand_node") not in (None, *ours):
            out.append(Issue(WARNING, "loadout", f"{name} carries {nodes['right_hand_node']} and still the base's "
                                                 f"{nodes['left_hand_node']} in the left hand", subject=name))
    for tier, body in sorted(res.chosen_bodies.items()):
        brec = w.pdxmesh(body)
        inf = ctx.naming.unit_entity(ctx.owner, "infantry", tier)
        mine = w.pdxmesh(w.resolve(inf, "pdxmesh") or "") or {}
        if brec is None or not w.game.in_base_game(brec.get("file") or ""):
            out.append(Issue(ERROR, "loadout", f"tier {tier} body {body!r} isn't a base-game pdxmesh", subject=inf))
            continue
        derived = res.derived.get(tier)
        if derived is not None and (mine.get("file") or "").lower() == derived.file.lower():
            mine = {**mine, "file": derived.base_file}  # a derived body of the chosen body (rule derived_body)
        if (mine.get("file") or "").lower() != brec["file"].lower():
            out.append(Issue(ERROR, "loadout", f"{inf} should wear tier {tier}'s body {body} ({brec['file']}), "
                                               f"not {mine.get('file')}", subject=inf))
        shapes = (w.mesh_summary(brec["file"]) or {}).get("shapes", [])
        if shapes and shapes[0].get("shader") != "PdxMeshColor":
            out.append(Issue(INFO, "loadout", f"tier {tier} body {body} uses {shapes[0].get('shader')}: its uniform "
                                              "takes no unit colours", subject=inf))
    for name in sorted(ents):
        nodes = {b.lower() for a in w.resolve(name, "attach") or [] for b, t in a.items() if b != "name"}
        held = {t for a in w.resolve(name, "attach") or [] for b, t in a.items() if b != "name"}
        if held & {r.entity for r in res.parts if r.kind == "hoplon"} and "right_hand_node" not in nodes:
            out.append(Issue(WARNING, "loadout", f"{name} carries a shield and nothing in its right hand (no weapon "
                                                 "part, and the base holds nothing there)", subject=name))
    shields = {r.entity for r in res.parts if r.kind == "hoplon"}
    for name in sorted(ents):
        carried = {t for a in w.resolve(name, "attach") or [] for b, t in a.items() if b != "name"} & shields
        if carried:
            anims = (w.pdxmesh(w.resolve(name, "pdxmesh") or "") or {}).get("animations") or {}
            if any(anims.get(i) != t for i, t in STANCE_TYPES["spear_shield"].items()):
                out.append(Issue(WARNING, "loadout", f"{name} carries {sorted(carried)} but doesn't use the "
                                                     "spear_shield animations: the left arm may not hold it up",
                                 subject=name))
    return out


@rule("texture_quality")
def texture_quality(ctx: Context) -> list[Issue]:
    """Q6: each generated part's texel density, painted detail and shading, normal-map tilt and
    gloss, against the base game's measured ranges (textures.quality.RANGES): warnings."""
    from eu4unitgen.library.scanner import _walk, parse_pdx_mesh
    from eu4unitgen.textures.dds import DDSError, read_dds
    from eu4unitgen.textures.quality import measure, outside

    res = ctx.generated.loadout
    if res is None:
        return []
    out, seen = [], set()
    for r in res.parts:
        if r.mesh_file in seen:
            continue
        seen.add(r.mesh_file)
        f = ctx.generated.files.get(r.mesh_file)
        if f is None:
            continue
        try:
            node = next(n for n in _walk(parse_pdx_mesh(f.data)) if n["name"] == "mesh")
            mesh = node["props"]
            material = next(c for c in node["children"] if c["name"] == "material")["props"]
            folder = r.mesh_file.rsplit("/", 1)[0]
            tex = {k: ctx.generated.files.get(f"{folder}/{material[slot][0]}") if material.get(slot) else None
                   for k, slot in (("diffuse", "diff"), ("normal", "n"), ("specular", "spec"))}
            if tex["diffuse"] is None or "u0" not in mesh:
                continue
            imgs = {k: read_dds(v.data) if v is not None else None for k, v in tex.items()}
            m = measure(mesh["u0"], mesh["p"], mesh["tri"], imgs["diffuse"], imgs["normal"], imgs["specular"])
        except (DDSError, OSError, ValueError, StopIteration, KeyError):
            continue
        found = outside(m)
        if found:
            out.append(Issue(WARNING, "texture_quality", f"{r.entity}: {', '.join(found)} (base-game range; "
                                                         "SPEC Track R quality measurements)", r.mesh_file, r.entity))
    return out


DERIVED_TRI_BUDGET = 4500  # the largest base-game infantry body: western_generic_02_type1, 4,360 (1.37.5)


@rule("derived_body")
def derived_body(ctx: Context) -> list[Issue]:
    """Derived bodies (SPEC Track R, core.derived_body): a base-game body (never DLC) whose every
    block is kept (skeleton, locators, material, vertices, skin rows, triangles) with only skinned
    additions appended; their weights stay on their declared bone chains; within the triangle
    budget; its material's textures resolve in its folder; its name is no game file's; its
    infantry entity wears it."""
    from eu4unitgen.core.derived_body import DerivedBodyError, bone_indices, check_derived, removed_triangles
    from eu4unitgen.library.scanner import _walk, parse_pdx_mesh

    res = ctx.generated.loadout
    if res is None or not res.derived:
        return []
    w, game, out = ctx.world, ctx.world.game, []
    for tier, rec in sorted(res.derived.items()):
        f = ctx.generated.files.get(rec.file)
        if f is None:
            out.append(Issue(ERROR, "derived_body", f"tier {tier}: {rec.file} wasn't generated", subject=rec.entity))
            continue
        if not game.in_base_game(rec.base_file):
            what = "a DLC file" if game.exists(rec.base_file) else "missing from the install"
            out.append(Issue(ERROR, "derived_body", f"base body {rec.base_file} is {what}; derived bodies come from "
                                                    "base-game files only", f.path))
            continue
        if f.derived_from != rec.base_file:
            out.append(Issue(ERROR, "derived_body", f"doesn't record its base body ({f.derived_from!r})", f.path))
        if game.exists(rec.file):
            out.append(Issue(ERROR, "derived_body", "has a game file's name: it would replace it for everyone", f.path))
        base = game.read(rec.base_file)
        try:
            removed = removed_triangles(base, rec.base_file, rec.removed) if rec.removed else []
        except DerivedBodyError as exc:
            out.append(Issue(ERROR, "derived_body", str(exc), f.path))
            continue
        names = {k: p.rsplit("/", 1)[-1] for k, p in (rec.textures or {}).items()}
        problems, counts = check_derived(base, f.data, removed, names or None)
        for kind, path in sorted((rec.textures or {}).items()):
            tex = ctx.generated.files.get(path)
            if tex is None:
                if not game.in_base_game(path):  # a base-game texture used as it is (e.g. the recipe left it) is fine
                    problems.append(f"its {kind} texture {path} is neither generated nor a base-game file")
            elif not tex.derived_from:
                problems.append(f"its derived {kind} texture {path} doesn't record its base texture")
        out += [Issue(ERROR, "derived_body", f"{p} (base {rec.base_file})", f.path) for p in problems]
        if problems:
            continue
        tris = counts["base_triangles"] - counts["removed_triangles"] + counts["added_triangles"]
        if tris > DERIVED_TRI_BUDGET:
            out.append(Issue(ERROR, "derived_body", f"{tris} triangles (budget {DERIVED_TRI_BUDGET})", f.path))
        tree = parse_pdx_mesh(f.data)
        mesh = next(n for n in _walk(tree) if n["name"] == "mesh")
        names = {i: b for b, i in bone_indices(next(n for n in _walk(tree) if n["name"] == "skeleton")).items()}
        skin = next(c for c in mesh["children"] if c["name"] == "skin")["props"]
        first = counts["base_vertices"]
        if sum(n for _, _, n in rec.additions) != counts["added_vertices"]:
            out.append(Issue(ERROR, "derived_body", f"{counts['added_vertices']} added vertices, but its additions "
                                                    f"account for {sum(n for _, _, n in rec.additions)}", f.path))
        for kind, chain, n in rec.additions:
            used = {names.get(skin["ix"][v * 4 + k]) for v in range(first, first + n) for k in range(4)
                    if skin["ix"][v * 4 + k] >= 0}
            if not used <= set(chain):
                out.append(Issue(ERROR, "derived_body", f"{kind}: weights on {sorted(used - set(chain))}, outside its "
                                                        f"chain {list(chain)}", f.path))
            first += n
        folder = rec.file.rsplit("/", 1)[0]
        material = next(c for c in mesh["children"] if c["name"] == "material")["props"]
        for slot in ("diff", "n", "spec"):
            for tex in material.get(slot, []):
                if not w.exists(f"{folder}/{tex}"):
                    out.append(Issue(ERROR, "derived_body", f"texture {tex} doesn't resolve in {folder}", f.path))
        if rec.additions and rec.deform is None:
            out.append(Issue(WARNING, "derived_body", f"tier {tier}: the deformation check didn't run (no Blender "
                                                      "or no animations): skinned additions unchecked", f.path))
        elif rec.deform is not None:
            from eu4unitgen.core.deform_check import judge

            for severity, message in judge(rec.deform, rec.accept_clipping):
                out.append(Issue(severity, "derived_body", f"tier {tier}: deformation {message}", f.path))
        pdx = w.pdxmesh(w.resolve(rec.entity, "pdxmesh") or "") or {}
        if w.resolve(rec.entity, "pdxmesh") != rec.pdxmesh or (pdx.get("file") or "").lower() != rec.file.lower():
            out.append(Issue(ERROR, "derived_body", f"{rec.entity} doesn't wear {rec.pdxmesh} ({rec.file})",
                             subject=rec.entity))
        out.append(Issue(INFO, "derived_body", f"tier {tier}: {rec.entity} wears {rec.file} = {rec.base_file}"
                                               + "".join(f" - {x}" for x in rec.removed)
                                               + f" ({counts['removed_triangles']} triangles removed) + "
                                               f"{', '.join(k for k, _, _ in rec.additions) or 'nothing'} "
                                               f"({counts['added_vertices']} vertices, {counts['added_triangles']} "
                                               "triangles added); its textures resolve from the base folder "
                                               "(verified in game, test 10)", f.path))
    return out


# --------------------------------------------------------------------------- naming

@rule("naming")
def naming(ctx: Context) -> list[Issue]:
    """Generated entity and pack names are exactly the ones naming.py produces."""
    n, tag, gfx = ctx.naming, ctx.owner, ctx.nation.gfx_culture
    expected = set()
    fx = ctx.generated.fixtures
    for owner in (tag, gfx, *(fx.unpacked_owners if fx else [])):
        for t in TIERS:
            expected |= {n.unit_entity(owner, u, t) for u in UNIT_TYPES} | {n.rider_entity(owner, t)}
    out = []
    names = set(ctx.gen_entities())
    # Optional: an owner's own cannon entity, made only when its cannon is retextured (Phase 2).
    optional = {n.cannon_entity(owner, t) for owner in (tag, gfx) for t in TIERS}
    optional |= {n.part(ctx.nation.set_name, kind, t, role) for kind in PART_KINDS for t in TIERS
                 for role in LOADOUT_ROLES}  # Phase 3 parts
    optional |= {n.crew_entity(tag, t) for t in TIERS}  # a loadout's own gun crew
    for name in sorted(names - expected - optional):
        out.append(Issue(ERROR, "naming", f"unexpected generated entity name {name!r}", subject=name))
    for name in sorted(expected - names):
        out.append(Issue(ERROR, "naming", f"expected entity {name!r} was not generated", subject=name))
    packs = set(ctx.gen_defs("sprite_packs"))
    for want in (n.set_sprite_pack(ctx.nation.set_name), n.gfx_sprite_pack(gfx)):
        if want not in packs:
            out.append(Issue(ERROR, "naming", f"sprite pack {want!r} was not generated", subject=want))
    return out


# --------------------------------------------------------------------------- entity chain

@rule("missing_nodes")
def missing_nodes(ctx: Context) -> list[Issue]:
    """Every pack slot, clone, attach, pdxmesh, mesh, texture, animation type and .anim resolves."""
    out = []
    for e in ctx.graph.missing():
        if e.dst.kind == "anim_id":
            continue  # reported by state_animations with more context
        what = {"entity": "entity", "pdxmesh": "pdxmesh", "file": "file", "anim_type": "animation type"}[e.dst.kind]
        log = {"attach:infantry_node": 'Failed to find entity "X" for attachment in gunman',
               "file": "Could not find mesh file"}.get(e.label, "")
        msg = f"{e.src.kind} {e.src.name!r} -> {e.label} -> {what} {e.dst.name!r} not found"
        if e.dst.kind == "entity":
            unowned = [d.get("dlc") for d in ctx.world.defs("entities", e.dst.name) if d.get("dlc")]
            if unowned:
                msg += f" (only in DLC not marked owned: {', '.join(sorted(set(unowned)))})"
        out.append(Issue(ERROR, "missing_nodes", msg + (f"  [error.log: {log}]" if log else ""), subject=e.src.name))
    return out


@rule("clone_order")
def clone_order(ctx: Context) -> list[Issue]:
    """A clone's parent must be defined earlier in load order (file name order, then file
    order), as the engine resolves clones while loading. Otherwise error.log has
    "Couldn't find parent clone entity X" and the unit falls back to another culture's model.
    Errors for generated entities, warnings for the mod's own (they break the same way)."""
    w, out = ctx.world, []
    for name, defs in sorted(w.idx.entities.items()):
        for rec in defs:
            if not rec.get("clone") or rec["origin"] == VANILLA or not w.loaded(rec):
                continue
            if w.clone_parent_loaded_before(rec) is not None:
                continue
            later = sorted({d["defined_in"] for d in w.defs("entities", rec["clone"]) if w.loaded(d)})
            why = (f"it is only defined after it in load order ({', '.join(later)})" if later
                   else "it is not defined anywhere")
            out.append(Issue(ERROR if rec["origin"] == GENERATED else WARNING, "clone_order",
                             f"{name} clones {rec['clone']!r}, but {why}  "
                             f"[error.log: Couldn't find parent clone entity {rec['clone']}]",
                             rec["defined_in"], name))
    return out


@rule("dlc_dependency")
def dlc_dependency(ctx: Context) -> list[Issue]:
    """References that resolve only through a DLC: they break for players without it."""
    out = []
    for e in ctx.graph.edges:
        if e.ok and e.origin and e.origin not in (VANILLA, MOD, GENERATED):
            out.append(Issue(WARNING, "dlc_dependency",
                             f"{e.src.name!r} -> {e.label} -> {e.dst.name!r} comes from {e.origin}", subject=e.src.name))
    return out


@rule("state_animations")
def state_animations(ctx: Context) -> list[Issue]:
    """Each state's animation id exists in the entity's pdxmesh; riders need horse_* ids and
    artillery crews need artillery_attack (SPEC §5). A missing id gives a frozen/T-posed unit."""
    w, out = ctx.world, []
    for name in sorted(ctx.gen_entities()):
        pdx_name = w.resolve(name, "pdxmesh")
        pdx = w.pdxmesh(pdx_name) if pdx_name else None
        if pdx is None:
            continue
        ids = set(pdx.get("animations") or {})
        for st in w.resolve(name, "states") or []:
            if st.get("name") not in LIVE_STATES:
                out.append(Issue(WARNING, "state_animations", f"{name}: state {st.get('name')!r} is not a live vanilla state"))
            anim = st.get("animation")
            if anim and anim not in ids:
                out.append(Issue(ERROR, "state_animations",
                                 f"{name}: state {st.get('name')!r} uses animation id {anim!r}, which pdxmesh {pdx_name!r} lacks",
                                 subject=name))
        if name.endswith("_rider") and not RIDER_IDS <= ids:
            out.append(Issue(ERROR, "state_animations",
                             f"{name}: rider pdxmesh {pdx_name!r} lacks {sorted(RIDER_IDS - ids)}", subject=name))
    # Vanilla ships crews without artillery_attack: every base-game culture except westerngfx
    # (1.37.5). A pdxmesh vanilla already uses as a crew is accepted as it is.
    vanilla_crews = None
    for name in sorted(ctx.gen_entities()):
        for attach in w.resolve(name, "attach") or []:
            crew = attach.get("infantry_node")
            if crew:
                crew_pdx_name = w.resolve(crew, "pdxmesh") or ""
                crew_pdx = w.pdxmesh(crew_pdx_name)
                if crew_pdx is not None and "artillery_attack" not in (crew_pdx.get("animations") or {}):
                    if vanilla_crews is None:
                        vanilla_crews = _vanilla_crew_pdxmeshes(w)
                    if crew_pdx_name in vanilla_crews:
                        continue
                    out.append(Issue(ERROR, "state_animations",
                                     f"{name}: crew {crew!r} pdxmesh {crew_pdx_name!r} lacks animation id "
                                     "'artillery_attack' and vanilla never uses it as a crew", subject=name))
    return out


def _vanilla_crew_pdxmeshes(w: World) -> set[str]:
    out = set()
    for name, defs in w.idx.entities.items():
        if not any(d["origin"] == VANILLA and w.loaded(d) for d in defs):
            continue
        for attach in w.resolve(name, "attach") or []:
            crew = attach.get("infantry_node")
            if crew and all(d["origin"] == VANILLA for d in w.defs("entities", crew)):
                out.add(w.resolve(crew, "pdxmesh") or "")
    out.discard("")
    return out


@rule("event_nodes")
def event_nodes(ctx: Context) -> list[Issue]:
    """State events fire from a locator in the mesh (muzzle, cannon_muzzle)."""
    w, out = ctx.world, []
    for node in sorted(ctx.graph.reached, key=lambda n: n.name):
        if node.kind != "entity":
            continue
        pdx = w.pdxmesh(w.resolve(node.name, "pdxmesh") or "")
        summary = w.mesh_summary(pdx["file"]) if pdx and pdx.get("file") else None
        if summary is None:
            continue
        for st in w.resolve(node.name, "states") or []:
            for ev in st.get("events", []):
                if ev.get("node") and ev["node"] not in summary["locators"]:
                    out.append(Issue(WARNING, "event_nodes",
                                     f"{node.name}: event node {ev['node']!r} is not a locator in {pdx['file']}"))
    return out


@rule("version_collisions")
def version_collisions(ctx: Context) -> list[Issue]:
    """A generated entity name defined elsewhere: the game keeps the higher version and logs
    'Set a higher version on entity X'. Checked against vanilla, all DLC and the mod."""
    out = []
    for name, ours in sorted(ctx.gen_entities().items()):
        mine = ours.get("version") or 0
        for other in ctx.world.defs("entities", name):
            if other["origin"] == GENERATED and other is not ours:
                out.append(Issue(ERROR, "version_collisions", f"{name} generated twice", other["defined_in"], name))
            elif other["origin"] != GENERATED:
                where = other.get("dlc") or other["origin"]
                theirs = other.get("version") or 0
                sev = ERROR if theirs >= mine else WARNING
                out.append(Issue(sev, "version_collisions",
                                 f"{name} also defined in {where} {other['defined_in']} (version {theirs}, ours {mine})"
                                 + (": theirs wins or ties" if sev == ERROR else ": ours wins"), other["defined_in"], name))
    return out


@rule("orphan_entities")
def orphan_entities(ctx: Context) -> list[Issue]:
    """Generated entities no sprite pack reaches (directly, or as rider/crew). The test suite's
    case-b TAG set is unpacked on purpose and is checked by test_fixtures instead."""
    fx = ctx.generated.fixtures
    unpacked = tuple(f"{t}_" for t in (fx.unpacked_owners if fx else []))
    from_packs, todo = set(), [r for r in ctx.graph.roots if r.kind == "pack"]
    while todo:
        n = todo.pop()
        if n in from_packs:
            continue
        from_packs.add(n)
        todo += [e.dst for e in ctx.graph.edges_from(n) if e.ok and e.dst.kind == "entity"]
    return [Issue(WARNING, "orphan_entities", f"{name} is not used by any sprite pack", subject=name)
            for name in sorted(ctx.gen_entities())
            if Node("entity", name) not in from_packs and not (unpacked and name.startswith(unpacked))]


# --------------------------------------------------------------------------- packs

@rule("sprite_packs")
def sprite_packs(ctx: Context) -> list[Issue]:
    n, tag, gfx = ctx.naming, ctx.owner, ctx.nation.gfx_culture
    packs, out = ctx.gen_defs("sprite_packs"), []
    tag_pack, gfx_pack = packs.get(n.set_sprite_pack(ctx.nation.set_name)), packs.get(n.gfx_sprite_pack(gfx))
    for pack, owner in ((tag_pack, tag), (gfx_pack, gfx)):
        if pack is None:
            continue
        for utype in UNIT_TYPES:
            slots = pack.get(utype, {})
            for t in TIERS:
                want = n.unit_entity(owner, utype, t)
                if slots.get(str(t - 1)) != want:
                    out.append(Issue(ERROR, "sprite_packs", f"{pack['name']}: sprite_{utype} {t - 1} should be {want}",
                                     pack["defined_in"]))
    want_trigger = set_trigger(ctx.nation.tags, ctx.nation.cultures)
    if tag_pack is not None and tag_pack.get("country_trigger") != want_trigger:
        out.append(Issue(ERROR, "sprite_packs", f"{tag_pack['name']}: country_trigger should be '{want_trigger}'",
                         tag_pack["defined_in"]))
    if gfx_pack is not None:
        if gfx_pack.get("graphical_cultures") != [gfx]:
            out.append(Issue(ERROR, "sprite_packs", f"{gfx_pack['name']}: graphical_cultures should be {{ {gfx} }}",
                             gfx_pack["defined_in"]))
        if gfx_pack.get("country_trigger") != "always = no":
            out.append(Issue(ERROR, "sprite_packs", f"{gfx_pack['name']}: country_trigger should be 'always = no'",
                             gfx_pack["defined_in"]))
    for name, defs in ctx.world.idx.sprite_packs.items():
        if len({d["defined_in"] for d in defs if ctx.world.loaded(d)}) > 1 and any(d["origin"] == GENERATED for d in defs):
            out.append(Issue(ERROR, "sprite_packs", f"sprite pack {name} is defined in more than one file"))
    return out


def _excludes_tag(trigger: str | None, tag: str) -> bool:
    """True only when the trigger provably can't match ``tag``. Anything else might."""
    if trigger is None:
        return False
    t = trigger.strip()
    if t == "always = no":
        return True
    m = re.fullmatch(r"tag = ([A-Z0-9]{3})", t)
    return bool(m) and m.group(1) != tag


def _culture_matches(trigger: str | None, culture: str, groups: dict[str, str]) -> bool | None:
    """Could a pack's trigger match a country whose primary culture is ``culture`` (any tag)?
    False only when provably not; None = maybe (depends on the tag, religion, ...)."""
    from eu4unitgen.library.tags import TagFacts, trigger_matches

    if trigger is None:
        return None
    return trigger_matches(trigger, TagFacts(tag="", primary_culture=culture, culture_group=groups.get(culture)))


def _cultures(ctx: Context) -> dict[str, str]:
    """culture -> group from vanilla and (with mod checks) the mod."""
    from eu4unitgen.library.tags import culture_groups

    if getattr(ctx, "_culture_groups", None) is None:
        groups = culture_groups(ctx.world.game)
        if ctx.world.mod is not None:
            groups.update(culture_groups(ctx.world.mod))
        ctx._culture_groups = groups
    return ctx._culture_groups


@rule("cultures")
def cultures(ctx: Context) -> list[Issue]:
    """Every culture in the set's trigger exists (common/cultures, vanilla or the mod)."""
    out = []
    if not ctx.nation.cultures:
        return out
    groups = _cultures(ctx)
    for c in ctx.nation.cultures:
        if c in groups:
            continue
        if ctx.world.mod is None:
            out.append(Issue(WARNING, "cultures", f"culture {c!r} isn't a vanilla culture; turn on mod checks so the "
                                                  "mod's common/cultures is read", subject=c))
        else:
            out.append(Issue(ERROR, "cultures", f"culture {c!r} isn't in common/cultures (vanilla or the mod): "
                                                "the pack would never match by culture", subject=c))
    return out


@rule("pack_order")
def pack_order(ctx: Context) -> list[Issue]:
    """The set's pack has an order entry; flag other packs that might match any of the set's tags or
    cultures at >= that order (the higher order wins, in-game test 3)."""
    n, tags, out = ctx.naming, ctx.nation.tags, []
    groups = _cultures(ctx) if ctx.nation.cultures else {}
    pack = n.set_sprite_pack(ctx.nation.set_name)
    orders = [d for d in ctx.world.idx.pack_order.get(pack, []) if ctx.world.loaded(d)]
    if not orders:
        return [Issue(ERROR, "pack_order", f"no sprite_packs_order entry for {pack}")]
    ours = orders[-1]["order"]
    for other, pdefs in sorted(ctx.world.idx.sprite_packs.items()):
        if other == pack:
            continue
        odefs = [d for d in ctx.world.idx.pack_order.get(other, []) if ctx.world.loaded(d)]
        order = odefs[-1]["order"] if odefs else 0  # vanilla: no entry means 0
        if order < ours:
            continue
        fx = ctx.generated.fixtures
        for p in pdefs:
            hit = [t for t in tags if not _excludes_tag(p.get("country_trigger"), t)] if ctx.world.loaded(p) else []
            if ctx.world.loaded(p) and not p.get("graphical_cultures"):
                for c in ctx.nation.cultures:
                    m = _culture_matches(p.get("country_trigger"), c, groups)
                    if m is not False:
                        hit.append(f"{'every' if m else 'maybe some'} {c} country")
            if hit:
                if fx and other == fx.rival_pack:
                    out.append(Issue(INFO, "pack_order", f"{other} (order {order}) is the pack-priority test's "
                                                         f"deliberate rival of {pack} (order {ours})", p["defined_in"]))
                    continue
                out.append(Issue(WARNING, "pack_order",
                                 f"{other} (order {order}, trigger '{p.get('country_trigger')}') may match {', '.join(hit)} "
                                 f"and is not below {pack} (order {ours})", p["defined_in"]))
    gfx_pack = n.gfx_sprite_pack(ctx.nation.gfx_culture)
    if gfx_pack in ctx.world.idx.pack_order:
        out.append(Issue(WARNING, "pack_order", f"{gfx_pack} has an order entry; vanilla gfx packs have none"))
    return out


@rule("graphical_culture")
def graphical_culture(ctx: Context) -> list[Issue]:
    gfx, out = ctx.nation.gfx_culture, []
    if gfx not in ctx.world.graphical_cultures:
        out.append(Issue(ERROR, "graphical_culture", f"{gfx} is not in common/graphicalculturetype.txt"))
    if ctx.world.graphical_cultures.count(gfx) > 1:
        out.append(Issue(WARNING, "graphical_culture", f"{gfx} is listed more than once"))
    vanilla = ctx.world.game.read("common/graphicalculturetype.txt")
    if vanilla:
        lost = [g for g in parse_bytes(vanilla).bare() if g not in ctx.world.graphical_cultures]
        if lost:
            out.append(Issue(WARNING, "graphical_culture", f"the mod's list drops vanilla cultures: {', '.join(lost)}"))
    return out


# --------------------------------------------------------------------------- units, loc, tech

def _unit_docs(ctx: Context) -> dict[str, tuple[str, Block]]:
    return {u.key: (ctx.naming.unit_file(u.key), parse_bytes(ctx.generated.files[ctx.naming.unit_file(u.key)].data))
            for u in ctx.nation.units if ctx.naming.unit_file(u.key) in ctx.generated.files}


def _tech_groups(ctx: Context) -> set[str]:
    groups = set()
    data = ctx.world.read("common/technology.txt")
    if data:
        blk = parse_bytes(data).get_block("groups")
        groups |= {p.key.value for p in blk.pairs()} if blk is not None else set()
    for layer in (ctx.world.game, ctx.world.mod):
        if layer is None:
            continue
        for _, _, raw in layer.iter_files("common/units", (".txt",)):
            v = parse_bytes(raw).get_str("unit_type")
            if v:
                groups.add(v)
    return groups


@rule("units")
def units(ctx: Context) -> list[Issue]:
    out, groups = [], None
    for unit in ctx.nation.units:
        path = ctx.naming.unit_file(unit.key)
        if path not in ctx.generated.files:
            out.append(Issue(ERROR, "units", f"unit file for {unit.key} was not generated", path))
    for key, (path, doc) in _unit_docs(ctx).items():
        utype = doc.get_str("type")
        if utype not in UNIT_TYPES:
            out.append(Issue(ERROR, "units", f"type {utype!r} is not a land unit type", path))
            continue
        if doc.get_str("maneuver") != str(MANEUVER[utype]):
            out.append(Issue(WARNING, "units", f"maneuver {doc.get_str('maneuver')} differs from vanilla {utype} ({MANEUVER[utype]})", path))
        ut = doc.get_str("unit_type")
        if utype == "artillery" and ut:
            out.append(Issue(WARNING, "units", "vanilla artillery has no unit_type", path))
        elif utype != "artillery" and ut:
            groups = groups if groups is not None else _tech_groups(ctx)
            if ut not in groups:
                out.append(Issue(ERROR, "units", f"unit_type {ut!r} is not a tech group in technology.txt or any unit", path))
        for k in ("offensive_morale", "defensive_morale", "offensive_fire", "defensive_fire",
                  "offensive_shock", "defensive_shock"):
            v = doc.get_str(k)
            if v is None or not re.fullmatch(r"\d+", v):
                out.append(Issue(ERROR, "units", f"{k} missing or not a whole number", path))
        others = [p for p in ctx.world.game.iter_files("common/units", (".txt",)) if p[0].lower().endswith(f"/{key}.txt")]
        if others:
            out.append(Issue(ERROR, "units", f"{key} clashes with vanilla {others[0][0]}", path))
    return out


@rule("localisation")
def localisation(ctx: Context) -> list[Issue]:
    out = []
    for lang in ctx.project.languages:
        path = ctx.naming.loc_file(ctx.nation.set_name, lang)
        f = ctx.generated.files.get(path)
        if f is None:
            out.append(Issue(ERROR, "localisation", f"no {lang} localisation generated", path))
            continue
        keys = set(re.findall(r"^ ([A-Za-z0-9_.]+):\d* ", f.data.decode("utf-8-sig"), re.M))
        for unit in ctx.nation.units:
            for key in (ctx.naming.loc_name_key(unit.key), ctx.naming.loc_desc_key(unit.key)):
                if key not in keys:
                    out.append(Issue(ERROR, "localisation", f"missing key {key}", path))
    return out


@rule("unit_names")
def unit_names(ctx: Context) -> list[Issue]:
    """F7: no two units share a name (the unit list would show them alike), and each unit's name
    is in the localisation file exactly as the set gives it."""
    out = []
    seen: dict[str, str] = {}
    for unit in ctx.nation.units:
        if unit.name in seen:
            out.append(Issue(ERROR, "unit_names", f"{unit.key} and {seen[unit.name]} are both named {unit.name!r}",
                             subject=unit.key))
        seen.setdefault(unit.name, unit.key)
    for lang in ctx.project.languages:
        path = ctx.naming.loc_file(ctx.nation.set_name, lang)
        f = ctx.generated.files.get(path)
        if f is None:
            continue  # the localisation rule reports it
        text = f.data.decode("utf-8-sig")
        for unit in ctx.nation.units:
            key = ctx.naming.loc_name_key(unit.key)
            m = re.search(rf'^ {re.escape(key)}:\d* "(.*)"$', text, re.M)
            if m and m.group(1) != unit.name:
                out.append(Issue(ERROR, "unit_names", f"{key} is {m.group(1)!r} in the file, not {unit.name!r}", path))
    return out


@rule("tech_enables")
def tech_enables(ctx: Context) -> list[Issue]:
    rel = ctx.naming.mil_tech_file()
    data = ctx.world.read(rel)
    if data is None:
        return [Issue(ERROR, "tech_enables", "no mil.txt in the mod or the game", rel)]
    techs = [t for t in parse_bytes(data).get_all("technology") if isinstance(t, Block)]
    enabled: dict[str, int] = {}
    for t in techs:
        for v in t.get_all("enable"):
            if isinstance(v, Scalar):
                enabled[v.value] = enabled.get(v.value, 0) + 1
    out = []
    for unit in ctx.nation.units:
        count = enabled.get(unit.key, 0)
        if count == 0:
            out.append(Issue(ERROR, "tech_enables", f"{unit.key} is never enabled in mil.txt", rel))
        elif count > 1:
            out.append(Issue(WARNING, "tech_enables", f"{unit.key} is enabled {count} times", rel))
    where = [(int(v.value), i) for i, t in enumerate(techs) for v in t.get_all("sprite_level")
             if isinstance(v, Scalar) and v.value.isdigit()]
    levels = sorted(lv for lv, _ in where)
    if [lv for lv in levels if lv <= 4] != [1, 2, 3, 4]:
        out.append(Issue(ERROR, "tech_enables", f"mil.txt sprite levels are {levels}; need 1-4 exactly once", rel))
    else:
        techs_of = [i for _, i in sorted(where) if _ <= 4]
        if techs_of != sorted(techs_of) or len(set(techs_of)) != 4:
            out.append(Issue(ERROR, "tech_enables", f"sprite levels 1-4 sit at MIL techs {techs_of}; "
                                                    "they must rise with the tech", rel))
    unit_paths = {p.lower() for p in ctx.generated.files} | ctx.world.mod_paths
    for key in sorted(enabled):
        if key.startswith(ctx.naming.prefix) and f"common/units/{key}.txt".lower() not in unit_paths:
            out.append(Issue(ERROR, "tech_enables", f"mil.txt enables {key}, which has no unit file", rel))
    mil = ctx.generated.files.get(rel)
    if mil is not None and mil.based_on == VANILLA:
        out.append(Issue(INFO, "tech_enables", "validated against vanilla mil.txt plus the enable lines "
                                               "from INTEGRATION.md", rel))
    return out


@rule("integration")
def integration(ctx: Context) -> list[Issue]:
    """INTEGRATION.md exists and carries every change it is meant to hand over."""
    integ, path = ctx.generated.integration, ctx.naming.integration_file(ctx.nation.set_name)
    f = ctx.generated.files.get(path)
    if integ is None or f is None:
        return [Issue(ERROR, "integration", "no INTEGRATION.md generated", path)]
    text, out = f.data.decode("utf-8"), []
    keys = [u.key for u in ctx.nation.units]
    if integ.historical_units != keys:
        out.append(Issue(ERROR, "integration", "historical_units doesn't list the generated units in order", path))
    listed = {k for units in integ.mil_enables.values() for k in units}
    for u in ctx.nation.units:
        if u.key not in integ.mil_enables.get(u.mil_tech, []) or f"enable = {u.key}" not in text:
            out.append(Issue(ERROR, "integration", f"{u.key} missing from the MIL enables (tech {u.mil_tech})", path))
    for extra in sorted(listed - set(keys)):
        out.append(Issue(ERROR, "integration", f"{extra} is listed but not generated", path))
    if f"graphical_culture = {ctx.nation.gfx_culture}" not in text:
        out.append(Issue(ERROR, "integration", "country lines don't set graphical_culture", path))
    if list(integ.tags) != list(ctx.nation.tags):
        out.append(Issue(ERROR, "integration", f"hands over tags {integ.tags}, the set has {ctx.nation.tags}", path))
    for t in ctx.nation.tags:
        cfile = integ.country_files.get(t)
        if cfile and not ctx.world.exists(cfile):
            out.append(Issue(ERROR, "integration", f"{t}: {cfile} doesn't exist in the game or the mod", path))
        if len(ctx.nation.tags) > 1 and f"- {t}:" not in text:
            out.append(Issue(ERROR, "integration", f"{t}'s country file isn't listed", path))
    return out


@rule("unit_colours")
def unit_colours(ctx: Context) -> list[Issue]:
    """Unit colours come from common/country_colors color1-3 (in-game test 5). Note when the tag
    has no entry anywhere and gets none from this set (the engine default applies); check that
    INTEGRATION.md and any generated copy of the file carry the set's colours."""
    from eu4unitgen.library.tags import unit_colours as read_colours

    wanted = ctx.nation.unit_colours or {}
    rel = ctx.naming.country_colours_file()
    out = []
    entries = read_colours(ctx.world.game)
    if ctx.world.mod is not None:
        entries.update(read_colours(ctx.world.mod))
    if ctx.world.gen.exists(rel):
        gen = read_colours(_OneFile(rel, ctx.world.gen.read(rel)))
        for tag, cols in wanted.items():
            if gen.get(tag) != list(cols):
                out.append(Issue(ERROR, "unit_colours", f"{tag}'s entry doesn't carry the set's unit colours", rel))
        entries.update(gen)
    if wanted:
        integ = ctx.generated.integration
        if integ is None or (integ.unit_colours or {}) != wanted or not integ.unit_colours_snippet:
            out.append(Issue(ERROR, "unit_colours", "INTEGRATION.md doesn't hand over the unit colours"))
    if ctx.nation.cultures:
        out.append(Issue(INFO, "unit_colours", "countries matched by culture ("
                                               + ", ".join(ctx.nation.cultures) + ") keep their own country_colors "
                                               "entries; unit colours here apply to the set's tags only"))
    for tag in ctx.nation.tags:
        if tag not in wanted and tag not in entries:
            out.append(Issue(INFO, "unit_colours", f"{tag} has no entry in {rel} and the set gives none: its units "
                                                   "use the engine's default unit colours (red in test 5; UNVERIFIED). "
                                                   "Set unit colours to choose them.", rel))
    return out


COLOUR_EFFECTS = ("restore_country_color", "change_country_color")
ALL, SUBJECTS, SCOPED = "every country", "subjects only", "the countries it scopes to"


def _reach(key: str, val: Block, outer: str) -> str:
    """How far a scope reaches: every_country (limited to ``is_subject = yes``: subjects only),
    every_subject_country (subjects only); anything else keeps the outer reach."""
    if key == "every_subject_country":
        return SUBJECTS
    if key == "every_country":
        limit = val.get_block("limit")
        return SUBJECTS if limit is not None and limit.get_str("is_subject") == "yes" else ALL
    return outer


class _ScriptWalk:
    """Follows a mod's script (read over the game) from a block through the events and scripted
    effects it calls, collecting the colour effects reached. Triggers and limits aren't evaluated:
    a reached effect *may* run."""

    def __init__(self, files):
        self.files = files
        self._effects: dict[str, tuple[str, Block]] | None = None
        self._events: dict[str, tuple[str, Block] | None] = {}
        self._docs: dict[str, Block] = {}
        self._event_files: list[tuple[str, bytes]] | None = None
        self.found: list[tuple[str, str, str]] = []  # (effect, call chain, reach)

    def _doc(self, rel: str, data: bytes) -> Block:
        if rel not in self._docs:
            self._docs[rel] = parse_bytes(data)
        return self._docs[rel]

    def effects(self) -> dict[str, tuple[str, Block]]:
        if self._effects is None:
            self._effects = {}
            for rel, _dlc, data in self.files.iter_files("common/scripted_effects", (".txt",)):
                for p in self._doc(rel, data).pairs():
                    if isinstance(p.value, Block):
                        self._effects[p.key.value] = (rel, p.value)
        return self._effects

    def event(self, eid: str) -> tuple[str, Block] | None:
        if eid not in self._events:
            self._events[eid] = None
            pat = re.compile(rb"\bid\s*=\s*\"?" + re.escape(eid.encode()) + rb"[\"\s}]")
            if self._event_files is None:
                self._event_files = [(rel, data) for rel, _dlc, data in self.files.iter_files("events", (".txt",))]
            for rel, data in self._event_files:
                if not pat.search(data):
                    continue
                for p in self._doc(rel, data).pairs():
                    if isinstance(p.value, Block) and p.value.get_str("id") == eid:
                        self._events[eid] = (rel, p.value)
                        break
                if self._events[eid] is not None:
                    break
        return self._events[eid]

    def walk(self, block: Block, chain: str, wide: str = SCOPED, seen: frozenset = frozenset()) -> None:
        for p in block.pairs():
            key, val = p.key.value, p.value
            if key in COLOUR_EFFECTS:
                self.found.append((key, chain, wide))
            elif key in ("country_event", "province_event") and isinstance(val, Block):
                self._call_event(val.get_str("id"), chain, wide, seen)
            elif key == "events" and isinstance(val, Block):  # on_actions' event lists
                for eid in val.bare():
                    self._call_event(eid, chain, wide, seen)
            elif key == "random_events" and isinstance(val, Block):
                for q in val.pairs():
                    if isinstance(q.value, Scalar):
                        self._call_event(q.value.value, chain, wide, seen)
            elif key in self.effects() and key not in seen:
                rel, body = self.effects()[key]
                self.walk(body, f"{chain} -> scripted effect {key} ({rel})", wide, seen | {key})
            elif isinstance(val, Block):
                self.walk(val, chain, _reach(key, val, wide), seen)

    def _call_event(self, eid: str | None, chain: str, wide: str, seen: frozenset) -> None:
        if not eid or f"event:{eid}" in seen:
            return
        ev = self.event(eid)
        if ev is not None:
            rel, body = ev
            self.walk(body, f"{chain} -> event {eid} ({rel})", wide, seen | {f"event:{eid}"})


def _block_has(block: Block, keys: tuple[str, ...]) -> bool:
    return any(p.key.value in keys or (isinstance(p.value, Block) and _block_has(p.value, keys))
               for p in block.pairs())


@rule("colour_scripts")
def colour_scripts(ctx: Context) -> list[Issue]:
    """Mod scripts that change or restore country colours. In-game test 9: Color-Your-Subject's
    startup event ran ``restore_country_color`` on every country, and MAC's units lost their
    country_colors unit colours (primary became the map colour); with it disabled they were right.
    Warn when an effect is reachable from ``on_startup``; note other effects that reach every country.
    Needs mod checks (the mod's scripts, read over the game)."""
    from eu4unitgen.library.scanner import LayeredFiles

    if ctx.world.mod is None:
        return []
    files = LayeredFiles(ctx.world.mod, ctx.world.game)
    walk = _ScriptWalk(files)
    tags = ", ".join(ctx.nation.tags) or f"the {ctx.nation.set_name} countries"
    out: list[Issue] = []
    startup_events: set[str] = set()
    for rel, _dlc, data in files.iter_files("common/on_actions", (".txt",)):
        for p in walk._doc(rel, data).pairs("on_startup"):
            if isinstance(p.value, Block):
                walk.walk(p.value, f"on_startup ({rel})")
    for effect, chain, wide in dict.fromkeys(walk.found):
        startup_events.update(re.findall(r"-> event (\S+)", chain))
        if wide == SUBJECTS:
            out.append(Issue(INFO, "colour_scripts", f"{chain} -> {effect} on subjects at game start: {tags} "
                                                     "is affected only while it is a subject", subject=effect))
            continue
        if effect == "restore_country_color":
            why = (f"in in-game test 9 this replaced the set's unit colours ({tags} showed its map colour as the "
                   "primary unit colour)")
        else:
            why = f"it may also replace the unit colours of the countries it recolours, {tags} included (UNVERIFIED)"
        out.append(Issue(WARNING, "colour_scripts",
                         f"{chain} -> {effect} on {wide} at game start: {why}. Keep it off the set's countries "
                         "(SPEC: 'Mod scripts that recolour countries')", subject=effect))
    # outside the startup chain: colour effects inside every_country in the mod's own events/decisions
    for folder in ("events", "decisions", "common/on_actions", "common/scripted_effects"):
        for rel, _dlc, data in ctx.world.mod.iter_files(folder, (".txt",)):
            if b"restore_country_color" not in data and b"change_country_color" not in data:
                continue
            for p in _script_units(walk._doc(rel, data)):
                name = p.value.get_str("id") or p.key.value
                if name in startup_events or p.key.value == "on_startup":
                    continue
                if any(_block_has(b, COLOUR_EFFECTS) for b in _every_country_blocks(p.value)):
                    out.append(Issue(INFO, "colour_scripts",
                                     f"{name} changes or restores country colours on every country: when it "
                                     f"runs, {tags} may lose the set's unit colours (UNVERIFIED outside startup)",
                                     rel, subject=name))
    return out


def _script_units(doc: Block) -> list[Pair]:
    """Top-level script units of a file: events, on_actions, scripted effects, or each decision."""
    units = []
    for p in doc.pairs():
        if not isinstance(p.value, Block):
            continue
        if p.key.value in ("country_decisions", "province_decisions"):
            units += [q for q in p.value.pairs() if isinstance(q.value, Block)]
        else:
            units.append(p)
    return units


def _every_country_blocks(block: Block) -> list[Block]:
    found = []
    for p in block.pairs():
        if isinstance(p.value, Block):
            if _reach(p.key.value, p.value, SCOPED) == ALL:
                found.append(p.value)
            else:
                found += _every_country_blocks(p.value)
    return found


class _OneFile:
    """Just enough of GameFiles for library.tags.unit_colours over one in-memory file."""

    def __init__(self, rel: str, data: bytes):
        self.rel, self.data = rel, data

    def iter_files(self, rel_dir: str, suffixes: tuple[str, ...], recursive: bool = False):
        yield self.rel, None, self.data


@rule("country_files")
def country_files(ctx: Context) -> list[Issue]:
    """Generated country-file copies (test mod only): culture listed, historical units exist."""
    out = []
    unit_paths = {p.lower() for p in ctx.generated.files}
    for f in ctx.generated.files.values():
        if not re.match(r"^common/countries/.+\.txt$", f.path):
            continue
        doc = parse_bytes(f.data)
        gfx = doc.get_str("graphical_culture")
        if gfx not in ctx.world.graphical_cultures:
            out.append(Issue(ERROR, "country_files", f"graphical_culture {gfx!r} is not a listed graphical culture", f.path))
        units = doc.get_block("historical_units")
        for unit in units.bare() if units is not None else []:
            rel = ctx.naming.unit_file(unit)
            if rel.lower() not in unit_paths and not ctx.world.exists(rel):
                out.append(Issue(ERROR, "country_files", f"historical unit {unit} has no unit file", f.path))
    return out


def check_descriptor(text: str, expect_path: str | None = None) -> list[Issue]:
    """A launcher .mod descriptor: the keys the user's own descriptors use (name, version,
    tags, supported_version, path), with path pointing at the mod folder."""
    doc, out = parse_bytes(text.encode("utf-8")), []
    for w in doc.warnings:
        out.append(Issue(ERROR, "descriptor", w))
    for key in ("name", "version", "supported_version"):
        if not doc.get_str(key):
            out.append(Issue(ERROR, "descriptor", f"missing {key}"))
    if expect_path is not None and doc.get_str("path") != expect_path:
        out.append(Issue(ERROR, "descriptor", f"path is {doc.get_str('path')!r}, expected {expect_path!r}"))
    if "\\" in (doc.get_str("path") or ""):
        out.append(Issue(ERROR, "descriptor", "path must use forward slashes"))
    return out


# --------------------------------------------------------------------------- in-game test suite

@rule("test_fixtures")
def test_fixtures(ctx: Context) -> list[Issue]:
    """The test suite's extra files do what each check needs (SPEC §10), re-read from the files:
    every fallback tag's patched history leaves no sprite pack able to match it; case b's
    TAG_* entities exist and are all generated; case c's country file uses the nation's gfx
    culture; the rival pack targets the main tag one order step from its pack."""
    fx = ctx.generated.fixtures
    if fx is None:
        return []
    w, out = ctx.world, []
    groups = culture_groups(w.game)
    for case in fx.cases:
        f = ctx.generated.files.get(case.history_file)
        if f is None:
            out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: {case.history_file} was not generated"))
            continue
        facts = apply_history(TagFacts(case.tag, graphical_culture=case.gfx), f.data, groups)
        if facts.primary_culture != fx.culture:
            out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: {case.tag} has primary culture "
                                                     f"{facts.primary_culture!r} at the start, not {fx.culture!r}", f.path))
        cov = coverage(w.idx, facts)
        if cov.packs:
            out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: {case.tag} may still be matched by "
                                                     f"{sorted({p for p, _, _ in cov.packs})}", f.path))
        theirs = [e for e in cov.entities if any(d["origin"] != GENERATED for d in w.defs("entities", e))]
        if theirs:
            out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: vanilla/DLC entities exist for {case.tag}: "
                                                     f"{theirs[:4]}", f.path))
        if case.case == "b":
            want = {ctx.naming.unit_entity(case.tag, u, t) for u in UNIT_TYPES for t in TIERS}
            missing = sorted(want - set(cov.entities))
            if missing:
                out.append(Issue(ERROR, "test_fixtures", f"case b: {case.tag} lacks {missing[:4]}"))
        elif cov.entities:
            out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: {case.tag} has TAG_* entities "
                                                     f"{cov.entities[:4]}; only case b may"))
        if case.country_file:
            cf = ctx.generated.files.get(case.country_file)
            gfx = parse_bytes(cf.data).get_str("graphical_culture") if cf else None
            if gfx != case.gfx:
                out.append(Issue(ERROR, "test_fixtures", f"case {case.case}: {case.country_file} has graphical_culture "
                                                         f"{gfx!r}, expected {case.gfx!r}", case.country_file))
    rival = ctx.gen_defs("sprite_packs").get(fx.rival_pack)
    if rival is None:
        out.append(Issue(ERROR, "test_fixtures", f"rival pack {fx.rival_pack} was not generated"))
    elif rival.get("country_trigger") != f"tag = {ctx.nation.tag}":
        out.append(Issue(ERROR, "test_fixtures", f"{fx.rival_pack}: country_trigger should be 'tag = {ctx.nation.tag}'",
                         rival["defined_in"]))
    orders = {p: [d["order"] for d in w.idx.pack_order.get(p, []) if d["origin"] == GENERATED]
              for p in (fx.rival_pack, ctx.naming.set_sprite_pack(ctx.nation.set_name))}
    (rival_o, main_o) = orders.values()
    if len(rival_o) != 1 or len(main_o) != 1 or abs(rival_o[0] - main_o[0]) != 1:
        out.append(Issue(ERROR, "test_fixtures", f"pack orders {orders} should be one step apart, one entry each"))
    return out


# --------------------------------------------------------------------------- notes

@rule("test_owners")
def test_owners(ctx: Context) -> list[Issue]:
    """TEST ONLY (fix C): each patched province history gives its tag the owner, controller and a core
    at the start date."""
    from eu4unitgen.library.tags import START_DATE, _history_entries

    out = []
    for rel, tag in sorted(ctx.generated.test_owners.items()):
        f = ctx.generated.files.get(rel)
        if f is None:
            out.append(Issue(ERROR, "test_owners", f"{rel} wasn't generated", rel))
            continue
        doc = parse_bytes(f.data)
        now = {p.key.value: p.value.value for p in _history_entries(doc, START_DATE)
               if p.key.value in ("owner", "controller") and isinstance(p.value, Scalar)}
        cores = {p.value.value for p in _history_entries(doc, START_DATE)
                 if p.key.value == "add_core" and isinstance(p.value, Scalar)}
        if now.get("owner") != tag or now.get("controller") != tag or tag not in cores:
            out.append(Issue(ERROR, "test_owners", f"{tag} should own, control and core it at the start: {now}", rel))
        else:
            out.append(Issue(INFO, "test_owners", f"TEST ONLY: {tag} owns it at the start", rel))
    return out


@rule("generator_notes")
def generator_notes(ctx: Context) -> list[Issue]:
    return [Issue(INFO, "generator_notes", note) for note in ctx.generated.notes]


# Which rules cover each generated output (CLAUDE.md rule 4: every generator has a validator rule).
COVERAGE = {
    r"^gfx/entities/.+\.asset$": ["parse_clean", "naming", "missing_nodes", "clone_order", "state_animations", "event_nodes",
                                  "version_collisions", "orphan_entities", "dlc_dependency"],
    r"^gfx/sprite_packs/.+\.txt$": ["parse_clean", "naming", "sprite_packs"],
    r"^gfx/sprite_packs_order/.+\.txt$": ["parse_clean", "pack_order"],
    r"^common/units/.+\.txt$": ["parse_clean", "units"],
    r"^localisation/.+\.yml$": ["file_encoding", "localisation"],
    r"^common/technologies/mil\.txt$": ["parse_clean", "tech_enables"],
    r"^common/graphicalculturetype\.txt$": ["parse_clean", "graphical_culture"],
    r"^INTEGRATION\.md$": ["integration"],
    r"^common/countries/.+\.txt$": ["parse_clean", "country_files"],
    r"^history/countries/.+\.txt$": ["parse_clean", "test_fixtures"],
    r"\.mod$": ["descriptor"],
}

# Rules that check a file outside a GeneratedSet (called directly by their generator).
STANDALONE_RULES = {"descriptor": check_descriptor}
