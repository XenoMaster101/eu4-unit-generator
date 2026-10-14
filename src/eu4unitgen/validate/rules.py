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

from eu4unitgen.core.generator import BINARY_SUFFIXES, SNIPPET, GeneratedSet
from eu4unitgen.core.naming import TIERS, UNIT_TYPES, Naming
from eu4unitgen.core.nation import MANEUVER, Nation
from eu4unitgen.core.project import Project
from eu4unitgen.library.tags import TagFacts, apply_history, coverage, culture_groups
from eu4unitgen.pdxscript.parser import Block, Scalar, parse_bytes
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
    matches its format, size and mip count (SPEC §4: a mismatch shows as a black/wrong unit)."""
    from eu4unitgen.textures.dds import DDSError, DDSFormat, check_matches, read_header

    out = []
    game = ctx.world.game
    for f in ctx.generated.files.values():
        if not f.path.lower().endswith(".dds"):
            continue
        base = f.derived_from
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
    for f in ctx.generated.files.values():
        if not f.path.lower().endswith(".dds"):
            continue
        if f.path.lower() not in named:
            out.append(Issue(ERROR, "texture_paths", "generated texture isn't in the folder of a mesh whose generated "
                                                     "pdxmesh names it by bare file name", f.path))
        if w.game.exists(f.path):
            where = "base-game" if w.game.in_base_game(f.path) else "DLC"
            out.append(Issue(ERROR, "texture_paths", f"same path as a {where} texture: the mod's file would replace "
                                                     "it for every nation", f.path))
        elif w.mod is not None and w.mod.exists(f.path):
            out.append(Issue(ERROR, "texture_paths", "same path as a file already in the mod", f.path))
    return out


# --------------------------------------------------------------------------- naming

@rule("naming")
def naming(ctx: Context) -> list[Issue]:
    """Generated entity and pack names are exactly the ones naming.py produces."""
    n, tag, gfx = ctx.naming, ctx.nation.tag, ctx.nation.gfx_culture
    expected = set()
    fx = ctx.generated.fixtures
    for owner in (tag, gfx, *(fx.unpacked_owners if fx else [])):
        for t in TIERS:
            expected |= {n.unit_entity(owner, u, t) for u in UNIT_TYPES} | {n.rider_entity(owner, t)}
    out = []
    names = set(ctx.gen_entities())
    # Optional: an owner's own cannon entity, made only when its cannon is retextured (Phase 2).
    optional = {n.cannon_entity(owner, t) for owner in (tag, gfx) for t in TIERS}
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
    n, tag, gfx = ctx.naming, ctx.nation.tag, ctx.nation.gfx_culture
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
    if tag_pack is not None and tag_pack.get("country_trigger") != f"tag = {tag}":
        out.append(Issue(ERROR, "sprite_packs", f"{tag_pack['name']}: country_trigger should be 'tag = {tag}'",
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


@rule("pack_order")
def pack_order(ctx: Context) -> list[Issue]:
    """The TAG pack has an order entry; flag other packs that might match the tag at >= that order."""
    n, tag, out = ctx.naming, ctx.nation.tag, []
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
            if ctx.world.loaded(p) and not _excludes_tag(p.get("country_trigger"), tag):
                if fx and other == fx.rival_pack:
                    out.append(Issue(INFO, "pack_order", f"{other} (order {order}) is the pack-priority test's "
                                                         f"deliberate rival of {pack} (order {ours})", p["defined_in"]))
                    continue
                out.append(Issue(WARNING, "pack_order",
                                 f"{other} (order {order}, trigger '{p.get('country_trigger')}') may match {tag} "
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
    if integ.country_file and not ctx.world.game.exists(integ.country_file):
        out.append(Issue(ERROR, "integration", f"{integ.country_file} doesn't exist in the game", path))
    return out


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
