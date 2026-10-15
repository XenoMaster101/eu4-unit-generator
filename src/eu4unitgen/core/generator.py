"""Phase 1 generator: every script file for one nation's land units, no new meshes.

Output (SPEC §10 Phase 1):
  * 12 TAG entities and 12 graphical-culture default entities (+ riders), full definitions
  * TAG sprite pack, gfx-culture sprite pack, pack-order entry for the TAG pack
  * unit files, localisation, ``enable =`` lines in mil.txt
  * graphicalculturetype.txt with the new culture added
  * INTEGRATION.md: what to merge by hand into files the tool doesn't own (country file,
    and mil.txt / graphicalculturetype.txt when the mod has its own)

Entities point at the base culture's vanilla pdxmeshes, weapons and cannons. Their
names come from the vanilla index, never from string-building; generated names come
from core.naming.

No generated entity uses ``clone =``. The engine resolves a clone while it loads the
files, so the parent must be defined earlier in load order (file name order), and every
vanilla culture file (``western_units.asset``) sorts after a prefixed mod file. The 1.37.5
in-game test logged "Couldn't find parent clone entity" for every clone. Instead each
entity spells out the pdxmesh, states and attaches it would have inherited. Attaches
and pdxmesh names are looked up when the unit is built, not at load, so they may point
at entities and pdxmeshes in later files (the infantry's weapon attach already does).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eu4unitgen.core.naming import TIERS, UNIT_TYPES, Naming
from eu4unitgen.core.nation import STAT_KEYS, Nation, UnitSpec, default_units
from eu4unitgen.core.project import Project
from eu4unitgen.library.scanner import GameFiles, Index, find_pack, pack_tiers, resolved
from eu4unitgen.library.tags import START_DATE, country_files, parse_date
from eu4unitgen.pdxscript.parser import Block, Pair, Scalar, pair, parse_bytes
from eu4unitgen.pdxscript.render import render

STATES = ("idle", "walk", "attack", "retreat")  # SPEC §5: the only live states on vanilla units

# File roles
OWNED = "owned"  # written entirely by the tool
PATCHED = "patched"  # an existing game/mod file plus the tool's additions
SNIPPET = "snippet"  # for the user to read or merge by hand; not loaded by the game
BINARY_SUFFIXES = (".dds",)  # generated files that aren't script text


class GenerationError(ValueError):
    pass


@dataclass
class GeneratedFile:
    path: str  # mod-relative, POSIX
    data: bytes
    role: str = OWNED
    based_on: str | None = None  # PATCHED: "vanilla" or "mod"
    game_path: str | None = None  # PATCHED from vanilla: the game path it copies, when it differs from ``path``
    derived_from: str | None = None  # derived texture: the base-game file it was made from (never DLC, CLAUDE.md rule 2)

    @property
    def text(self) -> str:
        return self.data.decode("utf-8-sig" if self.data.startswith(b"\xef\xbb\xbf") else "latin-1")


@dataclass
class GeneratedSet:
    nation: Nation
    files: dict[str, GeneratedFile] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    integration: "Integration | None" = None
    fixtures: "TestFixtures | None" = None  # in-game test suite only

    def add(self, f: GeneratedFile) -> None:
        if f.path in self.files:
            raise GenerationError(f"two generators wrote {f.path}")
        self.files[f.path] = f

    def script_files(self) -> list[GeneratedFile]:
        return [f for f in self.files.values() if f.role != SNIPPET]

    def package_files(self) -> list[GeneratedFile]:
        """The drop-in package: files the tool creates, plus INTEGRATION.md. Patched copies of
        game files (mil.txt, graphicalculturetype.txt) stay out; INTEGRATION.md carries
        their changes. They remain in the set so the validator sees the merged result."""
        return [f for f in self.files.values() if f.role != PATCHED]


def _utf8(text: str) -> bytes:
    return text.encode("utf-8")


def _utf8_bom(text: str) -> bytes:
    return b"\xef\xbb\xbf" + text.encode("utf-8")


# --------------------------------------------------------------------------- base chain

class BaseChain:
    """The vanilla base-game set of one graphical culture (e.g. westerngfx), per tier.

    Base game only: DLC entities are never copied or used as meshes (SPEC §9).
    """

    def __init__(self, idx: Index, gfx: str):
        pack = find_pack(idx, gfx=gfx)
        if pack is None:
            raise GenerationError(f"no base-game sprite pack lists graphical culture {gfx!r}")
        self.idx, self.gfx, self.pack = idx, gfx, pack
        self.tiers = pack_tiers(idx, pack)
        missing = [t for t in TIERS for u in UNIT_TYPES if u not in self.tiers.get(str(t), {})]
        if missing:
            raise GenerationError(f"{pack['name']} doesn't cover all 4 tiers of all 3 unit types")

    def slot(self, utype: str, tier: int) -> dict:
        return self.tiers[str(tier)][utype]

    def states(self, entity: str) -> list[dict]:
        found = resolved(self.idx, entity, "states") or []
        return [s for s in found if s.get("name") in STATES and s.get("animation")]

    def attaches(self, entity: str) -> list[dict]:
        """The attach blocks ``entity`` ends up with, following its clone chain."""
        return [_attach(a) for a in resolved(self.idx, entity, "attach") or []]

    def flat(self, entity: str) -> dict:
        """What a full copy of ``entity`` needs: pdxmesh, live states (with their events),
        attaches.

        Raises when the vanilla entity carries anything the template can't restate
        (extra keys), rather than dropping it silently.
        """
        pdxmesh = resolved(self.idx, entity, "pdxmesh")
        if not pdxmesh:
            raise GenerationError(f"base entity {entity!r} has no pdxmesh along its clone chain")
        states = self.states(entity)
        if resolved(self.idx, entity, "extra"):
            raise GenerationError(f"base entity {entity!r} sets keys the entity template can't copy: "
                                  f"{sorted(resolved(self.idx, entity, 'extra'))}")
        return {"pdxmesh": pdxmesh, "states": states, "attaches": self.attaches(entity)}


# --------------------------------------------------------------------------- entities

def _attach(raw: dict) -> dict:
    """Scanner attach record ({"name": ..., bone: entity}) -> template form."""
    return {"name": raw.get("name"), "nodes": [(k, v) for k, v in raw.items() if k != "name"]}


def _override_attaches(inherited: list[dict], ours: list[dict]) -> list[dict]:
    """Our attaches replace inherited ones with the same ``name``; the rest are kept in order.
    UNVERIFIED (SPEC §3): whether clones replace attaches by name; vanilla restates the named
    attach on every clone, so a full copy with the named one swapped matches what it gets."""
    by_name = {a["name"]: a for a in ours if a["name"]}
    out = [by_name.pop(a["name"]) if a["name"] in by_name else a for a in inherited]
    return out + [a for a in ours if not a["name"] or a["name"] in by_name]


def _entity(name: str, *, version: int, pdxmesh: str, states: list[dict], attaches: list[dict]) -> dict[str, Any]:
    return {"name": name, "pdxmesh": pdxmesh, "version": version, "states": states, "attaches": attaches}


def unit_entities(owner: str, chain: BaseChain, naming: Naming, version: int) -> list[dict]:
    """The 4-tier set for one owner (a TAG or a graphical culture), in tier order."""
    out = []
    for tier in TIERS:
        inf, cav, art = (chain.slot(u, tier) for u in UNIT_TYPES)
        infantry = naming.unit_entity(owner, "infantry", tier)
        rider = naming.rider_entity(owner, tier)

        if not inf.get("pdxmesh") or not inf.get("weapon_node"):
            raise GenerationError(f"{inf['entity']}: no pdxmesh or weapon attach to copy")
        out.append(_entity(
            infantry, version=version, pdxmesh=inf["pdxmesh"], states=chain.states(inf["entity"]),
            attaches=[{"name": None, "nodes": [(inf["weapon_node"], inf["weapon_entity"])]}]))

        # Rider: the vanilla rider's horse_* states and weapon, on this tier's infantry pdxmesh
        # (same .mesh as the infantry, SPEC §5, §7.2).
        base = chain.flat(cav["rider_entity"])
        out.append(_entity(rider, version=version, pdxmesh=inf["pdxmesh"], states=base["states"],
                           attaches=base["attaches"]))
        base = chain.flat(cav["entity"])
        out.append(_entity(
            naming.unit_entity(owner, "cavalry", tier), version=version, pdxmesh=base["pdxmesh"],
            states=base["states"], attaches=_override_attaches(
                base["attaches"], [{"name": "rider", "nodes": [("Saddle_Node", rider)]}])))
        base = chain.flat(art["entity"])
        out.append(_entity(
            naming.unit_entity(owner, "artillery", tier), version=version, pdxmesh=base["pdxmesh"],
            states=base["states"], attaches=_override_attaches(
                base["attaches"], [{"name": "gunman", "nodes": [("infantry_node", infantry),
                                                                ("cannon_node", art["cannon_entity"])]}])))
    return out


def _sprites(owner: str, naming: Naming) -> list[tuple[str, list[tuple[int, str]]]]:
    return [(u, [(t - 1, naming.unit_entity(owner, u, t)) for t in TIERS]) for u in UNIT_TYPES]


# --------------------------------------------------------------------------- patched files

def _read_current(rel: str, game: GameFiles, mod: GameFiles | None) -> tuple[bytes, str]:
    """The mod's copy if it has one (it replaces vanilla), else vanilla."""
    if mod is not None:
        data = mod.read(rel)
        if data is not None:
            return data, "mod"
    data = game.read(rel)
    if data is None:
        raise GenerationError(f"{rel} not found in the mod or the EU4 install")
    return data, "vanilla"


def patch_graphical_cultures(data: bytes, new: list[str]) -> bytes:
    doc = parse_bytes(data)
    have = set(doc.bare())
    for gfx in new:
        if gfx not in have:
            doc.append(Scalar(gfx))
            have.add(gfx)
    return doc.to_bytes()


def patch_sprite_levels(data: bytes, levels: dict[int, int]) -> bytes:
    """TEST ONLY: move ``sprite_level = N`` to MIL tech index ``levels[N]``.
    Lets a test nation reach every sprite tier in minutes. Never used for a real package."""
    doc = parse_bytes(data)
    techs = [v for v in doc.get_all("technology") if isinstance(v, Block)]
    for level, tech in levels.items():
        if not 0 <= tech < len(techs):
            raise GenerationError(f"sprite level {level}: MIL tech {tech} doesn't exist (mil.txt has {len(techs)})")
        for t in techs:
            for p in list(t.pairs("sprite_level")):
                if isinstance(p.value, Scalar) and p.value.value == str(level):
                    t.remove(p)
    for level, tech in sorted(levels.items()):
        target = techs[tech]
        year = list(target.pairs("year"))
        new = pair("sprite_level", str(level))
        if year:
            target.insert_after(year[-1], new)
        else:
            target.append(new)
    return doc.to_bytes()


def patch_mil_enables(data: bytes, units: list[UnitSpec]) -> bytes:
    """Add ``enable = <unit>`` to the right technology block, after its last enable.
    Idempotent: units already enabled anywhere in the file are left alone."""
    doc = parse_bytes(data)
    techs = [v for v in doc.get_all("technology") if isinstance(v, Block)]
    already = {v.value for t in techs for v in t.get_all("enable") if isinstance(v, Scalar)}
    for unit in units:
        if unit.key in already:
            continue
        if unit.mil_tech >= len(techs):
            raise GenerationError(f"{unit.key}: MIL tech {unit.mil_tech} doesn't exist (mil.txt has {len(techs)})")
        tech = techs[unit.mil_tech]
        enables = list(tech.pairs("enable"))
        new = pair("enable", unit.key)
        if enables:
            tech.insert_after(enables[-1], new)
        else:
            tech.append(new)
        already.add(unit.key)
    return doc.to_bytes()


@dataclass
class Integration:
    """Changes that belong in files the tool doesn't own, as data. INTEGRATION.md renders it for
    a human; the test mod applies it to copies of the vanilla files."""
    tag: str
    set_name: str
    country_file: str | None  # vanilla path of the tag's country file, if the tag exists in vanilla
    graphical_culture: str
    colour: tuple[int, int, int] | None
    historical_units: list[str]
    mil_enables: dict[int, list[str]]  # MIL tech index -> unit keys
    graphical_cultures_added: list[str]
    unit_colours: tuple[tuple[int, int, int], ...] | None = None  # common/country_colors entry, if set
    unit_colours_snippet: str = ""


def patch_history_culture(data: bytes, culture: str, date: tuple[int, int, int] = START_DATE) -> bytes:
    """TEST ONLY: set every ``primary_culture`` in effect at ``date`` (undated, and in dated
    blocks up to ``date``) to ``culture``, so no culture-based sprite pack matches the tag.
    Later dated changes and everything else stay byte-for-byte."""
    doc = parse_bytes(data)
    found = 0
    for p in doc.pairs():
        when = parse_date(p.key.value)
        targets = [p] if when is None else (list(p.value.pairs()) if isinstance(p.value, Block) and when <= date else [])
        for t in targets:
            if t.key.value == "primary_culture":
                t.value = Scalar(culture)
                found += 1
    if not found:
        raise GenerationError("history file has no primary_culture in effect at the start date")
    return doc.to_bytes()


def patch_country_gfx(data: bytes, gfx: str) -> bytes:
    """TEST ONLY: set a country file's graphical_culture; everything else stays byte-for-byte."""
    doc = parse_bytes(data)
    found = list(doc.pairs("graphical_culture"))
    if not found:
        raise GenerationError("country file has no graphical_culture")
    for p in found:
        p.value = Scalar(gfx)
    return doc.to_bytes()


@dataclass
class FallbackCase:
    """One fallback check of the in-game test suite (SPEC §10)."""
    case: str  # "a": no pack, no TAG entities; "b": TAG entities, no pack; "c": new gfx culture, gfx pack only
    tag: str
    history_file: str  # vanilla path, patched to the suite culture
    vanilla_gfx: str | None
    gfx: str | None  # the graphical culture in effect in the test mod
    country_file: str | None = None  # set when the test mod patches it (case c)


@dataclass
class TestFixtures:
    """What the in-game test suite deliberately adds around a unit set, so the validator can
    check it and tell deliberate oddities (an unpacked TAG set, a rival pack) from mistakes."""
    culture: str  # primary culture given to every fallback tag
    cases: list[FallbackCase]
    main_pack: str
    rival_pack: str
    rival_gfx: str
    main_order: int
    rival_order: int

    @property
    def unpacked_owners(self) -> list[str]:
        return [c.tag for c in self.cases if c.case == "b"]


def patch_country_colours(data: bytes, snippet: str, tag: str) -> bytes:
    """Apply an Integration's country_colors entry: replace the tag's block (or append it).
    The rest of the file is kept byte-for-byte."""
    doc = parse_bytes(data)
    if b"\r\n" in data:  # match the file's line endings
        snippet = snippet.replace("\r\n", "\n").replace("\n", "\r\n")
    new = next(parse_bytes(snippet.encode("utf-8")).pairs(tag), None)
    if new is None:
        raise GenerationError(f"country_colors snippet has no {tag} entry")
    found = list(doc.pairs(tag))
    if found:
        found[0].value = new.value
        for extra in found[1:]:
            doc.remove(extra)
    else:
        doc.append(new)
    return doc.to_bytes()


def patch_country(data: bytes, integ: Integration) -> bytes:
    """Apply the country-file part of an Integration: graphical_culture, color (if set),
    historical_units. Everything else in the file is kept byte-for-byte."""
    doc = parse_bytes(data)

    def set_value(key: str, value: Scalar | Block) -> None:
        found = list(doc.pairs(key))
        if found:
            found[0].value = value
            for extra in found[1:]:
                doc.remove(extra)
        else:
            doc.append(Pair(Scalar(key), "=", value))

    set_value("graphical_culture", Scalar(integ.graphical_culture))
    if integ.colour is not None:
        set_value("color", Block([Scalar(str(c)) for c in integ.colour]))
    set_value("historical_units", Block([Scalar(u) for u in integ.historical_units]))
    return doc.to_bytes()


# --------------------------------------------------------------------------- top level

def generate(project: Project, nation: Nation, game: GameFiles, idx: Index, mod: GameFiles | None = None,
             test_sprite_techs: dict[int, int] | None = None, recipe=None) -> GeneratedSet:
    """Build every Phase 1 file for ``nation``. Reads vanilla through ``game``/``idx`` and, when
    given, the mod's current copies of shared files through ``mod``. Writes nothing.

    ``test_sprite_techs`` (test mod only) moves sprite levels to the given MIL techs, e.g.
    {2: 2, 3: 3, 4: 4}, unlocks each tier's units there and all tier-1 units at MIL 1.

    ``recipe`` (a core.recipe.TextureRecipe, Phase 2) retextures the TAG set's slots through
    pdxmesh meshsettings (core.retexture); the gfx-culture defaults stay on vanilla meshes."""
    naming, version = project.naming, project.entity_version
    chain = BaseChain(idx, nation.base_gfx)
    if not nation.units:
        sprite_mil = {int(k): v["mil_tech"] for k, v in idx.sprite_levels.items()}
        if test_sprite_techs:
            sprite_mil.update(test_sprite_techs)
        if any(t not in sprite_mil for t in TIERS):
            raise GenerationError("mil.txt doesn't define sprite_level 1-4")
        first = {u: 1 for u in UNIT_TYPES} if test_sprite_techs else None
        nation.units = default_units(nation, naming, sprite_mil, first)
    out = GeneratedSet(nation)
    tag, gfx, set_name = nation.tag, nation.gfx_culture, nation.set_name

    # Entities: TAG set + gfx-culture defaults (SPEC §3: both, always).
    tag_entities = unit_entities(tag, chain, naming, version)
    if recipe is not None and recipe.active():
        from eu4unitgen.core.retexture import retexture_set

        meshes, textures, notes = retexture_set(naming, set_name, tag, tag_entities, recipe, chain, game, idx, version)
        out.add(GeneratedFile(naming.pdxmesh_file(set_name), _utf8(render("pdxmeshes.gfx.j2", meshes=meshes))))
        for f in textures:
            out.add(f)
        out.notes += notes
    out.add(GeneratedFile(naming.set_entities_file(set_name), _utf8(render(
        "entities.asset.j2", entities=tag_entities))))
    gfx_chain = BaseChain(idx, nation.gfx_set_base) if nation.gfx_set_base else chain
    out.add(GeneratedFile(naming.gfx_entities_file(gfx), _utf8(render(
        "entities.asset.j2", entities=unit_entities(gfx, gfx_chain, naming, version)))))

    # Sprite packs + order (order entry for the TAG pack only, SPEC §3).
    out.add(GeneratedFile(naming.set_pack_file(set_name), _utf8(render(
        "sprite_pack.txt.j2", pack=naming.set_sprite_pack(set_name), trigger=f"tag = {tag}",
        graphical_cultures=None, sprites=_sprites(tag, naming)))))
    out.add(GeneratedFile(naming.gfx_pack_file(gfx), _utf8(render(
        "sprite_pack.txt.j2", pack=naming.gfx_sprite_pack(gfx), trigger="always = no",
        graphical_cultures=[gfx], sprites=_sprites(gfx, naming)))))
    out.add(GeneratedFile(naming.pack_order_file(set_name), _utf8(render(
        "pack_order.txt.j2", orders=[(naming.set_sprite_pack(set_name), project.tag_pack_order)]))))

    # Units + localisation.
    for unit in nation.units:
        out.add(GeneratedFile(naming.unit_file(unit.key), _utf8(render(
            "unit.txt.j2", unit=unit, tech_group=nation.tech_group, stat_keys=STAT_KEYS))))
    entries = [(k, v) for u in nation.units
               for k, v in ((naming.loc_name_key(u.key), u.name), (naming.loc_desc_key(u.key), u.description))]
    for lang in project.languages:
        out.add(GeneratedFile(naming.loc_file(set_name, lang),
                              _utf8_bom(render("loc.yml.j2", language=lang, entries=entries))))

    # Shared single files: patch the mod's copy if it has one, else vanilla's.
    rel = naming.mil_tech_file()
    data, base = _read_current(rel, game, mod)
    if test_sprite_techs:
        data = patch_sprite_levels(data, test_sprite_techs)
        out.notes.append("TEST ONLY: sprite levels moved to MIL techs "
                         + ", ".join(f"{lvl}->{t}" for lvl, t in sorted(test_sprite_techs.items())))
    out.add(GeneratedFile(rel, patch_mil_enables(data, nation.units), PATCHED, base))
    rel = naming.graphical_culture_file()
    data, base = _read_current(rel, game, mod)
    out.add(GeneratedFile(rel, patch_graphical_cultures(data, [gfx]), PATCHED, base))

    # Hand-merge instructions for files the tool doesn't own.
    cfile = country_files(game).get(tag)
    out.integration = Integration(
        tag=tag, set_name=set_name, country_file=cfile, graphical_culture=gfx, colour=nation.colour,
        historical_units=[u.key for u in nation.units],
        mil_enables={t: [u.key for u in nation.units if u.mil_tech == t] for t in sorted({u.mil_tech for u in nation.units})},
        graphical_cultures_added=[gfx] if gfx not in idx.graphical_cultures else [],
        unit_colours=nation.unit_colours,
        unit_colours_snippet=render("country_colors_snippet.txt.j2", tag=tag, colours=nation.unit_colours)
        if nation.unit_colours else "")
    country_lines = render("country_snippet.txt.j2", gfx=gfx, colour=nation.colour,
                           units=out.integration.historical_units)
    out.add(GeneratedFile(naming.integration_file(set_name), _utf8(render(
        "integration.md.j2", integ=out.integration, country_lines=country_lines,
        mil_file=naming.mil_tech_file(), gct_file=naming.graphical_culture_file(),
        colours_file=naming.country_colours_file())), SNIPPET))

    if gfx not in idx.graphical_cultures:
        out.notes.append(
            f"{gfx} is a new graphical culture. UNVERIFIED: whether the engine also needs portrait, "
            f"city or advisor graphics for it; check error.log after the first launch.")
    return out
