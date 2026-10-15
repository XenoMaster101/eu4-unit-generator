"""Wrap a generated unit set into a standalone throwaway mod for in-game testing.

Layout, inside the EU4 user mod directory:
    eu4unitgen_test.mod                 launcher descriptor (with path=)
    eu4unitgen_test/descriptor.mod      same, without path
    eu4unitgen_test/...                 the package: entities, packs, units, loc, mil.txt, ...
    eu4unitgen_test/common/countries/<Tag>.txt
                                        vanilla country file with INTEGRATION.md applied
                                        (graphical_culture, historical_units, color if set)
    eu4unitgen_test/INTEGRATION.md      what was applied (from the package)
    eu4unitgen_test/.eu4unitgen/        manifest

``build_suite`` builds the in-game test suite instead: two variant mods
(eu4unitgen_test_v1, _v2; enable one at a time) that differ only in the pack-priority
orders, each holding the main tag's set plus the fallback cases (SPEC §10).

INTEGRATION.md changes are applied here only, never to a real mod. The test mod reads
nothing from any other mod. Export rules are the same as for a real mod: dry run first,
and files the tool didn't write are never overwritten.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen import __version__
from eu4unitgen.core.generator import (
    BASE, OWNED, PATCHED, BaseChain, FallbackCase, GeneratedFile, GeneratedSet, GenerationError, TestFixtures, generate,
    patch_country, patch_country_colours, patch_country_gfx, patch_history_culture, patch_province_owner,
    unit_entities,
)
from eu4unitgen.core.naming import MANIFEST_FILE, TEST_MOD_NAME, UNIT_TYPES
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.export.writer import ExportPlan, plan_export
from eu4unitgen.library.scanner import GameFiles, Index, find_pack
from eu4unitgen.library.tags import TagFacts, coverage
from eu4unitgen.pdxscript.render import render
from eu4unitgen.validate import Report, validate
from eu4unitgen.validate.rules import ERROR, check_descriptor

DESCRIPTOR = "descriptor"  # file role


def supported_version(idx: Index) -> str:
    """'release_1.37.5' -> 'v1.37.*' (the form the user's own descriptors use)."""
    branch = idx.game_version.get("eu4_branch", "")
    parts = branch.removeprefix("release_").split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts[:2]):
        raise GenerationError(f"can't read the game version from eu4_branch.txt ({branch!r})")
    return f"v{parts[0]}.{parts[1]}.*"


def descriptor_text(name: str, supported: str, path: str | None) -> str:
    lines = [f'version="{__version__}"', 'tags={', '\t"Graphics"', '\t"Military"', '}',
             f'name="{name}"', f'supported_version="{supported}"']
    if path is not None:
        lines.append(f'path="{path}"')
    return "\n".join(lines) + "\n"


@dataclass
class TestMod:
    package: GeneratedSet  # paths relative to the test mod folder
    report: Report
    plan: ExportPlan  # paths relative to the EU4 mod directory


def _main_package(project: Project, nation: Nation, game: GameFiles, idx: Index,
                  test_sprite_techs: dict[int, int] | None, recipe=None, loadout=None, runner=None,
                  part_cache: dict | None = None) -> GeneratedSet:
    """The unit set plus the main tag's country file with INTEGRATION.md applied."""
    package = generate(project, nation, game, idx, mod=None, test_sprite_techs=test_sprite_techs, recipe=recipe,
                       loadout=loadout, runner=runner, part_cache=part_cache)
    integ = package.integration
    for tag in nation.tags:
        cfile = integ.country_files.get(tag) if integ else None
        if not cfile:
            raise GenerationError(f"{tag} has no vanilla country file; the test mod needs vanilla tags")
        original = game.read(cfile)
        if original is None:
            raise GenerationError(f"{cfile} not found in the EU4 install")
        package.add(GeneratedFile(cfile, patch_country(original, integ), PATCHED, "vanilla"))
    if integ.unit_colours:
        rel = project.naming.country_colours_file()
        data = game.read(rel)
        if data is None:
            raise GenerationError(f"{rel} not found in the EU4 install")
        package.add(GeneratedFile(rel, patch_country_colours(data, integ.unit_colours_snippet, list(integ.unit_colours)),
                                  PATCHED, "vanilla"))
    return package


def _wrap(project: Project, nation: Nation, package: GeneratedSet, game: GameFiles, idx: Index, mod_dir: Path,
          name: str, report: Report | None = None, display: str | None = None,
          copies: dict[str, bytes] | None = None) -> TestMod:
    """Validate the package (unless ``report`` is given), then put it in a mod folder with both
    descriptors and plan the export. ``display``: the launcher name (default: the folder name).
    ``copies``: a base mod's files, copied unchanged where the package doesn't write that path."""
    if report is None:
        report = validate(project, nation, package, game, idx, mod=None)
    folder = Path(mod_dir) / name
    supported = supported_version(idx)
    outer = descriptor_text(display or name, supported, folder.as_posix())
    inner = descriptor_text(display or name, supported, None)
    report.issues += [i for i in check_descriptor(outer, folder.as_posix())]
    report.issues += [i for i in check_descriptor(inner)]

    wrapped = GeneratedSet(nation, notes=list(package.notes), integration=package.integration,
                           fixtures=package.fixtures)
    for f in package.files.values():
        wrapped.add(GeneratedFile(f"{name}/{f.path}", f.data, f.role, f.based_on, game_path=f.game_path or f.path))
    written = {f.path.lower() for f in package.files.values()}
    for rel, data in (copies or {}).items():
        if rel.lower() not in written:
            wrapped.add(GeneratedFile(f"{name}/{rel}", data, BASE))
    wrapped.add(GeneratedFile(f"{name}.mod", outer.encode("utf-8"), DESCRIPTOR))
    wrapped.add(GeneratedFile(f"{name}/descriptor.mod", inner.encode("utf-8"), DESCRIPTOR))
    plan = plan_export(mod_dir, wrapped, game, [str(i) for i in report.issues if i.severity == ERROR],
                       manifest_rel=f"{name}/{MANIFEST_FILE}")
    return TestMod(package, report, plan)


# --------------------------------------------------------------------------- test mod on a copy of a base mod

BASE_SKIP_DIRS = {".git", ".eu4unitgen"}  # never copied from the base mod
BASE_SKIP_FILES = {"descriptor.mod"}  # replaced by the test mod's own


def base_descriptor_name(base: Path) -> str:
    """The base mod's display name (its descriptor.mod ``name``), else its folder name."""
    from eu4unitgen.pdxscript.parser import parse_bytes

    desc = Path(base) / "descriptor.mod"
    if desc.is_file():
        name = parse_bytes(desc.read_bytes()).get_str("name")
        if name:
            return name
    return Path(base).name


def base_files(base: Path) -> dict[str, bytes]:
    """Every file of the base mod (posix paths), minus .git, the tool's folder and descriptor.mod."""
    base = Path(base)
    out = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if rel.parts[0] in BASE_SKIP_DIRS or rel.as_posix().lower() in BASE_SKIP_FILES:
            continue
        out[rel.as_posix()] = p.read_bytes()
    return out


def own_cores(package: GeneratedSet, tag: str, files) -> list[str]:
    """TEST ONLY (fix C): ``tag`` owns the provinces it has cores on, and its history capital, at the
    start date, so a releasable tag (no land in 1444) can be played straight away. ``files``: the
    base mod over the game (LayeredFiles). Returns the patched province history paths."""
    from eu4unitgen.library.tags import province_cores, province_file
    from eu4unitgen.pdxscript.parser import parse_bytes

    paths = province_cores(files, tag)
    hist = next((p for p, _, _ in files.iter_files("history/countries", (".txt",))
                 if p.rsplit("/", 1)[-1][:3].upper() == tag), None)
    capital = parse_bytes(files.read(hist)).get_str("capital") if hist else None
    cap_path = province_file(files, capital) if capital and capital.isdigit() else None
    if cap_path and cap_path not in paths:
        paths.append(cap_path)
    if not paths:
        raise GenerationError(f"{tag} has no cores and no capital province to own")
    for rel in paths:
        package.add(GeneratedFile(rel, patch_province_owner(files.read(rel), tag), PATCHED,
                                  "mod" if files.from_top(rel) else "vanilla"))
        package.test_owners[rel] = tag
    package.notes.append(f"TEST ONLY: {tag} owns {len(paths)} provinces at the start (its cores and capital): "
                         + ", ".join(p.rsplit("/", 1)[-1].removesuffix(".txt") for p in paths))
    return paths


def build_base_test_mod(project: Project, nation: Nation, game: GameFiles, idx: Index, mod_dir: Path, base: Path,
                        test_sprite_techs: dict[int, int] | None = None, recipe=None, loadout=None, runner=None,
                        part_cache: dict | None = None, give_cores: bool = False,
                        name: str | None = None) -> TestMod:
    """Fix B: a test mod that is a full copy of a base mod (e.g. Victoria's Secret) with the set
    integrated: INTEGRATION.md applied to copies of the base mod's own files (country file, mil.txt,
    graphicalculturetype.txt, country_colors), the package added, everything else copied unchanged.
    Enable it instead of the base mod. ``give_cores`` (fix C, TEST ONLY): the tag owns its cored
    provinces and capital at the start. The base mod is only read."""
    from eu4unitgen.library.scanner import LayeredFiles
    from eu4unitgen.library.tags import country_files

    base = Path(base)
    if not base.is_dir():
        raise GenerationError(f"base mod folder {base} not found")
    mod = GameFiles(base, include_dlc=False)
    try:
        files = LayeredFiles(mod, game)
        package = generate(project, nation, game, idx, mod=mod, test_sprite_techs=test_sprite_techs, recipe=recipe,
                           loadout=loadout, runner=runner, part_cache=part_cache)
        integ = package.integration
        cfiles = country_files(files)
        for tag in nation.tags:
            cfile = cfiles.get(tag)
            raw = files.read(cfile) if cfile else None
            if raw is None:
                raise GenerationError(f"{tag} has no country file in the base mod or the game")
            package.add(GeneratedFile(cfile, patch_country(raw, integ), PATCHED,
                                      "mod" if files.from_top(cfile) else "vanilla"))
        if integ.unit_colours:
            rel = project.naming.country_colours_file()
            raw = files.read(rel)
            if raw is None:
                raise GenerationError(f"{rel} not found in the base mod or the game")
            package.add(GeneratedFile(rel, patch_country_colours(raw, integ.unit_colours_snippet,
                                                                 list(integ.unit_colours)),
                                      PATCHED, "mod" if files.from_top(rel) else "vanilla"))
        if give_cores:
            for tag in nation.tags:
                own_cores(package, tag, files)
        report = validate(project, nation, package, game, idx, mod=mod)
    finally:
        mod.close()
    folder = project.naming.base_test_mod(base.name)
    display = name or f"{base_descriptor_name(base)} - unitgen test"
    return _wrap(project, nation, package, game, idx, mod_dir, folder, report=report, display=display,
                 copies=base_files(base))


def build_test_mod(project: Project, nation: Nation, game: GameFiles, idx: Index, mod_dir: Path,
                   name: str = TEST_MOD_NAME, test_sprite_techs: dict[int, int] | None = None,
                   recipe=None, loadout=None, runner=None, part_cache: dict | None = None) -> TestMod:
    """``test_sprite_techs`` (e.g. {2: 2, 3: 3, 4: 4}) moves sprite levels 2-4 to early MIL techs in
    this test mod's mil.txt only, so every tier can be seen in minutes. ``recipe`` (texture recipe)
    and ``loadout`` (parts, built through ``runner``): the set as generate() would write it."""
    package = _main_package(project, nation, game, idx, test_sprite_techs, recipe, loadout, runner, part_cache)
    return _wrap(project, nation, package, game, idx, mod_dir, name)


# --------------------------------------------------------------------------- test suite

# Vanilla looks each fallback tag should start from, so every outcome on the map can be told
# apart from the main set (western), the rival pack and the gfx default set.
CASE_GFX = {"a": ("africangfx",), "b": ("indiangfx",), "c": ("southamericagfx",)}
CASES = ("a", "b", "c")


@dataclass
class Suite:
    variants: list[TestMod]
    fixtures: TestFixtures
    main: TagFacts
    nation: Nation
    notes: list[str] = field(default_factory=list)


def _patched_facts(f: TagFacts, culture: str, groups: dict[str, str]) -> TagFacts:
    return dataclasses.replace(f, primary_culture=culture, culture_group=groups.get(culture))


def pick_fallback_tags(idx: Index, facts: dict[str, TagFacts], culture: str, groups: dict[str, str],
                       exclude: set[str], avoid_gfx: set[str]) -> dict[str, TagFacts]:
    """One tag per fallback case that no sprite pack or TAG_* entity can touch once its primary
    culture is ``culture`` (the clean main tag's): pick-test-tag's test, after the patch.
    Prefers the CASE_GFX look, then a look not in ``avoid_gfx`` or already picked, then size."""
    usable = [f for t, f in sorted(facts.items())
              if t not in exclude and f.history_file and f.country_file and f.primary_culture
              and coverage(idx, _patched_facts(f, culture, groups)).clean]
    out: dict[str, TagFacts] = {}
    for case in CASES:
        taken = {f.tag for f in out.values()}
        used = avoid_gfx | {f.graphical_culture for f in out.values()}

        def rank(f: TagFacts) -> tuple:
            return (f.graphical_culture not in CASE_GFX[case], f.graphical_culture in used, -f.provinces, f.tag)
        pool = sorted((f for f in usable if f.tag not in taken), key=rank)
        if not pool:
            raise GenerationError(f"no tag left for fallback case {case}")
        out[case] = pool[0]
    return out


def build_suite(project: Project, nation: Nation, game: GameFiles, idx: Index, mod_dir: Path,
                facts: dict[str, TagFacts], groups: dict[str, str], *, rival_gfx: str = "asiangfx",
                test_sprite_techs: dict[int, int] | None = None,
                fallback_tags: dict[str, str] | None = None) -> Suite:
    """The in-game test suite (SPEC §10), as two variant mods for at most two launches.

    Both hold the same files except the pack orders:
      * the main tag (``nation``, a clean tag such as HDA) with its full set and pack A, plus a
        rival pack B for the same tag listing the vanilla ``rival_gfx`` set.
        v1: A = N, B = N + 1.  v2: A = N + 1, B = N.  (N = project.tag_pack_order)
      * fallback case a: a tag with no pack and no TAG_* entities;
        case b: a tag with generated TAG_* entities (copies of base_gfx) and no pack;
        case c: a tag switched to the nation's graphical culture, covered only by its gfx pack.
        Each fallback tag gets the main tag's primary culture, so no culture pack matches it.
    """
    main = facts.get(nation.tag)
    if main is None or not coverage(idx, main).clean:
        raise GenerationError(f"{nation.tag} isn't clean (see pick-test-tag); the suite needs a clean main tag")
    culture = main.primary_culture
    if not culture:
        raise GenerationError(f"{nation.tag} has no primary culture at the start date")
    rival = find_pack(idx, gfx=rival_gfx)
    if rival is None:
        raise GenerationError(f"no base-game sprite pack for rival graphical culture {rival_gfx!r}")
    looks = {nation.base_gfx, nation.gfx_set_base or nation.base_gfx, rival_gfx, main.graphical_culture}
    if fallback_tags:
        picked = {}
        for case, tag in fallback_tags.items():
            f = facts.get(tag)
            if f is None or not f.history_file or not f.country_file:
                raise GenerationError(f"fallback tag {tag} owns no land at the start date or has no history/country file")
            if not coverage(idx, _patched_facts(f, culture, groups)).clean:
                raise GenerationError(f"fallback tag {tag} is still matched by a sprite pack or has TAG_* entities "
                                      f"with primary culture {culture}")
            picked[case] = f
    else:
        picked = pick_fallback_tags(idx, facts, culture, groups, {nation.tag}, looks)

    naming = project.naming
    cases = [FallbackCase(case, f.tag, f.history_file, f.graphical_culture,
                          nation.gfx_culture if case == "c" else f.graphical_culture,
                          f.country_file if case == "c" else None)
             for case, f in sorted(picked.items())]
    shown = {"the main set": nation.base_gfx, "the rival pack": rival_gfx,
             f"the {nation.gfx_culture} defaults": nation.gfx_set_base or nation.base_gfx,
             f"{nation.tag}'s native set": main.graphical_culture,
             **{f"case {c.case} ({c.tag}) native": c.vanilla_gfx for c in cases}}
    by_look: dict[str, list[str]] = {}
    for what, look in shown.items():
        by_look.setdefault(str(look), []).append(what)
    notes = [f"{' and '.join(whats)} share the {look} look; those outcomes can't be told apart on the map"
             for look, whats in sorted(by_look.items()) if len(whats) > 1]

    base_chain = BaseChain(idx, nation.base_gfx)
    sprites = [(u, sorted((int(i), e) for i, e in rival.get(u, {}).items())) for u in UNIT_TYPES]
    variants = []
    n = project.tag_pack_order
    for variant, (order_a, order_b) in ((1, (n, n + 1)), (2, (n + 1, n))):
        proj = dataclasses.replace(project, tag_pack_order=order_a)
        package = _main_package(proj, nation, game, idx, test_sprite_techs)
        package.add(GeneratedFile(naming.priority_test_pack_file(), render(
            "sprite_pack.txt.j2", pack=naming.priority_test_pack(), trigger=f"tag = {nation.tag}",
            graphical_cultures=None, sprites=sprites).encode("utf-8")))
        package.add(GeneratedFile(naming.priority_test_order_file(), render(
            "pack_order.txt.j2", orders=[(naming.priority_test_pack(), order_b)]).encode("utf-8")))
        for case in cases:
            raw = game.read(case.history_file)
            if raw is None:
                raise GenerationError(f"{case.history_file} not found in the EU4 install")
            package.add(GeneratedFile(case.history_file, patch_history_culture(raw, culture), PATCHED, "vanilla"))
            if case.case == "b":
                package.add(GeneratedFile(naming.fallback_entities_file(case.tag), render(
                    "entities.asset.j2", entities=unit_entities(case.tag, base_chain, naming,
                                                                proj.entity_version)).encode("utf-8"), OWNED))
            if case.country_file:
                raw = game.read(case.country_file)
                if raw is None:
                    raise GenerationError(f"{case.country_file} not found in the EU4 install")
                package.add(GeneratedFile(case.country_file, patch_country_gfx(raw, nation.gfx_culture),
                                          PATCHED, "vanilla"))
        package.fixtures = TestFixtures(culture, cases, naming.set_sprite_pack(nation.set_name), naming.priority_test_pack(),
                                        rival_gfx, order_a, order_b)
        package.notes += [f"TEST ONLY: variant {variant}: {naming.set_sprite_pack(nation.set_name)} = {order_a}, "
                          f"{naming.priority_test_pack()} ({rival_gfx}) = {order_b}"] + notes
        variants.append(_wrap(proj, nation, package, game, idx, mod_dir, naming.test_variant_mod(variant)))
    return Suite(variants, variants[0].package.fixtures, main, nation, notes)


def suite_checklist(suite: Suite, sprite_techs: dict[int, int] | None) -> str:
    """What to play and what to expect, per check, for the person running the suite."""
    fx, nation, tag = suite.fixtures, suite.nation, suite.nation.tag
    v1, v2 = (tm.plan.manifest_rel.split("/", 1)[0] for tm in suite.variants)
    techs = (", ".join(f"tier {lvl} at MIL {t}" for lvl, t in sorted(sprite_techs.items())) if sprite_techs
             else "vanilla sprite techs")
    base, rival = nation.base_gfx, fx.rival_gfx
    default_look = nation.gfx_set_base or nation.base_gfx
    lines = [
        "IN-GAME TEST CHECKLIST",
        f"Disable every other eu4unitgen mod (including {TEST_MOD_NAME}). Launch 1 with only {v1}, launch 2 with only {v2}.",
        f"Sprite tiers: tier 1 at start, {techs}. Look at other tags in observe mode or by switching tag; "
        "raise MIL tech the way you did in test 1.",
        "Mark each line PASS/FAIL and note what you actually saw.",
        "",
        f"LAUNCH 1 ({v1}): {fx.main_pack} order {fx.main_order}, {fx.rival_pack} order {fx.rival_order}",
        f"[ ] P1  {tag}: armies show the {rival} look (rival pack, higher order).",
        f"        If they show {base}: the higher order does NOT win -> do the tier checks below in launch 1.",
    ]
    for case in fx.cases:
        if case.case == "a":
            lines.append(f"[ ] Fa  {case.tag} (was {case.vanilla_gfx}, culture now {fx.culture}; no pack, no {case.tag}_* "
                         f"entities): expect its own {case.vanilla_gfx} look (graphical-culture pack fallback).")
        elif case.case == "b":
            lines.append(f"[ ] Fb  {case.tag} (was {case.vanilla_gfx}; {case.tag}_* entities copied from {base}, no pack): "
                         f"{base} look = engine still falls back to TAG_* names; {case.vanilla_gfx} look = it doesn't. "
                         "Record which.")
        else:
            lines.append(f"[ ] Fc  {case.tag} (was {case.vanilla_gfx}, now graphical_culture {case.gfx}, only the "
                         f"{case.gfx} gfx pack): expect the {default_look} look (the {case.gfx} default set). "
                         f"{case.vanilla_gfx} or {base} = the gfx pack isn't used.")
    lines += [
        "        Fallback lines only need checking in one launch; they don't change between variants.",
        "",
        f"LAUNCH 2 ({v2}): {fx.main_pack} order {fx.rival_order}, {fx.rival_pack} order {fx.main_order}",
        f"[ ] P2  {tag}: armies show the {base} look (main pack, now the higher order).",
        f"[ ] T1  {tag} tier 1: infantry, cavalry (horse + rider) and artillery (cannon + crew) all {base}.",
        f"[ ] T2  {tag} tier 2: same, after the tier-2 MIL tech.",
        f"[ ] T3  {tag} tier 3: same; the horse changes to the tier-3 horse.",
        f"[ ] T4  {tag} tier 4: same.",
        f"        Make one army per type (or mostly one type): the map shows one model per army.",
        "[ ] E   error.log after each launch: no clone/entity/hr_ lines, nothing mentioning the fallback tags' files.",
        "",
        "Result meanings: P1+P2 pass = higher order wins. P1 fails, P2 passes = the main pack always wins; "
        "P1 passes, P2 fails = the rival always wins (so file or pack name decides, not the order). "
        "Anything else: note exactly what showed.",
    ]
    lines += [f"NOTE: {n}" for n in suite.notes]
    return "\n".join(lines)


# --------------------------------------------------------------------------- texture test (Phase 2)

TEXTURE_TEST_HUE = 120  # greenish: unmistakable against vanilla colours


@dataclass
class TextureTest:
    mod: TestMod
    tier: int
    meshes: dict[str, tuple[str, str]]  # role -> (our pdxmesh, vanilla base pdxmesh)
    textures: list[str]  # generated DDS paths
    nation: Nation
    what: str  # what the edit does, for the checklist


def build_texture_test(project: Project, nation: Nation, game: GameFiles, idx: Index, mod_dir: Path,
                       test_sprite_techs: dict[int, int] | None = None, edit=None, what: str = "") -> TextureTest:
    """A texture test mod (eu4unitgen_test_textures, one launch): tier 1's infantry (and rider),
    horse and cannon retextured through pdxmesh meshsettings; tiers 2-4 plain (control).

    ``edit`` (BaseTextures -> Edited) defaults to a whole-texture hue shift of the diffuse, the
    in-game test 4 setup (SPEC §10: passed; full-path texture names render black, so only bare
    names beside the vanilla mesh are used)."""
    from eu4unitgen.core.retexture import Edited, Retexture, apply_retextures, retextured_mesh
    from eu4unitgen.textures.recolour import hue_shift

    if edit is None:
        edit, what = (lambda base: Edited(diffuse=hue_shift(base.diffuse, TEXTURE_TEST_HUE)),
                      f"diffuse hue shifted {TEXTURE_TEST_HUE} degrees, whole texture")
    naming, set_name, tier = project.naming, nation.set_name, 1
    package = _main_package(project, nation, game, idx, test_sprite_techs)
    chain = BaseChain(idx, nation.base_gfx)
    bases = {"infantry": chain.slot("infantry", tier)["pdxmesh"], "horse": chain.slot("cavalry", tier)["horse_pdxmesh"],
             "cannon": chain.slot("artillery", tier)["cannon_pdxmesh"]}
    entities = unit_entities(nation.tag, chain, naming, project.entity_version)
    meshes, made, textures = [], {}, []
    for role, base in bases.items():
        mesh, files = retextured_mesh(naming, set_name, Retexture(role, tier, base, edit), game, idx)
        meshes.append(mesh)
        made[role] = (mesh["name"], base)
        for f in files:
            package.add(f)
            textures.append(f.path)
    apply_retextures(entities, nation.tag, tier, {r: m for r, (m, _) in made.items()}, chain, naming,
                     project.entity_version)
    # The retextured TAG set replaces the plain one; the gfx-culture defaults stay as generated.
    rel = naming.set_entities_file(set_name)
    package.files[rel] = GeneratedFile(rel, render("entities.asset.j2", entities=entities).encode("utf-8"), OWNED)
    package.add(GeneratedFile(naming.pdxmesh_file(set_name), render("pdxmeshes.gfx.j2", meshes=meshes).encode("utf-8"),
                              OWNED))
    package.notes.append(f"TEST ONLY: tier {tier} retextured through pdxmesh meshsettings ({what})")
    return TextureTest(_wrap(project, nation, package, game, idx, mod_dir, naming.texture_test_mod()), tier, made,
                       textures, nation, what)


def texture_checklist(test: TextureTest, sprite_techs: dict[int, int] | None) -> str:
    tag, base = test.nation.tag, test.nation.base_gfx
    name = test.mod.plan.manifest_rel.split("/", 1)[0]
    techs = (", ".join(f"tier {lvl} at MIL {t}" for lvl, t in sorted(sprite_techs.items())) if sprite_techs
             else "vanilla sprite techs")
    lines = [
        "IN-GAME TEXTURE TEST CHECKLIST (meshsettings retexture)",
        f"Enable only {name} (disable every other eu4unitgen mod). Play {tag}. Sprite tiers: tier 1 at start, {techs}.",
        "Make one army per type: infantry, cavalry, artillery. Mark each line PASS/FAIL and note what you saw.",
        "",
        f"TIER {test.tier}: {test.what}",
    ]
    for role, (mesh, vbase) in test.meshes.items():
        what = {"infantry": "infantry and the cavalry rider", "horse": "cavalry horse", "cannon": "artillery cannon"}[role]
        lines.append(f"[ ] {role:8} {what}: edited look = PASS; plain {base} look = override ignored; "
                     f"black/untextured = texture didn't resolve. (base {vbase})")
    lines += [
        "",
        f"[ ] country  tier {test.tier} infantry: the country-colour areas still show {tag}'s colour.",
        "[ ] control  tiers 2-4: plain western units (after the MIL techs above).",
        "[ ] vanilla  another western country's tier-1 units are unchanged (no leak onto the shared .mesh).",
        "[ ] E        error.log: note any line with hr_, meshsettings, .dds, texture or mesh.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- parts (Phase 3, in-game test 8)

STANCE_LOOK = {
    "spear_shield": "the shield arm holds the shield up in front, the spear arm thrusts (no arm through the shield, "
                    "nothing floating).",
    "polearm_onehanded": "a one-handed thrust with the right arm; the left arm is free.",
    "polearm_twohanded": "both hands on the shaft, levelled in the attack like the halberd.",
}
ROLE_WHERE = {"infantry": "infantry (and the gun crew, unless a crew line follows)",
              "rider": "cavalry rider", "crew": "artillery gun crew"}


def _describe_part(r) -> str:
    from eu4unitgen.core.parts import KINDS

    p, k = r.params, KINDS[r.kind]
    bits = [k.label.split(" (")[0].lower(), f"on {r.bone}"]
    if "point" in p:
        bits.append("point " + ("+Z (up/forward, like the halberd)" if p["point"] == "up" else "-Z (like the sabre)"))
    if r.kind == "crest":
        bits.append("front-to-back" if p.get("orientation") == "fore_aft" else "side-to-side (transverse)")
    names = {"country1": "primary", "country2": "secondary", "country3": "tertiary"}
    for zone, value in r.zones.items():
        zd = next(z for z in k.zones if z.name == zone)
        if isinstance(value, str):
            bits.append(f"{zd.label.lower()} in the {names[value]} unit colour")
        elif tuple(value) != zd.colour:
            bits.append(f"{zd.label.lower()} fixed rgb {' '.join(map(str, value))}")
    if r.blazon:
        bits.append(f"blazon {Path(r.blazon).name} (must read the right way round)")
    return ", ".join(bits)


def loadout_checklist(generated: GeneratedSet, mod_name: str, sprite_techs: dict[int, int] | None) -> str:
    """In-game test 8: what each tier's bodies should carry, from the loadout the set was built with."""
    res, nation = generated.loadout, generated.nation
    techs = (", ".join(f"tier {lvl} at MIL {t}" for lvl, t in sorted(sprite_techs.items())) if sprite_techs
             else "vanilla sprite techs")
    lines = [
        "IN-GAME TEST 8 CHECKLIST (procedural parts: the full set)",
        f"Enable only {mod_name} (disable every other eu4unitgen mod). Play {nation.tag}. Sprite tiers: tier 1 at "
        f"start, {techs}.",
        "At each tier make one army per type (infantry, cavalry, artillery); watch them stand, march and fight. "
        "PASS/FAIL + notes per line.",
    ]
    colours = (nation.unit_colours or {}).get(nation.tag) if isinstance(nation.unit_colours, dict) else None
    if colours:
        lines.append("Unit colours: " + ", ".join(f"{n} {' '.join(map(str, c))}"
                                                 for n, c in zip(("primary", "secondary", "tertiary"), colours)))
    by_entity = {r.entity: r for r in res.parts}
    for tier in sorted({tier for tier, _ in res.slots}):
        lines += ["", f"TIER {tier}"]
        for role in ("infantry", "crew", "rider"):
            if (tier, role) not in res.slots:
                if role == "crew" and (tier, "infantry") in res.slots:
                    lines.append(f"[ ] {tier}C  artillery gun crew: the infantry entity itself, so it carries the "
                                 "infantry's parts and stance")
                continue
            parts = [by_entity[e] for e in res.carried.get((tier, role), [])]
            lines.append(f"[ ] {tier}{role[0].upper()}  {ROLE_WHERE[role]}: "
                         + ("; ".join(_describe_part(r) for r in parts) or "no parts of its own: the plain base infantry"))
        for entity, (stance, _) in sorted(res.stances.items()):
            if entity.endswith((f"_INFANTRY_{tier}", f"_{tier}_crew")):
                lines.append(f"[ ] {tier}S  {entity}: {stance} animations; " + STANCE_LOOK.get(stance, ""))
    lines += [
        "",
        "GENERAL",
        "[ ] G1 every part follows its bone through idle, walk and attack (no floating, no lag).",
        "[ ] G2 hoplon face: the country colour (or the fixed colour) as listed; rim bronze; the inside is not see-through.",
        "[ ] G3 blazon (if any): upright and not mirrored, seen from the front of the unit. If not: note how it is turned.",
        "[ ] G4 sizes: a part far too big or small, or sunk into / floating off the body: note which.",
        "[ ] E  error.log: note any line with hr_, .mesh, entity, attach, texture or animation.",
    ]
    return "\n".join(lines)
