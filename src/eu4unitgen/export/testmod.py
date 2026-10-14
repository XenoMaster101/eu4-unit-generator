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

INTEGRATION.md changes are applied here only, never to a real mod. The test mod reads
nothing from any other mod. Export rules are the same as for a real mod: dry run first,
and files the tool didn't write are never overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eu4unitgen import __version__
from eu4unitgen.core.generator import PATCHED, GeneratedFile, GeneratedSet, GenerationError, generate, patch_country
from eu4unitgen.core.naming import MANIFEST_FILE, TEST_MOD_NAME
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.export.writer import ExportPlan, plan_export
from eu4unitgen.library.scanner import GameFiles, Index
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


def build_test_mod(project: Project, nation: Nation, game: GameFiles, idx: Index, mod_dir: Path,
                   name: str = TEST_MOD_NAME, test_sprite_techs: dict[int, int] | None = None) -> TestMod:
    """``test_sprite_techs`` (e.g. {2: 2, 3: 3, 4: 4}) moves sprite levels 2-4 to early MIL techs in
    this test mod's mil.txt only, so every tier can be seen in minutes."""
    package = generate(project, nation, game, idx, mod=None, test_sprite_techs=test_sprite_techs)
    integ = package.integration
    if not integ or not integ.country_file:
        raise GenerationError(f"{nation.tag} has no vanilla country file; the test mod needs a vanilla tag")
    original = game.read(integ.country_file)
    if original is None:
        raise GenerationError(f"{integ.country_file} not found in the EU4 install")
    package.add(GeneratedFile(integ.country_file, patch_country(original, integ), PATCHED, "vanilla"))

    report = validate(project, nation, package, game, idx, mod=None)
    folder = Path(mod_dir) / name
    supported = supported_version(idx)
    outer = descriptor_text(name, supported, folder.as_posix())
    inner = descriptor_text(name, supported, None)
    report.issues += [i for i in check_descriptor(outer, folder.as_posix())]
    report.issues += [i for i in check_descriptor(inner)]

    wrapped = GeneratedSet(nation, notes=list(package.notes), integration=integ)
    for f in package.files.values():
        wrapped.add(GeneratedFile(f"{name}/{f.path}", f.data, f.role, f.based_on, game_path=f.game_path or f.path))
    wrapped.add(GeneratedFile(f"{name}.mod", outer.encode("utf-8"), DESCRIPTOR))
    wrapped.add(GeneratedFile(f"{name}/descriptor.mod", inner.encode("utf-8"), DESCRIPTOR))
    plan = plan_export(mod_dir, wrapped, game, [str(i) for i in report.issues if i.severity == ERROR],
                       manifest_rel=f"{name}/{MANIFEST_FILE}")
    return TestMod(package, report, plan)
