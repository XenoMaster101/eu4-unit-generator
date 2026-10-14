"""Generate / validate / export as a session: what the GUI (and the CLI) call.

A ``Session`` loads the EU4 install once (index, tag facts) and runs the same steps as
``cli generate`` and ``cli test-mod`` for a ``SetSpec``. Nothing here knows about Qt.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from eu4unitgen.core.generator import GeneratedSet, GenerationError, generate
from eu4unitgen.core.naming import TEST_MOD_NAME, NamingError, check_tag
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project, default_blender_path, default_mod_dir
from eu4unitgen.export.testmod import Suite, build_suite, build_test_mod, build_texture_test, suite_checklist, texture_checklist
from eu4unitgen.export.writer import ExportError, ExportPlan, apply_export, plan_package, stage_plan
from eu4unitgen.library.scanner import GameFiles, Index, build_index, find_pack
from eu4unitgen.library.tags import TagFacts, coverage, culture_groups, start_level, tag_facts
from eu4unitgen.validate import Report, validate


class SpecError(ValueError):
    """The unit-set form doesn't describe a valid set (bad tag, set name, ...)."""


@dataclass
class Settings:
    eu4_path: Path | None = None  # default: $EU4_PATH
    output_dir: Path = Path("output")
    owned_dlc: list[str] = field(default_factory=list)
    mod_folder: Path | None = None  # set: read it for collisions and shared-file copies (--check-mod)
    stage_dir: Path = Path("build") / "test_mod"
    mod_dir: Path | None = None  # EU4 user mod directory; default: the one under Documents
    prefix: str = "hr_"
    blender_path: Path | None = None  # default: the pinned Blender 4.2 LTS (core.project.default_blender_path)

    def __post_init__(self) -> None:
        if self.eu4_path is None and os.environ.get("EU4_PATH"):
            self.eu4_path = Path(os.environ["EU4_PATH"])
        if self.blender_path is None:
            self.blender_path = default_blender_path()
        for name in ("eu4_path", "output_dir", "mod_folder", "stage_dir", "mod_dir", "blender_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))

    def user_mod_dir(self) -> Path:
        return self.mod_dir or default_mod_dir()


@dataclass
class SetSpec:
    """The unit-set form. Several tags make one set per tag, each named after its tag."""
    tags: list[str]
    gfx: str
    set_name: str = ""
    base_gfx: str = "westerngfx"
    name: str = ""
    adjective: str = ""
    colour: tuple[int, int, int] | None = None
    tech_group: str = ""  # blank: the tag's 1444 tech group, else western
    gfx_set_base: str | None = None

    @staticmethod
    def parse_tags(text: str) -> list[str]:
        return [t for t in re.split(r"[\s,;]+", text.strip().upper()) if t]


@dataclass
class SetResult:
    nation: Nation
    generated: GeneratedSet
    report: Report
    plan: ExportPlan | None = None
    written: list[str] = field(default_factory=list)
    package_dir: Path | None = None
    integration_text: str = ""


@dataclass
class TestModResult:
    stage_dir: Path
    reports: list[tuple[str, Report]]  # (mod folder name, report)
    written: list[str]
    warnings: list[str]
    checklist: str = ""


STAGE_ENTRY = re.compile(rf"^{TEST_MOD_NAME}(_v\d+|_textures)?(\.mod)?$")


def clear_stage(stage_dir: Path) -> None:
    """Empty a stage folder, but only when everything in it is a staged eu4unitgen test mod
    (``eu4unitgen_test*`` folders and descriptors). Anything else: refuse and delete nothing."""
    import shutil

    stage_dir = Path(stage_dir)
    if not stage_dir.exists():
        return
    entries = list(stage_dir.iterdir())
    foreign = [e.name for e in entries if not STAGE_ENTRY.match(e.name)]
    if foreign:
        raise ExportError(f"{stage_dir} holds files that aren't staged test mods ({', '.join(sorted(foreign)[:5])}); "
                          "empty it by hand or pick another folder")
    for e in entries:
        if e.is_dir():
            shutil.rmtree(e)
        else:
            e.unlink()


class Session:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._game: GameFiles | None = None
        self._idx: Index | None = None
        self._facts: dict[str, TagFacts] | None = None
        self._loaded_from: Path | None = None

    # ------------------------------------------------------------------ game data
    def game(self) -> tuple[GameFiles, Index]:
        path = self.settings.eu4_path
        if path is None:
            raise SpecError("set the EU4 install folder (Settings) or EU4_PATH")
        if not (Path(path) / "common").is_dir():
            raise SpecError(f"{path} doesn't look like an EU4 install (no common/ folder)")
        if self._game is None or self._loaded_from != Path(path):
            self.close()
            self._game = GameFiles(Path(path))
            self._idx = build_index(self._game)
            self._loaded_from = Path(path)
        return self._game, self._idx

    def facts(self) -> dict[str, TagFacts]:
        game, _ = self.game()
        if self._facts is None:
            self._facts = tag_facts(game)
        return self._facts

    def available_dlc(self) -> list[str]:
        return sorted(self.game()[0].zips)

    def graphical_cultures(self) -> list[str]:
        return list(self.game()[1].graphical_cultures)

    def base_cultures(self) -> list[str]:
        """Vanilla cultures with a base-game sprite pack: the ones a set can copy."""
        idx = self.game()[1]
        return [g for g in idx.graphical_cultures if find_pack(idx, gfx=g) is not None]

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
        self._game = self._idx = self._facts = self._loaded_from = None

    # ------------------------------------------------------------------ building blocks
    def project(self, mod_root: Path | None = None, **kw) -> Project:
        s = self.settings
        check = mod_root is None and s.mod_folder is not None
        return Project(mod_root=mod_root or s.mod_folder or s.user_mod_dir() / "hellenistic_revolution",
                       eu4_path=s.eu4_path, prefix=s.prefix, owned_dlc=list(s.owned_dlc),
                       check_mod_collisions=check, **kw)

    def nations(self, spec: SetSpec) -> list[Nation]:
        if not spec.tags:
            raise SpecError("enter at least one tag")
        if len(spec.tags) > 1 and spec.set_name:
            raise SpecError("with several tags, leave the set name blank: each set is named after its tag")
        if len(set(spec.tags)) != len(spec.tags):
            raise SpecError("a tag is listed twice")
        out = []
        for tag in spec.tags:
            try:
                check_tag(tag)
                tech = spec.tech_group or self._tag_tech_group(tag) or "western"
                out.append(Nation(tag, spec.gfx, base_gfx=spec.base_gfx, name=spec.name, adjective=spec.adjective,
                                  colour=spec.colour, tech_group=tech, set_name=spec.set_name,
                                  gfx_set_base=spec.gfx_set_base or None))
            except (NamingError, ValueError) as exc:
                raise SpecError(str(exc)) from exc
        return out

    def _tag_tech_group(self, tag: str) -> str | None:
        f = self.facts().get(tag)
        return f.technology_group if f else None

    # ------------------------------------------------------------------ steps
    def validate(self, spec: SetSpec) -> list[SetResult]:
        """Generate in memory and validate; writes nothing."""
        game, idx = self.game()
        project = self.project()
        mod = project.mod_files()
        try:
            out = []
            for nation in self.nations(spec):
                generated = generate(project, nation, game, idx, mod)
                out.append(SetResult(nation, generated, validate(project, nation, generated, game, idx, mod)))
            return out
        finally:
            if mod is not None:
                mod.close()

    def generate(self, spec: SetSpec, write: bool = True) -> list[SetResult]:
        """Validate, then write each set's drop-in package to <output>/<set>/ (unless any set has
        validation errors or a blocked file: then nothing is written)."""
        game, _ = self.game()
        results = self.validate(spec)
        project = self.project()
        for r in results:
            r.plan = plan_package(self.settings.output_dir, r.generated, game, [str(i) for i in r.report.errors],
                                  mod_dirs=[self.settings.user_mod_dir(), project.mod_root])
            r.package_dir = self.settings.output_dir / r.nation.set_name
            integ = r.generated.files.get(project.naming.integration_file(r.nation.set_name))
            r.integration_text = integ.text if integ else ""
        if write and all(r.plan.can_apply for r in results):
            for r in results:
                r.written = apply_export(r.plan)
        return results

    def build_test_mod(self, spec: SetSpec, *, suite: bool = False, sprite_techs: dict[int, int] | None = None,
                       rival_gfx: str = "asiangfx", clear: bool = False, textures: bool = False) -> TestModResult:
        """Stage the test mod (or the two-variant suite, or the texture test) in settings.stage_dir,
        for copying into the EU4 mod folder by hand. One tag only."""
        if suite and textures:
            raise SpecError("pick either the test suite or the texture test, not both")
        if len(spec.tags) != 1:
            raise SpecError("the test mod takes exactly one tag")
        game, idx = self.game()
        facts = self.facts()
        tag = spec.tags[0]
        f = facts.get(tag)
        if f is None:
            raise SpecError(f"{tag} owns no land at 1444.11.11; pick another tag (pick-test-tag)")
        warnings = []
        cov = coverage(idx, f)
        if not cov.clean:
            warnings.append(f"{tag} is matched by {[p for p, _, _ in cov.packs]} / has {cov.entities[:4]}; "
                            "the map may not show the generated set")
        if sprite_techs:
            start = start_level(game, f.technology_group)
            if start is not None and min(sprite_techs.values()) <= start:
                warnings.append(f"{tag} starts at MIL {start}; tiers at MIL <= {start} show from day one")
        nation = self.nations(SetSpec(**{**spec.__dict__, "tech_group": spec.tech_group or f.technology_group or ""}))[0]
        mod_dir = self.settings.user_mod_dir()
        project = self.project(mod_root=mod_dir / TEST_MOD_NAME)
        checklist = ""
        built: Suite | None = None
        try:
            if textures:
                tt = build_texture_test(project, nation, game, idx, mod_dir, test_sprite_techs=sprite_techs)
                mods, checklist = [tt.mod], texture_checklist(tt, sprite_techs)
            elif suite:
                built = build_suite(project, nation, game, idx, mod_dir, facts, culture_groups(game),
                                    rival_gfx=rival_gfx, test_sprite_techs=sprite_techs)
                mods = built.variants
            else:
                mods = [build_test_mod(project, nation, game, idx, mod_dir, test_sprite_techs=sprite_techs)]
        except GenerationError as exc:
            raise SpecError(str(exc)) from exc
        reports = [(tm.plan.manifest_rel.split("/", 1)[0], tm.report) for tm in mods]
        stage = self.settings.stage_dir
        written: list[str] = []
        if all(tm.plan.can_apply for tm in mods):
            if clear:
                clear_stage(stage)
            targets = [stage_plan(tm.plan, stage, require_empty=i == 0) for i, tm in enumerate(mods)]
            for t in targets:
                written += apply_export(t)
        if built is not None:
            checklist = suite_checklist(built, sprite_techs)
        return TestModResult(stage, reports, written, warnings, checklist)
