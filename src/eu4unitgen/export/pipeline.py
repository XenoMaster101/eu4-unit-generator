"""Generate / validate / export as a session: what the GUI (and the CLI) call.

A ``Session`` loads the EU4 install once (index, tag facts) and runs the same steps as
``cli generate`` and ``cli test-mod`` for a ``SetSpec``. Nothing here knows about Qt.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eu4unitgen.core.generator import GeneratedSet, GenerationError, generate
from eu4unitgen.core.naming import TEST_MOD_NAME, TIERS, UNIT_TYPES, NamingError, check_tag
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import (
    DEFAULT_ENTITY_VERSION, DEFAULT_TAG_PACK_ORDER, Project, default_blender_path, default_mod_dir,
)
from eu4unitgen.export.testmod import (
    Suite, build_base_test_mod, build_suite, build_test_mod, build_texture_test, loadout_checklist, suite_checklist,
    texture_checklist,
)
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
    entity_version: int = DEFAULT_ENTITY_VERSION
    tag_pack_order: int = DEFAULT_TAG_PACK_ORDER

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
    """The unit-set form. Several tags share one set (one sprite pack, OR over the tags); the set
    name is then required. Unit colours: one (color1, color2, color3) for every tag, or {tag: ...}."""
    tags: list[str]
    gfx: str
    set_name: str = ""
    base_gfx: str = "westerngfx"
    name: str = ""
    adjective: str = ""
    colour: tuple[int, int, int] | None = None
    tech_group: str = ""  # blank: the tag's 1444 tech group, else western
    gfx_set_base: str | None = None
    unit_colours: Any = None  # common/country_colors color1-3: one triple for all tags or {tag: triple}
    texture_recipe: Any = None  # Phase 2: a core.recipe JSON file (Path) or a TextureRecipe
    loadout: Any = None  # Phase 3: a core.loadout JSON file (Path) or a Loadout
    cultures: list[str] = field(default_factory=list)  # primary cultures the set's pack also matches
    unit_names: dict = field(default_factory=dict)  # unit type -> in-game name (F7)

    @staticmethod
    def parse_tags(text: str) -> list[str]:
        return [t for t in re.split(r"[\s,;]+", text.strip().upper()) if t]

    @staticmethod
    def parse_cultures(text: str) -> list[str]:
        return [c for c in re.split(r"[\s,;]+", text.strip().lower()) if c]


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
    stage_dir: Path | None  # where it was written: the stage folder, the EU4 mod folder (apply), None (dry run)
    reports: list[tuple[str, Report]]  # (mod folder name, report)
    written: list[str]
    warnings: list[str]
    checklist: str = ""
    plans: list[ExportPlan] = field(default_factory=list)  # one per mod (the suite has two)
    info: str = ""  # the tag's 1444 facts, for the CLI


@dataclass
class PartsPreview:
    image: Any  # PIL: front and side views of the body with its parts, then each part alone
    path: Path  # where it was saved (<output>/part_previews/<set>/<tier>_<role>.png)
    body: str  # the body's pdxmesh
    parts: list[dict]  # {entity, kind, bone, params, fitted} per part
    notes: list[str]


@dataclass
class TexturePreview:
    base: str  # the base pdxmesh
    shader: str
    guide: Any  # PIL images: regions over the base diffuse, then base and edited as the shader tints them
    before: Any
    after: Any
    notes: list[str]


@dataclass
class Gallery:
    """The Result screen's 3D previews: (unit type, tier) -> image, with the validation they came
    from (its report is what the bottom bar shows) and per-tile render seconds."""
    images: dict
    result: Any  # SetResult
    seconds: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    thumbs: dict = field(default_factory=dict)  # part entity -> its picture (the Parts cards)
    parts: dict = field(default_factory=dict)  # (unit type, tier) -> {bone: part entity}


@dataclass
class BodyAreas:
    base: str  # the base pdxmesh
    mesh_file: str
    shader: str
    diffuse: Any  # PIL RGB: the base body texture (base game: shown, never saved)
    areas: dict  # area name -> 'L' mask at the diffuse's size
    skin: tuple  # the body's own skin tone (the skin region's mean), for "bare skin" areas
    specular: Any = None  # PIL: the base spec map (the live map's country colours and recipe spec ops)
    colours: list = field(default_factory=list)  # the unit colours previews tint with
    role: str = "infantry"


def finished_texture(body: BodyAreas, recipe, slot) -> Any:
    """The body texture with ``slot`` applied, tinted as PdxMeshColor tints it with the body's
    preview colours: the Character page's live map. Reads nothing from the game; the normal map
    isn't shown, so ``flatten`` works on a flat stand-in."""
    from PIL import Image

    from eu4unitgen.core.recipe import apply_slot
    from eu4unitgen.textures.recolour import shade

    res = apply_slot(recipe, slot, body.diffuse, body.specular, body.shader, body.role, body.areas,
                     lambda: Image.new("RGBA", body.diffuse.size, (128, 128, 255, 255)))
    return shade(res.diffuse or body.diffuse, res.specular or body.specular, body.colours or DEFAULT_UNIT_COLOURS,
                 body.shader)


# Unit colours for tags without a common/country_colors entry: the file's example template.
# UNVERIFIED engine default; in-game test 5 (HDA, no entry) showed red units.
DEFAULT_UNIT_COLOURS = [(205, 32, 26), (255, 255, 255), (70, 50, 40)]


STAGE_ENTRY = re.compile(rf"^{TEST_MOD_NAME}(_[a-z0-9_]+)?(\.mod)?$")  # incl. base-mod copies


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


def load_recipe(spec: "SetSpec"):
    """The spec's texture recipe: a TextureRecipe as it is (the GUI's), or loaded from a file."""
    from eu4unitgen.core.recipe import RecipeError, TextureRecipe

    if spec.texture_recipe is None or isinstance(spec.texture_recipe, TextureRecipe):
        return spec.texture_recipe
    try:
        return TextureRecipe.load(spec.texture_recipe)
    except RecipeError as exc:
        raise SpecError(str(exc)) from exc


class _Checked:
    """A Blender runner whose failures surface as SpecError (shown on the form, not as a traceback)."""

    def __init__(self, runner):
        self.inner = runner

    def run_job(self, *a, **kw):
        from eu4unitgen.blender.runner import BlenderError

        try:
            return self.inner.run_job(*a, **kw)
        except BlenderError as exc:
            raise SpecError(str(exc)) from exc


def horse_unit_colours(spec):
    """The set's unit colours, with the tags it leaves out given the first fresh horse's (by tier) whose cloth
    is the country colour's mask (loadout horses ``unit_colours``, SPEC M21): Byzantium has no vanilla
    ``country_colors`` entry, so without it the cataphract's caparison would take BYZ's map purple."""
    uc = spec.unit_colours
    lo = load_loadout(spec)
    suggested = next((h["unit_colours"] for _, h in sorted((lo.horses if lo else {}).items()) if h.get("unit_colours")), None)
    if not suggested or not spec.tags:
        return uc
    if uc is None:
        return {t: [tuple(c) for c in suggested] for t in spec.tags}
    if isinstance(uc, dict):
        return {**{t: [tuple(c) for c in suggested] for t in spec.tags}, **uc}
    return uc  # one triple for every tag: the user's choice


def load_loadout(spec: "SetSpec"):
    """The spec's parts loadout: a Loadout as it is (the GUI's), or loaded from a file."""
    from eu4unitgen.core.loadout import Loadout, LoadoutError

    if spec.loadout is None or isinstance(spec.loadout, Loadout):
        return spec.loadout
    try:
        return Loadout.load(spec.loadout)
    except LoadoutError as exc:
        raise SpecError(str(exc)) from exc


class Session:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._runner = None
        self._runner_for: Path | None = None
        self.part_cache: dict[str, bytes] = {}  # built part meshes, by job entry (core.parts.build_parts)
        self._base_facts: dict[tuple, dict[str, TagFacts]] = {}  # (mod folder, tags) -> facts through the mod
        self._game: GameFiles | None = None
        self._idx: Index | None = None
        self._facts: dict[str, TagFacts] | None = None
        self._loaded_from: Path | None = None
        self._unit_colours: dict | None = None

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

    def mod_facts(self, mod_folder: Path, tags: list[str]) -> dict[str, TagFacts]:
        """Fix A: tag facts with ``mod_folder`` read over the game (its country_tags, countries and
        history win), including ``tags`` that own no land at the start (a mod's releasable tag)."""
        from eu4unitgen.library.scanner import LayeredFiles

        key = (str(Path(mod_folder).resolve()), tuple(sorted(tags)))
        if key not in self._base_facts:
            game, _ = self.game()
            mod = GameFiles(Path(mod_folder), include_dlc=False)
            try:
                self._base_facts[key] = tag_facts(LayeredFiles(mod, game), tags=list(tags))
            finally:
                mod.close()
        return self._base_facts[key]

    def country_list(self) -> list[tuple[str, str, bool]]:
        """(tag, name, mod-only) for the nation picker, by name: the game's tags and the mod
        folder's (Settings), with the mod's localisation winning (F6)."""
        from eu4unitgen.library.scanner import GameFiles
        from eu4unitgen.library.tags import country_names

        game, _ = self.game()
        folder = self.settings.mod_folder
        mod = GameFiles(Path(folder), include_dlc=False) if folder and Path(folder).is_dir() else None
        try:
            names = country_names(game, mod)
        finally:
            if mod is not None:
                mod.close()
        return sorted(((t, n, o) for t, (n, o) in names.items()), key=lambda x: (x[1].lower(), x[0]))

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
        self._unit_colours = None
        self._base_facts = {}
        self._bodies = None

    # ------------------------------------------------------------------ building blocks
    def project(self, mod_root: Path | None = None, **kw) -> Project:
        s = self.settings
        check = mod_root is None and s.mod_folder is not None
        return Project(mod_root=mod_root or s.mod_folder or s.user_mod_dir() / "hellenistic_revolution",
                       eu4_path=s.eu4_path, prefix=s.prefix, owned_dlc=list(s.owned_dlc),
                       entity_version=s.entity_version, tag_pack_order=s.tag_pack_order,
                       check_mod_collisions=check, **kw)

    def nations(self, spec: SetSpec, base_mod: Path | None = None) -> list[Nation]:
        """The set the form describes (a list of one: callers loop over results)."""
        if not spec.tags and not spec.cultures:
            raise SpecError("enter at least one tag or culture")
        if len(set(spec.tags)) != len(spec.tags):
            raise SpecError("a tag is listed twice")
        if len(spec.tags) > 1 and not spec.set_name:
            raise SpecError("several tags share one set: give the set a name")
        if not spec.tags and not spec.set_name:
            raise SpecError("a set triggered by culture only needs a set name")
        try:
            for tag in spec.tags:
                check_tag(tag)
            tech = spec.tech_group
            if not tech:
                groups = {t: self._tag_tech_group(t, spec.tags, base_mod) for t in spec.tags}
                known = sorted({g for g in groups.values() if g})
                if len(known) > 1:
                    raise SpecError("the tags start in different tech groups ("
                                    + ", ".join(f"{t} {g}" for t, g in groups.items() if g)
                                    + "); set the tech group for the shared units")
                tech = known[0] if known else "western"
            return [Nation(list(spec.tags), spec.gfx, base_gfx=spec.base_gfx, name=spec.name,
                           adjective=spec.adjective, colour=spec.colour, tech_group=tech, set_name=spec.set_name,
                           gfx_set_base=spec.gfx_set_base or None, unit_colours=horse_unit_colours(spec),
                           cultures=list(spec.cultures), unit_names=dict(spec.unit_names or {}))]
        except (NamingError, ValueError) as exc:
            if isinstance(exc, SpecError):
                raise
            raise SpecError(str(exc)) from exc

    def _tag_tech_group(self, tag: str, tags: list[str], base_mod: Path | None = None) -> str | None:
        """From the tag's history: through the mod when one is read (``base_mod``, or mod checks), so a
        mod's own or landless tag gets its group (fix A); else vanilla's 1444 facts."""
        folder = base_mod or self.settings.mod_folder
        facts = self.mod_facts(folder, tags) if folder else self.facts()
        f = facts.get(tag)
        return f.technology_group if f else None

    # ------------------------------------------------------------------ steps
    def _layered(self, mod_folder: Path | None):
        """The game, or a mod folder read over it."""
        from eu4unitgen.library.scanner import LayeredFiles

        game, _ = self.game()
        return LayeredFiles(GameFiles(Path(mod_folder), include_dlc=False), game) if mod_folder else game

    def runner(self):
        """The Blender runner (pinned version checked on first use); SpecError when Blender is unusable."""
        from eu4unitgen.blender.runner import BlenderError, BlenderRunner

        path = self.settings.blender_path
        if self._runner is None or self._runner_for != path:
            try:
                self._runner = BlenderRunner(path, log_dir=self.settings.output_dir / "blender_logs")
            except BlenderError as exc:
                raise SpecError(str(exc)) from exc
            self._runner_for = path
        return _Checked(self._runner)

    def _parts_args(self, spec: SetSpec) -> dict:
        """generate()'s loadout arguments; Blender only when the loadout changes something."""
        loadout = load_loadout(spec)
        if loadout is None or not loadout.changes_anything():
            return {}
        return {"loadout": loadout, "runner": self.runner() if loadout.needs_blender() else None,
                "part_cache": self.part_cache}

    def validate(self, spec: SetSpec) -> list[SetResult]:
        """Generate in memory and validate; writes nothing (parts are built in a temporary folder)."""
        game, idx = self.game()
        project = self.project()
        mod = project.mod_files()
        recipe = load_recipe(spec)
        parts = self._parts_args(spec)
        try:
            out = []
            for nation in self.nations(spec):
                try:
                    generated = generate(project, nation, game, idx, mod, recipe=recipe, **parts)
                except GenerationError as exc:
                    raise SpecError(str(exc)) from exc
                out.append(SetResult(nation, generated, validate(project, nation, generated, game, idx, mod)))
            return out
        finally:
            if mod is not None:
                mod.close()

    def gallery_prints(self, spec: SetSpec, result: SetResult) -> dict[tuple[str, int], str]:
        """(unit type, tier) -> export.gallery.fingerprint of the unit as ``gallery`` would render it
        (a unit that can't be assembled gets its error, so it's rendered to show the note)."""
        from eu4unitgen.export.gallery import GalleryError, entity_tree, fingerprint
        from eu4unitgen.validate.graph import World

        game, idx = self.game()
        project = self.project()
        mod = project.mod_files()
        colours, _ = self._preview_colours(spec)
        owner = project.naming.entity_owner(result.nation.tags, result.nation.set_name)
        out = {}
        try:
            world = World(game, idx, mod, result.generated, project.owned_dlc)
            for tier in TIERS:
                for utype in UNIT_TYPES:
                    try:
                        tree = entity_tree(world, project.naming.unit_entity(owner, utype, tier))
                        out[(utype, tier)] = fingerprint(tree, world, colours)
                    except GalleryError as exc:
                        out[(utype, tier)] = f"error: {exc}"
        finally:
            if mod is not None:
                mod.close()
        return out

    def gallery(self, spec: SetSpec, units: list[tuple[str, int]] | None = None, size: int = 256,
                result: SetResult | None = None, thumbs: int = 0, camera: dict | None = None,
                lit: dict | None = None, dump_dir: Path | None = None) -> "Gallery":
        """Whole units, posed at their idle frame, as the game assembles them (Track F, F9): the
        generated set (``result``: a validation already run, else one now) over the mod and the
        game, each unit's entity tree staged and rendered in one background Blender run.
        ``units``: (unit type, tier), default all 12. ``camera``: one fixed framing for every tile
        (render_unit's ``camera``, e.g. gallery.GAME_CLOSE) instead of each unit fitted to its tile.
        ``lit``: render_unit's lit mode (EEVEE with the normal maps on or off) instead of Workbench.
        ``dump_dir``: each unit's posed meshes (render_unit's ``dump``) copied there as ``<type>_<tier>_posed.json``
        (the rider's clearance over its horse is measured from them). Writes nothing else outside a temporary
        folder."""
        import tempfile

        from PIL import Image

        from eu4unitgen.blender.runner import BlenderError
        from eu4unitgen.export.gallery import GalleryError, entity_tree, parts_of, stage
        from eu4unitgen.validate.graph import World

        result = result or self.validate(spec)[0]
        game, idx = self.game()
        project = self.project()
        mod = project.mod_files()
        colours, notes = self._preview_colours(spec)
        naming = project.naming
        owner = naming.entity_owner(result.nation.tags, result.nation.set_name)
        wanted = units or [(u, t) for t in TIERS for u in UNIT_TYPES]
        images: dict[tuple[str, int], Any] = {}
        seconds: dict[tuple[str, int], float] = {}
        pictures: dict[str, Any] = {}
        parts: dict[tuple[str, int], dict] = {}
        try:
            world = World(game, idx, mod, result.generated, project.owned_dlc)
            with tempfile.TemporaryDirectory(prefix="eu4unitgen_gallery_") as tmp:
                folder = Path(tmp)
                jobs = []
                for utype, tier in wanted:
                    name = f"{utype}_{tier}"
                    try:
                        tree = entity_tree(world, naming.unit_entity(owner, utype, tier))
                        parts[(utype, tier)] = parts_of(tree)
                        jobs.append({"name": name, "root": stage(tree, world, colours, folder / name)})
                    except GalleryError as exc:
                        notes.append(f"{utype} tier {tier}: {exc}")
                if jobs:
                    try:
                        res = self.runner().run_job("render_unit", {"units": jobs, "frame": 1, "size": size,
                                                                    "thumbs": thumbs, "out_dir": tmp, "camera": camera,
                                                                    "lit": lit, "dump": bool(dump_dir)},
                                                 timeout=900)
                        if dump_dir:
                            import shutil
                            Path(dump_dir).mkdir(parents=True, exist_ok=True)
                            for unit, path in (res.result.get("dumps") or {}).items():
                                shutil.copyfile(path, Path(dump_dir) / f"{unit}_posed.json")
                    except BlenderError as exc:
                        raise SpecError(f"the gallery needs Blender: {exc}") from exc
                    for utype, tier in wanted:
                        path = res.result["tiles"].get(f"{utype}_{tier}")
                        if path:
                            with Image.open(path) as im:
                                images[(utype, tier)] = im.convert("RGB")
                            seconds[(utype, tier)] = res.result["seconds"].get(f"{utype}_{tier}", 0.0)
                    for entity, path in (res.result.get("thumbs") or {}).items():
                        with Image.open(path) as im:
                            pictures[entity] = im.convert("RGB")
        finally:
            if mod is not None:
                mod.close()
        return Gallery(images, result, seconds, notes, pictures, parts)

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
                       rival_gfx: str = "asiangfx", clear: bool = False, textures: bool = False,
                       write: str = "stage", fallback_tags: dict[str, str] | None = None,
                       base_mod: Path | None = None, give_cores: bool = False,
                       base_name: str | None = None) -> TestModResult:
        """Build the test mod (or the two-variant suite, or the texture test). ``write``: "stage"
        (settings.stage_dir, for copying into the EU4 mod folder by hand), "apply" (straight into
        the EU4 mod folder) or "dry" (nothing written; plans returned). A shared set may have several
        tags; the suite and the texture test take one. The plain test mod carries the spec's
        texture recipe and parts loadout; with a loadout its checklist is in-game test 8's.

        ``base_mod`` (fix B): the test mod is a copy of that mod with the set integrated (enable it
        instead of the base mod); ``give_cores`` (fix C, TEST ONLY) makes each tag own its cored
        provinces and capital at the start; ``base_name`` overrides the launcher name."""
        if write not in ("stage", "apply", "dry"):
            raise SpecError(f"write must be stage, apply or dry, not {write!r}")
        if suite and textures:
            raise SpecError("pick either the test suite or the texture test, not both")
        if (suite or textures) and len(spec.tags) != 1:
            raise SpecError("the test suite and the texture test take exactly one tag")
        if base_mod is not None and (suite or textures):
            raise SpecError("a base-mod test mod is the plain test mod; untick the suite and the texture test")
        if give_cores and base_mod is None:
            raise SpecError("giving a tag its cores is for a test mod on a base mod")
        if not spec.tags:
            raise SpecError("enter at least one tag")
        game, idx = self.game()
        facts = self.mod_facts(base_mod, spec.tags) if base_mod else self.facts()
        warnings = []
        for t in spec.tags:
            ft = facts.get(t)
            if ft is None:
                raise SpecError(f"{t} owns no land at 1444.11.11; pick another tag (pick-test-tag)")
            if base_mod and ft.provinces == 0 and not give_cores:
                warnings.append(f"{t} owns no land at the start: release it in game, or tick 'give the tag its cores'")
            cov = coverage(idx, ft)
            if not cov.clean:
                warnings.append(f"{t} is matched by {[p for p, _, _ in cov.packs]} / has {cov.entities[:4]}; "
                                "the map may not show the generated set")
            if sprite_techs:
                start = start_level(self._layered(base_mod), ft.technology_group)
                if start is not None and min(sprite_techs.values()) <= start:
                    warnings.append(f"{t} starts at MIL {start}; tiers at MIL <= {start} show from day one")
        tag, f = spec.tags[0], facts[spec.tags[0]]
        nation = self.nations(spec, base_mod)[0]
        mod_dir = self.settings.user_mod_dir()
        project = self.project(mod_root=mod_dir / TEST_MOD_NAME)
        checklist = ""
        built: Suite | None = None
        try:
            if textures:
                tt = build_texture_test(project, nation, game, idx, mod_dir, test_sprite_techs=sprite_techs)
                mods, checklist = [tt.mod], texture_checklist(tt, sprite_techs)
            elif base_mod is not None:
                tm = build_base_test_mod(project, nation, game, idx, mod_dir, Path(base_mod),
                                         test_sprite_techs=sprite_techs, recipe=load_recipe(spec),
                                         give_cores=give_cores, name=base_name, **self._parts_args(spec))
                mods = [tm]
                if tm.package.loadout is not None:
                    checklist = loadout_checklist(tm.package, tm.plan.manifest_rel.split("/", 1)[0], sprite_techs,
                                                  load_recipe(spec))
            elif suite:
                built = build_suite(project, nation, game, idx, mod_dir, facts, culture_groups(game),
                                    rival_gfx=rival_gfx, test_sprite_techs=sprite_techs, fallback_tags=fallback_tags)
                mods = built.variants
            else:
                tm = build_test_mod(project, nation, game, idx, mod_dir, test_sprite_techs=sprite_techs,
                                    recipe=load_recipe(spec), **self._parts_args(spec))
                mods = [tm]
                if tm.package.loadout is not None:
                    checklist = loadout_checklist(tm.package, TEST_MOD_NAME, sprite_techs, load_recipe(spec))
        except GenerationError as exc:
            raise SpecError(str(exc)) from exc
        reports = [(tm.plan.manifest_rel.split("/", 1)[0], tm.report) for tm in mods]
        stage: Path | None = None
        written: list[str] = []
        if write != "dry" and all(tm.plan.can_apply for tm in mods):
            if write == "stage":
                stage = self.settings.stage_dir
                if clear:
                    clear_stage(stage)
                targets = [stage_plan(tm.plan, stage, require_empty=i == 0) for i, tm in enumerate(mods)]
            else:
                stage, targets = mod_dir, [tm.plan for tm in mods]
            for t in targets:
                written += apply_export(t)
        if built is not None:
            checklist = suite_checklist(built, sprite_techs)
        info = "test mod for " + "; ".join(
            f"{t} ({facts[t].primary_culture}, {facts[t].technology_group}, was {facts[t].graphical_culture})"
            for t in spec.tags)
        return TestModResult(stage, reports, written, warnings, checklist, [tm.plan for tm in mods], info)

    # ------------------------------------------------------------------ textures (Phase 2)
    def _slot_base(self, spec: SetSpec, tier: int, role: str, base: str | None):
        from eu4unitgen.core.generator import BaseChain
        from eu4unitgen.core.retexture import base_textures, default_base

        game, idx = self.game()
        loadout = load_loadout(spec)
        if base is None and role in ("infantry", "rider") and loadout is not None:
            base = loadout.bodies.get(tier)
        try:
            name = base or default_base(BaseChain(idx, spec.base_gfx), role, tier)
            return base_textures(name, game, idx)
        except GenerationError as exc:
            raise SpecError(str(exc)) from exc

    def texture_preview(self, spec: SetSpec, tier: int, role: str, recipe, slot) -> TexturePreview:
        """One slot's edit, before/after, tinted the way PdxMeshColor tints it with the tag's unit
        colours (common/country_colors; not the map colour). Writes nothing."""
        from eu4unitgen.core.recipe import RecipeError, apply_slot
        from eu4unitgen.textures.recolour import shade
        from eu4unitgen.textures.regions import guide

        b = self._slot_base(spec, tier, role, slot.base)
        try:
            res = apply_slot(recipe, slot, b.diffuse, b.specular, b.shape["shader"], role,
                             lambda: self._areas(b), lambda: b.image("normal"))
        except RecipeError as exc:
            raise SpecError(str(exc)) from exc
        colours, notes = self._preview_colours(spec)
        notes = list(res.notes) + notes
        shader = b.shape["shader"]
        return TexturePreview(b.pdxmesh, shader, guide(b.diffuse, res.regions),
                              shade(b.diffuse, b.specular, colours, shader),
                              shade(res.diffuse or b.diffuse, res.specular or b.specular, colours, shader), notes)

    @staticmethod
    def _areas(b) -> dict:
        from eu4unitgen.textures.areas import area_masks

        return area_masks(b.game.read(b.mesh_file), b.diffuse.size, b.mesh_file)

    def body_areas(self, spec: SetSpec, tier: int, role: str = "infantry", base: str | None = None) -> BodyAreas:
        """The slot's base body texture and its named texture areas (the Character page's map)."""
        from eu4unitgen.textures.areas import skin_colour
        from eu4unitgen.textures.regions import auto_regions

        b = self._slot_base(spec, tier, role, base)
        regions = auto_regions(b.diffuse, b.specular, b.shape["shader"] == "PdxMeshColor", hide=role == "horse")
        areas = self._areas(b)
        colours, _ = self._preview_colours(spec)
        return BodyAreas(b.pdxmesh, b.mesh_file, b.shape["shader"], b.diffuse.convert("RGB"), areas,
                         skin_colour(b.diffuse, regions["skin"], [areas[a] for a in ("head", "hands") if a in areas]),
                         b.specular, colours, role)

    def _preview_colours(self, spec: SetSpec) -> tuple[list, list[str]]:
        """The colours previews tint country regions with: the set's unit colours, else the tag's
        common/country_colors entry, else the file's template (with a note)."""
        tag = spec.tags[0] if spec.tags else ""
        own = spec.unit_colours.get(tag) if isinstance(spec.unit_colours, dict) else spec.unit_colours
        colours = [tuple(c) for c in own] if own else self.unit_colours().get(tag)
        if colours is None:
            return DEFAULT_UNIT_COLOURS, [f"{tag or 'the tag'} has no common/country_colors entry: previewed with the "
                                          "default unit colours (red/white/brown)"]
        return colours, []

    def unit_colours(self) -> dict:
        """tag -> [color1, color2, color3] from common/country_colors (what units are tinted with)."""
        from eu4unitgen.library.tags import unit_colours

        game, _ = self.game()  # reloads (and clears this cache) when the install path changed
        if self._unit_colours is None:
            self._unit_colours = unit_colours(game)
        return self._unit_colours

    def export_texture_guide(self, spec: SetSpec, tier: int, role: str, base: str | None = None) -> list[Path]:
        """Write PNGs to paint masks over (the base diffuse, its region map, its spec map) to
        <output>/texture_guides/<set>/. Derived from base-game files: user output only, never the repo."""
        from eu4unitgen.core.naming import Naming
        from eu4unitgen.textures.regions import auto_regions, guide

        b = self._slot_base(spec, tier, role, base)
        naming = Naming(self.settings.prefix)
        name = spec.set_name or (spec.tags[0].lower() if spec.tags else "")
        try:
            folder = self.settings.output_dir / naming.texture_guide_dir(name)
        except NamingError as exc:
            raise SpecError(str(exc)) from exc
        folder.mkdir(parents=True, exist_ok=True)
        regions = auto_regions(b.diffuse, b.specular, b.uses_country_colour, hide=role == "horse")
        out = []
        for kind, img in (("diffuse", b.diffuse.convert("RGB")), ("regions", guide(b.diffuse, regions)),
                          ("spec", b.specular.convert("RGBA"))):
            path = folder / naming.texture_guide(tier, role, kind)
            img.save(path)
            out.append(path)
        return out

    # ------------------------------------------------------------------ parts (Phase 3)
    def _body(self, spec: SetSpec, tier: int, loadout=None, role: str = "infantry", build: bool = False):
        """The body a tier's parts are fitted on: the loadout's body for that tier, else the base
        culture's infantry pdxmesh (riders and crews use the same .mesh). The tier's infantry with a
        derived body is fitted on its base after removals; with ``build`` the returned bytes are the
        whole derived body (its additions built in Blender), for the preview.
        -> (pdxmesh, mesh file, mesh bytes, BodyFit)."""
        from eu4unitgen.core.fitting import measure_body
        from eu4unitgen.core.generator import BaseChain
        from eu4unitgen.core.loadout import check_body

        game, idx = self.game()
        loadout = loadout if loadout is not None else load_loadout(spec)
        try:
            name = (loadout.bodies.get(tier) if loadout else None) or BaseChain(idx, spec.base_gfx).slot(
                "infantry", tier)["pdxmesh"]
            check_body(name, game, idx)
        except GenerationError as exc:
            raise SpecError(str(exc)) from exc
        file = idx.pdxmesh(name)["file"]
        data = game.read(file)
        if data is None:
            raise SpecError(f"body mesh {file} not found in the EU4 install")
        spec_d = (loadout.derived.get(tier) if loadout else None) if role == "infantry" else None
        if spec_d:
            from eu4unitgen.core.derived_body import DerivedBodyError, append_additions, build_derived, removed_triangles

            try:
                removed = removed_triangles(data, file, spec_d["remove"]) if spec_d["remove"] else []
                fit = measure_body(append_additions(data, [], removed), file)
                if build and spec_d["add"]:
                    data, _ = build_derived(self.runner(), data, spec_d["add"], remove=removed, base_file=file)
                else:
                    data = append_additions(data, [], removed)
            except DerivedBodyError as exc:
                raise SpecError(f"tier {tier}: derived body: {exc}") from exc
            return name, file, data, fit
        return name, file, data, measure_body(data, file)

    def fitted(self, spec: SetSpec, tier: int, role: str, kind: str) -> dict:
        """A part kind's values on this tier's body before any override: defaults, then fitting
        (lengths scaled to the body, the crest measured on the head). Fast: no Blender."""
        from eu4unitgen.core.parts import PartError, PartSpec, resolve

        *_, fit = self._body(spec, tier)
        try:
            return resolve(PartSpec(kind), role, fit).params
        except PartError as exc:
            raise SpecError(str(exc)) from exc

    def parts_preview(self, spec: SetSpec, tier: int, role: str, loadout) -> PartsPreview:
        """Render one slot in background Blender: the tier's body in its bind pose with the slot's
        parts on their bones (front and side), and each part alone. Country zones and the body are
        tinted with the unit colours, as PdxMeshColor would. Saved to
        <output>/part_previews/<set>/<tier>_<role>.png (it shows a base-game body: user output only)."""
        import tempfile

        from PIL import Image, ImageDraw

        from eu4unitgen.core.fitting import rigid
        from eu4unitgen.core.loadout import LoadoutError, RoleLoadout, check_slot
        from eu4unitgen.core.naming import Naming
        from eu4unitgen.core.parts import PartBuild, PartError, build_parts, resolve
        from eu4unitgen.core.retexture import base_textures
        from eu4unitgen.textures.dds import DDSFormat, full_mip_count, write_dds
        from eu4unitgen.textures.recolour import shade

        game, idx = self.game()
        naming = Naming(self.settings.prefix)
        set_name = spec.set_name or (spec.tags[0].lower() if spec.tags else "")
        slot = loadout.slots.get((tier, role))
        notes: list[str] = []
        if slot is None and role == "crew":
            slot = loadout.slots.get((tier, "infantry"))
            notes.append("no crew slot: the gun crew is the infantry entity, shown with the infantry's parts")
        slot = slot or RoleLoadout()
        try:
            check_slot(tier, role, slot)
        except LoadoutError as exc:
            raise SpecError(str(exc)) from exc
        body_name, body_file, body_data, fit = self._body(spec, tier, loadout, role, build=True)
        notes += fit.notes
        colours, more = self._preview_colours(spec)
        notes += more
        builds = []
        try:
            for spec_part in slot.parts:
                part = resolve(spec_part, role, fit, loadout.points)
                blazon = loadout.image(spec_part.blazon) if spec_part.blazon else None
                builds.append(PartBuild(naming.part(set_name, spec_part.kind, tier, role), tier, part, blazon))
        except (PartError, LoadoutError) as exc:
            raise SpecError(str(exc)) from exc
        built = build_parts(self.runner(), naming, set_name, builds, self.settings.entity_version, self.part_cache)
        skeleton = self._skeleton(body_data)

        def dds(img) -> bytes:
            w, h = img.size
            return write_dds(img.convert("RGBA"), DDSFormat("DXT1", w, h, full_mip_count(w, h)))

        with tempfile.TemporaryDirectory(prefix="eu4unitgen_preview_") as tmp:
            folder = Path(tmp)
            body_path = folder / Path(body_file).name
            body_path.write_bytes(body_data)
            try:
                bt = base_textures(body_name, game, idx)
                d, sp = self._recipe_body(spec, tier, bt, notes)
                (folder / bt.shape["diffuse"]).write_bytes(dds(shade(d, sp, colours, bt.shape["shader"])))
            except GenerationError as exc:
                notes.append(f"body untextured in the preview: {exc}")
            parts_args = []
            for b in builds:
                imgs = built.images(b)
                diffuse = shade(imgs["diffuse"], imgs["specular"], colours, b.part.shader)
                tex = naming.part_texture(set_name, b.part.kind, tier, "diffuse", role)
                (folder / tex).write_bytes(dds(diffuse))
                mesh_path = folder / f"{b.name}.mesh"
                mesh_path.write_bytes(built.raw_meshes[b.name])
                bind = (skeleton.get(b.part.bone) or {}).get("bind")
                if bind is None:
                    notes.append(f"{body_file} has no {b.part.bone} bone: {b.part.kind} shown at the origin")
                    bind = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
                bind, scaled = rigid(bind)
                if scaled:
                    notes.append(f"{body_file}: {b.part.bone}'s bind pose is scaled; the preview uses its rotation "
                                 "(the game animates from the .anim)")
                parts_args.append({"name": b.name, "mesh": str(mesh_path), "matrix": list(bind) + [0, 0, 0, 1]})
            head = [b.name for b in builds if b.part.bone == "head"]
            res = self.runner().run_job("render_parts", {"body": str(body_path), "parts": parts_args,
                                                         "views": ["front", "side"], "size": 360,
                                                         "close": head, "close_views": ["three_quarter", "side", "rear"],
                                                         "thumbs": bool(parts_args), "out_dir": tmp}, timeout=300)
            views = [Image.open(res.result["views"][v]).convert("RGB") for v in ("front", "side")]
            closes = [(v, Image.open(res.result["views"][f"{v}_close"]).convert("RGB"))
                      for v in ("three_quarter", "side", "rear") if f"{v}_close" in res.result["views"]]
            for _, img in closes:
                img.load()
            thumbs = [(n, Image.open(path).convert("RGB")) for n, path in res.result.get("thumbs", {}).items()]
            for img in views + [t for _, t in thumbs]:
                img.load()
        thumb = 150
        close_h = 240 if closes else 0
        sheet = Image.new("RGB", (720, 360 + (thumb + 20 if thumbs else 0) + close_h), (40, 40, 42))
        sheet.paste(views[0], (0, 0))
        sheet.paste(views[1], (360, 0))
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"tier {tier} {role}: front", fill=(230, 230, 230))
        draw.text((366, 4), "side (left, shield side)", fill=(230, 230, 230))
        for i, (name, img) in enumerate(thumbs[:4]):
            x = i * (thumb + 30)
            sheet.paste(img.resize((thumb, thumb)), (x, 360 + 18))
            draw.text((x + 2, 362), name.removeprefix(f"{naming.prefix}{set_name}_"), fill=(230, 230, 230))
        y0 = 360 + (thumb + 20 if thumbs else 0)
        for i, (view, img) in enumerate(closes):
            sheet.paste(img.resize((220, 220)), (i * 240, y0 + 18))
            draw.text((i * 240 + 2, y0 + 2), f"head close-up: {view.replace('_', ' ')}", fill=(230, 230, 230))
        out = self.settings.output_dir / naming.part_preview_dir(set_name) / naming.part_preview(tier, role)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
        parts = [{"entity": b.name, "kind": b.part.kind, "bone": b.part.bone, "params": b.part.params,
                  "fitted": b.part.fitted, "shader": b.part.shader} for b in builds]
        return PartsPreview(sheet, out, body_name, parts, notes)

    def _recipe_body(self, spec: SetSpec, tier: int, bt, notes: list[str]):
        """The body's diffuse and spec as the set's texture recipe leaves them (its tier's infantry
        slot dresses the body of every role), for previews."""
        from eu4unitgen.core.recipe import RecipeError, apply_slot

        recipe = load_recipe(spec)
        slot = recipe.slots.get((tier, "infantry")) if recipe is not None else None
        if slot is None or not slot.ops:
            return bt.diffuse, bt.specular
        if slot.base and slot.base != bt.pdxmesh:
            notes.append(f"the recipe's tier {tier} slot retextures {slot.base}, not this body: shown untouched")
            return bt.diffuse, bt.specular
        try:
            res = apply_slot(recipe, slot, bt.diffuse, bt.specular, bt.shape["shader"], "infantry",
                             lambda: self._areas(bt), lambda: bt.image("normal"))
        except RecipeError as exc:
            raise SpecError(f"texture recipe slot {tier}/infantry: {exc}") from exc
        notes.append(f"body textured with the recipe's tier {tier} slot")
        return res.diffuse or bt.diffuse, res.specular or bt.specular

    def body_candidates(self) -> list[dict]:
        """Base-game pdxmeshes a loadout can pick as a tier's body: infantry and rider animation ids,
        the parts' bones (head, both hand nodes), a base-game .mesh. Sorted by file; one entry per
        .mesh (the first pdxmesh naming it). Cached per install."""
        from eu4unitgen.core.loadout import BODY_BONES, check_body
        from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh

        game, idx = self.game()
        if getattr(self, "_bodies", None) is None:
            out, seen = [], set()
            for name in sorted(idx.pdxmeshes):
                try:
                    rec = check_body(name, game, idx)
                except GenerationError:
                    continue
                stem = Path(rec["file"]).stem.lower()
                if rec["file"].lower() in seen:
                    if name.lower() == stem:  # prefer the generic name over a tag's (DAN_INFANTRY_2)
                        for b in out:
                            if b["file"].lower() == rec["file"].lower():
                                b["pdxmesh"] = name
                    continue
                try:
                    summary = summarize_mesh(parse_pdx_mesh(game.read(rec["file"])))
                except (ValueError, IndexError, TypeError):
                    continue
                if not set(BODY_BONES) <= set(summary["bones"]):
                    continue
                seen.add(rec["file"].lower())
                shader = (summary["shapes"] or [{}])[0].get("shader")
                out.append({"pdxmesh": name, "file": rec["file"], "shader": shader,
                            "crew": "artillery_attack" in (rec.get("animations") or {})})
            self._bodies = sorted(out, key=lambda b: b["file"].lower())
        return self._bodies

    @staticmethod
    def _skeleton(body_data: bytes) -> dict:
        from eu4unitgen.library.scanner import parse_pdx_mesh, summarize_mesh

        return summarize_mesh(parse_pdx_mesh(body_data))["skeleton"]

    # ------------------------------------------------------------------ Blender (Phase 2 step 8)
    def check_blender(self) -> dict:
        """Run the pinned-toolchain check in background Blender (version + io_pdx_mesh enabled)."""
        from eu4unitgen.blender.runner import BlenderError, BlenderRunner

        try:
            runner = BlenderRunner(self.settings.blender_path, log_dir=self.settings.output_dir / "blender_logs")
            return runner.check_addon()
        except BlenderError as exc:
            raise SpecError(str(exc)) from exc
