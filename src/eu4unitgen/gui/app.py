"""Nation wizard: Settings -> Character -> Unit set -> Textures -> Parts -> Validate -> Generate -> Test mod.

Thin by design (CLAUDE.md): every action builds Settings/SetSpec from the forms and calls
export.pipeline.Session. Long steps (loading the install takes several seconds) run on a
worker thread, one at a time.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QListWidget, QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from eu4unitgen.core.generator import GenerationError
from eu4unitgen.core.loadout import LoadoutError
from eu4unitgen.core.naming import TEST_MOD_NAME, UNIT_TYPES, NamingError
from eu4unitgen.core.parts import PartError
from eu4unitgen.core.recipe import RecipeError
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.export.writer import ExportError
from eu4unitgen.core.character import CharacterError
from eu4unitgen.core.finishes import FinishError
from eu4unitgen.core.history import History, Snapshot
from eu4unitgen.gui.character_page import CharacterPage
from eu4unitgen.gui.home_page import HomePage
from eu4unitgen.gui.result_page import ResultPage
from eu4unitgen.gui.pages import GeneratePage, SettingsPage, TestModPage, UnitSetPage, ValidatePage
from eu4unitgen.gui.parts_page import PartsPage
from eu4unitgen.gui.textures_page import TexturesPage

EXPECTED = (SpecError, ExportError, GenerationError, NamingError, RecipeError, LoadoutError, PartError, CharacterError,
            FinishError, OSError)
KEYS = ("eu4_path", "output_dir", "mod_folder", "stage_dir", "mod_dir", "prefix", "blender_path")


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn, self.signals = fn, _Signals()

    def run(self) -> None:
        try:
            result = self.fn()
        except EXPECTED as exc:
            self.signals.failed.emit(str(exc))
        except Exception:  # noqa: BLE001 - shown to the user, not swallowed
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.done.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None, run_async: bool = True, store: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("eu4unitgen: nation wizard")
        self.resize(1100, 760)
        self.store = store
        self.session = session or Session(self._stored_settings())
        self.run_async = run_async
        self.pool = QThreadPool.globalInstance()
        self._busy = False
        self._tasks: list[_Task] = []
        app = QApplication.instance()
        if app is not None and getattr(app, "_wheel_guard", None) is None:
            from eu4unitgen.gui.widgets import install_wheel_guard
            app._wheel_guard = install_wheel_guard(app)
        self.theme_choice = str(store.value("theme", "system")) if store is not None else "system"
        if app is not None:
            from eu4unitgen.gui.theme import apply_theme
            self.theme_mode = apply_theme(app, self.theme_choice)  # before any widget, so none keeps the system's

        self.settings_page = SettingsPage(self, self.session.settings)
        self.character_page = CharacterPage(self)
        self.unit_page = UnitSetPage(self)
        self.textures_page = TexturesPage(self)
        self.parts_page = PartsPage(self)
        self.validate_page = ValidatePage(self)
        self.generate_page = GeneratePage(self)
        self.test_page = TestModPage(self)
        self.pages = [self.settings_page, self.character_page, self.unit_page, self.textures_page, self.parts_page,
                      self.validate_page, self.generate_page, self.test_page]
        if store is not None:
            self.unit_page.tags.setText(str(store.value("tags", "")))
            self.unit_page.cultures.setText(str(store.value("cultures", "")))
            self.unit_page.set_name.setText(str(store.value("set_name", "")))
            self.unit_page.adjective.setText(str(store.value("adjective", "")))
            if store.value("gfx"):
                self.unit_page.gfx.setEditText(str(store.value("gfx")))
            recipe = str(store.value("texture_recipe", "") or "")
            if recipe and Path(recipe).is_file():
                try:
                    self.textures_page.load_file(Path(recipe))
                except RecipeError:
                    pass
                self.textures_page.use.setChecked(str(store.value("use_texture_recipe", "")) == "true")
            loadout = str(store.value("loadout", "") or "")
            if loadout and Path(loadout).is_file():
                try:
                    self.parts_page.load_file(Path(loadout))
                except LoadoutError:
                    pass
                self.parts_page.use.setChecked(str(store.value("use_loadout", "")) == "true")

        self.steps = QListWidget()
        self.steps.addItems([p.title for p in self.pages])
        self.steps.setFixedWidth(170)
        self.stack = QStackedWidget()
        for p in self.pages:
            self.stack.addWidget(p)
        self.steps.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.steps.currentRowChanged.connect(lambda _: self.character_edited())  # edits on other pages count too
        self.steps.setCurrentRow(0)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.steps)
        lay.addWidget(self.stack, 1)
        # Track F: the Home screen first; every page above behind "Advanced editor"
        self.home = HomePage(self)
        advanced = QWidget()
        al = QVBoxLayout(advanced)
        al.setContentsMargins(0, 0, 0, 0)
        back = QPushButton("← Simple view")
        back.setFlat(True)
        back.setToolTip("Back to the describe-and-generate screen")
        back.clicked.connect(self.show_home)
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 0)
        bar.addWidget(back)
        bar.addStretch(1)
        al.addLayout(bar)
        al.addWidget(central, 1)
        self.result_page = ResultPage(self)
        self.views = QStackedWidget()
        self.views.addWidget(self.home)
        self.views.addWidget(advanced)
        self.views.addWidget(self.result_page)
        from eu4unitgen.core.army import Army

        self.army = Army()
        self._result_token = 0
        self._result_dirty = False
        self._tile_prints: dict = {}  # (unit type, tier) -> fingerprint of the picture on its tile
        self._skins: dict[str, tuple] = {}  # body pdxmesh -> its skin tone (read once)
        self.setCentralWidget(self.views)
        self.statusBar().showMessage("Set the EU4 install and press Load.")
        self.history = History(self.character_state())
        self._update_title()
        i = self.settings_page.theme.findData(self.theme_choice)
        self.settings_page.theme.blockSignals(True)
        self.settings_page.theme.setCurrentIndex(max(i, 0))
        self.settings_page.theme.blockSignals(False)

    def set_theme(self, choice: str) -> None:
        """Light, dark or follow Windows: the whole app, at once (gui.theme)."""
        from eu4unitgen.gui.theme import apply_theme

        app = QApplication.instance()
        self.theme_choice = choice
        self.theme_mode = apply_theme(app, choice)
        self.home.restyle()
        self.result_page.restyle()
        if self.store is not None:
            self.store.setValue("theme", choice)

    # ------------------------------------------------------------------ views (Track F)
    def show_home(self) -> None:
        self.views.setCurrentIndex(0)

    def show_advanced(self, page=None) -> None:
        self.views.setCurrentIndex(1)
        if page is not None:
            self.steps.setCurrentRow(self.pages.index(page))

    def _home_status(self, version: str | None = None) -> None:
        """The Home screen's line: EU4 (and its version once loaded), owned DLC, Blender."""
        if version:
            self._eu4_version = version
        version = getattr(self, "_eu4_version", None)
        page = self.settings_page
        eu4 = page.eu4.text().strip()
        ok_eu4 = bool(eu4) and (Path(eu4) / "common").is_dir()
        blender = page.blender.text().strip()
        n = len(page.owned())
        self.home.set_status([
            (ok_eu4, f"EU4 {version}" if (ok_eu4 and version) else ("EU4 found" if ok_eu4 else "EU4 not found")),
            (True, f"{n} DLC ticked"),
            (bool(blender) and Path(blender).is_file(), "Blender found" if blender and Path(blender).is_file()
             else "Blender not found (3D parts and previews need it)")])

    # ------------------------------------------------------------------ plumbing
    def _stored_settings(self) -> Settings:
        if self.store is None:
            return Settings()
        values = {k: self.store.value(k) for k in KEYS if self.store.value(k)}
        dlc = self.store.value("owned_dlc") or []
        return Settings(**values, owned_dlc=[dlc] if isinstance(dlc, str) else list(dlc))

    def _sync_settings(self) -> None:
        """Forms -> session settings. A different EU4 path reloads the install on next use."""
        self.session.settings = self.settings_page.settings()
        if self.store is not None:
            s = self.session.settings
            for k in KEYS:
                v = getattr(s, k)
                self.store.setValue(k, "" if v is None else str(v))
            self.store.setValue("owned_dlc", s.owned_dlc)
            u = self.unit_page
            for k, v in (("tags", u.tags.text()), ("cultures", u.cultures.text()), ("set_name", u.set_name.text()),
                         ("adjective", u.adjective.text()),
                         ("gfx", u.gfx.currentText())):
                self.store.setValue(k, v)
            self.store.setValue("texture_recipe", self.textures_page.recipe_path.text().strip())
            self.store.setValue("use_texture_recipe", "true" if self.textures_page.use.isChecked() else "false")
            self.store.setValue("loadout", self.parts_page.loadout_path.text().strip())
            self.store.setValue("use_loadout", "true" if self.parts_page.use.isChecked() else "false")

    def run(self, page, label: str, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        """Run ``fn`` (off the GUI thread unless run_async is False), then ``on_done`` with its result.
        Failures go to the page's message line."""
        if self._busy:
            return
        self._busy = True
        page.show_message("")
        page.set_busy(True)
        self.statusBar().showMessage(f"{label}…")

        def finish() -> None:
            self._busy = False
            page.set_busy(False)
            if page is getattr(self, "result_page", None) and self._result_dirty:  # an edit waited: after this step
                QTimer.singleShot(0, self._restart_if_dirty)

        def done(result: Any) -> None:
            finish()
            self.statusBar().showMessage(f"{label}: done", 5000)
            on_done(result)

        def failed(text: str) -> None:
            finish()
            self.statusBar().showMessage(f"{label}: failed", 5000)
            page.show_message(text.replace("\n", "<br>"), error=True)

        if not self.run_async:
            try:
                result = fn()
            except EXPECTED as exc:
                return failed(str(exc))
            return done(result)
        task = _Task(fn)
        task.signals.done.connect(done)
        task.signals.failed.connect(failed)
        self._tasks = [t for t in self._tasks if t is not task] + [task]  # keep signals alive
        self.pool.start(task)

    # ------------------------------------------------------------------ actions
    def auto_setup(self, load: bool = True) -> list[str]:
        """Find the EU4 install and the owned DLC when the settings don't name them, then load the
        install in the background. Never replaces what the user set. Returns what it did."""
        from eu4unitgen.core.install import find_blender, find_eu4, owned_dlc

        page, did = self.settings_page, []
        if not page.eu4.text().strip():
            found = find_eu4()
            if found is not None:
                page.eu4.setText(str(found))
                did.append(f"found EU4 at {found}")
        eu4 = Path(page.eu4.text().strip()) if page.eu4.text().strip() else None
        if eu4 is not None and not page.owned():
            names, note = owned_dlc(eu4)
            if names:
                page.set_owned(names)
            did.append(note + ("; untick any you don't own" if names else ""))
        current = page.blender.text().strip()
        if not current or not Path(current).is_file():  # a working path is the user's choice: kept
            exe, note = find_blender()
            if exe is not None:
                page.blender.setText(str(exe))
            did.append(note)
        if eu4 is None:
            did.append("EU4 install not found: set its folder in 1. Settings")
        elif load:
            self.character_page.show_hint(loaded=False, loading=True)
            self.load_install()
        if did:
            page.info.setText("Auto-setup: " + "; ".join(did) + ".")
        self._home_status()
        return did

    def load_install(self) -> None:
        self._sync_settings()
        s = self.session

        def work():
            game, idx = s.game()
            return (s.available_dlc(), s.graphical_cultures(), s.base_cultures(), idx.game_version.get("eu4_branch", "?"),
                    s.body_candidates(), s.country_list())

        def done(result) -> None:
            dlc, cultures, bases, version, bodies, countries = result
            self.home.fill_nations(countries)
            self.parts_page.fill_bodies(bodies)
            self.settings_page.fill_dlc(dlc)
            self.unit_page.fill(cultures, bases)
            self.test_page.fill(bases)
            self.settings_page.info.setText(f"Loaded {version}: {len(cultures)} graphical cultures, "
                                            f"{len(bases)} you can copy from, {len(dlc)} DLC folders.")
            self.statusBar().showMessage("Install loaded. Tick your owned DLC, then fill in the unit set.", 8000)
            self.character_page.show_hint(loaded=True)
            self._home_status(version.replace("release_", ""))
        self.run(self.settings_page, "Loading EU4 install", work, done)

    def current_spec(self, page) -> SetSpec | None:
        """The unit-set form plus the texture recipe and parts loadout; a bad form shows on ``page``."""
        spec = self.unit_page.spec()
        try:
            spec.texture_recipe = self.textures_page.recipe_for_spec()
            spec.loadout = self.parts_page.loadout_for_spec()
        except SpecError as exc:
            page.show_message(str(exc), error=True)
            return None
        return spec

    # ------------------------------------------------------------------ the Character page
    def load_character(self, path: Path) -> None:
        """A character spec into the Unit set, Textures and Parts pages (its paths stay relative to its folder)."""
        from eu4unitgen.core.character import CharacterSpec
        from eu4unitgen.core.loadout import Loadout
        from eu4unitgen.core.recipe import TextureRecipe

        try:
            spec = CharacterSpec.load(path)
        except CharacterError as exc:
            return self.character_page.show_message(str(exc), error=True)
        if self.adopt_character(spec, path):
            self.history.reset(self.character_state())
            self._update_title()
            self.character_page.show_message(
                f"Loaded {path}: {spec.name or 'unnamed'}; the Unit set, Textures and Parts pages now show it. "
                "Press Show body to paint its areas.")

    def adopt_character(self, spec, path: Path | None = None) -> bool:
        """Show ``spec`` in the Unit set, Textures, Parts and Character pages (``path``: its file;
        None: made on the Home screen, not saved yet). False (with a message) if it doesn't load."""
        from eu4unitgen.core.loadout import Loadout
        from eu4unitgen.core.recipe import TextureRecipe

        page = self.character_page
        try:
            recipe = TextureRecipe.from_json(spec.textures, spec.root) if spec.textures is not None else None
            loadout = Loadout.from_json(spec.loadout, spec.root) if spec.loadout is not None else None
        except (RecipeError, LoadoutError) as exc:
            page.show_message(str(exc), error=True)
            return False
        label = f"(in {path.name})" if path else "(made on the Home screen)"
        page.spec = spec
        page.spec_path.setText(str(path) if path else "")
        page.name.setText(spec.name)
        page.masks.clear()
        self.unit_page.load_set(spec.set, spec.unit_colours())
        self.textures_page.set_recipe(recipe, label if recipe else "")
        if recipe is None:
            self.textures_page.recipe.root = spec.root
        if loadout is not None:
            loadout.hair = dict(spec.hair)
        self.parts_page.set_loadout(loadout, label if loadout else "")
        if loadout is None:
            self.parts_page.loadout.root = spec.root
        if page.body is not None:
            page.show_areas(page.body, self.textures_page.recipe)
        return True

    # ------------------------------------------------------------------ the Home screen (Track F)
    def home_generate(self) -> None:
        """Home's description, image chips, country and Tiers choice -> the army; on to the Result
        screen, which checks it and renders the gallery."""
        from eu4unitgen.core.army import Army
        from eu4unitgen.core.look import LookError

        home = self.home
        tag = home.tag()
        if not tag:
            return home._check_ready()  # the For: field shows what's missing
        self._sync_settings()
        try:
            home.look().check()
        except (LookError, ValueError) as exc:
            return home.show_message(str(exc), error=True)
        try:
            own = self.session.unit_colours().get(tag)
        except SpecError:
            own = None
        self.army = Army(home.text(), list(home.image_items), tag, home.tiers.currentData(), [],
                         [list(c) for c in own] if own else None)
        self.result_page.clear_gallery()
        self._tile_prints = {}
        self.views.setCurrentWidget(self.result_page)
        self.result_regenerate()

    # ------------------------------------------------------------------ the Result screen (Track F)
    def _result_base(self, tag: str) -> dict:
        """The spec a look is compiled into: the same country's spec as the pages hold it (so what the
        Advanced editor added stays), else a new one."""
        old = self.character_page.spec
        name = self.army.text.strip()[:80]
        if old is not None and old.set.get("tags") == [tag]:
            return {**old.to_json(), "name": name}
        gfx = self.unit_page.gfx.currentText().strip() or "hellenicgfx"
        return {"schema": 1, "name": name, "set": {"tags": [tag], "gfx": gfx, "set_name": tag.lower(),
                                                   "base_gfx": self.unit_page.base.currentText() or "westerngfx"}}

    def result_regenerate(self) -> None:
        """The army -> spec (shown in the pages too) -> validation -> the gallery, row by row, in the
        background. Only units whose assembled model changed are rendered again; the others keep
        their picture, and a picture being redone is dimmed until it arrives. An edit made
        meanwhile restarts it once the current step ends."""
        from eu4unitgen.core.character import CharacterSpec
        from eu4unitgen.core.look import DEFAULT_SKIN, LookError, compile_look

        page, army = self.result_page, self.army
        interp = army.interpretation(self.home.interp)
        try:
            look = army.look(self.home.interp)
            look.check()
        except (LookError, ValueError) as exc:
            return page.show_message(str(exc), error=True)
        page.show_army(army, look, interp, self._skins.get(look.body_base, DEFAULT_SKIN))
        if self._busy:
            self._result_dirty = True
            return
        self._result_dirty = False
        self._result_token += 1
        token = self._result_token
        page.mark_stale(page.tiles)
        page.show_waiting("checking the army and building its parts…")
        page.show_status("Checking…")
        base = self._result_base(army.tag)
        session, skins = self.session, dict(self._skins)

        def work():
            skin = skins.get(look.body_base)
            if skin is None:
                first = CharacterSpec.from_json(compile_look(look, base), Path.cwd())
                skin = session.body_areas(first.set_spec(), 1, "infantry").skin
            spec = CharacterSpec.from_json(compile_look(look, base, {t: skin for t in (1, 2, 3, 4)}), Path.cwd())
            setspec = spec.set_spec()
            result = session.validate(setspec)[0]
            return spec, setspec, result, skin, session.gallery_prints(setspec, result)

        def done(res) -> None:
            spec, setspec, result, skin, prints = res
            self._skins[look.body_base] = skin
            if self.adopt_character(spec):
                self.character_edited()
            page.show_army(army, look, interp, skin)
            page.show_report(result.report)
            self._result_spec = (setspec, result)
            if self._restart_if_dirty():
                return
            changed = [k for k, v in prints.items() if self._tile_prints.get(k) != v or k not in page.images]
            page.mark_stale([k for k in page.tiles if k not in changed], False)
            self._render_rows(token, setspec, result, [u for u in UNIT_TYPES if any(k[0] == u for k in changed)],
                              changed, prints)

        self.run(page, "Checking the army", work, done)

    def _restart_if_dirty(self) -> bool:
        if self._result_dirty:
            self._result_dirty = False
            self.result_regenerate()
            return True
        return False

    def _render_rows(self, token: int, setspec, result, rows: list[str], changed: list, prints: dict) -> None:
        """One Blender run per unit type with changed units (only those), so tiles appear as they're
        ready."""
        if token != self._result_token or not rows:
            return
        page, utype = self.result_page, rows[0]
        units = [k for k in changed if k[0] == utype]
        page.show_waiting("rendering…", units)

        def done(gallery) -> None:
            if token != self._result_token:
                return
            page.show_gallery(gallery, units)
            self._tile_prints.update({k: prints[k] for k in units})
            if gallery.notes:
                page.show_message("; ".join(gallery.notes))
            if not self._restart_if_dirty():
                self._render_rows(token, setspec, result, rows[1:], changed, prints)

        self.run(page, f"Rendering the {utype}", lambda: self.session.gallery(setspec, units, 200, result, thumbs=120),
                 done)

    def result_edit(self, *edits) -> None:
        """Result-screen changes (core.army.Edit): recorded, replayed on top of the description."""
        for e in edits:
            self.army.add(e)
        self.result_regenerate()

    def result_chip_remove(self, n: int, from_image: bool) -> None:
        interp = self.army.interpretation(self.home.interp)
        if from_image:
            del self.army.image_items[n - (len(interp.items) - len(self.army.image_items))]
        else:
            self.army.text = interp.text_without(n)
        self.home.image_items = list(self.army.image_items)
        self.home.set_text(self.army.text)  # Home shows the same description
        self.result_regenerate()

    def result_chip_swap(self, n: int, entry: dict, from_image: bool) -> None:
        interp = self.army.interpretation(self.home.interp)
        if from_image:
            interp.swap(n, entry)
            self.army.image_items[n - (len(interp.items) - len(self.army.image_items))] = interp.items[n]
        else:
            self.army.text = interp.text_swapped(n, entry)
        self.home.image_items = list(self.army.image_items)
        self.home.set_text(self.army.text)
        self.result_regenerate()

    def result_emblem(self, name: str) -> None:
        """A built-in emblem in the army's second unit colour, drawn into the output folder."""
        from eu4unitgen.core.army import Edit
        from eu4unitgen.core.emblems import write

        look = self.army.look(self.home.interp)
        colours = look.colours or self.army.base_colours or [[150, 28, 32], [195, 150, 60], [230, 222, 200]]
        path = write(name, self.session.settings.output_dir / "emblems", tuple(colours[1]))
        self.result_edit(Edit("emblem", (), str(path)))

    def confirm(self, title: str, text: str, details: str = "") -> bool:
        """Ask before writing files (tests replace this)."""
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        if details:
            box.setDetailedText(details)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        return box.exec() == QMessageBox.Yes

    def _result_setspec(self):
        spec = getattr(self, "_result_spec", None)
        if spec is None:
            self.result_page.show_message("Generate the army first.", error=True)
        return spec

    def result_test(self) -> None:
        """Build the test mod for this army and stage it (copying it into EU4's mod folder is yours:
        Windows' Controlled Folder Access may block the app from writing there)."""
        got = self._result_setspec()
        if got is None:
            return
        setspec, _ = got
        page = self.result_page
        mod_dir = self.session.settings.user_mod_dir()

        def done(res) -> None:
            if not res.written:
                problems = [str(i) for _, r in res.reports for i in r.errors][:5]
                return page.show_message("The test mod wasn't staged: " + ("; ".join(problems) or "see the Advanced "
                                         "editor's 8. Test mod page"), error=True)
            page.show_message(
                f"Test mod ready: {len(res.written)} files in <b>{res.stage_dir}</b>.<br>"
                f"Copy everything inside that folder into <b>{mod_dir}</b> (with Explorer), tick "
                f"\"{TEST_MOD_NAME}\" in the EU4 launcher and start a game as {', '.join(setspec.tags)}."
                + ("<br>" + "<br>".join(f"Note: {w}" for w in res.warnings) if res.warnings else ""))

        self.run(page, "Building the test mod", lambda: self.session.build_test_mod(setspec), done)

    def result_export(self) -> None:
        """A dry run first; the file list to confirm; then the package written to the output
        folder (never into a mod folder, never over files the tool didn't write)."""
        from eu4unitgen.export.writer import ExportError, apply_export

        got = self._result_setspec()
        if got is None:
            return
        setspec, _ = got
        page = self.result_page

        def done(results) -> None:
            blocked = [r for r in results if not r.plan.can_apply]
            if blocked:
                reasons = [f"{f.path}: {f.reason}" for r in blocked for f in r.plan.files if f.action == "blocked"]
                reasons += [e for r in blocked for e in r.plan.validation_errors]
                return page.show_message("Export blocked: " + "; ".join(reasons[:5]), error=True)
            files = [f"{f.action:9} {f.path}" for r in results for f in r.plan.files]
            new = sum(1 for r in results for f in r.plan.files if f.action != "unchanged")
            where = ", ".join(str(r.package_dir) for r in results)
            if not self.confirm("Export", f"Write {new} file(s) to {where}?\n\nNothing outside that folder is "
                                          "touched, and no file the app didn't write is replaced.", "\n".join(files)):
                return page.show_message("Export cancelled; nothing written.")
            try:
                written = [p for r in results for p in apply_export(r.plan)]
            except (ExportError, OSError) as exc:
                return page.show_message(f"Export failed: {exc}", error=True)
            page.show_message(f"Exported {len(written)} file(s) to <b>{where}</b>. Next: add them to your mod as "
                              "INTEGRATION.md in that folder says.")

        self.run(page, "Preparing the export", lambda: self.session.generate(setspec, write=False), done)

    def save_character(self, path: Path) -> None:
        """The pages -> a character spec at ``path`` (recipe and loadout paths rebased onto its folder)."""
        from eu4unitgen.core.character import CharacterSpec, rebase_paths

        page = self.character_page
        old = page.spec
        try:
            self.textures_page.store()
            self.parts_page.store()
            recipe = self.textures_page.recipe if self.textures_page.use.isChecked() else None
            loadout = self.parts_page.loadout if self.parts_page.use.isChecked() else None
            dst = path.parent
            painted = set(self.textures_page.recipe.images)  # written into dst below, so not rebased
            spec = CharacterSpec(
                set=self.unit_page.set_section(old.set if old else None), name=page.name.text().strip(),
                textures=rebase_paths(recipe.to_json(), Path(recipe.root), dst, painted) if recipe else None,
                loadout=rebase_paths(loadout.to_json(), Path(loadout.root), dst) if loadout else None,
                hair=dict(old.hair) if old else {}, export=dict(old.export) if old else {},
                test=dict(rebase_paths(old.test, old.root, dst)) if old else {}, root=dst)
            spec = CharacterSpec.from_json(spec.to_json(), dst)  # checks it the way loading will
            tp_recipe = self.textures_page.recipe
            same = Path(tp_recipe.root).resolve() == dst.resolve()
            masks = tp_recipe.write_images(dst) if recipe else []
            spec.save(path)
        except (SpecError, CharacterError, OSError) as exc:
            return page.show_message(str(exc), error=True)
        wrote = f" and {len(masks)} painted mask(s) beside it" if masks else ""
        if not same:  # paths in the pages are relative to the old folder: read it all back from the new one
            self.load_character(path)
            return page.show_message(f"Saved {path}{wrote}. It's in a new folder, so the undo history starts here.")
        page.spec = spec
        page.spec_path.setText(str(path))
        self.history.mark_saved(self.character_state())
        self._update_title()
        page.show_message(f"Saved {path}{wrote}.")

    # ------------------------------------------------------------------ the Character page's edit history (U3)
    def character_state(self) -> Snapshot | None:
        """The character as the pages show it now (one undo step); None while a form is invalid."""
        tp, pp, page = self.textures_page, self.parts_page, self.character_page
        try:
            tp.store()
            pp.store()
            s = self.unit_page.set_section(page.spec.set if page.spec else None)
        except SpecError:
            return None
        return Snapshot({"name": page.name.text().strip(), "set": s,
                         "textures": tp.recipe.to_json(), "use_textures": tp.use.isChecked(),
                         "loadout": pp.loadout.to_json(), "use_loadout": pp.use.isChecked()}, tp.recipe.images)

    def character_edited(self) -> None:
        """Record the pages' state as an undo step if it changed (every Character page edit calls this)."""
        if getattr(self, "history", None) is None:
            return
        self.history.push(self.character_state())
        self._update_title()

    def restore_character(self, state: Snapshot) -> None:
        """Put a snapshot back into the Unit set, Textures, Parts and Character pages."""
        from eu4unitgen.core.character import CharacterSpec
        from eu4unitgen.core.loadout import Loadout
        from eu4unitgen.core.recipe import TextureRecipe

        tp, pp, page = self.textures_page, self.parts_page, self.character_page
        d = state.doc
        page.name.setText(d["name"])
        self.unit_page.load_set(d["set"], CharacterSpec(set=d["set"]).unit_colours())
        recipe = TextureRecipe.from_json(d["textures"], tp.recipe.root)
        recipe.images = dict(state.images)
        tp.set_recipe(recipe, tp.recipe_path.text())
        tp.use.setChecked(d["use_textures"])
        loadout = Loadout.from_json(d["loadout"], pp.loadout.root)
        loadout.hair = dict(pp.loadout.hair)
        pp.set_loadout(loadout, pp.loadout_path.text())
        pp.use.setChecked(d["use_loadout"])
        if page.body is not None:
            page.show_areas(page.body, tp.recipe)
        self._update_title()

    def character_undo(self) -> None:
        self.character_edited()  # an edit made on another page since the last step is a step too
        state = self.history.undo()
        if state is None:
            return self.character_page.show_message("Nothing to undo.")
        self.restore_character(state)
        self.character_page.show_message("Undone.")

    def character_redo(self) -> None:
        if self.history.current != self.character_state():
            return self.character_page.show_message("Nothing to redo: you changed something since the undo.")
        state = self.history.redo()
        if state is None:
            return self.character_page.show_message("Nothing to redo.")
        self.restore_character(state)
        self.character_page.show_message("Redone.")

    def character_reset(self) -> None:
        """Back to the file as last loaded or saved (itself a step, so it can be undone)."""
        self.character_edited()
        saved = self.history.saved
        if saved is None or not self.history.dirty:
            return self.character_page.show_message("No unsaved changes.")
        self.restore_character(saved)
        self.history.push(saved)
        self._update_title()
        self.character_page.show_message("Back to the saved file (Undo brings your changes back).")

    def _update_title(self) -> None:
        h = getattr(self, "history", None)
        dirty = h is not None and h.dirty
        path = self.character_page.spec_path.text().strip()
        name = Path(path).name if path else "new character"
        self.setWindowTitle(f"eu4unitgen: {name}{' *' if dirty else ''}")
        self.character_page.show_dirty(dirty, h.can_undo if h else False, h.can_redo if h else False)

    def character_areas(self) -> None:
        self._sync_settings()
        page = self.character_page
        spec = self.unit_page.spec()
        try:
            spec.loadout = self.parts_page.loadout_for_preview()  # its tier bodies are what gets painted
            self.textures_page.store()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        tier, role = page.slot_key()
        slot = self.textures_page.recipe.slots.get((tier, role))
        recipe = self.textures_page.recipe
        self.run(page, "Reading body areas",
                 lambda: self.session.body_areas(spec, tier, role, slot.base if slot else None),
                 lambda body: page.show_areas(body, recipe))

    def apply_character_finishes(self, key, chosen: dict, skin: tuple, order: list[str]) -> None:
        """Area finishes -> the recipe slot ``key`` (replacing the slot's earlier finishes)."""
        from eu4unitgen.core.finishes import with_finishes
        from eu4unitgen.core.recipe import SlotRecipe

        tp = self.textures_page
        tp.store()
        slot = tp.recipe.slots.get(key) or SlotRecipe()
        slot.ops = with_finishes(slot.ops, chosen, skin, order)
        if slot.ops or slot.base or slot.protect_mask:
            tp.recipe.slots[key] = slot
        else:
            tp.recipe.slots.pop(key, None)
        if slot.ops:
            tp.use.setChecked(True)
        tp.reshow_slot()

    def character_preview(self) -> None:
        page = self.character_page
        tier, role = page.slot_key()
        self.textures_page.tier.setCurrentText(str(tier))
        self.textures_page.role.setCurrentText(role)
        self.textures_page.reshow_slot()
        self._sync_settings()
        spec = self.unit_page.spec()
        try:
            spec.loadout = self.parts_page.loadout_for_spec()
            slot = self.textures_page.current_slot()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        recipe = self.textures_page.recipe
        self.run(page, "Previewing textures", lambda: self.session.texture_preview(spec, tier, role, recipe, slot),
                 page.show_preview)

    def character_render(self) -> None:
        self._sync_settings()
        page = self.character_page
        spec = self.current_spec(page)
        if spec is None:
            return
        tier, role = page.slot_key()
        try:
            loadout = self.parts_page.loadout_for_preview()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        self.run(page, "Rendering the unit", lambda: self.session.parts_preview(spec, tier, role, loadout),
                 page.show_render)

    def check_blender(self) -> None:
        self._sync_settings()
        page = self.settings_page

        def done(res: dict) -> None:
            page.show_message(f"Blender {res['blender']} OK; io_pdx_mesh {res['addon_version']} enabled "
                              f"({res['module']}).")
        self.run(page, "Checking Blender", self.session.check_blender, done)

    def validate(self) -> None:
        self._sync_settings()
        spec = self.current_spec(self.validate_page)
        if spec is not None:
            self.run(self.validate_page, "Validating", lambda: self.session.validate(spec),
                     self.validate_page.show_results)

    def generate(self) -> None:
        self._sync_settings()
        spec = self.current_spec(self.generate_page)
        if spec is not None:
            self.run(self.generate_page, "Generating", lambda: self.session.generate(spec),
                     self.generate_page.show_results)

    def texture_preview(self) -> None:
        self._sync_settings()
        page = self.textures_page
        spec = self.unit_page.spec()
        try:
            spec.loadout = self.parts_page.loadout_for_spec()  # its tier bodies are what gets retextured
            slot = page.current_slot()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        tier, role = page.slot_key()
        recipe = page.recipe
        self.run(page, "Previewing textures", lambda: self.session.texture_preview(spec, tier, role, recipe, slot),
                 page.show_preview)

    def export_texture_guide(self) -> None:
        self._sync_settings()
        page = self.textures_page
        spec = self.unit_page.spec()
        try:
            spec.loadout = self.parts_page.loadout_for_spec()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        tier, role = page.slot_key()
        base = page.base.text().strip() or None
        self.run(page, "Exporting painting guide",
                 lambda: self.session.export_texture_guide(spec, tier, role, base), page.show_guide)

    def parts_preview(self) -> None:
        self._sync_settings()
        page = self.parts_page
        spec = self.unit_page.spec()
        try:
            loadout = page.loadout_for_preview()
        except SpecError as exc:
            return page.show_message(str(exc), error=True)
        tier, role = page.slot_key()
        self.run(page, "Rendering parts preview", lambda: self.session.parts_preview(spec, tier, role, loadout),
                 page.show_preview)

    def build_test_mod(self) -> None:
        self._sync_settings()
        spec, opts = self.current_spec(self.test_page), self.test_page.options()
        if spec is None:
            return
        self.run(self.test_page, "Building test mod", lambda: self.session.build_test_mod(spec, **opts),
                 self.test_page.show_results)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._sync_settings()
        self.pool.waitForDone(10000)
        self.session.close()
        super().closeEvent(event)


def log_path() -> Path:
    """Where the GUI writes errors when it has no console (pythonw.exe, the desktop shortcut)."""
    import os
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "state")
    return base / "eu4unitgen" / "gui.log"


def _install_error_reporting() -> None:
    """Under pythonw.exe there is no stderr: send output to log_path() and show uncaught errors
    in a message box instead of letting the app vanish silently."""
    if sys.stderr is None or sys.stdout is None:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        from PySide6.QtWidgets import QMessageBox
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "eu4unitgen: unexpected error", f"{text}\n(also written to {log_path()})")
    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    _install_error_reporting()
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    window = MainWindow(store=QSettings("eu4unitgen", "eu4unitgen"))
    window.show()
    window.auto_setup()
    return app.exec()
