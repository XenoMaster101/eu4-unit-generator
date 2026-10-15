"""Nation wizard: Settings -> Unit set -> Textures -> Parts -> Validate -> Generate -> Test mod.

Thin by design (CLAUDE.md): every action builds Settings/SetSpec from the forms and calls
export.pipeline.Session. Long steps (loading the install takes several seconds) run on a
worker thread, one at a time.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QListWidget, QMainWindow, QStackedWidget, QWidget

from eu4unitgen.core.generator import GenerationError
from eu4unitgen.core.loadout import LoadoutError
from eu4unitgen.core.naming import NamingError
from eu4unitgen.core.parts import PartError
from eu4unitgen.core.recipe import RecipeError
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.export.writer import ExportError
from eu4unitgen.gui.pages import GeneratePage, SettingsPage, TestModPage, UnitSetPage, ValidatePage
from eu4unitgen.gui.parts_page import PartsPage
from eu4unitgen.gui.textures_page import TexturesPage

EXPECTED = (SpecError, ExportError, GenerationError, NamingError, RecipeError, LoadoutError, PartError, OSError)
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

        self.settings_page = SettingsPage(self, self.session.settings)
        self.unit_page = UnitSetPage(self)
        self.textures_page = TexturesPage(self)
        self.parts_page = PartsPage(self)
        self.validate_page = ValidatePage(self)
        self.generate_page = GeneratePage(self)
        self.test_page = TestModPage(self)
        self.pages = [self.settings_page, self.unit_page, self.textures_page, self.parts_page, self.validate_page,
                      self.generate_page, self.test_page]
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
        self.steps.setCurrentRow(0)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.steps)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Set the EU4 install and press Load.")

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
    def load_install(self) -> None:
        self._sync_settings()
        s = self.session

        def work():
            game, idx = s.game()
            return (s.available_dlc(), s.graphical_cultures(), s.base_cultures(), idx.game_version.get("eu4_branch", "?"),
                    s.body_candidates())

        def done(result) -> None:
            dlc, cultures, bases, version, bodies = result
            self.parts_page.fill_bodies(bodies)
            self.settings_page.fill_dlc(dlc)
            self.unit_page.fill(cultures, bases)
            self.test_page.fill(bases)
            self.settings_page.info.setText(f"Loaded {version}: {len(cultures)} graphical cultures, "
                                            f"{len(bases)} you can copy from, {len(dlc)} DLC folders.")
            self.statusBar().showMessage("Install loaded. Tick your owned DLC, then fill in the unit set.", 8000)
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
    return app.exec()
