"""The app's window: a top bar on every screen (Home · New unit · Library · Setup · Help, the status ticks for EU4,
Blender and the AI, the version) over one screen at a time. New unit builds a unit from library parts
(``gui.builder``); New part (from Home and the Library) makes a part from a picture (``gui.new_unit``).

Thin by design (CLAUDE.md): screens collect input and show results; runs are ``ref3d.flow`` in their own detached
process; checks are ``core.checks``; Add to mod is ``export.modexport``. Short jobs (Add to mod) run on a worker
thread through ``run``. The old keyword/vanilla-parts builder is in ``archive/src/gui_nation_wizard``.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout,
    QWidget,
)

from eu4unitgen import __version__
from eu4unitgen.gui.theme import tokens

NAV = (("home", "Home", "The start: pick what to make, or watch the example"),
       ("build", "New unit", "Build a unit (infantry, cavalry, artillery) from the parts in your library, tier by tier"),
       ("units", "Library", "Your parts (horses, soldiers, guns) and units: previews, tags, edits, Add to mod"),
       ("setup", "Setup", "Paths, checks and the one-time setup of Blender and the AI"),
       ("help", "Help", "The tour, the example, the picture guide, questions and answers, updates"))
STATUS_MS = 30000


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
        except Exception:  # noqa: BLE001 - shown to the user in plain words (core.errors), not swallowed
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.done.emit(result)


class TopBar(QFrame):
    def __init__(self, window):
        super().__init__()
        self.setObjectName("topbar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 4, 16, 0)
        brand = QLabel("⚔ EU4 Unit Generator")
        brand.setObjectName("brand")
        brand.setToolTip("Makes Europa Universalis IV units from your pictures")
        lay.addWidget(brand)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("muted")
        ver.setToolTip("This version. Help → Check for updates looks for a newer one.")
        lay.addWidget(ver)
        lay.addSpacing(18)
        self.buttons: dict[str, QPushButton] = {}
        for key, text, tip in NAV:
            b = QPushButton(text)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: window.go(k))
            lay.addWidget(b)
            self.buttons[key] = b
        lay.addStretch(1)
        self.ticks: dict[str, QPushButton] = {}
        for key, text in (("eu4", "EU4"), ("blender", "Blender"), ("ai", "AI")):
            t = QPushButton(f"… {text}")
            t.setObjectName("link")
            t.clicked.connect(lambda _=False: window.go("setup"))
            lay.addWidget(t)
            self.ticks[key] = t

    def select(self, key: str) -> None:
        for k, b in self.buttons.items():
            b.setChecked(k == key)

    def set_status(self, found: dict[str, tuple[bool, str, str]]) -> None:
        """``found``: key -> (ok, short text, what to do when not ok)."""
        t = tokens()
        for key, (ok, text, fix) in found.items():
            b = self.ticks[key]
            b.setText(f"{'✓' if ok else '✗'} {text}")
            b.setStyleSheet(f"color: {t['ok'] if ok else t['error']}; text-decoration: none; padding: 4px 6px;")
            b.setToolTip(("Ready. " if ok else f"Not ready: {fix} ") + "Click to open Setup.")


def status() -> dict[str, tuple[bool, str, str]]:
    """The top bar's ticks from the quick checks (no WSL call: that's Setup's full check)."""
    from eu4unitgen.core import checks
    c = {x.key: x for x in checks.run_all(quick=True)}
    ai_ok = c["ai_env"].ok and c.get("weights", c["ai_env"]).ok
    ai_fix = next((x.fix for x in (c["ai_env"], c.get("weights")) if x is not None and not x.ok), "")
    return {"eu4": (c["eu4_path"].ok, "EU4", c["eu4_path"].fix), "blender": (c["blender"].ok, "Blender", c["blender"].fix),
            "ai": (ai_ok, "AI ready" if ai_ok else "AI", ai_fix)}


class MainWindow(QMainWindow):
    def __init__(self, store: QSettings | None = None, run_async: bool = True):
        super().__init__()
        self.setWindowTitle(f"EU4 Unit Generator {__version__}")
        self.resize(1240, 820)
        self.store = store
        self.run_async = run_async
        self.pool = QThreadPool.globalInstance()
        self._tasks: list[_Task] = []
        self._busy = False
        app = QApplication.instance()
        self.theme_choice = str(store.value("theme", "system")) if store is not None else "system"
        if app is not None:
            from eu4unitgen.gui.theme import apply_theme
            from eu4unitgen.gui.widgets import install_wheel_guard
            self.theme_mode = apply_theme(app, self.theme_choice)  # before any widget, so none keeps the system's
            if getattr(app, "_wheel_guard", None) is None:
                app._wheel_guard = install_wheel_guard(app)

        from eu4unitgen.gui.builder import BuildUnitScreen
        from eu4unitgen.gui.help_page import HelpPage
        from eu4unitgen.gui.home import HomeScreen
        from eu4unitgen.gui.my_units import MyUnitsScreen
        from eu4unitgen.gui.new_unit import NewUnitScreen
        from eu4unitgen.gui.setup_page import AboutPage, SetupPage
        self.bar = TopBar(self)
        self.stack = QStackedWidget()
        self.home = HomeScreen(self)
        self.new_unit = NewUnitScreen(self)
        self.builder = BuildUnitScreen(self)
        self.my_units = MyUnitsScreen(self)
        self.setup_page = SetupPage(self)
        self.help_page = HelpPage(self)
        self.about_page = AboutPage(self)
        self.screens = {"home": self.home, "new": self.new_unit, "build": self.builder, "units": self.my_units,
                        "setup": self.setup_page,
                        "help": self.help_page, "about": self.about_page}
        for w in self.screens.values():
            self.stack.addWidget(w)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.bar)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.restyle()
        self.go("home")
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(STATUS_MS)
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start()
        QTimer.singleShot(0, self.refresh_status)

    # ------------------------------------------------------------------ screens
    def go(self, key: str, **kw) -> None:
        """Show screen ``key`` (home, new (a part), build (a unit), units (the Library), setup, help, about); ``kw``
        go to its ``enter``."""
        w = self.screens[key]
        self.stack.setCurrentWidget(w)
        self.bar.select({"about": "help", "new": "units"}.get(key, key))
        if hasattr(w, "enter"):
            w.enter(**kw)

    def show_home(self) -> None:
        self.go("home")

    show_unit = show_home  # Setup's and About's back button

    def show_setup(self) -> None:
        self.go("setup")

    def show_about(self) -> None:
        self.go("about")

    def first_screen(self) -> None:
        """Home; Setup first (and its wizard) while EU4, Blender or the AI environment is missing."""
        found = status()
        if not all(ok for ok, _, _ in found.values()):
            self.go("setup")
            QTimer.singleShot(400, self.setup_page.open_wizard)

    def refresh_status(self) -> None:
        try:
            self.bar.set_status(status())
        except Exception:  # noqa: BLE001 - the ticks are a hint; Setup shows the full checks
            pass

    def restyle(self) -> None:
        from eu4unitgen.gui.base import css
        self.setStyleSheet(css(tokens()))
        for w in self.screens.values():
            if hasattr(w, "restyle"):
                w.restyle()

    def set_theme(self, choice: str) -> None:
        from eu4unitgen.gui.theme import apply_theme
        self.theme_choice = choice
        self.theme_mode = apply_theme(QApplication.instance(), choice)
        self.restyle()
        if self.store is not None:
            self.store.setValue("theme", choice)

    # ------------------------------------------------------------------ plumbing
    def run(self, page, label: str, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        """Run ``fn`` off the GUI thread (unless run_async is False), then ``on_done`` with its result; a failure is
        shown on ``page`` in plain words (core.errors), the details behind a button."""
        if self._busy:
            return
        self._busy = True
        page.show_message(f"{label}…")
        page.set_busy(True)

        def finish() -> None:
            self._busy = False
            page.set_busy(False)

        def done(result: Any) -> None:
            finish()
            page.show_message("")
            on_done(result)

        def failed(text: str) -> None:
            from eu4unitgen.core import errors
            finish()
            x = errors.explain(text.strip().splitlines()[-1] if text.strip() else text)
            page.show_message(f"{label} didn't work. {x['what']} What to do: {x['todo']}", error=True)
            self.details(text)

        if not self.run_async:
            try:
                result = fn()
            except Exception:  # noqa: BLE001 - as above
                return failed(traceback.format_exc())
            return done(result)
        task = _Task(fn)
        task.signals.done.connect(done)
        task.signals.failed.connect(failed)
        self._tasks = [t for t in self._tasks if t is not task] + [task]  # keep signals alive
        self.pool.start(task)

    def details(self, text: str) -> None:
        self._last_details = text

    def confirm(self, title: str, text: str, details: str = "") -> bool:
        """Ask before doing something that writes or deletes (tests replace this)."""
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        if details:
            box.setDetailedText(details)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        return box.exec() == QMessageBox.Yes

    def inform(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.pool.waitForDone(10000)
        super().closeEvent(event)


def log_path() -> Path:
    """Where the GUI writes errors when it has no console (pythonw.exe, the desktop shortcut)."""
    from eu4unitgen.core import config
    return config.logs_dir() / "gui.log"


def _install_error_reporting() -> None:
    """Under pythonw.exe there is no stderr: send output to log_path() and show uncaught errors in a message box
    instead of letting the app vanish silently."""
    if sys.stderr is None or sys.stdout is None:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "EU4 Unit Generator: unexpected error", f"{text}\n(also written to {log_path()})")
    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    from eu4unitgen.core import layout
    layout.use_root_temp()  # temp files and caches inside the app's root, not the user profile
    _install_error_reporting()
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.png"))))
    if sys.platform == "win32":  # the taskbar shows our icon, not Python's
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("eu4unitgen.app")
        except (AttributeError, OSError):
            pass
    from eu4unitgen.core import config
    window = MainWindow(store=QSettings(str(config.config_dir() / "gui.ini"), QSettings.IniFormat))  # not the registry
    window.show()
    window.first_screen()
    return app.exec()
