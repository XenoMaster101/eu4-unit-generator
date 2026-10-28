"""The Setup screen (Track P3.3, P4): the settings file's paths, the checks, and the setup steps for a new machine.

Each step starts in a window of its own (``core.setup.command``) so its progress shows; nothing is typed but the
Linux password Ubuntu asks for and, for DINOv3, a Hugging Face token pasted here (passed to that one download, never
stored). The paths are ``core.config``'s settings file. Also the About page: the licence of every model and tool
(``core.licences``).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QTextBrowser,
    QVBoxLayout, QWidget,
)

from eu4unitgen.gui.base import Page, css
from eu4unitgen.gui.theme import tokens

FIELDS = (("eu4_path", "EU4 install", True), ("mod_dir", "EU4 mod folder", True), ("workspace", "Workspace (runs)", True),
          ("ai_python", "AI environment python", False), ("ai_root", "AI folder (weights, TRELLIS.2)", True),
          ("wsl_distro", "WSL distribution (blank: default)", None), ("wsl_python", "TRELLIS.2 python in WSL", None),
          ("blender_path", "Blender 4.2", False))


def _scrolled(page: Page, inner: QWidget) -> None:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.NoFrame)
    sc.setWidget(inner)
    page.layout_.addWidget(sc, 1)


def _top(page: Page, window, title: str) -> None:
    top = QHBoxLayout()
    h = QLabel(title)
    h.setObjectName("h1")
    top.addWidget(h)
    top.addStretch(1)
    page.layout_.addLayout(top)


def detect() -> dict:
    """What a new machine's settings can be filled with by themselves (an empty setting only): the EU4 install from
    Steam's library list, Blender 4.2, the EU4 mod folder, the AI environment the setup step made. Saved. -> what was
    found."""
    from eu4unitgen.core import config, install, setup
    c = config.load()
    found = {}
    if not c.eu4():
        p = install.find_eu4()
        if p:
            found["eu4_path"] = str(p)
    if not c.blender_path:
        exe, _ = install.find_blender()
        if exe:
            found["blender_path"] = str(exe)
    if not c.mod_dir and c.mods().is_dir():
        found["mod_dir"] = str(c.mods())
    if found:
        config.update(**found)
    py = setup.adopt_ai_env()
    if py:
        found["ai_python"] = py
    return found


class SetupPage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        _top(self, window, "Setup")
        inner = QWidget()
        lay = QVBoxLayout(inner)
        # the settings file
        box = QGroupBox("Paths (saved in one settings file)")
        g = QGridLayout(box)
        self.edits: dict[str, QLineEdit] = {}
        for r, (key, label, folder) in enumerate(FIELDS):
            e = QLineEdit()
            self.edits[key] = e
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(e, r, 1)
            if folder is not None:
                b = QPushButton("Browse…")
                b.setObjectName("ghost")
                b.clicked.connect(lambda _=False, k=key, d=folder: self._browse(k, d))
                g.addWidget(b, r, 2)
        save = QPushButton("Save settings")
        save.clicked.connect(self.save)
        g.addWidget(save, len(FIELDS), 2)
        self.where = QLabel()
        self.where.setObjectName("muted")
        g.addWidget(self.where, len(FIELDS), 0, 1, 2)
        lay.addWidget(box)
        # the checks
        cbox = QGroupBox("Checks")
        cl = QVBoxLayout(cbox)
        self.checks = QTextBrowser()
        self.checks.setHtml("Press \"Check again\".")
        self.checks.setMinimumHeight(190)
        cl.addWidget(self.checks)
        again = QPushButton("Check again")
        again.clicked.connect(self.run_checks)
        cl.addWidget(again, 0, Qt.AlignLeft)
        lay.addWidget(cbox)
        # the setup wizard (the big pieces: Blender, WSL, TRELLIS.2, the AI environment and weights)
        sbox = QGroupBox("Set up the AI side and Blender (once per computer)")
        sg = QVBoxLayout(sbox)
        self.steps_state = QLabel()
        self.steps_state.setWordWrap(True)
        sg.addWidget(self.steps_state)
        wiz = QPushButton("Open the setup wizard")
        wiz.clicked.connect(self.open_wizard)
        sg.addWidget(wiz, 0, Qt.AlignLeft)
        lay.addWidget(sbox)
        lay.addWidget(self.message)
        lay.addStretch(1)
        _scrolled(self, inner)
        self.restyle()

    def restyle(self) -> None:
        t = tokens()
        self.setStyleSheet(css(t) + f"""
QGroupBox {{ color: {t['text']}; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
""")

    def showEvent(self, e):  # noqa: N802 - Qt name
        super().showEvent(e)
        from eu4unitgen.core import config
        found = detect()
        if found:
            self.show_message("Found and saved: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in found.items()) + ".")
        c = config.load()
        for key, e2 in self.edits.items():
            e2.setText(str(getattr(c, key) or ""))
        self.where.setText(f"Settings file: {config.config_dir() / config.FILE}")
        self.edits["workspace"].setPlaceholderText(str(config.default_workspace()))
        self.edits["mod_dir"].setPlaceholderText(str(c.mods()))
        self.refresh_steps()

    def _browse(self, key: str, folder: bool) -> None:
        cur = self.edits[key].text()
        path = QFileDialog.getExistingDirectory(self, key, cur) if folder else QFileDialog.getOpenFileName(self, key, cur)[0]
        if path:
            self.edits[key].setText(path)

    def save(self) -> None:
        from eu4unitgen.core import config
        p = config.update(**{k: e.text().strip() for k, e in self.edits.items()})
        self.show_message(f"Saved ({config.config_dir() / config.FILE}).")
        del p

    def run_checks(self) -> None:
        from eu4unitgen.core import checks
        t = tokens()
        rows = []
        for c in checks.run_all():
            col = t["ok"] if c.ok else (t["error"] if c.blocks else t["warn"])
            rows.append(f"<span style='color:{col}'>{'✔' if c.ok else '✖'}</span> {c.what}"
                        + ("" if c.ok else f"<br>&nbsp;&nbsp;&nbsp;→ {c.fix}"))
        self.checks.setHtml("<br>".join(rows))
        self.refresh_steps()

    def refresh_steps(self) -> None:
        from eu4unitgen.core import setup
        try:
            done = setup.state()
        except Exception:  # noqa: BLE001 - a check that can't run shows as to do
            done = {}
        self.steps_state.setText("   ".join(f"{'✔' if done.get(s.key) else '○'} {s.title}" for s in setup.steps()))

    def open_wizard(self) -> None:
        from eu4unitgen.gui.setup_wizard import open_wizard
        open_wizard(self)
        self.refresh_steps()
        self.run_checks()


class AboutPage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        _top(self, window, "About and licences")
        from eu4unitgen import __version__
        from eu4unitgen.core import licences
        inner = QWidget()
        lay = QVBoxLayout(inner)
        intro = QLabel(f"<b>eu4unitgen {__version__}</b> (GPL-3.0). Makes Europa Universalis IV units from your images. "
                       "It never copies Paradox files: everything from the game is read from your own install. "
                       "Images you use must be yours or ones you may use; your rights statement is kept with each unit.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        from eu4unitgen.core import inventory
        table = QLabel(licences.html() + "<h3>What is installed (inventory)</h3>" + inventory.html())
        table.setWordWrap(True)
        table.setTextFormat(Qt.RichText)
        table.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(table)
        lay.addStretch(1)
        _scrolled(self, inner)
        self.setStyleSheet(css(tokens()))

    def restyle(self) -> None:
        self.setStyleSheet(css(tokens()))
