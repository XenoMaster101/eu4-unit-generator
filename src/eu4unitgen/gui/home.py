"""The Home screen: "What should your soldiers look like?". Make a part from a picture (a soldier, a horse, later a
gun) or drop one; build a unit from the library; watch the example or take the tour. The text box is kept for later
(text to image) and is off. Parts are made in the New part steps (``gui.new_unit``), units in Build a unit
(``gui.builder``); the old keyword builder is archived."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget

from eu4unitgen.gui.base import DropImage, Page, card, clear_layout, ghost, label, pixmap, primary

TYPES = (("soldier", "🛡  A soldier", "One person from a front picture. Your infantry, your riders and your gun crews.",
          "A soldier part from one picture: used as infantry (foot animations), as a rider on any horse, and as the "
          "crew of any gun"),
         ("horse", "🐎  A horse", "From a side picture. Any of your soldiers can ride it.",
          "A horse part from one picture, tagged by style (light, barded, steppe, Arab) and region"),
         ("gun", "💣  A gun  (coming)", "From a three-quarter picture, crewed by your soldiers.",
          "Being built: a gun from a three-quarter picture, with your soldiers as its crew. Not available yet."))


class HomeScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        col = QVBoxLayout()
        col.setSpacing(12)
        h1 = label("What should your soldiers look like?", "h1")
        h1.setAlignment(Qt.AlignCenter)
        sub = label("Start from a picture. The app turns it into a 3D unit for Europa Universalis IV, one step at a "
                    "time, and adds it to your mod.", "muted")
        sub.setAlignment(Qt.AlignCenter)
        col.addWidget(h1)
        col.addWidget(sub)
        col.addWidget(label("Make a part from a picture", "h2"))
        grid = QGridLayout()
        self.cards = {}
        for i, (key, title, text, tip) in enumerate(TYPES):
            c = card(title, text, tip)
            c.setMinimumHeight(110)
            if key == "gun":
                c.setEnabled(False)
            else:
                c.clicked.connect(lambda _=False, k=key: window.go("new", kind=k))
            grid.addWidget(c, 0, i)
            self.cards[key] = c
        col.addLayout(grid)
        self.build_card = card("🧩  Build a unit from your library",
                               "Infantry, cavalry or artillery, put together from your parts tier by tier. The "
                               "starter parts are already there. No AI needed.",
                               "Pick the parts for each tier (e.g. tier 1 a light horse, tier 3 a barded one)")
        self.build_card.clicked.connect(lambda _=False: window.go("build"))
        col.addWidget(self.build_card)
        row = QHBoxLayout()
        self.drop = DropImage("…or drop a picture here\n(a soldier from the front, or a horse from the side)", 300, 150)
        self.drop.picked.connect(self.picture)
        row.addWidget(self.drop, 2)
        side = QVBoxLayout()
        self.describe = QPlainTextEdit()
        self.describe.setPlaceholderText("Describe it (coming later: text to image)")
        self.describe.setEnabled(False)
        self.describe.setFixedHeight(80)
        self.describe.setToolTip("Coming later: describe a soldier in words and the app draws the picture for you. "
                                 "For now, start from a picture.")
        side.addWidget(self.describe)
        self.example = primary("▶  Show me an example", lambda: window.help_page.show_example(),
                               "Walks through a finished unit (the cataphract), step by step, with its real results")
        self.tour = ghost("Take the 2-minute tour", lambda: window.help_page.show_tour(),
                          "A short walk through the app's screens, with captions")
        side.addWidget(self.example)
        side.addWidget(self.tour)
        side.addStretch(1)
        sw = QWidget()
        sw.setLayout(side)
        row.addWidget(sw, 1)
        col.addLayout(row)
        self.recent_title = label("Your latest parts and units", "h2")
        col.addWidget(self.recent_title)
        self.recent = QHBoxLayout()
        col.addLayout(self.recent)
        self.hint = label("New here? Press “Show me an example” to see a finished unit being made, step by step. "
                          "Then make a soldier or a horse above, or build a unit from the starter parts.", "explain")
        col.addWidget(self.hint)
        wrap = QHBoxLayout()
        wrap.addStretch(1)
        box = QWidget()
        box.setLayout(col)
        box.setMaximumWidth(980)
        wrap.addWidget(box, 6)
        wrap.addStretch(1)
        self.layout_.addSpacing(24)
        self.layout_.addLayout(wrap)
        self.layout_.addStretch(1)
        self.layout_.addWidget(self.message)
        self.restyle()

    def enter(self) -> None:
        self.fill_recent()

    def picture(self, path: Path) -> None:
        """A picture dropped on Home: a wide one is a horse, else a soldier (the New part steps)."""
        from PIL import Image
        try:
            with Image.open(path) as im:
                w, h = im.size
        except OSError:
            return self.show_message(f"{Path(path).name} isn't a picture the app can read (PNG, JPG or WEBP).", error=True)
        if w > 1.15 * h:
            self.window_.go("new", kind="horse", image=path)
        else:
            self.window_.go("new", kind="soldier", image=path)

    def fill_recent(self) -> None:
        from eu4unitgen.gui.my_units import thumb_of
        from eu4unitgen.ref3d import flow
        clear_layout(self.recent)
        rs = flow.runs()[:4]
        self.recent_title.setVisible(bool(rs))
        for r in rs:
            from eu4unitgen.gui.progress import state_text
            b = card(r["name"].replace("_", " "), f"{r['kind']} · {state_text(r)}", "Open it in the Library",
                     picture=pixmap(thumb_of(r), 64, 64))
            b.setFixedWidth(260)
            b.clicked.connect(lambda _=False, f=r["folder"]: self.window_.go("units", folder=f))
            self.recent.addWidget(b)
        self.recent.addStretch(1)
