"""The Home screen: "What should your soldiers look like?". Two choices: Build a unit (from the library; works
without the AI) and Make a new part from a picture (needs the AI); the example and the tour; the latest parts and
units. Text to image stays on the roadmap only. Units are built in ``gui.builder``, parts in ``gui.new_unit``; the
old keyword builder is archived."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from eu4unitgen.gui.base import Page, card, clear_layout, ghost, label

CHOICES = (("build", "build", "Build a unit",
            "Infantry, cavalry or artillery, put together from your library tier by tier. The starter parts are "
            "already there, so this works without the AI.",
            "Pick the parts for each tier (e.g. tier 1 a light horse, tier 3 a barded one); a few minutes"),
           ("new", "part", "Make a new part",
            "From a picture (a soldier from the front, a horse from the side, a gun from three-quarter; needs the AI "
            "setup) or from a 3D model you import (.glb or .fbx; no AI needed). Made once, used in as many units as "
            "you like.",
            "The guided steps: your picture or model, its name and licence, then the app builds it and checks it "
            "(a few minutes for a model, 15 to 30 for a picture)"))


class HomeScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        col = QVBoxLayout()
        col.setSpacing(14)
        h1 = label("What should your soldiers look like?", "h1")
        h1.setAlignment(Qt.AlignCenter)
        sub = label("Make Europa Universalis IV units from your own pictures and 3D models: build them from a library "
                    "of parts, and add them to your mod.", "muted")
        sub.setAlignment(Qt.AlignCenter)
        col.addWidget(h1)
        col.addWidget(sub)
        col.addSpacing(6)
        from eu4unitgen.gui import icons
        from eu4unitgen.gui.theme import tokens
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        self.cards = {}
        for i, (key, ic, title, text, tip) in enumerate(CHOICES):
            c = card(title, text, tip, icon=ic)
            c.setMinimumHeight(118)
            c.clicked.connect(lambda _=False, k=key: window.go(k))
            grid.addWidget(c, 0, i)
            self.cards[key] = c
        col.addLayout(grid)
        self.build_card = self.cards["build"]
        strip = QHBoxLayout()
        strip.addStretch(1)
        self.steps = []
        for i, (ic, text) in enumerate((("part", "Make parts"), ("build", "Build a unit"), ("check", "Add to mod"))):
            if i:
                strip.addWidget(label("→", "muted", wrap=False))
            pic = QLabel()
            pic.setPixmap(icons.pixmap(ic, tokens()["accent_fg"], 18))
            strip.addWidget(pic)
            step = label(f"{i + 1}  {text}", "h3", wrap=False)
            self.steps.append(step)
            strip.addWidget(step)
        strip.addStretch(1)
        how = QWidget()
        how.setLayout(strip)
        how.setToolTip("Parts (horses, soldiers, guns) are made once from pictures; a unit is put together from them "
                       "tier by tier; Add to mod writes it into your EU4 mod.")
        col.addWidget(label("How it works", "step"), 0, Qt.AlignHCenter)
        col.addWidget(how)
        self.ai_note = label("", "explain")
        self.ai_note.setAlignment(Qt.AlignCenter)
        col.addWidget(self.ai_note)
        row = QHBoxLayout()
        row.addStretch(1)
        self.example = ghost("Show me an example", lambda: window.help_page.show_example(),
                             "Walks through a finished unit (the cataphract), step by step, with its real results")
        self.example.setIcon(icons.icon("example", tokens()["secondary"], 18))
        self.tour = ghost("Take the 2-minute tour", lambda: window.help_page.show_tour(),
                          "A short walk through the app's screens, with captions")
        self.tour.setIcon(icons.icon("tour", tokens()["secondary"], 18))
        row.addWidget(self.example)
        row.addWidget(self.tour)
        row.addStretch(1)
        col.addLayout(row)
        col.addSpacing(8)
        self.recent_title = label("Your latest parts and units", "h2")
        col.addWidget(self.recent_title)
        self.recent = QHBoxLayout()
        self.recent.setSpacing(10)
        col.addLayout(self.recent)
        wrap = QHBoxLayout()
        wrap.addStretch(1)
        box = QWidget()
        box.setLayout(col)
        box.setMaximumWidth(1000)
        wrap.addWidget(box, 6)
        wrap.addStretch(1)
        self.layout_.addSpacing(28)
        self.layout_.addLayout(wrap)
        self.layout_.addStretch(1)
        self.layout_.addWidget(self.message)
        self.restyle()

    def enter(self) -> None:
        from eu4unitgen.core import library
        ok, why = library.ai_ready()
        self.ai_note.setText("" if ok else why)
        self.ai_note.setVisible(not ok)
        self.fill_recent()

    def picture(self, path: Path) -> None:
        """A picture handed to Home (the tour, a file opened with the app): a wide one is a horse, else a soldier (the
        Make a part steps)."""
        from PIL import Image
        try:
            with Image.open(path) as im:
                w, h = im.size
        except OSError:
            return self.show_message(f"{Path(path).name} isn't a picture the app can read (PNG, JPG or WEBP).", error=True)
        self.window_.go("new", kind="horse" if w > 1.15 * h else "soldier", image=path)

    def fill_recent(self) -> None:
        from eu4unitgen.core import library
        from eu4unitgen.gui.base import thumb
        from eu4unitgen.ref3d import flow
        clear_layout(self.recent)
        rs = flow.runs()[:4]
        self.recent_title.setVisible(bool(rs))
        for r in rs:
            from eu4unitgen.gui.progress import state_text
            src = library.part_render(r) if r["kind"] in library.KINDS else library.run_thumb(r)
            b = card(library.display(r["name"]), "", "Open it in the Library", picture=thumb(src, 72, 64),
                     status=status_of(r, state_text(r)))
            b.setFixedWidth(240)
            b.setMinimumHeight(88)
            b.clicked.connect(lambda _=False, f=r["folder"]: self.window_.go("units", folder=f))
            self.recent.addWidget(b)
        self.recent.addStretch(1)


def status_of(r: dict, words: str) -> tuple[str, str]:
    """A run's status dot: green ready, amber being made or paused, red failed."""
    st = r.get("state", "new")
    if (r.get("status") or {}).get("done"):
        return "ok", "ready"
    return ("error" if st == "failed" else "warn"), words
