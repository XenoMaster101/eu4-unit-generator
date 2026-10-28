"""Help: the tour, the example (a finished unit, step by step, with its real results), the picture guide with good
and bad examples, answers to common questions, Check for updates (GitHub Releases), About and licences, the logs."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from eu4unitgen.gui.base import Page, Turntable, card, clear_layout, ghost, label, open_folder, pil_pixmap, pixmap, primary

FAQ = (
    ("How long does a unit take?", "About 15 to 30 minutes for a soldier or a horse (most of it is the AI making the "
     "3D model), a few minutes for a cavalry unit made from a horse and soldier you already have. You can close the app "
     "while it runs."),
    ("What is a tier?", "EU4 changes the look of a country's army four times as its military technology grows: tier 1 "
     "from the start, then at military tech 12, 21 and 26. Each unit you make fills one tier."),
    ("Soldier, horse, cavalry: what's the difference?", "A soldier is one person: on foot it's your infantry and the "
     "crew of your guns. A horse is just the horse. Cavalry puts one of your soldiers on one of your horses; you can mix "
     "and reuse them."),
    ("What makes a good picture?", "One subject on a plain background, the whole of it in frame: a soldier from the "
     "front with the arms a little out, a horse from the side. The picture guide shows good and bad examples."),
    ("What does “achievement-safe” mean?", "EU4 turns achievements off when a mod changes the game's rules. An "
     "achievement-safe mod only changes graphics: it re-dresses one graphical culture's unit (every country of that "
     "culture shows it) and leaves the rules alone."),
    ("What is a graphical culture?", "The family of looks the game gives a country's units, buildings and portraits "
     "(westerngfx, easterngfx, muslimgfx and so on)."),
    ("“Let each country paint its cloth” – what does it do?", "The cloth (tunic, caparison, shield face) takes each "
     "country's own unit colour in game, like the game's own units. Off, it keeps your picture's colours everywhere."),
    ("Windows says it blocked the app from the mod folder.", "That's Windows' Controlled Folder Access. The app saves "
     "the mod elsewhere and tells you exactly what to copy; or allow the app in Windows Security → Ransomware "
     "protection."),
    ("A run stopped or failed.", "Open it in My units: it says what happened and what to do, and Resume carries on "
     "from the last finished step. The 3D steps wait while EU4 is open (it holds the graphics card)."),
    ("Which pictures may I use?", "Your own, ones you made with an AI tool whose output you own, or ones licensed for "
     "this use. Don't use characters, uniforms or logos you don't have the rights to. Each unit keeps your rights "
     "statement."),
    ("Where are my files?", "Everything is in the app's one folder (chosen at install): your units in work\\projects, "
     "your pictures in work\\references, result sheets in work\\result sheets. Nothing of the game is ever copied."),
)


class HelpPage(Page):
    def __init__(self, window):
        super().__init__(window)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.addWidget(label("Help", "h1"))
        grid = QGridLayout()
        items = (("▶  Show me an example", "A finished unit (the cataphract), step by step, with its real results.",
                  self.show_example),
                 ("🎞  Take the tour", "The app's screens in two minutes, with captions. Can be saved as a video.",
                  self.show_tour),
                 ("🖼  The picture guide", "What makes a good picture for a soldier, a horse or a gun, with good and bad "
                  "examples.", self.show_guide),
                 ("⟳  Check for updates", "Looks for a newer version on GitHub Releases.", self.check_updates),
                 ("ⓘ  About and licences", "The version, the licences of every model and tool, what is installed.",
                  lambda: window.go("about")),
                 ("📁  Open the logs", "The app's log files, to send with a problem report.", self._logs))
        for i, (title, text, fn) in enumerate(items):
            c = card(title, text)
            c.clicked.connect(lambda _=False, f=fn: f())
            grid.addWidget(c, i // 3, i % 3)
        lay.addLayout(grid)
        self.update_text = label("", rich=True)
        lay.addWidget(self.update_text)
        lay.addWidget(label("Questions and answers", "h2"))
        for q, a in FAQ:
            lay.addWidget(label(f"<b>{q}</b><br>{a}", rich=True))
        theme = QHBoxLayout()
        theme.addWidget(label("Colours:", wrap=False))
        for key, text in (("system", "like Windows"), ("light", "light"), ("dark", "dark")):
            theme.addWidget(ghost(text, lambda k=key: window.set_theme(k), f"The app's colours: {text}"))
        theme.addStretch(1)
        lay.addLayout(theme)
        lay.addWidget(self.message)
        lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        self.layout_.addWidget(sc, 1)
        self.restyle()

    def show_example(self) -> None:
        ExampleDialog(self.window_).exec()

    def show_tour(self) -> None:
        from eu4unitgen.gui.tour import TourDialog
        TourDialog(self.window_).exec()

    def show_guide(self, kind: str = "soldier") -> None:
        GuideDialog(self.window_, kind).exec()

    def check_updates(self) -> None:
        from eu4unitgen.core import updates

        def done(u) -> None:
            link = f" <a href='{u.url}'>Open the download page</a>" if u.url else ""
            self.update_text.setText(f"<p><b>{u.text}</b>{link}</p>")
        self.window_.run(self, "Checking for updates", lambda: updates.check(), done)

    def _logs(self) -> None:
        from eu4unitgen.core import config
        d = config.logs_dir()
        d.mkdir(parents=True, exist_ok=True)
        open_folder(d)


class ExampleDialog(QDialog):
    """The example's steps (core.example), one per screen, with Back and Next."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from eu4unitgen.core import example
        self.setWindowTitle("Example: the cataphract, step by step")
        self.resize(1120, 780)
        lay = QVBoxLayout(self)
        self.step = label("", "step")
        self.title = label("", "h1")
        self.text = label("")
        self.text.setStyleSheet("font-size: 14px;")
        for w in (self.step, self.title, self.text):
            lay.addWidget(w)
        self.pics = QGridLayout()
        lay.addLayout(self.pics, 1)
        self.lines = label("", rich=True)
        lay.addWidget(self.lines)
        row = QHBoxLayout()
        self.prev = ghost("← Back", lambda: self.show_step(self.i - 1))
        self.next = primary("Next →", lambda: self.show_step(self.i + 1))
        row.addWidget(self.prev)
        row.addWidget(self.next)
        row.addStretch(1)
        row.addWidget(ghost("Close", self.accept))
        lay.addLayout(row)
        self.steps = example.steps() or []
        self.i = 0
        if not self.steps:
            self.title.setText("The example isn't on this computer")
            self.text.setText("The example is the cataphract made with this app: its horse, its soldier and the cavalry "
                              "unit joining them. Its pictures belong to the app's author, so they aren't included in "
                              "the download. Make your own unit instead: New unit in the top bar walks you through it, "
                              "and the tour (Help) shows every screen.")
            self.prev.hide()
            self.next.hide()
        else:
            self.show_step(0)

    def show_step(self, i: int) -> None:
        if not (0 <= i < len(self.steps)):
            return self.accept() if i >= len(self.steps) else None
        self.i = i
        s = self.steps[i]
        self.step.setText(f"STEP {i + 1} OF {len(self.steps)}")
        self.title.setText(s.title.split(". ", 1)[-1])
        self.text.setText(s.text)
        clear_layout(self.pics)
        if s.frames:
            t = Turntable(420)
            t.set_frames(s.frames)
            self.pics.addWidget(t, 0, 0, Qt.AlignCenter)
        n = max(1, len(s.images))
        size = 420 if n <= 2 else 300 if n <= 3 else 220
        for k, (caption, path) in enumerate(s.images):
            pic = label("")
            pic.setAlignment(Qt.AlignCenter)
            pic.setPixmap(pixmap(path, size, size))
            cap = label(caption, "explain")
            cap.setAlignment(Qt.AlignCenter)
            box = QVBoxLayout()
            box.addWidget(pic)
            box.addWidget(cap)
            cell = QWidget()
            cell.setLayout(box)
            cols = 2 if n <= 2 else 3 if n <= 6 else 4
            self.pics.addWidget(cell, k // cols, k % cols)
        self.lines.setText("".join(f"<p>{x}</p>" for x in s.lines if x))
        self.prev.setEnabled(i > 0)
        self.next.setText("Finish" if i == len(self.steps) - 1 else "Next →")


class GuideDialog(QDialog):
    """The picture guide: for each kind, what works, why, and good and bad examples."""

    def __init__(self, parent=None, kind: str = "soldier"):
        super().__init__(parent)
        from eu4unitgen.core import guide, guide_pictures
        self.setWindowTitle("The picture guide")
        self.resize(980, 720)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        for key, title in (("soldier", "A soldier"), ("horse", "A horse"), ("artillery", "A gun (coming)")):
            w = QWidget()
            tl = QVBoxLayout(w)
            tl.addWidget(label(guide.WHY.get(key, ""), "explain"))
            tl.addWidget(label("<ul>" + "".join(f"<li>{x}</li>" for x in guide.GUIDE[key] + guide.GUIDE["all"]) + "</ul>",
                               rich=True))
            grid = QGridLayout()
            for i, (good, caption, im) in enumerate(guide_pictures.examples(key)):
                pic = label("")
                pic.setPixmap(pil_pixmap(im, 160, 160))
                cap = label(("✓ " if good else "✗ ") + caption, "good" if good else "bad")
                cap.setMaximumWidth(180)
                box = QVBoxLayout()
                box.addWidget(pic)
                box.addWidget(cap)
                cell = QWidget()
                cell.setLayout(box)
                grid.addWidget(cell, i // 4, i % 4)
            tl.addLayout(grid)
            tl.addStretch(1)
            tabs.addTab(w, title)
            if key == kind:
                tabs.setCurrentWidget(w)
        lay.addWidget(tabs)
        lay.addWidget(ghost("Close", self.accept), 0, Qt.AlignRight)
