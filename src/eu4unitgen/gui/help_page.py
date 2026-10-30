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
    ("Can I use a 3D model instead of a picture?", "Yes: Make a part, then Import a 3D model (.glb preferred, or "
     ".fbx). No AI is needed. Give its licence (your own work, CC0, or CC BY with the credit, which the app writes "
     "into the mod) and pick the profile: standard for normal proportions, stylised for exaggerated ones (a big "
     "chibi head is kept whole and rigid). Rigged models fit best; the app fits the limbs to the game's skeleton."),
    ("What are “the checks before showing”?", "Every part is checked before you see it: the arms brought down from "
     "a T-pose, the limbs on the skeleton, weapons and shields rigid and in the hand, clipping and folds in the "
     "animations. It makes up to three passes, trying a fix between them, and tells you in plain words what's left."),
    ("Cavalry without a horse?", "Build a unit → Cavalry → On foot: each tier's soldier stands in as the cavalry, "
     "holding the base-game weapon you pick for that tier (a yari, a matchlock, a halberd, a sabre, a musket...) "
     "and playing the infantry animations that go with it."),
    ("What is a culture pack?", "A mod that adds a new graphical culture (e.g. mynationgfx) with your unit's whole "
     "set. Give it to any nation with graphical_culture = mynationgfx in its country file. Add to mod → “A culture "
     "pack”; the mod's README.txt lists every file and how to install it."),
    ("What's in an exported mod?", "Every export writes a README.txt into the mod: each folder and file, what the "
     "game uses it for, and how to install it. Add to mod shows the same text under “What's in it”."),
    ("How do I update the app?", "Help → Check for updates. If there's a newer version it shows what changed, and "
     "Download and install updates the app in place: your units, settings, library and downloads stay. No "
     "uninstalling."),
    ("Windows said “Windows protected your PC”.", "The installer isn't code-signed (that needs a paid yearly "
     "certificate). Click More info → Run anyway. The release page's SHA256SUMS.txt lets you check the download. "
     "Updates installed from inside the app don't show the warning."),
    ("What does the app download?", "Setup's first page lists every download: what it is, why it's needed, its "
     "size, licence, where it comes from and where it goes. Only Blender and io_pdx_mesh are required; the AI "
     "(about 40 GB) is only for making parts from your own pictures."),
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
    ("A run stopped or failed.", "Open it in the Library: it says what happened and what to do, and Resume carries on "
     "from the last finished step. The 3D steps wait while EU4 is open (it holds the graphics card)."),
    ("Which pictures and models may I use?", "Your own, ones you made with an AI tool whose output you own, or ones "
     "licensed for this use (CC0, or CC BY with its credit). Each part keeps its source, author and licence, and a "
     "CC BY credit is written into the mod for you."),
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
        items = (("example", "Show me an example", "A finished unit (the cataphract), step by step, with its real "
                  "results.", self.show_example),
                 ("tour", "Take the tour", "The app's screens in two minutes, with captions. Can be saved as a video.",
                  self.show_tour),
                 ("part", "The picture guide", "What makes a good picture for a soldier, a horse or a gun, with good and "
                  "bad examples.", self.show_guide),
                 ("refresh", "Check for updates", "Looks for a newer version on the app's GitHub page, shows what "
                  "changed, and installs it in place: your units, settings and downloads stay.", self.check_updates),
                 ("info", "About and licences", "The version, the licences of every model and tool, what is installed.",
                  lambda: window.go("about")),
                 ("folder", "Open the logs", "The app's log files, to send with a problem report.", self._logs))
        for i, (ic, title, text, fn) in enumerate(items):
            c = card(title, text, icon=ic)
            c.clicked.connect(lambda _=False, f=fn: f())
            grid.addWidget(c, i // 3, i % 3)
        lay.addLayout(grid)
        self.update_text = label("", rich=True)
        self.update_text.setOpenExternalLinks(True)
        lay.addWidget(self.update_text)
        self.update_btn = primary("Download and install the update", self.install_update,
                                  "Downloads the new version (its checksum is checked), closes the app, updates it in "
                                  "place and starts it again")
        self.update_btn.setVisible(False)
        lay.addWidget(self.update_btn, 0, Qt.AlignLeft)
        self._update = None
        lay.addWidget(label("Questions and answers", "h2"))
        for q, a in FAQ:
            lay.addWidget(label(f"<b>{q}</b><br>{a}", rich=True))
        theme = QHBoxLayout()
        theme.addWidget(label("Design:", wrap=False))
        for key, text in (("campaign", "Campaign (charcoal and gold)"), ("admiralty", "Admiralty (navy and bronze)")):
            theme.addWidget(ghost(text, lambda k=key: window.set_theme(k), f"The app's look: {text}"))
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
            import html
            link = f" <a href='{u.url}'>Open the release page</a>" if u.url else ""
            notes = ""
            if u.newer and u.notes:
                notes = ("<p><b>What's new:</b></p><pre style='white-space: pre-wrap'>"
                         f"{html.escape(u.notes[:4000])}</pre>")
            size = f" ({u.setup_size / 2 ** 20:.0f} MB)" if u.setup_size else ""
            self.update_text.setText(f"<p><b>{html.escape(u.text)}</b>{link}</p>{notes}")
            self._update = u if (u.newer and u.setup_url and u.sums_url) else None
            self.update_btn.setText(f"Download and install {u.latest.lstrip('v')}{size}")
            self.update_btn.setVisible(self._update is not None)
        self.window_.run(self, "Checking for updates", lambda: updates.check(), done)

    def install_update(self) -> None:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from eu4unitgen.core import updates
        u = self._update
        if u is None:
            return
        got = [0, u.setup_size]

        def progress(done_, total) -> None:  # from the download thread: only numbers, read by the timer below
            got[0], got[1] = done_, total or got[1]

        timer = QTimer(self)

        def tick() -> None:
            if not self.window_._busy:  # finished or failed (a failure's message stays)
                return timer.stop()
            self.show_message(f"Downloading {u.setup_name}: {got[0] / 2 ** 20:.0f}"
                              + (f" of {got[1] / 2 ** 20:.0f} MB" if got[1] else " MB"))
        timer.timeout.connect(tick)
        timer.start(500)

        def done(path) -> None:
            timer.stop()
            try:
                updates.launch(path)
            except OSError as exc:
                return self.show_message(str(exc), error=True)
            self.show_message("Checked. The app closes now; the installer updates it and starts it again.")
            QTimer.singleShot(1500, QApplication.quit)
        self.window_.run(self, "Downloading the update", lambda: updates.download(u, progress=progress), done)

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
            # the pictures come with the app; the results (renders of the game's own bodies) are made here
            self.steps = [example.Step(
                "Make the example yourself",
                "The example is the cataphract: a horse, a soldier and the cavalry unit joining them. Its pictures "
                "(below) come with the app, but its results are made on your computer. Press New unit, choose Soldier "
                "and press “Use the example picture” on the first step; then do the same for a Horse, and last make "
                "Cavalry from the two. Once they're made, this example walks through every step with their real "
                "results.",
                [(c, str(p)) for k in ("soldier", "horse") for c, p in example.pictures(k)])]
            self.show_step(0)
            self.step.setText("")
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
