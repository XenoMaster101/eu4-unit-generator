"""New unit: a guided flow, one action per screen ("Step 2 of 4"), each with what happens and why.

  soldier or horse   1 the picture (the guide for its kind: what works, why, good and bad examples, the picture's own
                     checks), or a 3D model to import (.glb or .fbx: ``ref3d.importer``; it takes TRELLIS.2's place)
                     2 name and rights (an imported model: its licence, CC0, CC BY with its credit, or own work)   3 where it goes (the tier, the country colour)   4 make it (the
                     checks, then the run's stages as they go)
  cavalry            1 the horse   2 the rider (each picked from the Library, or made here: that runs the horse's or
                     soldier's own steps, then comes back)   3 name and tier   4 make it (it waits for the horse and
                     soldier to be ready by itself)

Thin by design: the picture checks are ``core.guide``, the run is ``ref3d.flow`` (detached: closing the app doesn't
stop it), the checks before a run are ``core.checks``.
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from eu4unitgen.gui.base import DropImage, Page, card, clear_layout, ghost, label, pil_pixmap, pixmap, primary
from eu4unitgen.gui.progress import StageList

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
RIGHTS_EXAMPLE = "Original character I made with ChatGPT; I own the output"
TIERS = ((1, "Tier 1", "From the start (1444); the game's first look"),
         (2, "Tier 2", "From military tech 12 (about 1557)"),
         (3, "Tier 3", "From military tech 21 (about 1674)"),
         (4, "Tier 4", "From military tech 26 (about 1745); the last look"))
WORD = {"soldier": "soldier", "horse": "horse", "cavalry": "cavalry unit", "gun": "gun"}
FLOWS = {
    "soldier": (("picture", "Your picture", "Drop in a picture of one soldier."),
                ("about", "Name and rights", "Give it a name and say where the picture comes from."),
                ("slot", "Where it goes", "Choose when it appears in game, and its colours."),
                ("make", "Make it", "Check everything is ready, then let the app build it.")),
    "horse": (("picture", "Your picture", "Drop in a picture of the horse."),
              ("about", "Name and rights", "Give it a name and say where the picture comes from."),
              ("slot", "Where it goes", "Choose when it appears in game, and its colours."),
              ("make", "Make it", "Check everything is ready, then let the app build it.")),
    "gun": (("picture", "Your picture", "Drop in a picture of the gun, from three-quarter."),
            ("about", "Name and rights", "Give it a name and say where the picture comes from."),
            ("slot", "Where it goes", "Choose when it appears in game."),
            ("make", "Make it", "Check everything is ready, then let the app build it.")),
    "cavalry": (("horse", "The horse", "Pick the horse, or make a new one from a picture."),
                ("rider", "The rider", "Pick the soldier who rides it, or make a new one."),
                ("slot", "Name and tier", "Name the unit and choose when it appears in game."),
                ("make", "Make it", "The app puts the soldier on the horse and checks the result.")),
}


def suggest_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", Path(path).stem.lower()).strip("_")[:40] or "unit"


def name_problem(name: str) -> str:
    from eu4unitgen.ref3d import flow
    if not name:
        return "Give it a name."
    if not NAME.match(name):
        return "Use lower-case letters, digits and _ only, starting with a letter (e.g. royal_guard)."
    if flow.folder_of(name).exists():
        return f"You already have a unit called {name}: pick another name."
    return ""


class NewUnitScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        self.kind = "soldier"
        self.image: Path | None = None
        self.model: Path | None = None  # Import a model: the .glb or .fbx in place of a picture
        self.started: str | None = None  # the run started from the Make it step
        self.parent_draft: dict | None = None  # making a horse or soldier for a cavalry unit: where to go back to
        self.picks = {"horse": None, "soldier": None}
        self.i = 0
        head = QVBoxLayout()
        self.step_label = label("", "step")
        self.title = label("", "h1")
        self.explain = label("", "explain")
        head.addWidget(self.step_label)
        head.addWidget(self.title)
        head.addWidget(self.explain)
        self.layout_.addLayout(head)
        self.stack = QStackedWidget()
        self.steps = {"picture": self._picture_step(), "about": self._about_step(), "slot": self._slot_step(),
                      "make": self._make_step(), "horse": self._pick_step("horse"), "rider": self._pick_step("soldier")}
        for w in self.steps.values():
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(w)
            self.stack.addWidget(sc)
            w._scroll = sc
        self.layout_.addWidget(self.stack, 1)
        foot = QHBoxLayout()
        self.back = ghost("← Back", self.go_back, "The step before (nothing is lost)")
        foot.addWidget(self.back)
        foot.addWidget(self.message, 1)
        self.next_why = label("", "hint")
        foot.addWidget(self.next_why)
        self.next = primary("Next →", self.go_next)
        foot.addWidget(self.next)
        self.layout_.addLayout(foot)
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self._poll)
        self.restyle()
        self._show()

    # ------------------------------------------------------------------ the flow
    def enter(self, kind: str | None = None, image: Path | None = None, horse: str | None = None,
              soldier: str | None = None, keep: bool = False) -> None:
        """Start a new draft of ``kind`` (a soldier when none is given and nothing is going on), optionally with its
        picture, or a cavalry unit with its horse or soldier already picked."""
        if keep or (kind is None and (self.image or self.started or self.parent_draft)):
            return self._show()
        self.kind = kind or "soldier"
        self.image, self.started, self.i = None, None, 0
        self.model = None
        self.model_name.setText("No model chosen yet.")
        self._set_how(False)
        self.licence.setCurrentIndex(0)
        self.credit.setText("")
        self.picks = {"horse": horse, "soldier": soldier}
        if kind in (None, "soldier", "horse", "cavalry") and self.parent_draft and kind != "cavalry":
            pass  # a horse or soldier made for a cavalry unit keeps its way back
        else:
            self.parent_draft = None
        self.name.setText("")
        self.rights.setText("")
        self.regions.setText("")
        self.style.setCurrentIndex(0)
        self.mask.setChecked(True)
        self.tier_group.button(1).setChecked(True)
        self.drop.show_image(None)
        self.checks_out.setText("")
        if image:
            self.set_picture(Path(image))
        if self.kind == "cavalry":
            self.i = 2 if horse and soldier else 1 if horse else 0
        self._show()

    def flow(self):
        return FLOWS[self.kind]

    def _show(self) -> None:
        key, title, explain = self.flow()[self.i]
        n = len(self.flow())
        what = WORD[self.kind]
        sub = f" (for the cavalry unit {self.parent_draft['name']})" if self.parent_draft and self.parent_draft.get("name") else (
            " (for a cavalry unit)" if self.parent_draft else "")
        part = "NEW PART: " if self.kind in ("soldier", "horse", "gun") else "NEW "
        self.step_label.setText(f"{part}{what.upper()}{sub.upper()}  ·  STEP {self.i + 1} OF {n}")
        self.title.setText(title)
        self.explain.setText(explain + " " + self._why(key))
        self.stack.setCurrentWidget(self.steps[key]._scroll)
        self.back.setEnabled(self.i > 0 or self.parent_draft is not None)
        self.back.setText("← Back" if self.i > 0 or not self.parent_draft else "← Back to the cavalry unit")
        getattr(self, f"_enter_{key}", lambda: None)()
        self._ready()

    def _why(self, key: str) -> str:
        from eu4unitgen.core import guide
        return {"picture": guide.WHY.get(self.kind, ""),
                "about": "The name is used for its folder and its name in game. The rights statement is kept with the "
                         "unit: only use pictures you made or may use.",
                "slot": "EU4 changes how a country's army looks four times as its military technology grows; each look "
                        "is a tier.",
                "make": "This takes a while (about 15 to 30 minutes). You can close the app: it keeps going.",
                "horse": "Cavalry is a horse with a soldier on it, made separately so you can mix and reuse them.",
                "rider": "Any of your soldiers can ride. The app fits them to this horse."}.get(key, "")

    def _ready(self) -> None:
        """Next is on only when this step is done; the reason shows beside it."""
        key = self.flow()[self.i][0]
        why = ""
        if key == "picture" and self.importing() and self.model is None:
            why = "Choose a model file first."
        elif key == "picture" and not self.importing() and self.image is None:
            why = "Drop in a picture first."
        elif key == "about" and self.importing():
            from eu4unitgen.ref3d import importer
            why = name_problem(self.name.text().strip()) or importer.problem(self.model, self.licence.currentData(),
                                                                             self.credit.text())
        elif key == "about":
            why = name_problem(self.name.text().strip()) or ("Say where the picture comes from (the rights)."
                                                             if not self.rights.text().strip() else "")
        elif key == "horse" and not self.picks["horse"]:
            why = "Pick a horse (or make one)."
        elif key == "rider" and not self.picks["soldier"]:
            why = "Pick a soldier (or make one)."
        elif key == "slot" and self.kind == "cavalry":
            why = name_problem(self.name.text().strip())
        self.next_why.setText(why)
        last = self.i == len(self.flow()) - 1
        self.next.setVisible(not last)
        self.next.setEnabled(not why)

    def go_next(self) -> None:
        if self.i < len(self.flow()) - 1:
            self.i += 1
            self._show()

    def go_back(self) -> None:
        if self.i > 0:
            self.i -= 1
            return self._show()
        if self.parent_draft:
            self._return_to_parent(None)

    # ------------------------------------------------------------------ step 1: the picture
    def _picture_step(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        kinds = QHBoxLayout()
        self.kind_group = QButtonGroup(self)
        self.kind_cards = {}
        for key, ic, title, text in (("soldier", "soldier", "A soldier", "From the front, arms a little out"),
                                     ("horse", "horse", "A horse", "From the side"),
                                     ("gun", "gun", "A gun", "From three-quarter")):
            c = card(title, text, f"Make {title.lower()} part from a picture", checkable=True, icon=ic)
            c.setMaximumHeight(72)
            c.clicked.connect(lambda _=False, k=key: self._switch_kind(k))
            self.kind_group.addButton(c)
            self.kind_cards[key] = c
            kinds.addWidget(c)
        outer.addLayout(kinds)
        how = QHBoxLayout()
        self.how_group = QButtonGroup(self)
        self.from_picture = card("From a picture", "The AI makes the 3D model", "Make it from one picture (TRELLIS.2)",
                                 checkable=True)
        self.from_model = card("Import a 3D model", ".glb (best) or .fbx: CC0, CC BY or your own",
                               "Use a 3D model you made or may use, in place of the AI's", checkable=True)
        for c in (self.from_picture, self.from_model):
            c.setMaximumHeight(64)
            self.how_group.addButton(c)
            how.addWidget(c)
        self.from_picture.setChecked(True)
        self.from_picture.clicked.connect(lambda *_: self._set_how(False))
        self.from_model.clicked.connect(lambda *_: self._set_how(True))
        outer.addLayout(how)
        self.model_box = QWidget()
        mb = QVBoxLayout(self.model_box)
        mb.setContentsMargins(0, 0, 0, 0)
        pick = QHBoxLayout()
        pick.addWidget(primary("Choose a model file…", self.choose_model, "A .glb (textures inside it) or an .fbx"))
        self.model_name = label("No model chosen yet.", "explain")
        pick.addWidget(self.model_name, 1)
        mb.addLayout(pick)
        mb.addWidget(label("What works", "h2"))
        mb.addWidget(label("<ul><li>One model of the whole thing: a soldier standing with his arms a little out from his "
                           "sides, a horse standing, or a gun.</li><li>His weapon, shield and helmet can be part of "
                           "it: the app splits them off as it does for the AI's models.</li><li>A .glb with its "
                           "textures inside is best; an .fbx's textures must sit beside it. A rig and animations are "
                           "dropped: the app rigs it on the game's skeleton.</li><li>Any size and any number of "
                           "pieces or materials: they're joined and the colours put on one texture.</li><li>Only a "
                           "model you made, or one under CC0 or CC BY (you give its credit next).</li></ul>",
                           rich=True))
        mb.addStretch(1)
        self.model_box.hide()
        outer.addWidget(self.model_box, 1)
        self.picture_box = QWidget()
        row = QHBoxLayout(self.picture_box)
        row.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.picture_box, 1)
        left = QVBoxLayout()
        self.drop = DropImage(w=380, h=380)
        self.drop.picked.connect(self.set_picture)
        left.addWidget(self.drop)
        self.example_row = QHBoxLayout()
        left.addLayout(self.example_row)
        self.checks_out = label("", rich=True)
        left.addWidget(self.checks_out)
        left.addStretch(1)
        row.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(label("What works", "h2"))
        self.rules = label("", rich=True)
        right.addWidget(self.rules)
        right.addWidget(label("Good and bad pictures", "h2"))
        self.examples = QGridLayout()
        right.addLayout(self.examples)
        right.addStretch(1)
        row.addLayout(right, 1)
        return w

    def _switch_kind(self, kind: str) -> None:
        """What the part is: its guide, checks and steps follow (the picture, if any, is checked again)."""
        image, model, importing = self.image, self.model, self.from_model.isChecked()
        self.enter(kind=kind)
        if image:
            self.set_picture(image)
        if importing:
            self._set_how(True)
            if model:
                self.set_model(model)

    def importing(self) -> bool:
        return self.from_model.isChecked() and self.kind in ("soldier", "horse", "gun")

    def _set_how(self, model: bool) -> None:
        """From a picture or from an imported model: the step shows the one asked for."""
        self.from_model.setChecked(model)
        self.from_picture.setChecked(not model)
        self.model_box.setVisible(model)
        self.picture_box.setVisible(not model)
        self.mask.setChecked(not model)  # an imported model keeps its own colours (1:1) unless asked
        if hasattr(self, "next_why"):
            self._ready()

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose your 3D model", "", "3D models (*.glb *.fbx)")
        if path:
            self.set_model(Path(path))

    def set_model(self, path: Path) -> None:
        from eu4unitgen.ref3d import importer
        why = importer.problem(Path(path), "own")
        if why:
            return self.show_message(why, error=True)
        self.model = Path(path)
        mb = self.model.stat().st_size / 1e6
        self.model_name.setText(f"✓ {self.model.name} ({mb:.1f} MB)")
        if not self.name.text().strip():
            self.name.setText(suggest_name(self.model))
        meta = importer.metadata(self.model)  # a suggestion from the file itself (Sketchfab writes its licence)
        if meta.get("licence"):
            self.licence.setCurrentIndex(self.licence.findData(meta["licence"]))
        if meta.get("credit") and not self.credit.text().strip():
            self.credit.setText(meta["credit"])
        self.show_message("")
        self._ready()

    def _enter_picture(self) -> None:
        from eu4unitgen.core import example, guide, guide_pictures
        self.rules.setText("<ul>" + "".join(f"<li>{x}</li>" for x in guide.GUIDE[self.kind] + guide.GUIDE["all"]) + "</ul>")
        if getattr(self, "_examples_kind", None) == self.kind:
            return
        self._examples_kind = self.kind
        for k, c in self.kind_cards.items():
            c.setChecked(k == self.kind)
            c.setVisible(not self.parent_draft)  # a horse or soldier for a cavalry run stays what it is
        clear_layout(self.example_row)
        for caption, path in example.pictures(self.kind):
            self.example_row.addWidget(ghost("Use the example picture" if len(example.pictures(self.kind)) == 1
                                             else f"Example: {caption.split(':')[0].replace('the ', '', 1)}",
                                             lambda *_, p=path: self.use_example(p),
                                             f"Try the app with {caption} (comes with the app; its rights are filled "
                                             "in for you)"))
        self.example_row.addStretch(1)
        clear_layout(self.examples)
        for i, (good, caption, im) in enumerate(guide_pictures.examples(self.kind)):
            pic = label("")
            pic.setPixmap(pil_pixmap(im, 120, 120))
            pic.setToolTip(caption)
            cap = label(("✓ " if good else "✗ ") + caption, "good" if good else "bad")
            cap.setMaximumWidth(140)
            box = QVBoxLayout()
            box.addWidget(pic)
            box.addWidget(cap)
            cell = QWidget()
            cell.setLayout(box)
            self.examples.addWidget(cell, i // 3, i % 3)

    def use_example(self, path: Path) -> None:
        from eu4unitgen.core import example
        self.set_picture(path)
        self.rights.setText(example.RIGHTS)

    def set_picture(self, path: Path) -> None:
        from PIL import Image

        from eu4unitgen.core import guide
        try:
            with Image.open(path) as im:
                im.load()
        except OSError:
            return self.show_message(f"{Path(path).name} isn't a picture the app can read (PNG, JPG or WEBP).", error=True)
        self.image = Path(path)
        self.drop.show_image(self.image)
        warn = guide.check(self.image, self.kind)
        view = guide.VIEW.get(self.kind, "")
        self.checks_out.setText(
            "".join(f"<p style='color:{self._tok('warn')}'>⚠ {x}</p>" for x in warn) +
            ("<p>You can still go on; the result may be worse. Press Back on the next screen to change it.</p>" if warn
             else f"<p style='color:{self._tok('ok')}'>✓ This looks like a good {view}.</p>"))
        if not self.name.text().strip():
            self.name.setText(suggest_name(self.image))
        self.show_message("")
        self._ready()

    def _tok(self, k: str) -> str:
        from eu4unitgen.gui.theme import tokens
        return tokens()[k]

    # ------------------------------------------------------------------ step 2: name and rights
    def _about_step(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Name", "h2"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. royal_guard")
        self.name.setToolTip("Lower-case letters, digits and _ only. Used for its folder and, made readable, its name in "
                             "game (royal_guard → “Royal Guard Infantry”).")
        self.name.textChanged.connect(lambda *_: self._name_changed())
        lay.addWidget(self.name)
        self.name_hint = label("", "explain")
        lay.addWidget(self.name_hint)
        self.licence_box = QWidget()
        lb = QVBoxLayout(self.licence_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.addWidget(label("The model's licence", "h2"))
        from eu4unitgen.ref3d import importer
        self.licence = QComboBox()
        self.licence.addItem("Choose…", None)
        for k, text in importer.LICENCES.items():
            self.licence.addItem(text, k)
        self.licence.setToolTip("Only a model you made, or one its author released under CC0 or CC BY. It's saved with "
                                "the part; a CC BY credit is written into your mod with it.")
        self.licence.currentIndexChanged.connect(lambda *_: self._ready())
        lb.addWidget(self.licence)
        self.credit = QLineEdit()
        self.credit.setPlaceholderText("Credit: who made it and where it's from, e.g. \u201cKnight by Jane Doe, "
                                       "sketchfab.com/\u2026\u201d")
        self.credit.setToolTip("Needed for CC BY (the licence asks for it); for CC0 and your own work it's optional")
        self.credit.textChanged.connect(lambda *_: self._ready())
        lb.addWidget(self.credit)
        lb.addWidget(label("Don't import models of characters, uniforms or logos you don't have the rights to.",
                           "explain"))
        self.licence_box.hide()
        lay.addWidget(self.licence_box)
        self.rights_box = QWidget()
        rbx = QVBoxLayout(self.rights_box)
        rbx.setContentsMargins(0, 0, 0, 0)
        rbx.addWidget(label("Where does the picture come from? (rights)", "h2"))
        self.rights = QLineEdit()
        self.rights.setPlaceholderText(f"e.g. {RIGHTS_EXAMPLE}")
        self.rights.setToolTip("Who made the picture, and that you may use it in your mod. It's saved with the unit. "
                               "Don't use pictures of characters, uniforms or logos you don't have the rights to.")
        self.rights.textChanged.connect(lambda *_: self._ready())
        rbx.addWidget(self.rights)
        ex = QHBoxLayout()
        ex.addWidget(label(f"Example: “{RIGHTS_EXAMPLE}”", "explain"))
        ex.addWidget(ghost("Use this wording", lambda: self.rights.setText(RIGHTS_EXAMPLE),
                           "Only if it's true for your picture"))
        ex.addStretch(1)
        rbx.addLayout(ex)
        rbx.addWidget(label("Other good answers: “Drawn by me”, “Photo I took myself”, “Licensed CC-BY 4.0 from "
                            "<artist>, credited in the mod”.", "explain"))
        lay.addWidget(self.rights_box)
        from eu4unitgen.core import library
        lay.addWidget(label("Tags (optional)", "h2"))
        lay.addWidget(label("They decide which parts the app suggests when you build a unit for a country. You can "
                            "change them later in the Library.", "explain"))
        self.style_row = QWidget()
        sr = QHBoxLayout(self.style_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(label("Style", wrap=False))
        self.style = QComboBox()
        self.style.addItem("no style", None)
        for s in library.STYLES:
            self.style.addItem(library.STYLE_TEXT[s], s)
        self.style.setToolTip("What kind of horse it is. A country's graphical culture suggests a style (westerngfx: "
                              "barded first; muslimgfx: Arab; a nomad: steppe).")
        sr.addWidget(self.style)
        sr.addStretch(1)
        lay.addWidget(self.style_row)
        self.regions = QLineEdit()
        self.regions.setPlaceholderText("Regions or peoples it fits, e.g. Byzantine, Greek")
        self.regions.setToolTip("Separated by commas. A country whose culture or culture group matches gets this part "
                                "suggested first.")
        lay.addWidget(self.regions)
        lay.addStretch(1)
        return w

    def _enter_about(self) -> None:
        self.style_row.setVisible(self.kind == "horse")  # a style is a horse's; regions are any part's
        self.licence_box.setVisible(self.importing())
        self.rights_box.setVisible(not self.importing())

    def _name_changed(self) -> None:
        from eu4unitgen.ref3d.unitstage import unit_names
        n = self.name.text().strip()
        bad = name_problem(n)
        self.name.setProperty("bad", "true" if bad else "false")
        self.name.style().unpolish(self.name)
        self.name.style().polish(self.name)
        if bad:
            self.name_hint.setText(bad)
        else:
            from eu4unitgen.ref3d import flow
            what = flow.FOOT_TYPE.get(self.kind, "cavalry")
            self.name_hint.setText(f"In game: “{unit_names(n)[what]}”.")
        self._ready()

    # ------------------------------------------------------------------ step 3: where it goes
    def _slot_step(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.cav_name_box = QWidget()
        cl = QVBoxLayout(self.cav_name_box)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(label("Name", "h2"))
        self.cav_name = QLineEdit()
        self.cav_name.setToolTip("Lower-case letters, digits and _ only")
        self.cav_name.textChanged.connect(lambda t: (self.name.setText(t), self._ready()))
        cl.addWidget(self.cav_name)
        lay.addWidget(self.cav_name_box)
        lay.addWidget(label("When does it appear in game?", "h2"))
        grid = QGridLayout()
        self.tier_group = QButtonGroup(self)
        for i, (t, title, text) in enumerate(TIERS):
            b = card(title, text, f"{title}: {text}. Each country's army changes its look at these points.", checkable=True)
            self.tier_group.addButton(b, t)
            grid.addWidget(b, 0, i)
        self.tier_group.button(1).setChecked(True)
        lay.addLayout(grid)
        self.use_note = label("", "explain")
        lay.addWidget(self.use_note)
        lay.addWidget(label("Colours", "h2"))
        self.mask = QCheckBox("Let each country paint its cloth in its own colour")
        self.mask.setChecked(True)
        self.mask.setToolTip("The cloth (tunic, caparison, shield face) takes each country's unit colour in game, like "
                             "the game's own units. Off: it keeps the picture's colours everywhere.")
        lay.addWidget(self.mask)
        lay.addStretch(1)
        return w

    def _enter_slot(self) -> None:
        cav = self.kind == "cavalry"
        self.cav_name_box.setVisible(cav)
        self.mask.setVisible(not cav)
        if cav and not self.cav_name.text().strip() and self.picks["soldier"] and self.picks["horse"]:
            self.cav_name.setText(f"{self.picks['soldier']}_on_{self.picks['horse']}"[:40])
        self.use_note.setText({
            "soldier": "On foot it is your infantry at this tier (and the crew of your guns). From the Library you can "
                       "later put it on a horse.",
            "horse": "It replaces the game's horse at this tier; the game's own rider sits on it until you give it one "
                     "of your soldiers (Cavalry on Home).",
            "cavalry": "The cavalry at this tier: your horse with your soldier riding."}[self.kind])

    # ------------------------------------------------------------------ cavalry steps 1-2: pick a horse, a soldier
    def _pick_step(self, kind: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grid = QGridLayout()
        lay.addLayout(grid)
        new = ghost(f"Make a new {kind} from a picture", lambda: self._make_child(kind),
                    f"Runs the {kind}'s own steps (picture, name, make it), then brings you back here")
        lay.addWidget(new, 0, Qt.AlignLeft)
        empty = label("", "explain")
        lay.addWidget(empty)
        lay.addStretch(1)
        w._grid, w._group, w._empty, w._kind = grid, QButtonGroup(w), empty, kind
        return w

    def _fill_pick(self, w) -> None:
        from eu4unitgen.gui.my_units import thumb_of
        from eu4unitgen.gui.progress import state_text
        from eu4unitgen.ref3d import flow
        clear_layout(w._grid)
        for b in w._group.buttons():
            w._group.removeButton(b)
        rs = [r for r in flow.runs() if r["kind"] == w._kind and r["state"] != "failed"]
        for i, r in enumerate(rs):
            b = card(" ".join(w.capitalize() for w in r["name"].split("_")), state_text(r), "Pick this one", checkable=True,
                     picture=pixmap(thumb_of(r), 96, 96))
            b.setFixedWidth(360)
            b.setChecked(r["name"] == self.picks[w._kind])
            b.clicked.connect(lambda _=False, n=r["name"]: self._pick(w._kind, n))
            w._group.addButton(b)
            w._grid.addWidget(b, i // 3, i % 3)
        w._empty.setText("" if rs else f"You have no {w._kind} yet: make one from a picture.")

    def _enter_horse(self) -> None:
        self._fill_pick(self.steps["horse"])

    def _enter_rider(self) -> None:
        self._fill_pick(self.steps["rider"])

    def _pick(self, kind: str, name: str) -> None:
        self.picks[kind] = name
        self._ready()

    def _make_child(self, kind: str) -> None:
        """A horse or soldier made for this cavalry unit: its own steps, then back here with it picked."""
        self.parent_draft = {"picks": dict(self.picks), "i": self.i, "name": self.cav_name.text().strip(),
                             "tier": self.tier_group.checkedId(), "child": kind}
        parent = self.parent_draft
        self.enter(kind=kind)
        self.parent_draft = parent
        self._show()

    def _return_to_parent(self, made: str | None) -> None:
        p = self.parent_draft
        self.parent_draft = None
        self.kind, self.image, self.started = "cavalry", None, None
        self.picks = dict(p["picks"])
        if made:
            self.picks[p["child"]] = made
        self.i = p["i"] + (1 if made else 0)
        self.cav_name.setText(p.get("name") or "")
        self.tier_group.button(p.get("tier") or 1).setChecked(True)
        self._show()

    # ------------------------------------------------------------------ step 4: make it
    def _make_step(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.summary = label("", rich=True)
        lay.addWidget(self.summary)
        self.checks_box = label("", rich=True)
        lay.addWidget(self.checks_box)
        row = QHBoxLayout()
        self.go = primary("Make it", self.start, "Starts it; it keeps going if you close the app (about 15 to 30 "
                                                 "minutes)")
        row.addWidget(self.go)
        self.recheck = ghost("Check again", self._run_checks, "Run the checks again (after closing EU4, freeing space…)")
        row.addWidget(self.recheck)
        row.addStretch(1)
        lay.addLayout(row)
        self.stages = StageList()
        self.stages.hide()
        lay.addWidget(self.stages)
        after = QHBoxLayout()
        self.see = primary("See it in the Library →", lambda: self.window_.go("units", folder=self.started),
                           "Its page: the progress now, then its preview, report, edits and Add to mod")
        self.see.hide()
        after.addWidget(self.see)
        self.back_to_parent = primary("Back to the cavalry unit →", lambda: self._return_to_parent(Path(self.started).name),
                                      "It keeps being made; the cavalry unit waits for it by itself")
        self.back_to_parent.hide()
        after.addWidget(self.back_to_parent)
        after.addStretch(1)
        lay.addLayout(after)
        lay.addStretch(1)
        return w

    def _enter_make(self) -> None:
        what = WORD[self.kind]
        self.go.setText(f"Make my {what}")
        tier = self.tier_group.checkedId()
        if self.kind == "cavalry":
            body = (f"<b>{self.name.text().strip()}</b>: the soldier <b>{self.picks['soldier']}</b> riding the horse "
                    f"<b>{self.picks['horse']}</b>, cavalry tier {tier}.")
        elif self.importing():
            body = (f"<b>{self.name.text().strip()}</b>: a {what} from your 3D model <b>{self.model.name if self.model else '?'}"
                    f"</b> ({self.licence.currentText()}), tier {tier}"
                    f"{', its cloth in each country’s colour' if self.mask.isChecked() else ''}.")
        else:
            body = (f"<b>{self.name.text().strip()}</b>: a {what} from <b>{self.image.name if self.image else '?'}</b>, "
                    f"tier {tier}{', its cloth in each country’s colour' if self.mask.isChecked() else ''}.")
        self.summary.setText(f"<h3>You're making</h3>{body}")
        if self.started:
            self.stages.show()
            self._poll()
            self.timer.start()
        else:
            self.stages.hide()
            self.see.hide()
            self.back_to_parent.hide()
            self.go.setEnabled(True)
            self._run_checks()

    def _run_checks(self) -> None:
        from eu4unitgen.core import checks
        from eu4unitgen.gui.theme import tokens
        t = tokens()
        self.checks_box.setText("<h3>Checking your computer…</h3>")

        def done(found) -> None:
            rows = []
            for c in found:
                col = t["ok"] if c.ok else (t["error"] if c.blocks else t["warn"])
                rows.append(f"<span style='color:{col}'>{'✓' if c.ok else '✗'}</span> {c.what}"
                            + ("" if c.ok else f"<br>&nbsp;&nbsp;&nbsp;→ {c.fix}"))
            stop = [c for c in checks.blocking(found) if c.key != "eu4"]
            self._blocking = stop
            self.checks_box.setText("<h3>Your computer</h3>" + "<br>".join(rows)
                                    + ("<p><b>Fix the ✗ lines first (Setup in the top bar helps).</b></p>" if stop else ""))
            self.go.setEnabled(not stop and not self.started)
        self.window_.run(self, "Checking", lambda: checks.run_all(), done)

    def start(self) -> None:
        from eu4unitgen.ref3d import flow
        if getattr(self, "_blocking", None):
            return self.show_message("Fix the ✗ lines above first.", error=True)
        name = self.name.text().strip()
        try:
            if self.kind == "cavalry":
                folder = flow.new_request(name, None, "", "cavalry", self.tier_group.checkedId(),
                                          horse=self.picks["horse"], soldier=self.picks["soldier"])
            elif self.importing():
                folder = flow.new_request(name, None, "", self.kind, self.tier_group.checkedId(),
                                          options={"country_mask": self.mask.isChecked()},
                                          model={"path": str(self.model), "licence": self.licence.currentData(),
                                                 "credit": self.credit.text()})
            else:
                folder = flow.new_request(name, self.image, self.rights.text().strip(), self.kind,
                                          self.tier_group.checkedId(), options={"country_mask": self.mask.isChecked()})
        except (FileExistsError, ValueError, OSError) as exc:
            return self.show_message(str(exc), error=True)
        if self.kind in ("soldier", "horse", "gun"):
            from eu4unitgen.core import library
            library.set_tags(folder, self.style.currentData(), self.regions.text())
        flow.start(folder)
        self.started = str(folder)
        self.go.setEnabled(False)
        self.stages.show()
        self.show_message("Started. You can close the app: it keeps going.")
        if self.parent_draft:
            self.back_to_parent.show()
        else:
            self.see.show()
        self.timer.start()
        self._poll()

    def _poll(self) -> None:
        if not self.started or not self.isVisible():
            return
        try:
            st = self.stages.show_run(self.started)
        except (OSError, ValueError):
            return
        if st.get("done"):
            self.timer.stop()
            self.show_message("Ready. Open it in the Library to see it turn, or use it in a unit (New unit).")

    def hideEvent(self, e):  # noqa: N802 - Qt name
        super().hideEvent(e)
        self.timer.stop()
