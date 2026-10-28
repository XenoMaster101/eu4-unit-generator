"""The unit screen (Track P2): a unit from one image, in one flow.

Drop in an image, give it a name and the rights statement, choose horse or rider and the unit slot, press Generate.
The stages show with their times and what's left; then the result sheet next to the reference, the quality report,
simple edits, and "Add to mod". Thin by design: the run is ``ref3d.flow`` in its own detached process (it keeps
going when the window closes; reopening the app shows it again, and a stopped run resumes); edits are
``ref3d.edits``; Add to mod is ``export.modexport``; checks are ``core.checks``; errors are ``core.errors``.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSlider, QVBoxLayout,
    QWidget,
)

from eu4unitgen.gui.home_page import css
from eu4unitgen.gui.pages import Page, open_folder
from eu4unitgen.gui.theme import tokens

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
POLL_MS = 3000
STATE_TEXT = {"running": "running", "done": "done", "failed": "failed: press Resume", "stopped": "stopped: press Resume",
              "new": "not started"}
VERDICT_COLOUR = {"good": "ok", "ok": "warn", "check": "error"}
EMBLEMS = [("none", None), ("the Palaiologan cross (tetragrammatic)", "tetragrammatic")]


def minutes(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{sec:.0f} s" if sec < 90 else f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


class DropImage(QLabel):
    """The image well: drop a picture on it, or click to choose one."""

    def __init__(self, on_image):
        super().__init__("Drop the reference image here\n(or click to choose)")
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 220)
        self.setCursor(Qt.PointingHandCursor)
        self.on_image = on_image

    def dragEnterEvent(self, e):  # noqa: N802 - Qt name
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802 - Qt name
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.on_image(Path(urls[0]))

    def mousePressEvent(self, e):  # noqa: N802 - Qt name
        path, _ = QFileDialog.getOpenFileName(self, "Choose the reference image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.on_image(Path(path))


class UnitPage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.image: Path | None = None
        self.current: str | None = None  # the run shown (its folder)
        self.shown_sheet: str | None = None
        self.cloth: tuple | None = None
        self.emblem_upload: str | None = None

        top = QHBoxLayout()
        back = QPushButton("← Home")
        back.setObjectName("ghost")
        back.clicked.connect(lambda: window.show_home())
        top.addWidget(back)
        h = QLabel("A unit from one image")
        h.setObjectName("h1")
        top.addWidget(h)
        top.addStretch(1)
        for label, fn in (("Input guide", self.show_guide), ("Checks", self.show_checks), ("Setup", lambda: window.show_setup()),
                          ("About and licences", lambda: window.show_about())):
            b = QPushButton(label)
            b.setObjectName("ghost")
            b.clicked.connect(fn)
            top.addWidget(b)
        self.layout_.addLayout(top)

        body = QHBoxLayout()
        # ------------------------------------------------------------ left: the inputs and the runs
        left = QVBoxLayout()
        self.drop = DropImage(self.set_image)
        self.drop.setText("Drop the reference image here\n(or click to choose)")
        left.addWidget(self.drop)
        form = QGridLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. cataphract_horse")
        self.name.setToolTip("Lower case letters, digits and _; also the folder's name and the units' names in game")
        self.rights = QLineEdit()
        self.rights.setPlaceholderText("who made the image, and that you may use it")
        self.rights.setToolTip("Rule 8: say who made the image and that you may use it in your mod. Required; it is "
                               "recorded with the unit.")
        form.addWidget(QLabel("Name"), 0, 0)
        form.addWidget(self.name, 0, 1)
        form.addWidget(QLabel("Rights"), 1, 0)
        form.addWidget(self.rights, 1, 1)
        kinds = QHBoxLayout()
        self.kind_group = QButtonGroup(self)
        self.kind_buttons = {}
        for i, (key, label) in enumerate((("horse", "Horse"), ("rider", "Rider"), ("foot", "Foot soldier"))):
            rb = QRadioButton(label)
            self.kind_group.addButton(rb, i)
            self.kind_buttons[key] = rb
            kinds.addWidget(rb)
        self.kind_buttons["horse"].setChecked(True)
        self.kind_buttons["foot"].setEnabled(False)
        self.kind_buttons["foot"].setToolTip("On the roadmap: infantry comes after the cavalry (SPEC Track P)")
        self.kind_group.idToggled.connect(lambda *_: self._kind_changed())
        kinds.addStretch(1)
        kw = QWidget()
        kw.setLayout(kinds)
        form.addWidget(QLabel("Type"), 2, 0)
        form.addWidget(kw, 2, 1)
        self.tier = QComboBox()
        for t in (1, 2, 3, 4):
            self.tier.addItem(f"Cavalry, tier {t}", t)
        self.tier.setToolTip("The unit slot: which cavalry tier this unit is (tier 1 from the start; 2-4 by MIL tech)")
        form.addWidget(QLabel("Unit slot"), 3, 0)
        form.addWidget(self.tier, 3, 1)
        self.horse = QComboBox()
        self.horse.setToolTip("The horse this rider rides: one of your finished horses (its colours and width are used), "
                              "or the vanilla horse")
        self.horse_label = QLabel("Rides")
        form.addWidget(self.horse_label, 4, 0)
        form.addWidget(self.horse, 4, 1)
        self.mask = QCheckBox("Cloth as the country colour")
        self.mask.setToolTip("The cloth takes each country's unit colour in game (PdxMeshColor)")
        form.addWidget(self.mask, 5, 1)
        left.addLayout(form)
        self.warnings = QLabel()
        self.warnings.setObjectName("warn")
        self.warnings.setWordWrap(True)
        left.addWidget(self.warnings)
        row = QHBoxLayout()
        self.generate = QPushButton("Generate")
        self.generate.setObjectName("generate")
        self.generate.clicked.connect(self.go)
        row.addWidget(self.generate)
        row.addStretch(1)
        left.addLayout(row)
        left.addWidget(QLabel("Your units:"))
        self.list = QListWidget()
        self.list.setMaximumHeight(170)
        self.list.currentItemChanged.connect(lambda cur, _: self.show_run(cur.data(Qt.UserRole) if cur else None))
        left.addWidget(self.list)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(470)
        body.addWidget(lw)

        # ------------------------------------------------------------ right: progress, result, edits, add
        right = QVBoxLayout()
        self.run_title = QLabel("No run yet.")
        self.run_title.setObjectName("h2")
        right.addWidget(self.run_title)
        self.stage_box = QFrame()
        self.stage_box.setObjectName("card")
        grid = QGridLayout(self.stage_box)
        from eu4unitgen.ref3d import flow
        self.stage_rows: dict[str, QLabel] = {}
        for r, key in enumerate(flow.STAGES):
            grid.addWidget(QLabel(flow.LABELS[key]), r, 0)
            state = QLabel("–")
            state.setObjectName("muted")
            grid.addWidget(state, r, 1)
            self.stage_rows[key] = state
        self.left_time = QLabel()
        self.left_time.setWordWrap(True)
        grid.addWidget(self.left_time, len(flow.STAGES), 0, 1, 2)
        right.addWidget(self.stage_box)
        self.error_box = QFrame()
        self.error_box.setObjectName("card")
        el = QVBoxLayout(self.error_box)
        self.error_text = QLabel()
        self.error_text.setWordWrap(True)
        self.error_text.setObjectName("bad")
        el.addWidget(self.error_text)
        self.details = QPushButton("Details")
        self.details.setObjectName("ghost")
        self.details.clicked.connect(self.show_details)
        el.addWidget(self.details, 0, Qt.AlignLeft)
        self.error_box.hide()
        right.addWidget(self.error_box)
        self.sheet = QLabel("The result sheet shows here when the run is done.")
        self.sheet.setObjectName("cardpic")
        self.sheet.setAlignment(Qt.AlignCenter)
        self.sheet.setMinimumHeight(220)
        self.sheet.setCursor(Qt.PointingHandCursor)
        self.sheet.mousePressEvent = lambda e: self.open_sheet()
        right.addWidget(self.sheet, 1)
        self.quality = QLabel()
        self.quality.setWordWrap(True)
        self.quality.setTextFormat(Qt.RichText)
        right.addWidget(self.quality)
        right.addWidget(self._edits_box())
        right.addWidget(self._add_box())
        right.addWidget(self.message)
        rw = QWidget()
        rw.setLayout(right)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rw)
        scroll.setFrameShape(QFrame.NoFrame)
        body.addWidget(scroll, 1)
        self.layout_.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.refresh)
        self._kind_changed()
        self.restyle()

    # ------------------------------------------------------------------ the boxes
    def _edits_box(self) -> QGroupBox:
        box = QGroupBox("Edits (rebuild only what they need)")
        g = QGridLayout(box)
        self.e_plume = QCheckBox("Plume")
        self.e_tassels = QCheckBox("Head tassels")
        self.e_mask = QCheckBox("Cloth as the country colour")
        self.e_emblem = QComboBox()
        for label, key in EMBLEMS:
            self.e_emblem.addItem(label, key)
        self.e_emblem.addItem("upload a mask (white on black)…", "__upload__")
        self.e_emblem.activated.connect(self._emblem_pick)
        self.e_colour = QPushButton("Cloth colour…")
        self.e_colour.setObjectName("ghost")
        self.e_colour.clicked.connect(self._pick_colour)
        self.e_bright = QSlider(Qt.Horizontal)
        self.e_bright.setRange(70, 130)
        self.e_bright.setValue(100)
        self.e_bright.setToolTip("Brightness of every texture: 70-130%")
        self.e_side = QComboBox()
        for label, key in (("as in the image", "auto"), ("left forearm", "left"), ("right forearm", "right")):
            self.e_side.addItem(label, key)
        self.e_side.setToolTip("Which forearm carries the shield; the weapon moves to the other hand. The cavalry "
                               "animations raise the right arm, so a shield there sits high, behind the shoulder.")
        self.e_hem = QSlider(Qt.Horizontal)
        self.e_hem.setRange(50, 100)
        self.e_hem.setValue(100)
        self.e_hem.setToolTip("How far the cloth hangs below the body: 50-100% of what was built. Subtle on a caparison "
                              "that is one shape with the legs: only the cloth's own lower edge rises")
        self.e_cand = QComboBox()
        self.e_cand.setToolTip("TRELLIS.2 makes two models from your image; the one whose outline matches the reference "
                               "better is built. Pick the other here to compare.")
        rows = [(QLabel("3D model"), self.e_cand), (self.e_plume, self.e_tassels), (QLabel("Emblem"), self.e_emblem),
                (self.e_mask, self.e_colour),
                (QLabel("Brightness"), self.e_bright), (QLabel("Shield side"), self.e_side), (QLabel("Hem length"), self.e_hem)]
        for r, (a, b) in enumerate(rows):
            g.addWidget(a, r, 0)
            g.addWidget(b, r, 1)
        self.apply_edits = QPushButton("Apply edits")
        self.apply_edits.clicked.connect(self.edit)
        g.addWidget(self.apply_edits, len(rows), 1, Qt.AlignRight)
        self.edits_box = box
        return box

    def _add_box(self) -> QGroupBox:
        box = QGroupBox("Add to mod")
        g = QGridLayout(box)
        self.a_mode = QComboBox()
        self.a_mode.addItem("Normal mod: a new unit (stats, tech, colours)", "normal")
        self.a_mode.addItem("Achievement-safe: graphics only, re-dress a culture's cavalry", "achievement_safe")
        self.a_mode.currentIndexChanged.connect(lambda *_: self._mode_changed())
        self.a_tag = QLineEdit("BYZ")
        self.a_tag.setToolTip("The country (tag) that gets the unit, e.g. BYZ")
        self.a_gfx = QLineEdit("")
        self.a_gfx.setPlaceholderText("graphical culture, e.g. easterngfx")
        self.a_dir = QLineEdit()
        browse = QPushButton("Browse…")
        browse.setObjectName("ghost")
        browse.clicked.connect(self._pick_mod_dir)
        self.a_go = QPushButton("Add to mod")
        self.a_go.setObjectName("generate")
        self.a_go.clicked.connect(self.add_to_mod)
        g.addWidget(QLabel("Kind"), 0, 0)
        g.addWidget(self.a_mode, 0, 1, 1, 2)
        self.a_tag_label = QLabel("Country tag")
        g.addWidget(self.a_tag_label, 1, 0)
        g.addWidget(self.a_tag, 1, 1, 1, 2)
        g.addWidget(QLabel("Graphical culture"), 2, 0)
        g.addWidget(self.a_gfx, 2, 1, 1, 2)
        g.addWidget(QLabel("Mod folder"), 3, 0)
        g.addWidget(self.a_dir, 3, 1)
        g.addWidget(browse, 3, 2)
        self.a_note = QLabel()
        self.a_note.setWordWrap(True)
        self.a_note.setObjectName("muted")
        g.addWidget(self.a_note, 4, 0, 1, 3)
        g.addWidget(self.a_go, 5, 2)
        self.add_box = box
        self._mode_changed()
        return box

    def restyle(self) -> None:
        t = tokens()
        self.setStyleSheet(css(t) + f"""
QLabel#drop {{ background: {t['surface']}; color: {t['muted']}; border: 2px dashed {t['border']}; border-radius: 12px; }}
QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QLabel#cardpic {{ background: {t['alt']}; border-radius: 6px; color: {t['muted']}; }}
QLabel#h2 {{ font-size: 17px; font-weight: 600; }}
QLabel#bad {{ color: {t['error']}; font-weight: 600; }}
QLabel#good {{ color: {t['ok']}; }}
QGroupBox {{ color: {t['text']}; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
QListWidget {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px; }}
""")
        self._tok = t

    def showEvent(self, e):  # noqa: N802 - Qt name
        super().showEvent(e)
        from eu4unitgen.core import config
        if not self.a_dir.text():
            self.a_dir.setText(str(config.load().mods()))
        self.refresh_list()
        self.timer.start()

    def hideEvent(self, e):  # noqa: N802 - Qt name
        super().hideEvent(e)
        self.timer.stop()

    # ------------------------------------------------------------------ inputs
    def kind(self) -> str:
        return next(k for k, b in self.kind_buttons.items() if b.isChecked())

    def _kind_changed(self) -> None:
        rider = self.kind() == "rider"
        self.horse.setVisible(rider)
        self.horse_label.setVisible(rider)
        self.mask.setChecked(rider)
        if self.image:
            self._warn()

    def _mode_changed(self) -> None:
        safe = self.a_mode.currentData() == "achievement_safe"
        self.a_tag.setVisible(not safe)
        self.a_tag_label.setVisible(not safe)
        self.a_note.setText("Graphics only: nothing in common/, so the checksum and achievements stay as they are. "
                            "Every country of that graphical culture shows this unit at its tier."
                            if safe else "A mod of its own named after the unit, with its stats, tech and colours. It "
                            "changes the game's checksum: no achievements while it's on.")

    def _pick_mod_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "The EU4 mod folder", self.a_dir.text())
        if d:
            self.a_dir.setText(d)

    def _pick_colour(self) -> None:
        c = QColorDialog.getColor(QColor(*(self.cloth or (150, 30, 30))), self, "Cloth colour")
        if c.isValid():
            self.cloth = (c.red(), c.green(), c.blue())
            self.e_colour.setText(f"Cloth colour: {self.cloth}")

    def _emblem_pick(self, _i) -> None:
        if self.e_emblem.currentData() == "__upload__":
            path, _ = QFileDialog.getOpenFileName(self, "An emblem mask (white on black)", "", "Images (*.png *.jpg)")
            if path:
                self.emblem_upload = path
                self.e_emblem.setItemText(self.e_emblem.currentIndex(), f"upload: {Path(path).name}")
            else:
                self.e_emblem.setCurrentIndex(0)

    def set_image(self, path: Path) -> None:
        self.image = Path(path)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.image = None
            return self.show_message(f"{Path(path).name} isn't an image this app can read.", error=True)
        self.drop.setPixmap(pix.scaled(self.drop.width() - 8, self.drop.height() - 8, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
        if not self.name.text().strip():
            self.name.setText(re.sub(r"[^a-z0-9_]+", "_", Path(path).stem.lower()).strip("_")[:40] or "unit")
        self._warn()
        self.show_message("")

    def _warn(self) -> None:
        from eu4unitgen.core import guide
        found = guide.check(self.image, self.kind()) if self.image else []
        self.warnings.setText("\n".join("⚠ " + w for w in found))

    def problems(self) -> list[str]:
        from eu4unitgen.ref3d import flow
        out = []
        if self.image is None:
            out.append("Drop in a reference image.")
        name = self.name.text().strip()
        if not NAME.match(name):
            out.append("The name: lower case letters, digits and _, starting with a letter.")
        elif flow.folder_of(name).exists():
            out.append(f"There is already a unit called {name}: pick another name (or resume it from the list).")
        if not self.rights.text().strip():
            out.append("The rights statement: who made the image and that you may use it (rule 8).")
        return out

    # ------------------------------------------------------------------ actions
    def go(self) -> None:
        from eu4unitgen.core import checks, guide
        from eu4unitgen.ref3d import flow
        st = flow.status(Path(self.current)) if self.current else None
        resume = st is not None and st.get("state") in ("stopped", "failed") or (st is not None and st.get("state") == "new")
        if not resume:
            bad = self.problems()
            if bad:
                return self.show_message(" ".join(bad), error=True)
        found = checks.run_all()
        stop = checks.blocking([c for c in found if c.key != "eu4"])
        if stop:
            return self.show_message("Before a run: " + " ".join(f"{c.what} {c.fix}" for c in stop), error=True)
        warn = [f"{c.what} {c.fix}" for c in found if not c.ok and (not c.blocks or c.key == "eu4")]
        if not resume:
            warn += guide.check(self.image, self.kind())
        if warn and not self.window_.confirm("Before you start", "Start anyway?\n\n" + "\n\n".join(warn)):
            return
        if not resume:
            horse = self.horse.currentData() if self.kind() == "rider" else None
            folder = flow.new_request(self.name.text().strip(), self.image, self.rights.text().strip(), self.kind(),
                                      int(self.tier.currentData()), horse, {"country_mask": self.mask.isChecked()})
            self.current = str(folder)
        flow.start(Path(self.current))
        self.show_message(("Resumed" if resume else "Started") + ". You can close the app; it keeps running.")
        QTimer.singleShot(1500, self.refresh_list)

    def current_edits(self) -> dict:
        e = {"plume": self.e_plume.isChecked(), "tassels": self.e_tassels.isChecked(),
             "country_mask": self.e_mask.isChecked(), "brightness": self.e_bright.value() / 100.0,
             "shield_side": self.e_side.currentData(), "hem": self.e_hem.value() / 100.0, "cloth_colour": self.cloth,
             "candidate": self.e_cand.currentData()}
        em = self.e_emblem.currentData()
        e["emblem"] = self.emblem_upload if em == "__upload__" else em
        return e

    def edit(self) -> None:
        from eu4unitgen.ref3d import flow
        if not self.current:
            return
        back = flow.set_edits(Path(self.current), self.current_edits())
        if back is None:
            return self.show_message("Nothing changed.")
        flow.start(Path(self.current))
        self.show_message("Rebuilding the game model with your edits (about 1.5 to 5 minutes)." if back == "build"
                          else "Applying your edits (a few seconds, then the preview).")

    def add_to_mod(self) -> None:
        from eu4unitgen.core import config
        from eu4unitgen.export import modexport
        from eu4unitgen.ref3d import flow
        if not self.current:
            return
        folder = Path(self.current)
        req = flow.request(folder)
        mode = self.a_mode.currentData()
        gfx = self.a_gfx.text().strip() or ("westerngfx" if mode == "achievement_safe" else f"{req['name']}gfx")
        tag = self.a_tag.text().strip().upper() or "BYZ"
        cfg = config.load()
        mod_dir = Path(self.a_dir.text().strip() or cfg.mods())
        what = (f"Write {modexport.mod_name(req['name'])} into {mod_dir}"
                + (f" for {tag} (graphical culture {gfx})?" if mode == "normal" else f", re-dressing {gfx}'s cavalry at tier "
                   f"{req['tier']} (graphics only)?"))
        if not self.window_.confirm("Add to mod", what):
            return

        def work():
            return modexport.add_to_mod(folder / "preview" / "preview.json", req["name"], mode, tag, gfx, int(req["tier"]),
                                        mod_dir, cfg.eu4(), cfg.exports())

        def done(r) -> None:
            if r.notes and any(n.startswith("validation") for n in r.notes):
                return self.show_message("It didn't pass the game checks: " + "; ".join(r.notes), error=True)
            chk = ("No checksummed files: achievements stay." if not r.checksum_files else
                   f"{len(r.checksum_files)} files change the checksum (no achievements), e.g. "
                   + ", ".join(r.checksum_files[:4]) + ".")
            if r.staged_at:
                QMessageBox.information(self, "Copy it by hand", "\n".join(r.steps))
                open_folder(r.staged_at)
                return self.show_message(f"Staged at {r.staged_at}. {chk}")
            self.show_message(f"Written to {r.written_to} ({len(r.files)} files). {chk} Enable "
                              f"\"{r.mod_name}\" in the launcher.")
        self.window_.run(self, "Adding the unit to your mod", work, done)

    def show_guide(self) -> None:
        from eu4unitgen.core.guide import GUIDE
        k = self.kind() if self.kind() in GUIDE else "rider"
        text = ("<b>A good reference</b><br><br><b>For a horse</b><ul>" + "".join(f"<li>{x}</li>" for x in GUIDE["horse"])
                + "</ul><b>For a rider or soldier</b><ul>" + "".join(f"<li>{x}</li>" for x in GUIDE["rider"]) + "</ul>"
                + "".join(f"<p>{x}</p>" for x in GUIDE["all"]))
        del k
        QMessageBox.information(self, "Input guide", text)

    def show_checks(self) -> None:
        from eu4unitgen.core import checks
        found = checks.run_all()
        text = "<br>".join(("✔ " if c.ok else "✖ ") + c.what + ("" if c.ok else f"<br>&nbsp;&nbsp;&nbsp;→ {c.fix}") for c in found)
        QMessageBox.information(self, "Checks", text)

    def show_details(self) -> None:
        st = self._st or {}
        box = QMessageBox(self)
        box.setWindowTitle("Details")
        box.setText("The technical details of the error (send them with a report).")
        box.setDetailedText(st.get("trace") or st.get("error") or "")
        box.exec()

    def open_sheet(self) -> None:
        if self.shown_sheet:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.shown_sheet))

    # ------------------------------------------------------------------ the runs
    def refresh_list(self) -> None:
        from eu4unitgen.ref3d import flow
        rs = flow.runs()
        keep = self.current
        self.list.blockSignals(True)
        self.list.clear()
        self.horse.clear()
        self.horse.addItem("the vanilla horse", None)
        for r in rs:
            it = QListWidgetItem(f"{r['name']}  ·  {r['kind']}, tier {r.get('tier', 1)}  ·  {STATE_TEXT.get(r['state'], r['state'])}")
            it.setData(Qt.UserRole, r["folder"])
            self.list.addItem(it)
            if r["folder"] == keep or (keep is None and self.list.count() == 1):
                self.list.setCurrentItem(it)
            if r["kind"] == "horse" and r["state"] == "done":
                self.horse.addItem(r["name"], r["name"])
        self.list.blockSignals(False)
        cur = self.list.currentItem()
        self.show_run(cur.data(Qt.UserRole) if cur else None)

    def refresh(self) -> None:
        if self.current:
            self.show_run(self.current)

    def show_run(self, folder: str | None) -> None:
        from eu4unitgen.core import errors
        from eu4unitgen.ref3d import flow
        self.current = folder
        st = flow.status(Path(folder)) if folder else None
        self._st = st
        req = flow.request(Path(folder)) if folder and (Path(folder) / "request.json").is_file() else None
        state = (st or {}).get("state", "new") if folder else None
        self.generate.setText("Resume" if state in ("stopped", "failed", "new") and folder else "Generate")
        self.generate.setEnabled(state != "running")
        done = bool(st and st.get("done"))
        self.edits_box.setEnabled(done)
        self.add_box.setEnabled(done)
        if req is None:
            self.run_title.setText("No run yet.")
            for lab in self.stage_rows.values():
                lab.setText("–")
            self.left_time.setText("")
            self.error_box.hide()
            return
        self.e_plume.setVisible(req["kind"] == "rider")
        self.e_side.setEnabled(req["kind"] == "rider")
        self.e_tassels.setVisible(req["kind"] == "horse")
        e = req.get("edits") or {}
        from eu4unitgen.ref3d import flow as _flow
        cands = _flow.candidates(Path(folder))
        self.e_cand.blockSignals(True)
        self.e_cand.clear()
        if cands:
            for i, c in enumerate(cands["candidates"]):
                tag = " (best outline)" if i == cands["best"] else ""
                self.e_cand.addItem(f"candidate {i + 1}: outline {c['iou']:.0%}{tag}", None if i == cands["best"] else i + 1)
            pick = e.get("candidate")
            self.e_cand.setCurrentIndex((int(pick) - 1) if pick else cands["best"])
        else:
            self.e_cand.addItem("one model (from before candidates)", None)
        self.e_cand.blockSignals(False)
        self.e_plume.setChecked(e.get("plume", True))
        self.e_tassels.setChecked(e.get("tassels", True))
        self.e_mask.setChecked(bool(e.get("country_mask", (req.get("options") or {}).get("country_mask"))))
        self.run_title.setText(f"{req['name']} ({req['kind']}, tier {req.get('tier', 1)}): {STATE_TEXT.get(state, state)}")
        est = flow.estimates(req["kind"])
        left_s, now = 0.0, time.time()
        for key, lab in self.stage_rows.items():
            s = ((st or {}).get("stages") or {}).get(key) or {}
            ss = s.get("state", "waiting")
            if ss == "done":
                text = f"done in {minutes(s['seconds'])}" if s.get("seconds") else "done"
            elif ss == "running" and state == "running":
                took = now - s.get("started", now)
                text = f"running: {minutes(took)} of about {minutes(est[key])}"
                left_s += max(0.0, est[key] - took)
            elif ss == "waiting_eu4":
                text = "waiting: close EU4 (it holds the graphics card)"
            else:
                text = f"waiting (about {minutes(est[key])})"
                left_s += est[key]
            lab.setText(text)
            lab.setObjectName("good" if ss == "done" else "bad" if ss == "waiting_eu4" else "muted")
            lab.style().unpolish(lab)
            lab.style().polish(lab)
        if state == "running":
            self.left_time.setText(f"About {minutes(left_s)} left. You can close the app; the run continues.")
        elif state == "stopped":
            self.left_time.setText("Stopped (the app or PC was closed). Resume continues from the last finished stage.")
        elif done:
            self.left_time.setText("Done.")
        else:
            self.left_time.setText("")
        if st and st.get("error") and state in ("failed", "stopped"):
            x = errors.explain(st.get("error") or "", next((k for k, v in (st.get("stages") or {}).items()
                                                           if v.get("state") in ("running", "waiting")), None))
            self.error_text.setText(f"{x['what']}<br>What to do: {x['todo']}")
            self.error_box.show()
        else:
            self.error_box.hide()
        sheet = (st or {}).get("sheet")
        if sheet and sheet != self.shown_sheet and Path(sheet).is_file():
            pix = QPixmap(sheet)
            if not pix.isNull():
                self.sheet.setPixmap(pix.scaled(max(500, self.sheet.width() - 8), max(220, self.sheet.height() - 8),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.shown_sheet = sheet
            self.show_quality(Path(folder))
        elif not sheet:
            self.sheet.setPixmap(QPixmap())
            self.sheet.setText("The result sheet shows here when the run is done.")
            self.shown_sheet = None
            self.quality.setText("")

    def show_quality(self, folder: Path) -> None:
        import json

        from eu4unitgen.ref3d.quality import TITLES
        p = folder / "preview" / "quality.json"
        if not p.is_file():
            return
        rep = json.loads(p.read_text(encoding="utf-8"))
        t = getattr(self, "_tok", tokens())
        rows = [f"<b>{rep['summary']}</b>"]
        for k, title in TITLES.items():
            x = rep["measures"].get(k)
            if x:
                col = t.get(VERDICT_COLOUR[x["verdict"]], t["text"])
                rows.append(f"<span style='color:{col}'>●</span> <b>{title}</b>: {x['text']}")
        self.quality.setText("<br>".join(rows))
