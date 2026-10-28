"""My units: a gallery of everything made (a picture, its name, kind and state), and each unit's own page: the
turntable (the unit turning, as the game draws it), the result sheet, the quality report in plain words, its
progress while it is being made, and what to do with it: edit the look, use a soldier as a rider (or a horse for
cavalry), add it to a mod, delete it.

Thin by design: runs are ``ref3d.flow``; edits ``ref3d.edits`` (only what they need is rebuilt); Add to mod
``export.modexport``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLineEdit,
    QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from eu4unitgen.gui.base import Page, Turntable, card, ghost, label, open_file, open_folder, pixmap, primary
from eu4unitgen.gui.progress import StageList, state_text
from eu4unitgen.gui.widgets import FlowLayout

KIND_TEXT = {"soldier": "soldier", "horse": "horse", "cavalry": "cavalry", "rider": "rider"}
FILTERS = (("all", "All"), ("soldier", "Soldiers"), ("horse", "Horses"), ("cavalry", "Cavalry"))
EMBLEMS = [("none", None), ("the Palaiologan cross (tetragrammatic)", "tetragrammatic")]
VERDICT_COLOUR = {"good": "ok", "ok": "warn", "check": "error"}


def thumb_of(r: dict) -> str | None:
    """A unit's picture for the gallery: its turntable's three-quarter frame, its close render, else its reference."""
    from eu4unitgen.ref3d import flow
    f = Path(r["folder"])
    frames = flow.turntable_frames(f)
    if frames:
        return str(frames[min(1, len(frames) - 1)])
    g = f / "preview" / "gallery" / f"close_three_quarter_{r.get('tier', 1)}.png"
    if g.is_file():
        return str(g)
    return r.get("image")


class MyUnitsScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        self.filter = "all"
        self.stack = QStackedWidget()
        self.layout_.addWidget(self.stack, 1)
        # the gallery
        g = QWidget()
        gl = QVBoxLayout(g)
        head = QHBoxLayout()
        head.addWidget(label("My units", "h1"))
        head.addStretch(1)
        self.filter_group = QButtonGroup(self)
        for key, text in FILTERS:
            b = ghost(text, tip=f"Show {text.lower()}")
            b.setCheckable(True)
            b.setChecked(key == "all")
            b.clicked.connect(lambda _=False, k=key: self._set_filter(k))
            self.filter_group.addButton(b)
            head.addWidget(b)
        head.addSpacing(12)
        head.addWidget(primary("＋  New unit", lambda: window.go("new"), "Make a soldier or a cavalry unit"))
        gl.addLayout(head)
        self.empty = label("Nothing here yet. Press “New unit” to make your first soldier, or “Show me an example” "
                           "on Home to see how it goes.", "explain")
        gl.addWidget(self.empty)
        box = QWidget()
        self.flow_layout = FlowLayout(box)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(box)
        gl.addWidget(sc, 1)
        self.stack.addWidget(g)
        self.gallery_page = g
        self.detail = UnitDetail(self)
        self.stack.addWidget(self.detail)
        self.layout_.addWidget(self.message)
        self.timer = QTimer(self)
        self.timer.setInterval(4000)
        self.timer.timeout.connect(self._tick)
        self.restyle()

    def enter(self, folder: str | None = None) -> None:
        self.fill()
        if folder:
            self.open(folder)
        else:
            self.stack.setCurrentWidget(self.gallery_page)
        self.timer.start()

    def hideEvent(self, e):  # noqa: N802 - Qt name
        super().hideEvent(e)
        self.timer.stop()

    def _tick(self) -> None:
        if self.stack.currentWidget() is self.detail:
            self.detail.refresh()
        else:
            self.fill()

    def _set_filter(self, key: str) -> None:
        self.filter = key
        self.fill()

    def fill(self) -> None:
        from eu4unitgen.ref3d import flow
        self.flow_layout.clear()
        rs = [r for r in flow.runs() if self.filter == "all" or r["kind"] == self.filter
              or (self.filter == "cavalry" and r["kind"] == "rider")]
        for r in rs:
            b = card(r["name"].replace("_", " "), f"{KIND_TEXT.get(r['kind'], r['kind'])}, tier {r.get('tier', 1)}\n"
                     f"{state_text(r)}", "Open it: its preview, quality report, edits and Add to mod",
                     picture=pixmap(thumb_of(r), 150, 150))
            b.setFixedSize(380, 176)
            b.clicked.connect(lambda _=False, f=r["folder"]: self.open(f))
            self.flow_layout.addWidget(b)
        self.empty.setVisible(not rs)

    def open(self, folder: str) -> None:
        self.detail.show_unit(folder)
        self.stack.setCurrentWidget(self.detail)

    def back(self) -> None:
        self.fill()
        self.stack.setCurrentWidget(self.gallery_page)


class UnitDetail(QWidget):
    def __init__(self, screen: MyUnitsScreen):
        super().__init__()
        self.screen = screen
        self.window_ = screen.window_
        self.folder: str | None = None
        self.cloth: tuple | None = None
        self.emblem_upload: str | None = None
        self._shown_frames = None
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(ghost("← All units", screen.back))
        self.title = label("", "h1", wrap=False)
        top.addWidget(self.title)
        top.addStretch(1)
        outer.addLayout(top)
        self.state = label("", "explain")
        outer.addWidget(self.state)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.turn = Turntable(330)
        left.addWidget(self.turn)
        self.sheet_btn = ghost("Open the result sheet", self._open_sheet,
                               "The picture next to the unit in game (close, side and map zoom) and the report, full size")
        left.addWidget(self.sheet_btn)
        self.folder_btn = ghost("Open its folder", lambda: open_folder(Path(self.folder)) if self.folder else None,
                                "Every file of this unit (the models, textures, logs)")
        left.addWidget(self.folder_btn)
        left.addStretch(1)
        body.addLayout(left)
        right = QVBoxLayout()
        actions = QHBoxLayout()
        self.resume = primary("Resume", self._resume, "Carry on from the last finished step")
        self.edit_btn = ghost("Edit the look", lambda: self._toggle(self.edits_box),
                              "Colours, brightness, plume, shield side…: only what's needed is rebuilt")
        self.ride_btn = ghost("Use as rider", self._use_as_rider, "Put this soldier on one of your horses (a new cavalry unit)")
        self.add_btn = primary("Add to mod", lambda: self._toggle(self.add_box), "Write it into a mod folder you choose")
        self.del_btn = ghost("Delete", self._delete, "Delete this unit and its files (asks first)")
        self.del_btn.setObjectName("danger")
        for b in (self.resume, self.edit_btn, self.ride_btn, self.add_btn, self.del_btn):
            actions.addWidget(b)
        actions.addStretch(1)
        right.addLayout(actions)
        self.stages = StageList()
        right.addWidget(self.stages)
        self.quality = label("", rich=True)
        right.addWidget(self.quality)
        right.addWidget(self._edits())
        right.addWidget(self._add())
        self.message = label("", rich=True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.message)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # the text wraps: no sideways scrolling
        sc.setWidget(rw)
        body.addWidget(sc, 1)
        outer.addLayout(body, 1)

    # the page protocol the window's ``run`` uses
    def show_message(self, text: str, error: bool = False) -> None:
        from eu4unitgen.gui.theme import tokens
        self.message.setText(f'<span style="color:{tokens()["error"]}">{text}</span>' if text and error else text)

    def set_busy(self, busy: bool) -> None:
        self.add_go.setEnabled(not busy)

    def _toggle(self, box) -> None:
        box.setVisible(not box.isVisible())

    # ------------------------------------------------------------------ showing a unit
    def show_unit(self, folder: str) -> None:
        self.folder = folder
        self._shown_frames = None
        self.edits_box.hide()
        self.add_box.hide()
        self.show_message("")
        self.refresh()

    def refresh(self) -> None:
        from eu4unitgen.ref3d import flow
        if not self.folder or not (Path(self.folder) / "request.json").is_file():
            return
        req = flow.request(Path(self.folder))
        st = self.stages.show_run(self.folder)
        state = st.get("state", "new")
        done = bool(st.get("done"))
        kind = req["kind"]
        self.title.setText(req["name"].replace("_", " "))
        bits = [f"A {KIND_TEXT.get(kind, kind)}, tier {req.get('tier', 1)}"]
        if kind == "cavalry":
            bits.append(f"the soldier {req.get('soldier')} on the horse {req.get('horse')}")
        bits.append(f"rights: {req.get('rights', '')}")
        self.state.setText(" · ".join(bits))
        self.stages.setVisible(not done)
        self.resume.setVisible(state in ("stopped", "failed", "new"))
        for b in (self.edit_btn, self.add_btn):
            b.setEnabled(done)
        self.ride_btn.setVisible(kind in ("soldier", "horse"))
        self.ride_btn.setText("Use as rider" if kind == "soldier" else "Use for cavalry")
        self.ride_btn.setToolTip("Put this soldier on one of your horses (a new cavalry unit)" if kind == "soldier"
                                 else "Give this horse one of your soldiers as its rider (a new cavalry unit)")
        self.ride_btn.setEnabled(done)
        self.del_btn.setEnabled(state != "running")
        frames = [str(p) for p in flow.turntable_frames(Path(self.folder))]
        if frames != self._shown_frames:
            self.turn.set_frames(frames, thumb_of({"folder": self.folder, "tier": req.get("tier", 1), "image": req.get("image")}))
            self._shown_frames = frames
        self.sheet_btn.setEnabled(bool(st.get("sheet")) and Path(st.get("sheet") or "").is_file())
        self._show_quality()
        if done:
            self._fill_edits(req)
            self._fill_add(req)

    def _show_quality(self) -> None:
        from eu4unitgen.gui.theme import tokens
        from eu4unitgen.ref3d.quality import TITLES
        p = Path(self.folder) / "preview" / "quality.json"
        if not p.is_file():
            self.quality.setText("")
            return
        rep = json.loads(p.read_text(encoding="utf-8"))
        t = tokens()
        rows = [f"<h3>How it came out: {rep['summary']}</h3>"]
        for k, title in TITLES.items():
            x = rep["measures"].get(k)
            if x:
                col = t.get(VERDICT_COLOUR[x["verdict"]], t["text"])
                rows.append(f"<p><span style='color:{col}'>●</span> <b>{title}</b>: {x['text']}</p>")
        rows.append("<p style='font-size:12px'>● green: good · ● amber: usable · ● red: look at it in game first</p>")
        self.quality.setText("".join(rows))

    def _open_sheet(self) -> None:
        from eu4unitgen.ref3d import flow
        st = flow.status(Path(self.folder)) or {}
        if st.get("sheet"):
            open_file(Path(st["sheet"]))

    def _resume(self) -> None:
        from eu4unitgen.ref3d import flow
        flow.start(Path(self.folder))
        self.show_message("Resumed. You can close the app: it keeps going.")
        QTimer.singleShot(1500, self.refresh)

    def _use_as_rider(self) -> None:
        from eu4unitgen.ref3d import flow
        req = flow.request(Path(self.folder))
        if req["kind"] == "soldier":
            self.window_.go("new", kind="cavalry", soldier=req["name"])
        else:
            self.window_.go("new", kind="cavalry", horse=req["name"])

    def _delete(self) -> None:
        from eu4unitgen.ref3d import flow
        f = Path(self.folder)
        req = flow.request(f)
        if (flow.status(f) or {}).get("state") == "running":
            return self.show_message("It's being made right now; wait until it's finished (or paused).", error=True)
        root = flow.root().resolve()
        if root not in f.resolve().parents:
            return self.show_message("That folder isn't one of the app's units; nothing deleted.", error=True)
        users = [r["name"] for r in flow.runs() if r["kind"] == "cavalry" and req["name"] in (r.get("horse"), r.get("soldier"))]
        warn = (f"\n\nThe cavalry unit(s) {', '.join(users)} use it: they keep working, but their edits that rebuild the "
                "rider need it." if users else "")
        if not self.window_.confirm("Delete", f"Delete {req['name']} and all its files ({f})?{warn}\n\nYour picture "
                                              "stays in the references folder. This can't be undone."):
            return
        shutil.rmtree(f)
        self.screen.back()

    # ------------------------------------------------------------------ edits
    def _edits(self) -> QGroupBox:
        box = QGroupBox("Edit the look (only what an edit needs is rebuilt)")
        g = QGridLayout(box)
        self.e_cand = QComboBox()
        self.e_cand.setToolTip("The AI makes two 3D models from your picture; the one scored best on outline, colour and "
                               "movement is used. Pick the other to compare (rebuilds the game model, a few minutes).")
        self.e_plume = QCheckBox("Plume")
        self.e_plume.setToolTip("The helmet's plume on or off (seconds)")
        self.e_tassels = QCheckBox("Head tassels")
        self.e_tassels.setToolTip("The horse's painted head tassels and plume on or off (rebuilds, about 2 minutes)")
        self.e_mask = QCheckBox("Cloth in each country's colour")
        self.e_mask.setToolTip("The cloth takes each country's unit colour in game (rebuilds)")
        self.e_emblem = QComboBox()
        for text, key in EMBLEMS:
            self.e_emblem.addItem(text, key)
        self.e_emblem.addItem("upload a mask (white on black)…", "__upload__")
        self.e_emblem.setToolTip("An emblem painted on the horse's cloth panels: a built-in one, or your own as white on "
                                 "black (rebuilds)")
        self.e_emblem.activated.connect(self._emblem_pick)
        self.e_colour = ghost("Cloth colour…", self._pick_colour, "The cloth's colour when it isn't the country's (seconds)")
        self.e_bright = QSlider(Qt.Horizontal)
        self.e_bright.setRange(70, 130)
        self.e_bright.setValue(100)
        self.e_bright.setToolTip("Brightness of every texture, 70-130% (seconds). Units that are too dark vanish at map zoom.")
        self.e_side = QComboBox()
        for text, key in (("as in the picture", "auto"), ("left forearm", "left"), ("right forearm", "right")):
            self.e_side.addItem(text, key)
        self.e_side.setToolTip("Which forearm carries the shield; the weapon moves to the other hand (rebuilds). The "
                               "cavalry animations raise the right arm, so a shield there sits high.")
        self.e_hem = QSlider(Qt.Horizontal)
        self.e_hem.setRange(50, 100)
        self.e_hem.setValue(100)
        self.e_hem.setToolTip("How far the cloth hangs: 50-100% of what was built (seconds)")
        self.e_rows = [("3D model", self.e_cand), ("", self.e_plume), ("", self.e_tassels), ("", self.e_mask),
                       ("Emblem", self.e_emblem), ("", self.e_colour), ("Brightness", self.e_bright),
                       ("Shield side", self.e_side), ("Hem length", self.e_hem)]
        self.e_labels = {}
        for r, (text, w) in enumerate(self.e_rows):
            lab = label(text, wrap=False)
            self.e_labels[w] = lab
            g.addWidget(lab, r, 0)
            g.addWidget(w, r, 1)
        g.addWidget(primary("Apply the edits", self._apply, "Rebuilds only what the changes need, then the preview"),
                    len(self.e_rows), 1, Qt.AlignRight)
        box.hide()
        self.edits_box = box
        return box

    def _fill_edits(self, req: dict) -> None:
        from eu4unitgen.ref3d import flow
        kind = req["kind"]
        person = kind != "horse"
        shown = {self.e_plume: person, self.e_side: person, self.e_tassels: not person, self.e_emblem: not person,
                 self.e_hem: True, self.e_cand: kind != "cavalry"}
        for w, vis in shown.items():
            w.setVisible(vis)
            self.e_labels[w].setVisible(vis)
        e = req.get("edits") or {}
        cands = flow.candidates(Path(self.folder))
        self.e_cand.blockSignals(True)
        self.e_cand.clear()
        if cands:
            for i, c in enumerate(cands["candidates"]):
                tag = " (scored best)" if i == cands["best"] else ""
                self.e_cand.addItem(f"version {i + 1}: {flow.candidate_text(c)}{tag}", None if i == cands["best"] else i + 1)
            pick = e.get("candidate")
            self.e_cand.setCurrentIndex((int(pick) - 1) if pick else cands["best"])
        else:
            self.e_cand.addItem("one model", None)
        self.e_cand.blockSignals(False)
        self.e_plume.setChecked(e.get("plume", True))
        self.e_tassels.setChecked(e.get("tassels", True))
        self.e_mask.setChecked(bool(e.get("country_mask", (req.get("options") or {}).get("country_mask"))))
        self.e_bright.setValue(int(100 * float(e.get("brightness", 1.0))))
        self.e_hem.setValue(int(100 * float(e.get("hem", 1.0))))
        i = self.e_side.findData(e.get("shield_side", "auto"))
        self.e_side.setCurrentIndex(max(0, i))

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

    def current_edits(self) -> dict:
        e = {"plume": self.e_plume.isChecked(), "tassels": self.e_tassels.isChecked(),
             "country_mask": self.e_mask.isChecked(), "brightness": self.e_bright.value() / 100.0,
             "shield_side": self.e_side.currentData(), "hem": self.e_hem.value() / 100.0, "cloth_colour": self.cloth,
             "candidate": self.e_cand.currentData()}
        em = self.e_emblem.currentData()
        e["emblem"] = self.emblem_upload if em == "__upload__" else em
        return e

    def _apply(self) -> None:
        from eu4unitgen.ref3d import flow
        back = flow.set_edits(Path(self.folder), self.current_edits())
        if back is None:
            return self.show_message("Nothing changed.")
        flow.start(Path(self.folder))
        self.show_message("Rebuilding the game model with your edits (about 2 to 6 minutes)." if back == "build"
                          else "Applying your edits (a few seconds, then the preview, a few minutes).")
        QTimer.singleShot(1500, self.refresh)

    # ------------------------------------------------------------------ add to mod
    def _add(self) -> QGroupBox:
        box = QGroupBox("Add to mod")
        g = QGridLayout(box)
        self.a_mode = QComboBox()
        self.a_mode.addItem("A normal mod: a new unit for one country (stats, tech, colours)", "normal")
        self.a_mode.addItem("Achievement-safe: graphics only, re-dress a graphical culture's unit", "achievement_safe")
        self.a_mode.setToolTip("Normal: changes the game's checksum (no achievements while it's on). Achievement-safe: "
                               "only graphics, so achievements stay; every country of that graphical culture shows it.")
        self.a_mode.currentIndexChanged.connect(lambda *_: self._mode_changed())
        self.a_tag = QLineEdit("BYZ")
        self.a_tag.setToolTip("The country that gets the unit: its three-letter tag (BYZ is Byzantium, FRA France…)")
        self.a_gfx = QLineEdit()
        self.a_gfx.setPlaceholderText("graphical culture, e.g. easterngfx")
        self.a_gfx.setToolTip("The family of looks the game gives countries (westerngfx, easterngfx, muslimgfx…). "
                              "Normal mode: a new one named after the unit is made when left blank.")
        self.a_dir = QLineEdit()
        self.a_dir.setToolTip("Your EU4 mod folder (Documents\\Paradox Interactive\\Europa Universalis IV\\mod)")
        self.a_note = label("", "explain")
        self.a_go = primary("Add to mod", self.add_to_mod, "Checks it, asks you once, then writes it into the mod folder")
        g.addWidget(label("Kind", wrap=False), 0, 0)
        g.addWidget(self.a_mode, 0, 1, 1, 2)
        self.a_tag_label = label("Country (tag)", wrap=False)
        g.addWidget(self.a_tag_label, 1, 0)
        g.addWidget(self.a_tag, 1, 1, 1, 2)
        g.addWidget(label("Graphical culture", wrap=False), 2, 0)
        g.addWidget(self.a_gfx, 2, 1, 1, 2)
        g.addWidget(label("Mod folder", wrap=False), 3, 0)
        g.addWidget(self.a_dir, 3, 1)
        g.addWidget(ghost("Browse…", self._pick_mod_dir, "Pick the mod folder"), 3, 2)
        g.addWidget(self.a_note, 4, 0, 1, 3)
        g.addWidget(self.a_go, 5, 2)
        box.hide()
        self.add_box = box
        self._mode_changed()
        return box

    def _fill_add(self, req: dict) -> None:
        from eu4unitgen.core import config
        if not self.a_dir.text():
            self.a_dir.setText(str(config.load().mods()))

    def _mode_changed(self) -> None:
        safe = self.a_mode.currentData() == "achievement_safe"
        self.a_tag.setVisible(not safe)
        self.a_tag_label.setVisible(not safe)
        self.a_note.setText("Graphics only: nothing in common/, so the checksum and achievements stay. Every country of "
                            "that graphical culture shows this unit at its tier." if safe else
                            "A mod of its own named after the unit, with its stats, tech and colours. It changes the "
                            "game's checksum: no achievements while it's on.")

    def _pick_mod_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "The EU4 mod folder", self.a_dir.text())
        if d:
            self.a_dir.setText(d)

    def add_to_mod(self) -> None:
        from eu4unitgen.core import config
        from eu4unitgen.export import modexport
        from eu4unitgen.ref3d import flow
        folder = Path(self.folder)
        req = flow.request(folder)
        mode = self.a_mode.currentData()
        utype = flow.FOOT_TYPE[req["kind"]]
        gfx = self.a_gfx.text().strip() or ("westerngfx" if mode == "achievement_safe" else f"{req['name']}gfx")
        tag = self.a_tag.text().strip().upper() or "BYZ"
        cfg = config.load()
        mod_dir = Path(self.a_dir.text().strip() or cfg.mods())
        what = (f"Write {modexport.mod_name(req['name'])} into {mod_dir}"
                + (f" for {tag} (graphical culture {gfx})?" if mode == "normal" else
                   f", re-dressing {gfx}'s {utype} at tier {req['tier']} (graphics only)?"))
        if not self.window_.confirm("Add to mod", what):
            return

        def work():
            return modexport.add_to_mod(folder / "preview" / "preview.json", req["name"], mode, tag, gfx, int(req["tier"]),
                                        mod_dir, cfg.eu4(), cfg.exports(), utype=utype)

        def done(r) -> None:
            if r.notes and any(n.startswith("validation") for n in r.notes):
                return self.show_message("It didn't pass the game checks: " + "; ".join(r.notes), error=True)
            chk = ("No checksummed files: achievements stay." if not r.checksum_files else
                   f"{len(r.checksum_files)} files change the checksum (no achievements while it's on).")
            if r.staged_at:
                self.window_.inform("Copy it by hand", "Windows didn't let the app write into the mod folder "
                                    "(Controlled Folder Access). It's saved here instead; do this:\n\n" + "\n".join(r.steps))
                open_folder(r.staged_at)
                return self.show_message(f"Saved at {r.staged_at}; copy it as the steps say. {chk}")
            self.show_message(f"Done: written to {r.written_to} ({len(r.files)} files). {chk} In the EU4 launcher, "
                              f"tick “{r.mod_name}” under Mods and start the game.")
        self.window_.run(self, "Adding it to your mod", work, done)
