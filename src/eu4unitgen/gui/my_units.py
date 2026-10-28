"""The Library (was My units; SPEC Track L): the parts (horses, soldiers, guns: made once from a picture, or the
starter parts that come with the app) and the units put together from them. Each part's page: the turntable, the
result sheet, the quality report in plain words, its progress while it is being made, its tags, and what to do with
it: use it in a unit, edit the look, add it to a mod, delete it. A starter part is read-only.

Thin by design: parts and units are ``core.library``; runs ``ref3d.flow``; edits ``ref3d.edits`` (only what they need
is rebuilt); Add to mod ``export.modexport``."""
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

KIND_TEXT = {"soldier": "soldier", "horse": "horse", "gun": "gun", "cavalry": "cavalry unit", "rider": "cavalry unit",
             "unit": "unit"}
FILTERS = (("all", "All"), ("horse", "Horses"), ("soldier", "Soldiers"), ("gun", "Guns"), ("unit", "Units"))
EMBLEMS = [("none", None), ("the Palaiologan cross (tetragrammatic)", "tetragrammatic")]
VERDICT_COLOUR = {"good": "ok", "ok": "warn", "check": "error"}


def thumb_of(r: dict) -> str | None:
    """A run's picture for the gallery: its turntable's three-quarter frame, its close render, else its reference."""
    from eu4unitgen.core import library
    return library.run_thumb(r)


def unit_text(r: dict) -> str:
    """A unit card's line: each tier's parts ("tier 1: bay horse + cataphract")."""
    if r.get("kind") != "unit":
        return f"tier {r.get('tier', 1)}: {r.get('horse')} + {r.get('soldier')}".replace("_", " ")
    bits = []
    for t, s in sorted((r.get("tiers") or {}).items()):
        bits.append(f"tier {t}: " + " + ".join(ref.split(":", 1)[1].replace("_", " ") for ref in s.values()))
    return "; ".join(bits)


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
        head.addWidget(label("Library", "h1"))
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
        head.addWidget(ghost("＋  New part", lambda: window.go("new"),
                             "Make a horse, a soldier or a gun from a picture (needs the AI setup)"))
        head.addWidget(primary("＋  Build a unit", lambda: window.go("build"),
                               "Put a unit together from the parts here, tier by tier (no AI needed)"))
        gl.addLayout(head)
        gl.addWidget(label("Parts are made once from a picture and used in as many units as you like: infantry is a "
                           "soldier, cavalry a horse and a soldier riding it, artillery a gun and a soldier as its "
                           "crew.", "explain"))
        self.ai_note = label("", "explain")
        gl.addWidget(self.ai_note)
        self.empty = label("Nothing here yet. Press “New part” to make your first soldier or horse from a picture.",
                           "explain")
        gl.addWidget(self.empty)
        box = QWidget()
        bl = QVBoxLayout(box)
        self.parts_title = label("Parts", "h2")
        bl.addWidget(self.parts_title)
        pw = QWidget()
        self.flow_layout = FlowLayout(pw)
        bl.addWidget(pw)
        self.units_title = label("Units", "h2")
        bl.addWidget(self.units_title)
        uw = QWidget()
        self.units_layout = FlowLayout(uw)
        bl.addWidget(uw)
        bl.addStretch(1)
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
        from eu4unitgen.core import library
        from eu4unitgen.ref3d import flow
        self.flow_layout.clear()
        self.units_layout.clear()
        ps = [p for p in library.parts() if self.filter in ("all", p.kind)]
        for p in ps:
            tags = p.tag_text()
            state = "ready" if p.done else state_text({"status": flow.status(p.folder) or {}, "state": p.state})
            b = card(p.title, f"{KIND_TEXT[p.kind]}{' · starter' if p.starter else ''}\n"
                     f"{tags or 'no tags yet'}\n{state}",
                     "A starter part that comes with the app: use it in a unit, or make a copy to edit it" if p.starter
                     else "Open it: its preview, quality report, tags, edits, Use in a unit",
                     picture=pixmap(p.thumb or p.image, 130, 130))
            b.setFixedSize(360, 160)
            b.clicked.connect(lambda _=False, ref=p.ref: self.open_part(ref))
            self.flow_layout.addWidget(b)
        us = library.units() if self.filter in ("all", "unit") else []
        for r in us:
            b = card(r["name"].replace("_", " "), f"{library.TYPE_TEXT.get(r.get('type'), 'Cavalry')}\n{unit_text(r)}\n"
                     f"{state_text(r)}", "Open it: its tiers, preview, Add to mod", picture=pixmap(thumb_of(r), 130, 130))
            b.setFixedSize(360, 160)
            b.clicked.connect(lambda _=False, f=r["folder"]: self.open(f))
            self.units_layout.addWidget(b)
        self.parts_title.setVisible(bool(ps))
        self.units_title.setVisible(bool(us))
        self.empty.setVisible(not ps and not us)
        ok, why = library.ai_ready()
        self.ai_note.setText(why)
        self.ai_note.setVisible(not ok)

    def open(self, folder: str) -> None:
        self.detail.show_unit(folder)
        self.stack.setCurrentWidget(self.detail)

    def open_part(self, ref: str) -> None:
        from eu4unitgen.core import library
        p = library.part(ref)
        if p is None:
            return
        if p.starter:
            self.detail.show_starter(p)
            self.stack.setCurrentWidget(self.detail)
        else:
            self.open(str(p.folder))

    def back(self) -> None:
        self.fill()
        self.stack.setCurrentWidget(self.gallery_page)


class UnitDetail(QWidget):
    def __init__(self, screen: MyUnitsScreen):
        super().__init__()
        self.screen = screen
        self.window_ = screen.window_
        self.folder: str | None = None
        self.starter = None  # a core.library.Part when a starter part is shown
        self.cloth: tuple | None = None
        self.emblem_upload: str | None = None
        self._shown_frames = None
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(ghost("← Library", screen.back))
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
        self.ride_btn = primary("Use in a unit", self._use_as_rider, "Build a unit with this part (no AI needed)")
        self.add_btn = primary("Add to mod", lambda: self._toggle(self.add_box), "Write it into a mod folder you choose")
        self.del_btn = ghost("Delete", self._delete, "Delete this unit and its files (asks first)")
        self.del_btn.setObjectName("danger")
        self.copy_btn = ghost("Make a copy", self._copy, "A copy of this starter part in your library, to edit")
        self.tags_btn = ghost("Tags", lambda: self._toggle(self.tags_box), "Its style and regions: they drive the "
                              "suggestions when you build a unit for a country")
        for b in (self.resume, self.ride_btn, self.edit_btn, self.tags_btn, self.add_btn, self.copy_btn, self.del_btn):
            actions.addWidget(b)
        actions.addStretch(1)
        right.addLayout(actions)
        self.stages = StageList()
        right.addWidget(self.stages)
        self.quality = label("", rich=True)
        right.addWidget(self.quality)
        right.addWidget(self._tags())
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
        self.starter = None
        self._shown_frames = None
        self.edits_box.hide()
        self.add_box.hide()
        self.tags_box.hide()
        self.show_message("")
        self.refresh()

    def refresh(self) -> None:
        from eu4unitgen.ref3d import flow
        if getattr(self, "starter", None) is not None:
            return
        if not self.folder or not (Path(self.folder) / "request.json").is_file():
            return
        from eu4unitgen.core import library
        req = flow.request(Path(self.folder))
        kind = req["kind"]
        st = self.stages.show_run(self.folder) if kind != "unit" else (flow.status(Path(self.folder)) or {})
        state = st.get("state", "new")
        done = bool(st.get("done"))
        part = kind in library.KINDS
        self.title.setText(req["name"].replace("_", " "))
        if kind == "unit":
            bits = [f"{library.TYPE_TEXT[req['type']]}, put together from the library",
                    unit_text(req), "built" if done else "not built yet: putting units together comes in the next step"]
        else:
            bits = [f"A {KIND_TEXT.get(kind, kind)}" + ("" if part else f", tier {req.get('tier', 1)}")]
        if kind == "cavalry":
            bits.append(f"the soldier {req.get('soldier')} on the horse {req.get('horse')}")
        if part and req.get("tags"):
            bits.append("tags: " + library.Part("", "", kind, Path(), tags=req["tags"]).tag_text())
        bits.append(f"rights: {req.get('rights', '')}")
        self.state.setText(" · ".join(bits))
        self.stages.setVisible(not done and kind != "unit")
        self.resume.setVisible(state in ("stopped", "failed", "new") and kind != "unit")
        for b in (self.edit_btn, self.add_btn):
            b.setEnabled(done)
        self.edit_btn.setVisible(kind != "unit")
        self.ride_btn.setVisible(part)
        self.ride_btn.setEnabled(done)
        self.tags_btn.setVisible(part)
        self.copy_btn.setVisible(False)
        for b in (self.add_btn, self.del_btn, self.sheet_btn, self.folder_btn):
            b.setVisible(True)
        if part:
            self._fill_tags(req)
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
        """Use in a unit: Build a unit with this part picked for tier 1 (a horse: cavalry; a gun: artillery; a
        soldier: infantry, switch the type there to ride or crew)."""
        from eu4unitgen.ref3d import flow
        if self.starter is not None:
            return self.window_.go("build", pick=self.starter.ref)
        req = flow.request(Path(self.folder))
        self.window_.go("build", pick=f"run:{req['name']}")

    # ------------------------------------------------------------------ a starter part (read-only)
    def show_starter(self, p) -> None:
        self.starter = p
        self.folder = str(p.folder)
        self._shown_frames = None
        for box in (self.edits_box, self.add_box, self.tags_box):
            box.hide()
        self.show_message("")
        self.title.setText(p.title)
        self.state.setText(f"A starter {KIND_TEXT[p.kind]} that comes with the app · tags: {p.tag_text() or 'none'}"
                           f" · rights: {p.rights}")
        self.stages.hide()
        for b in (self.resume, self.edit_btn, self.tags_btn, self.add_btn, self.del_btn):
            b.setVisible(False)
        self.ride_btn.setVisible(True)
        self.ride_btn.setEnabled(True)
        self.copy_btn.setVisible(True)
        self.turn.set_frames([], p.thumb or p.image)
        self.sheet_btn.setVisible(False)  # its sheet shows the game's own bodies: made on this PC once it's in a unit
        self.folder_btn.setVisible(False)
        import json as _json
        d = _json.loads((p.folder / "part.json").read_text(encoding="utf-8"))
        q = d.get("quality") or {}
        from eu4unitgen.ref3d.quality import TITLES
        rows = [f"<h3>How it came out: {q.get('summary') or '—'}</h3>"]
        for k, title in TITLES.items():
            x = (q.get("measures") or {}).get(k)
            if x:
                rows.append(f"<p><b>{title}</b>: {x.get('text', '')}</p>")
        rows.append("<p>Read-only: it comes with the app. Use it in a unit as it is, or make a copy to change its "
                    "colours, emblem or tags.</p>")
        self.quality.setText("".join(rows))

    def _copy(self) -> None:
        self.show_message("Copying starter parts comes with the starter library (next steps).")

    # ------------------------------------------------------------------ tags
    def _tags(self) -> QGroupBox:
        from eu4unitgen.core import library
        box = QGroupBox("Tags (they decide which parts are suggested for a country)")
        g = QGridLayout(box)
        self.t_style = QComboBox()
        self.t_style.addItem("no style", None)
        for s in library.STYLES:
            self.t_style.addItem(library.STYLE_TEXT[s], s)
        self.t_style.setToolTip("What kind of horse it is. A country's graphical culture suggests a style (e.g. "
                                "westerngfx: barded first; muslimgfx: Arab; a nomad: steppe).")
        self.t_style_label = label("Style", wrap=False)
        self.t_regions = QLineEdit()
        self.t_regions.setPlaceholderText("e.g. Byzantine, Greek")
        self.t_regions.setToolTip("Peoples or regions it fits, separated by commas. A country whose culture or culture "
                                  "group matches gets it suggested first.")
        g.addWidget(self.t_style_label, 0, 0)
        g.addWidget(self.t_style, 0, 1)
        g.addWidget(label("Regions", wrap=False), 1, 0)
        g.addWidget(self.t_regions, 1, 1)
        g.addWidget(primary("Save the tags", self._save_tags, "Saved with the part; nothing is rebuilt"), 2, 1,
                    Qt.AlignRight)
        box.hide()
        self.tags_box = box
        return box

    def _fill_tags(self, req: dict) -> None:
        tags = req.get("tags") or {}
        horse = req["kind"] == "horse"
        self.t_style.setVisible(horse)
        self.t_style_label.setVisible(horse)
        self.t_style.setCurrentIndex(max(0, self.t_style.findData(tags.get("style"))))
        if not self.t_regions.hasFocus():
            self.t_regions.setText(", ".join(tags.get("regions") or []))

    def _save_tags(self) -> None:
        from eu4unitgen.core import library
        tags = library.set_tags(Path(self.folder), self.t_style.currentData(), self.t_regions.text())
        self.show_message("Tags saved: " + (library.Part("", "", "", Path(), tags=tags).tag_text() or "none") + ".")
        self.refresh()

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
