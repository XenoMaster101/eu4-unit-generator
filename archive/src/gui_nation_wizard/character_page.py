"""The Character page: one character spec (core.character), its body's named texture areas, and
what each area is painted as (core.finishes), with masks painted by hand on the body texture.

Thin by design: the spec loads into (and saves from) the Unit set, Textures and Parts pages; area
finishes become ordinary operations in the Textures page's recipe; the body map and its areas come
from export.pipeline.Session.body_areas. The page only draws the map and the brush strokes.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageChops
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QVBoxLayout, QWidget,
)

from eu4unitgen.core.finishes import FinishError, default_colour, finishes, paint_mask_name, split
from eu4unitgen.core.naming import MESH_ROLES, TIERS
from eu4unitgen.core.recipe import RecipeError, SlotRecipe
from eu4unitgen.export.pipeline import finished_texture
from eu4unitgen.gui.pages import Page, _path_row
from eu4unitgen.textures.areas import FEATHER_MAX, GROW_MAX, adjust, overlay

KEEP = "keep (base texture)"
VIEWS = ("Finished (live)", "Areas")
PAINT = "mask:paint_"  # rows the brush paints: mask:paint_<finish>[_<rrggbb>].png
MAP_SIZE = 512
SELECTED = (0, 230, 255)  # the selected row's tint on the map: a colour no finish uses


def _qimage(img: Image.Image) -> QImage:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    q = QImage()
    q.loadFromData(buf.getvalue(), "PNG")
    return q


def label(region: str) -> str:
    """A row's name in plain words: areas by name, painted masks by finish (and colour)."""
    if region.startswith("mask:paint_"):
        finish, _, colour = region[11:].removesuffix(".png").partition("_")
        return f"painted {finish}" + (f" #{colour}" if colour else "")
    return region.removeprefix("area:").removeprefix("mask:").replace("_", " ")


def _describe(region: str, base: str) -> str:
    """A row's tooltip in plain words."""
    from eu4unitgen.textures.areas import bands

    if region.startswith("mask:paint_"):
        return "Painted with the brush (saved next to the character file)"
    if region.startswith("mask:"):
        return f"A mask from the file {region[5:]}"
    name = region.removeprefix("area:")
    band = bands().get(name)
    if band:
        lo, hi = round(band["from"] * 100), round(band["to"] * 100)
        where = "from the bottom up" if band["axis"] == "height" else "from the body out"
        return f"Part of the {band['of'].replace('_', ' ')}: {lo}-{hi}% of its length, {where}"
    return f"The {name.replace('_', ' ')}, found from the body's bones"


def _pil_mask(q: QImage) -> Image.Image:
    return Image.frombytes("L", (q.width(), q.height()), bytes(q.constBits()), "raw", "L", q.bytesPerLine())


class PaintCanvas(QLabel):
    """The body map. With the brush on, the left button paints and the right button erases; the
    stroke shows in the brush colour and goes to the page when the button is let go, at the
    texture's own size. With the brush off, a click picks the area under it."""

    stroke_done = Signal(object, bool)  # PIL 'L' stroke mask at texture size, erase
    picked = Signal(int, int)  # texture pixel clicked with the brush off

    def __init__(self):
        super().__init__()
        self.setFixedSize(MAP_SIZE, MAP_SIZE)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setStyleSheet("background: #333")
        self.base: QImage | None = None
        self.stroke: QImage | None = None
        self.painting = False
        self.brush = 12  # px on screen
        self.colour = QColor(255, 255, 255)
        self._erase = False
        self._last: QPoint | None = None

    def show_map(self, img: Image.Image) -> None:
        self.base = _qimage(img)
        self._redraw()

    def _redraw(self) -> None:
        if self.base is None:
            return
        out = self.base.copy()
        if self.stroke is not None:
            p = QPainter(out)
            tint = QImage(self.stroke.size(), QImage.Format_ARGB32)
            tint.fill(QColor(40, 40, 40) if self._erase else self.colour)
            tint.setAlphaChannel(self.stroke)
            p.setOpacity(0.8)
            p.drawImage(0, 0, tint)
            p.end()
        self.setPixmap(QPixmap.fromImage(out).scaled(MAP_SIZE, MAP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _to_texture(self, pos) -> QPoint:
        return QPoint(int(pos.x() * self.base.width() / MAP_SIZE), int(pos.y() * self.base.height() / MAP_SIZE))

    def _stroke(self, pos) -> None:
        pt = self._to_texture(pos)
        p = QPainter(self.stroke)
        p.setPen(QPen(QColor(255, 255, 255), self.brush * self.base.width() / MAP_SIZE, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawLine(self._last or pt, pt)
        p.end()
        self._last = pt
        self._redraw()

    def mousePressEvent(self, e) -> None:  # noqa: N802 - Qt name
        if self.base is None:
            return
        if not self.painting:
            pt = self._to_texture(e.position())
            return self.picked.emit(pt.x(), pt.y())
        self._erase = e.button() == Qt.RightButton
        self.stroke = QImage(self.base.size(), QImage.Format_Grayscale8)
        self.stroke.fill(0)
        self._last = None
        self._stroke(e.position())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self.stroke is not None:
            self._stroke(e.position())

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self.stroke is None:
            return
        stroke, self.stroke, self._last = _pil_mask(self.stroke), None, None
        self.stroke_done.emit(stroke, self._erase)


class CharacterPage(Page):
    title = "2. Character"

    def __init__(self, window):
        super().__init__(window)
        self.spec = None  # the loaded core.character.CharacterSpec (keeps hair, export and test)
        self.body = None  # Session.body_areas result for the shown slot
        self.masks: dict[str, Image.Image] = {}  # mask:<png> rows (painted or from files), from the recipe
        self._colours: dict[str, tuple] = {}  # region -> the user's colour (tints any finish)
        self._edges: dict[str, tuple[int, int]] = {}  # region -> (grow, feather) in texture pixels
        self._quiet = False  # rebuilding the rows: their signals aren't edits
        self._finished: tuple | None = None  # (key, image): the live map, until the slot changes
        self._selected: str | None = None

        self.layout_.setSpacing(8)

        # ---- 1 Character file
        file_box = QGroupBox("1  Character file")
        fl = QVBoxLayout(file_box)
        top = QHBoxLayout()
        self.spec_path = QLineEdit()
        self.spec_path.setPlaceholderText("character spec .json (one file per character; paths in it are relative to "
                                          "its folder)")
        self.spec_path.setToolTip("The character file. Everything about the character is in it; painted masks are "
                                  "saved next to it.")
        load, save = QPushButton("Load"), QPushButton("Save")
        load.setToolTip("Open this character file into every page (Unit set, Textures, Parts)")
        save.setToolTip("Save the character from every page into this file, with its painted masks beside it")
        load.clicked.connect(self._load)
        save.clicked.connect(self._save)
        path_row = _path_row(self.spec_path, pick_dir=False)
        for b in path_row.findChildren(QPushButton):
            b.setToolTip("Find a character file on disk")
        top.addWidget(path_row, 1)
        top.addWidget(load)
        top.addWidget(save)
        fl.addLayout(top)
        name_row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("character name (free text)")
        self.name.setToolTip("A name for your own use; the game never shows it")
        self.name.editingFinished.connect(lambda: window.character_edited())
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self.name, 1)
        self.undo_btn, self.redo_btn = QPushButton("Undo"), QPushButton("Redo")
        self.undo_btn.setToolTip("Take back the last change (Ctrl+Z)")
        self.redo_btn.setToolTip("Put back a change you took back (Ctrl+Y)")
        self.reset_btn = QPushButton("Reset to saved file")
        self.reset_btn.setToolTip("Throw away every change since the file was loaded or saved (Undo brings them back)")
        self.undo_btn.clicked.connect(lambda: window.character_undo())
        self.redo_btn.clicked.connect(lambda: window.character_redo())
        self.reset_btn.clicked.connect(lambda: window.character_reset())
        self.dirty = QLabel()
        self.dirty.setToolTip("Whether the character has changes that aren't in the file yet")
        for w in (self.undo_btn, self.redo_btn, self.reset_btn, self.dirty):
            name_row.addWidget(w)
        fl.addLayout(name_row)
        self.layout_.addWidget(file_box)
        for keys, fn in (("Ctrl+Z", window.character_undo), ("Ctrl+Y", window.character_redo),
                         ("Ctrl+Shift+Z", window.character_redo)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

        # ---- 2 Body: which body, and its texture map
        body_box = QGroupBox("2  Body")
        ll = QVBoxLayout(body_box)
        slot_row = QHBoxLayout()
        self.tier = QComboBox()
        self.tier.addItems([str(t) for t in TIERS])
        self.tier.setToolTip("The unit's look changes with military technology; tier 1 is the earliest of the four")
        self.role = QComboBox()
        self.role.addItems(list(MESH_ROLES))
        self.role.setToolTip("Which soldier to dress; infantry also dresses the cavalry rider and the artillery crew")
        show = QPushButton("Show body")
        show.setToolTip("Show this soldier's body texture: the flat picture wrapped around the 3D model")
        show.clicked.connect(lambda: window.character_areas())
        self.view = QComboBox()
        self.view.addItems(VIEWS)
        self.view.setToolTip("Finished: the texture as the unit will wear it, in its country colours. "
                             "Areas: the plain texture with each area tinted in its finish's colour.")
        self.view.currentIndexChanged.connect(lambda _: self._draw_map(self._selected))
        for w in (QLabel("Tier"), self.tier, QLabel("Soldier"), self.role, show, QLabel("Show"), self.view):
            slot_row.addWidget(w)
        slot_row.addStretch(1)
        ll.addLayout(slot_row)
        self.canvas = PaintCanvas()
        self.canvas.setToolTip("The body texture. Brush off: click to find an area. Brush on: paint finishes.")
        self.canvas.stroke_done.connect(self.paint)
        self.canvas.picked.connect(self._pick_area)
        canvas_scroll = QScrollArea()
        canvas_scroll.setWidget(self.canvas)
        canvas_scroll.setAlignment(Qt.AlignCenter)
        ll.addWidget(canvas_scroll, 1)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        ll.addWidget(self.legend)

        # ---- 3 Areas and finishes
        areas_box = QGroupBox("3  Areas and finishes")
        al = QVBoxLayout(areas_box)
        hint = QLabel("Each row is a part of the body. Pick what it's made of; the map updates as you go. "
                      "Lower rows paint over higher ones.")
        hint.setWordWrap(True)
        al.addWidget(hint)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Area", "Finish", "Colour", "Grow", "Feather"])
        for col, tip in enumerate(("The part of the body", "What the area is made of", "The finish's colour; click "
                                   "to pick your own", "Make the area bigger (or smaller, below 0) by this many "
                                   "pixels", "Soften the area's edge over this many pixels")):
            self.table.horizontalHeaderItem(col).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 170)
        self.table.setColumnWidth(2, 60)
        self.table.setColumnWidth(3, 70)
        self.table.setColumnWidth(4, 70)
        self.table.setMinimumHeight(120)
        self.table.itemSelectionChanged.connect(self._highlight)
        al.addWidget(self.table, 1)

        # ---- 4 Paint
        paint = QGroupBox("4  Paint")
        pv = QVBoxLayout(paint)
        pl = QHBoxLayout()
        self.paint_btn = QPushButton("Brush on")
        self.paint_btn.setCheckable(True)
        self.paint_btn.setToolTip("Turn the brush on to paint on the map; off, a click on the map finds the area there")
        self.paint_btn.toggled.connect(self._toggle_paint)
        self.brush_finish = QComboBox()
        self.brush_finish.addItems([f"{k}: {v['label']}" for k, v in finishes().items()])
        self.brush_finish.setToolTip("What the brush paints: the same finishes as the areas")
        self.brush_finish.currentIndexChanged.connect(lambda _: self._brush_changed(own=True))
        self.brush_colour = QPushButton()
        self.brush_colour.clicked.connect(self._pick_brush_colour)
        self.brush = QSpinBox()
        self.brush.setRange(2, 80)
        self.brush.setValue(12)
        self.brush.setSuffix(" px")
        self.brush.setToolTip("How wide the brush is, in screen pixels")
        self.brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        for w in (self.paint_btn, QLabel("Paint"), self.brush_finish, self.brush_colour, QLabel("Size"), self.brush):
            pl.addWidget(w)
        pl.addStretch(1)
        pv.addLayout(pl)
        self.paint_hint = QLabel()
        self.paint_hint.setWordWrap(True)
        pv.addWidget(self.paint_hint)
        self._brush: tuple | None = None  # the brush's own colour (None: the finish's)
        self._toggle_paint(False)

        # ---- 5 Preview
        prev_box = QGroupBox("5  Preview")
        vl = QVBoxLayout(prev_box)
        act = QHBoxLayout()
        preview = QPushButton("Before and after")
        preview.setToolTip("The base texture next to yours, both in the unit's country colours")
        preview.clicked.connect(lambda: window.character_preview())
        render = QPushButton("Render unit")
        render.setToolTip("Draw the whole soldier in 3D with your textures and parts (needs Blender; takes a while)")
        render.clicked.connect(lambda: window.character_render())
        for w in (preview, render):
            act.addWidget(w)
        act.addStretch(1)
        vl.addLayout(act)
        self.after = QLabel("Before/after and renders show here.")
        self.after.setAlignment(Qt.AlignCenter)
        self.after_scroll = QScrollArea()
        self.after_scroll.setWidgetResizable(True)
        self.after_scroll.setWidget(self.after)
        self.after_scroll.setMinimumHeight(80)
        vl.addWidget(self.after_scroll, 1)

        # resizable panes: the body | (areas / paint / preview)
        self.right_split = QSplitter(Qt.Vertical)
        for w in (areas_box, paint, prev_box):
            self.right_split.addWidget(w)
        self.right_split.setChildrenCollapsible(False)
        self.right_split.setSizes([380, 110, 260])
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(body_box)
        self.split.addWidget(self.right_split)
        self.split.setChildrenCollapsible(False)
        self.split.setSizes([560, 560])
        self.layout_.addWidget(self.split, 1)
        self.layout_.addWidget(self.message)
        self.show_hint(loaded=False)

    def show_hint(self, loaded: bool, loading: bool = False) -> None:
        """What the empty map says: why it's empty and what to do next."""
        if self.body is not None:
            return
        if not loaded and not loading and self.window_.session._game is not None:
            loaded = True
        if loading:
            text = "Loading the EU4 install...\n\nThis takes a few seconds."
        elif not loaded:
            text = ("The EU4 install isn't loaded yet.\n\nGo to 1. Settings, check the EU4 folder,\n"
                    "and press \"Load EU4 install\".\nThen come back and press \"Show body\".")
        else:
            text = ("No body shown yet.\n\nPick a tier and role above\n"
                    "and press \"Show body\" to see\nthe body's texture and its areas.")
        self.canvas.clear()
        self.canvas.setText(text)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #333; color: #ddd; font-size: 13px")

    def set_busy(self, busy: bool) -> None:
        super().set_busy(busy)
        if not busy:  # buttons that are off for a reason stay off
            self.window_._update_title()
            for region in self.regions() if self.body else []:
                self._paint_button(region)
            self._brush_changed()

    def show_dirty(self, dirty: bool, can_undo: bool, can_redo: bool) -> None:
        self.dirty.setText("<b>● Unsaved changes</b>" if dirty else "All changes saved")
        self.dirty.setStyleSheet("color: #c60" if dirty else "color: #4a4")
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)
        self.reset_btn.setEnabled(dirty)

    # ------------------------------------------------------------------ slot
    def slot_key(self) -> tuple[int, str]:
        return int(self.tier.currentText()), self.role.currentText()

    def regions(self) -> list[str]:
        """The rows: the body's areas (broad before narrow), then this page's masks."""
        areas = [f"area:{a}" for a in (self.body.areas if self.body else {})]
        return areas + [m for m in self.masks if m not in areas]

    def show_areas(self, body, recipe) -> None:
        """Fill the map and the rows from Session.body_areas and the recipe's current choices."""
        self.body = body
        slot = recipe.slots.get(self.slot_key()) if recipe is not None else None
        _, chosen = split(slot.ops if slot else [])
        self.masks = {}
        for region in chosen:
            if region.startswith("mask:"):
                try:
                    m = recipe.image(region[5:]).convert("L")
                except Exception:  # noqa: BLE001 - a missing mask file shows as an empty mask row
                    m = Image.new("L", body.diffuse.size, 0)
                self.masks[region] = m if m.size == body.diffuse.size else m.resize(body.diffuse.size)
        self._colours = {r: tuple(c["colour"]) for r, c in chosen.items()
                         if c.get("colour") and tuple(c["colour"]) != default_colour(c["finish"], body.skin)}
        self._sizes = {r: sum(m.histogram()[128:]) for r, m in self._all_masks(raw=True).items()}  # px, for picking
        rows = self.regions()
        self.table.setRowCount(len(rows))
        names = [KEEP] + [f"{k}: {v['label']}" for k, v in finishes().items()]
        self._edges = {r: (c.get("grow", 0), c.get("feather", 0)) for r, c in chosen.items()}
        for i, region in enumerate(rows):
            name = QLabel(label(region))
            name.setToolTip(_describe(region, body.base))
            self.table.setCellWidget(i, 0, name)
            combo = QComboBox()
            combo.addItems(names)
            combo.setToolTip("What this area is made of; \"keep\" leaves the base texture")
            c = chosen.get(region)
            if c:
                combo.setCurrentIndex(1 + list(finishes()).index(c["finish"]))
            combo.currentIndexChanged.connect(lambda _=0, r=region: self._finish_changed(r))
            self.table.setCellWidget(i, 1, combo)
            btn = QPushButton()
            btn.clicked.connect(lambda _=False, r=region: self._pick_colour(r))
            self.table.setCellWidget(i, 2, btn)
            for col, (lo, hi), v in ((3, (-GROW_MAX, GROW_MAX), self._edges.get(region, (0, 0))[0]),
                                     (4, (0, FEATHER_MAX), self._edges.get(region, (0, 0))[1])):
                box = QSpinBox()
                box.setRange(lo, hi)
                box.setValue(v)
                box.setSuffix(" px")
                box.setToolTip("Make the area bigger (or smaller, below 0) by this many pixels" if col == 3
                               else "Soften the area's edge over this many pixels")
                box.valueChanged.connect(lambda _=0, r=region: self._edge_changed(r))
                self.table.setCellWidget(i, col, box)
            self._paint_button(region)
        self._draw_map()
        self._brush_changed()
        self.show_message(f"Body {body.base} ({body.mesh_file}); its own skin tone is rgb {' '.join(map(str, body.skin))}. "
                          "Pick a finish per area, or paint finishes on the map.")

    def _row(self, region: str) -> int:
        return self.regions().index(region)

    def _finish(self, region: str) -> str | None:
        combo = self.table.cellWidget(self._row(region), 1)
        i = combo.currentIndex() if combo else 0
        return list(finishes())[i - 1] if i > 0 else None

    def chosen(self) -> dict[str, dict]:
        out = {}
        for region in self.regions():
            f = self._finish(region)
            if f:
                grow, feather = self._edges.get(region, (0, 0))
                out[region] = {"finish": f, "colour": self._colours.get(region), "grow": grow, "feather": feather}
        return out

    def set_edge(self, region: str, grow: int = 0, feather: int = 0) -> None:
        """Grow (negative: shrink) and feather a row's area, in texture pixels (tests, scripted setups)."""
        row = self._row(region)
        self._quiet = True
        try:
            self.table.cellWidget(row, 3).setValue(grow)
        finally:
            self._quiet = False
        self.table.cellWidget(row, 4).setValue(feather)
        self._edge_changed(region)

    def _edge_changed(self, region: str) -> None:
        if self._quiet:
            return
        row = self._row(region)
        self._edges[region] = (self.table.cellWidget(row, 3).value(), self.table.cellWidget(row, 4).value())
        self._draw_map()
        if self._finish(region):
            self.apply_finishes()

    def set_finish(self, region: str, finish: str | None, colour: tuple | None = None) -> None:
        """Choose a finish (and a colour of your own) for a row (tests, and scripted setups)."""
        combo = self.table.cellWidget(self._row(region), 1)
        self._quiet = True
        try:
            combo.setCurrentIndex(0 if finish is None else 1 + list(finishes()).index(finish))
        finally:
            self._quiet = False
        self._colours.pop(region, None)
        if colour is not None:
            self._colours[region] = tuple(colour)
        self._paint_button(region)
        self._draw_map()
        self.apply_finishes()

    def _colour_of(self, region: str) -> tuple | None:
        f = self._finish(region)
        if f is None:
            return None
        return self._colours.get(region) or default_colour(f, self.body.skin)

    def _paint_button(self, region: str) -> None:
        btn = self.table.cellWidget(self._row(region), 2)
        c = self._colour_of(region)
        btn.setEnabled(self._finish(region) is not None)
        btn.setText("" if c else ("pick…" if self._finish(region) else "–"))
        btn.setStyleSheet(f"background: rgb{tuple(c)}" if c else "")
        btn.setToolTip("Pick a colour for this finish" if self._finish(region) else "Choose a finish first")

    def _finish_changed(self, region: str) -> None:
        if self._quiet:
            return
        self._colours.pop(region, None)  # a new finish starts from its own colour
        if self._finish(region) and default_colour(self._finish(region), self.body.skin) is None:
            self._pick_colour(region, apply=False)
        self._paint_button(region)
        self._draw_map()
        self.apply_finishes()

    def _pick_colour(self, region: str, apply: bool = True) -> None:
        start = QColor(*(self._colour_of(region) or (128, 128, 128)))
        c = QColorDialog.getColor(start, self, f"Colour for {label(region)}")
        if c.isValid():
            self._colours[region] = (c.red(), c.green(), c.blue())
            self._paint_button(region)
            self._draw_map()
            if apply:
                self.apply_finishes()

    # ------------------------------------------------------------------ map
    def _all_masks(self, raw: bool = False) -> dict[str, Image.Image]:
        out = {f"area:{k}": v for k, v in (self.body.areas if self.body else {}).items()}
        out.update(self.masks)
        if raw:
            return out
        return {r: adjust(m, *self._edges[r]) if self._edges.get(r, (0, 0)) != (0, 0) else m for r, m in out.items()}

    def finished_map(self) -> Image.Image:
        """The body texture with this slot's recipe applied, tinted with the unit colours (cached
        until the slot or its painted masks change)."""
        recipe = self.window_.textures_page.recipe
        slot = recipe.slots.get(self.slot_key()) or SlotRecipe()
        key = (json.dumps(slot.to_json(), sort_keys=True), tuple((k, id(v)) for k, v in sorted(recipe.images.items())))
        if self._finished is None or self._finished[0] != key:
            try:
                img = finished_texture(self.body, recipe, slot)
            except RecipeError as exc:
                self.show_message(f"Can't show the finished texture: {exc}", error=True)
                img = self.body.diffuse
            self._finished = (key, img)
        return self._finished[1]

    def _draw_map(self, highlight: str | None = None) -> None:
        """Finished: the texture as the unit will wear it, the selected row lightly tinted. Areas:
        the base texture, areas with a finish in its colour, the selected row in a bright tint."""
        if self.body is None:
            return
        self._selected = highlight
        masks = self._all_masks()
        colours = {r: self._colour_of(r) for r in self.regions() if self._colour_of(r)}
        legend = [f"{label(r)} → {self._finish(r)}" for r in colours]
        tint = {highlight: SELECTED} if highlight else {}
        if self.view.currentText() == VIEWS[0]:
            self.canvas.show_map(overlay(self.finished_map(), masks, tint, 0.45))
        else:
            self.canvas.show_map(overlay(self.body.diffuse, masks, {**colours, **tint}, 0.6))
        self.legend.setText("Finishes: " + (", ".join(legend) if legend else "none yet")
                            + (f"<br>Selected (in blue): <b>{label(highlight)}</b>" if highlight else ""))

    def _highlight(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        regions = self.regions()
        self._draw_map(regions[min(rows)] if rows and min(rows) < len(regions) else None)

    # ------------------------------------------------------------------ painting
    def _toggle_paint(self, on: bool) -> None:
        if on and self.body is None:
            self.paint_btn.setChecked(False)
            return self.show_message("Show the body areas first.", error=True)
        self.canvas.painting = on
        self.paint_btn.setText("Brush on" if on else "Brush off")
        self.canvas.setCursor(Qt.CrossCursor if on else Qt.PointingHandCursor)
        self.paint_hint.setText(
            "Left button paints the finish, right button erases paint. Painting over another painted finish "
            "replaces it." if on else "Brush off: click the map to find the area under the pointer.")

    def brush_finish_key(self) -> str:
        return list(finishes())[self.brush_finish.currentIndex()]

    def brush_colour_value(self) -> tuple | None:
        skin = self.body.skin if self.body else (196, 150, 120)
        return self._brush or default_colour(self.brush_finish_key(), skin)

    def set_brush(self, finish: str, colour: tuple | None = None) -> None:
        """Pick what the brush paints (tests, scripted setups)."""
        self.brush_finish.setCurrentIndex(list(finishes()).index(finish))
        self._brush = tuple(colour) if colour else None
        self._brush_changed()

    def _brush_changed(self, own: bool = False) -> None:
        if own:
            self._brush = None  # a new finish starts from its own colour
        c = self.brush_colour_value()
        self.brush_colour.setText("" if c else "pick…")
        self.brush_colour.setStyleSheet(f"background: rgb{tuple(c)}" if c else "")
        self.brush_colour.setToolTip("The colour the brush paints; click to pick another (it tints the finish)")
        if c:
            self.canvas.colour = QColor(*c)

    def _pick_brush_colour(self) -> None:
        start = QColor(*(self.brush_colour_value() or (128, 128, 128)))
        c = QColorDialog.getColor(start, self, "Brush colour")
        if c.isValid():
            self._brush = (c.red(), c.green(), c.blue())
            self._brush_changed()

    def paint(self, stroke: Image.Image, erase: bool = False) -> None:
        """A brush stroke (an 'L' mask at the texture's size): paint the brush's finish there, taking
        it out of every other painted finish; or, erasing, take it out of all of them."""
        if self.body is None:
            return
        recipe = self.window_.textures_page.recipe
        stroke = stroke.convert("L")
        if stroke.size != self.body.diffuse.size:
            stroke = stroke.resize(self.body.diffuse.size)
        chosen = self.chosen()
        target = None
        if not erase:
            colour = self.brush_colour_value()
            if colour is None:
                return self.show_message("Pick a colour for the brush first.", error=True)
            finish = self.brush_finish_key()
            target = "mask:" + paint_mask_name(finish, colour, self.body.skin)
            own = tuple(colour) != default_colour(finish, self.body.skin)
            old = chosen.get(target, {})
            chosen[target] = {"finish": finish, "colour": tuple(colour) if own else None,
                              "grow": old.get("grow", 0), "feather": old.get("feather", 0)}
            self.masks.setdefault(target, Image.new("L", stroke.size, 0))
        for region in [r for r in self.masks if r.startswith(PAINT)]:
            if region == target:
                m = ImageChops.lighter(self.masks[region], stroke)
            else:
                m = ImageChops.subtract(self.masks[region], stroke)
                if m.getbbox() is None:  # all of it painted over or erased: the row goes
                    chosen.pop(region, None)
                    recipe.images.pop(region[5:], None)
                    continue
            recipe.images[region[5:]] = m
        self.apply_finishes(chosen)
        self.show_areas(self.body, recipe)

    def _pick_area(self, x: int, y: int) -> None:
        """Select the narrowest row whose area holds texture pixel (x, y)."""
        if self.body is None:
            return
        rows = self.regions()
        hits = [r for r, m in self._all_masks().items() if r in rows and m.getpixel((x, y)) >= 128]
        if not hits:
            return self.show_message("No named area there (no area covers that part of the texture).")
        best = min(hits, key=lambda r: self._sizes.get(r, 0))
        self.table.selectRow(rows.index(best))
        others = [label(h) for h in hits if h != best]
        self.show_message(f"<b>{label(best)}</b>" + (f" (also in: {', '.join(others)})" if others else ""))

    # ------------------------------------------------------------------ recipe
    def apply_finishes(self, chosen: dict | None = None) -> None:
        """The rows' finishes (or ``chosen``) -> the Textures page's slot for this tier/role, replacing
        the old ones. Runs on every change, so the recipe (and the undo history) always matches."""
        if self.body is None:
            return self.show_message("Show the body areas first.", error=True)
        chosen = self.chosen() if chosen is None else chosen
        order = self.regions() + sorted(r for r in chosen if r not in self.regions())
        try:
            self.window_.apply_character_finishes(self.slot_key(), chosen, self.body.skin, order)
        except FinishError as exc:
            return self.show_message(str(exc), error=True)
        self.window_.character_edited()
        self._draw_map(self._selected)  # the live map follows the recipe
        n = len(chosen)
        self.show_message(f"Applied {n} area finish(es) to the texture recipe's slot "
                          f"{self.slot_key()[0]}/{self.slot_key()[1]}." if n else "Cleared this slot's area finishes.")

    def show_preview(self, preview) -> None:
        both = Image.new("RGB", (520, 260), (40, 40, 42))
        for i, img in enumerate((preview.before, preview.after)):
            both.paste(img.convert("RGB").resize((256, 256)), (i * 264, 2))
        self.after.setPixmap(QPixmap.fromImage(_qimage(both)))
        self.show_message("Before | after, tinted with the unit colours. " + "; ".join(preview.notes))

    def show_render(self, preview) -> None:
        w = max(300, min(900, self.after_scroll.viewport().width() - 10))
        self.after.setPixmap(QPixmap.fromImage(_qimage(preview.image)).scaledToWidth(w, Qt.SmoothTransformation))
        self.show_message(f"Saved to {preview.path}. " + "; ".join(preview.notes))

    # ------------------------------------------------------------------ files
    def _load(self) -> None:
        path = self.spec_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load character spec", str(Path.cwd()), "Character (*.json)")
        if path:
            self.window_.load_character(Path(path))

    def _save(self) -> None:
        path = self.spec_path.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save character spec", str(Path.cwd() / "character.json"),
                                                  "Character (*.json)")
        if path:
            self.window_.save_character(Path(path))


