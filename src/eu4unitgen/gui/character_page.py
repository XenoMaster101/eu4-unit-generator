"""The Character page: one character spec (core.character), its body's named texture areas, and
what each area is painted as (core.finishes), with masks painted by hand on the body texture.

Thin by design: the spec loads into (and saves from) the Unit set, Textures and Parts pages; area
finishes become ordinary operations in the Textures page's recipe; the body map and its areas come
from export.pipeline.Session.body_areas. The page only draws the map and the brush strokes.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QTableWidget, QVBoxLayout, QWidget,
)

from eu4unitgen.core.finishes import FinishError, finishes, split
from eu4unitgen.core.naming import MESH_ROLES, TIERS
from eu4unitgen.gui.pages import Page, _path_row
from eu4unitgen.textures.areas import overlay

KEEP = "keep (base texture)"
MAP_SIZE = 512
# area tints on the map; masks you paint show in white
TINTS = [(255, 210, 0), (0, 160, 255), (150, 0, 255), (0, 210, 90), (255, 120, 0), (255, 60, 160), (255, 0, 0),
         (0, 230, 230), (190, 255, 0), (120, 80, 255), (255, 170, 120)]


def _qimage(img: Image.Image) -> QImage:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    q = QImage()
    q.loadFromData(buf.getvalue(), "PNG")
    return q


def _pil_mask(q: QImage) -> Image.Image:
    return Image.frombytes("L", (q.width(), q.height()), bytes(q.constBits()), "raw", "L", q.bytesPerLine())


class MaskCanvas(QLabel):
    """The body texture with areas tinted; while painting, the left button adds to the mask and the
    right button erases. The mask lives at the texture's own size."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(MAP_SIZE, MAP_SIZE)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setStyleSheet("background: #333")
        self.base: QImage | None = None
        self.mask: QImage | None = None
        self.painting = False
        self.brush = 12
        self._last: QPoint | None = None

    def show_map(self, img: Image.Image) -> None:
        self.base = _qimage(img)
        if self.mask is None or self.mask.size() != self.base.size():
            self.mask = QImage(self.base.size(), QImage.Format_Grayscale8)
            self.mask.fill(0)
        self._redraw()

    def new_mask(self, existing: Image.Image | None = None) -> None:
        if self.base is None:
            return
        self.mask = QImage(self.base.size(), QImage.Format_Grayscale8)
        self.mask.fill(0)
        if existing is not None:
            m = existing.convert("L").resize((self.base.width(), self.base.height()))
            self.mask = QImage(m.tobytes(), m.width, m.height, m.width, QImage.Format_Grayscale8).copy()
        self._redraw()

    def mask_image(self) -> Image.Image | None:
        return _pil_mask(self.mask) if self.mask is not None else None

    def _redraw(self) -> None:
        if self.base is None:
            return
        out = self.base.copy()
        if self.painting and self.mask is not None:
            p = QPainter(out)
            tint = QImage(self.mask.size(), QImage.Format_ARGB32)
            tint.fill(QColor(255, 255, 255))
            tint.setAlphaChannel(self.mask)
            p.setOpacity(0.7)
            p.drawImage(0, 0, tint)
            p.end()
        self.setPixmap(QPixmap.fromImage(out).scaled(MAP_SIZE, MAP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _to_texture(self, pos) -> QPoint:
        sx = self.base.width() / MAP_SIZE
        sy = self.base.height() / MAP_SIZE
        return QPoint(int(pos.x() * sx), int(pos.y() * sy))

    def _stroke(self, pos, erase: bool) -> None:
        if not self.painting or self.mask is None or self.base is None:
            return
        pt = self._to_texture(pos)
        p = QPainter(self.mask)
        colour = QColor(0, 0, 0) if erase else QColor(255, 255, 255)
        p.setPen(QPen(colour, self.brush * self.base.width() / MAP_SIZE, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(self._last or pt, pt)
        p.end()
        self._last = pt
        self._redraw()

    def mousePressEvent(self, e) -> None:  # noqa: N802 - Qt name
        self._last = None
        self._stroke(e.position(), e.button() == Qt.RightButton)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._stroke(e.position(), bool(e.buttons() & Qt.RightButton))

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        self._last = None


class CharacterPage(Page):
    title = "2. Character"

    def __init__(self, window):
        super().__init__(window)
        self.spec = None  # the loaded core.character.CharacterSpec (keeps hair, export and test)
        self.body = None  # Session.body_areas result for the shown slot
        self.masks: dict[str, Image.Image] = {}  # mask:<png> regions painted or loaded on this page
        self._colours: dict[str, tuple] = {}  # region -> custom colour

        top = QHBoxLayout()
        self.spec_path = QLineEdit()
        self.spec_path.setPlaceholderText("character spec .json (one file per character; paths in it are relative to "
                                          "its folder)")
        load, save = QPushButton("Load"), QPushButton("Save")
        load.clicked.connect(self._load)
        save.clicked.connect(self._save)
        top.addWidget(_path_row(self.spec_path, pick_dir=False), 1)
        top.addWidget(load)
        top.addWidget(save)
        self.layout_.addLayout(top)
        name_row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("character name (free text)")
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self.name, 1)
        self.layout_.addLayout(name_row)

        slot_row = QHBoxLayout()
        self.tier = QComboBox()
        self.tier.addItems([str(t) for t in TIERS])
        self.role = QComboBox()
        self.role.addItems(list(MESH_ROLES))
        self.role.setToolTip("infantry also dresses the cavalry rider and the artillery crew")
        show = QPushButton("Show body areas")
        show.setToolTip("the tier's body texture (the Parts page's body for that tier) with its named areas")
        show.clicked.connect(lambda: window.character_areas())
        for w in (QLabel("Tier"), self.tier, QLabel("Role"), self.role, show):
            slot_row.addWidget(w)
        slot_row.addStretch(1)
        self.layout_.addLayout(slot_row)

        body = QHBoxLayout()
        left = QVBoxLayout()
        self.canvas = MaskCanvas()
        left.addWidget(self.canvas)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        left.addWidget(self.legend)
        body.addLayout(left)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Area", "Finish", "Colour"])
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 190)
        self.table.itemSelectionChanged.connect(self._highlight)
        right.addWidget(self.table, 1)

        paint = QGroupBox("Paint a mask (left button paints, right erases)")
        pl = QHBoxLayout(paint)
        self.mask_name = QLineEdit()
        self.mask_name.setPlaceholderText("mask name, e.g. greaves_band")
        self.brush = QSpinBox()
        self.brush.setRange(2, 80)
        self.brush.setValue(12)
        self.brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        self.paint_btn = QPushButton("Start painting")
        self.paint_btn.setCheckable(True)
        self.paint_btn.toggled.connect(self._toggle_paint)
        keep = QPushButton("Save mask")
        keep.setToolTip("writes <name>.png next to the character spec and adds it as an area (mask:<name>.png)")
        keep.clicked.connect(self._save_mask)
        for w in (self.mask_name, QLabel("Brush"), self.brush, self.paint_btn, keep):
            pl.addWidget(w)
        right.addWidget(paint)

        act = QHBoxLayout()
        apply = QPushButton("Apply to the texture recipe")
        apply.setToolTip("writes the finishes into the Textures page's slot for this tier/role (after its own "
                         "operations); the set then uses the recipe")
        apply.clicked.connect(self.apply_finishes)
        preview = QPushButton("Preview textures")
        preview.clicked.connect(lambda: window.character_preview())
        render = QPushButton("Render unit")
        render.setToolTip("the Parts page's 3D preview of this tier, body textured by the recipe")
        render.clicked.connect(lambda: window.character_render())
        for w in (apply, preview, render):
            act.addWidget(w)
        right.addLayout(act)
        self.after = QLabel()
        self.after.setMinimumHeight(260)
        right.addWidget(self.after)
        body.addLayout(right, 1)
        self.layout_.addLayout(body, 1)
        self.layout_.addWidget(self.message)

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
        for region in chosen:
            if region.startswith("mask:") and region not in self.masks:
                try:
                    self.masks[region] = recipe.image(region[5:]).convert("L")
                except Exception:  # noqa: BLE001 - a missing mask file shows as an empty mask row
                    self.masks[region] = Image.new("L", body.diffuse.size, 0)
        rows = self.regions()
        self.table.setRowCount(len(rows))
        names = [KEEP] + [f"{k}: {v['label']}" for k, v in finishes().items()]
        for i, region in enumerate(rows):
            label = QLabel(region.removeprefix("area:"))
            self.table.setCellWidget(i, 0, label)
            combo = QComboBox()
            combo.addItems(names)
            c = chosen.get(region)
            if c:
                combo.setCurrentIndex(1 + list(finishes()).index(c["finish"]))
                if finishes()[c["finish"]]["colour"] is None:
                    self._colours[region] = c["colour"]
            combo.currentIndexChanged.connect(lambda _=0, r=region: self._finish_changed(r))
            self.table.setCellWidget(i, 1, combo)
            btn = QPushButton()
            btn.clicked.connect(lambda _=False, r=region: self._pick_colour(r))
            self.table.setCellWidget(i, 2, btn)
            self._paint_button(region)
        self._draw_map()
        self.show_message(f"Body {body.base} ({body.mesh_file}); its own skin tone is rgb {' '.join(map(str, body.skin))}. "
                          "Pick a finish per area, then Apply.")

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
                out[region] = {"finish": f, "colour": self._colours.get(region)}
        return out

    def set_finish(self, region: str, finish: str | None, colour: tuple | None = None) -> None:
        """Choose a finish for a row (tests, and scripted setups)."""
        if colour is not None:
            self._colours[region] = tuple(colour)
        combo = self.table.cellWidget(self._row(region), 1)
        combo.setCurrentIndex(0 if finish is None else 1 + list(finishes()).index(finish))

    def _colour_of(self, region: str) -> tuple | None:
        f = self._finish(region)
        if f is None:
            return None
        c = finishes()[f]["colour"]
        return self._colours.get(region) if c is None else (self.body.skin if c == "skin" else tuple(c))

    def _paint_button(self, region: str) -> None:
        btn = self.table.cellWidget(self._row(region), 2)
        c = self._colour_of(region)
        custom = self._finish(region) is not None and finishes()[self._finish(region)]["colour"] is None
        btn.setEnabled(custom)
        btn.setText("" if c else ("pick…" if custom else "–"))
        btn.setStyleSheet(f"background: rgb{tuple(c)}" if c else "")

    def _finish_changed(self, region: str) -> None:
        if self._finish(region) and finishes()[self._finish(region)]["colour"] is None and region not in self._colours:
            self._pick_colour(region)
        self._paint_button(region)
        self._draw_map()

    def _pick_colour(self, region: str) -> None:
        start = QColor(*(self._colours.get(region) or (128, 128, 128)))
        c = QColorDialog.getColor(start, self, f"Colour for {region}")
        if c.isValid():
            self._colours[region] = (c.red(), c.green(), c.blue())
            self._paint_button(region)
            self._draw_map()

    # ------------------------------------------------------------------ map
    def _all_masks(self) -> dict[str, Image.Image]:
        out = {f"area:{k}": v for k, v in (self.body.areas if self.body else {}).items()}
        out.update(self.masks)
        return out

    def _draw_map(self, highlight: str | None = None) -> None:
        """Areas with a finish show in its colour; the selected row shows in a bright tint."""
        if self.body is None:
            return
        masks = self._all_masks()
        colours = {r: self._colour_of(r) for r in self.regions() if self._colour_of(r)}
        legend = [f"{r.removeprefix('area:')} → {self._finish(r)}" for r in colours]
        if highlight:
            colours[highlight] = TINTS[self.regions().index(highlight) % len(TINTS)]
        self.canvas.show_map(overlay(self.body.diffuse, masks, colours, 0.6))
        self.legend.setText("Finishes: " + (", ".join(legend) if legend else "none yet")
                            + (f"<br>Selected: <b>{highlight.removeprefix('area:')}</b>" if highlight else ""))

    def _highlight(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        regions = self.regions()
        self._draw_map(regions[min(rows)] if rows and min(rows) < len(regions) else None)

    # ------------------------------------------------------------------ masks
    def _toggle_paint(self, on: bool) -> None:
        if on and self.body is None:
            self.paint_btn.setChecked(False)
            return self.show_message("Show the body areas first.", error=True)
        self.canvas.painting = on
        self.paint_btn.setText("Stop painting" if on else "Start painting")
        if on:
            name = self.mask_name.text().strip()
            existing = self.masks.get(f"mask:{name}.png") if name else None
            self.canvas.new_mask(existing)
        self.canvas._redraw()

    def _save_mask(self) -> None:
        name = self.mask_name.text().strip()
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            return self.show_message("Name the mask first (letters, digits, _ and -).", error=True)
        if self.canvas.mask is None:
            return self.show_message("Paint the mask first.", error=True)
        try:
            path = self.save_mask(name, self.canvas.mask_image())
        except OSError as exc:
            return self.show_message(str(exc), error=True)
        self.paint_btn.setChecked(False)
        self.show_message(f"Saved {path}; it's now the row mask:{name}.png.")

    def save_mask(self, name: str, img: Image.Image) -> Path:
        """Write a painted mask beside the character spec (the recipe's folder) and add its row."""
        root = self.window_.textures_page.recipe.root
        path = Path(root) / f"{name}.png"
        img.convert("L").save(path)
        region = f"mask:{name}.png"
        self.masks[region] = img.convert("L")
        if self.body is not None:
            keep = self.chosen()
            self.show_areas(self.body, None)
            for r, c in keep.items():
                self.set_finish(r, c["finish"], c.get("colour"))
        return path

    # ------------------------------------------------------------------ recipe
    def apply_finishes(self) -> None:
        """The rows' finishes -> the Textures page's slot for this tier/role (replacing the old ones)."""
        if self.body is None:
            return self.show_message("Show the body areas first.", error=True)
        try:
            self.window_.apply_character_finishes(self.slot_key(), self.chosen(), self.body.skin, self.regions())
        except FinishError as exc:
            return self.show_message(str(exc), error=True)
        n = len(self.chosen())
        self.show_message(f"Applied {n} area finish(es) to the texture recipe's slot "
                          f"{self.slot_key()[0]}/{self.slot_key()[1]}." if n else "Cleared this slot's area finishes.")

    def show_preview(self, preview) -> None:
        both = Image.new("RGB", (520, 260), (40, 40, 42))
        for i, img in enumerate((preview.before, preview.after)):
            both.paste(img.convert("RGB").resize((256, 256)), (i * 264, 2))
        self.after.setPixmap(QPixmap.fromImage(_qimage(both)))
        self.show_message("Before | after, tinted with the unit colours. " + "; ".join(preview.notes))

    def show_render(self, preview) -> None:
        w = max(300, min(900, self.after.width() - 10))
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


