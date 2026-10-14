"""The Textures screen (Phase 2): a texture recipe, one slot at a time, with a 2D preview.

The form maps to core.recipe.SimpleSlot; previews and guide export run through
export.pipeline.Session (via the main window). Nothing here touches images itself except
showing the PNGs the pipeline returns.
"""
from __future__ import annotations

import io
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from eu4unitgen.core.naming import MESH_ROLES, TIERS
from eu4unitgen.core.recipe import RecipeError, SimpleSlot, SlotRecipe, TextureRecipe
from eu4unitgen.export.pipeline import SpecError
from eu4unitgen.gui.pages import Page, _path_row
from eu4unitgen.textures.recolour import BLENDS, GLOSS_PRESETS
from eu4unitgen.textures.regions import COUNTRY_REGIONS, REGIONS

PREVIEW_SIZE = 300
TARGET_LABELS = {"country1": "primary", "country2": "secondary", "country3": "tertiary", "cloth": "cloth"}


def _pixmap(img) -> QPixmap:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    pm = QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    return pm.scaled(PREVIEW_SIZE, PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _file_row(placeholder: str) -> tuple[QLineEdit, QWidget]:
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    return edit, _path_row(edit, pick_dir=False)


def _spin(lo: float, hi: float, value: float, step: float = 0.05) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class TexturesPage(Page):
    title = "3. Textures"

    def __init__(self, window):
        super().__init__(window)
        self.recipe = TextureRecipe(root=Path.cwd())
        self._slot_key: tuple[int, str] = (1, "infantry")

        top = QHBoxLayout()
        self.use = QCheckBox("Use a texture recipe for this set")
        self.recipe_path = QLineEdit()
        self.recipe_path.setPlaceholderText("recipe .json (paths inside it are relative to its folder)")
        load, save = QPushButton("Load"), QPushButton("Save")
        load.clicked.connect(self._load)
        save.clicked.connect(self._save)
        top.addWidget(self.use)
        top.addWidget(_path_row(self.recipe_path, pick_dir=False), 1)
        top.addWidget(load)
        top.addWidget(save)
        self.layout_.addLayout(top)

        slot_row = QHBoxLayout()
        self.tier = QComboBox()
        self.tier.addItems([str(t) for t in TIERS])
        self.role = QComboBox()
        self.role.addItems(list(MESH_ROLES))
        self.role.setToolTip("infantry also dresses the cavalry rider and the artillery crew")
        self.base = QLineEdit()
        self.base.setPlaceholderText("base pdxmesh (default: the base culture's own for this tier)")
        for w, label in ((self.tier, "Tier"), (self.role, "Slot"), (self.base, "Base")):
            slot_row.addWidget(QLabel(label))
            slot_row.addWidget(w, 1 if w is self.base else 0)
        self.tier.currentIndexChanged.connect(self._switch_slot)
        self.role.currentIndexChanged.connect(self._switch_slot)
        self.layout_.addLayout(slot_row)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.addWidget(self._protect_box())
        bl.addWidget(self._palette_box())
        bl.addWidget(self._cloth_box())
        bl.addWidget(self._pattern_box())
        bl.addWidget(self._spec_box())
        bl.addWidget(self._gloss_box())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview slot")
        self.preview_btn.clicked.connect(window.texture_preview)
        self.guide_btn = QPushButton("Export painting guide")
        self.guide_btn.clicked.connect(window.export_texture_guide)
        clear = QPushButton("Clear slot")
        clear.clicked.connect(self._clear_slot)
        for b in (self.preview_btn, self.guide_btn, clear):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.slots_label = QLabel()
        self.slots_label.setWordWrap(True)

        pv = QHBoxLayout()
        self.previews = []
        for caption in ("Regions (red/green/blue country, cyan metal, magenta skin, grey cloth)",
                        "Base, as the game tints it", "Edited, as the game tints it"):
            col = QVBoxLayout()
            img = QLabel()
            img.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
            img.setAlignment(Qt.AlignCenter)
            img.setStyleSheet("background:#222")
            cap = QLabel(caption)
            cap.setWordWrap(True)
            cap.setMaximumWidth(PREVIEW_SIZE)
            col.addWidget(img)
            col.addWidget(cap)
            pv.addLayout(col)
            self.previews.append(img)
        pv.addStretch(1)

        self.layout_.addWidget(scroll, 1)
        self.layout_.addLayout(buttons)
        self.layout_.addWidget(self.slots_label)
        self.layout_.addLayout(pv)
        self.layout_.addWidget(self.message)
        self._refresh_slots()

    # ------------------------------------------------------------------ form sections
    def _protect_box(self) -> QGroupBox:
        box = QGroupBox("Protect (left untouched by every colour change)")
        f = QFormLayout(box)
        self.protect_skin = QCheckBox("Skin (auto-detected; on horses the whole hide)")
        self.protect_skin.setChecked(True)
        self.protect_metal = QCheckBox("Metal (auto-detected from gloss)")
        self.protect_mask, row = _file_row("optional PNG: white = protect; replaces the auto mask")
        f.addRow(self.protect_skin)
        f.addRow(self.protect_metal)
        f.addRow("Protect mask", row)
        return box

    def _palette_box(self) -> QGroupBox:
        box = QGroupBox("Reference palette (dominant colours of an image)")
        f = QFormLayout(box)
        self.palette_image, row = _file_row("reference image (painting, photo, flag...)")
        targets = QHBoxLayout()
        self.palette_targets = {}
        for region in (*COUNTRY_REGIONS, "cloth"):
            cb = QCheckBox(TARGET_LABELS[region])
            cb.setChecked(region in COUNTRY_REGIONS[:2])
            self.palette_targets[region] = cb
            targets.addWidget(cb)
        targets.addStretch(1)
        tw = QWidget()
        tw.setLayout(targets)
        self.palette_fixed = QCheckBox("Fixed colours: clear the country colour where the palette paints")
        self.palette_fixed.setChecked(True)
        self.palette_fixed.setToolTip("The game multiplies the diffuse under country regions by the country colour "
                                      "(SPEC §4). Untick to keep country colours there (the palette then tints them).")
        f.addRow("Image", row)
        f.addRow("Recolour", tw)
        f.addRow(self.palette_fixed)
        return box

    def _cloth_box(self) -> QGroupBox:
        box = QGroupBox("Cloth colour (everything that isn't country colour, metal or skin)")
        h = QHBoxLayout(box)
        self.cloth_on = QCheckBox("Recolour cloth to")
        self.cloth_btn = QPushButton()
        self._cloth = QColor(110, 70, 40)
        self.cloth_btn.clicked.connect(self._pick_cloth)
        self.cloth_strength = _spin(0, 1, 1.0)
        h.addWidget(self.cloth_on)
        h.addWidget(self.cloth_btn)
        h.addWidget(QLabel("strength"))
        h.addWidget(self.cloth_strength)
        h.addStretch(1)
        self._paint_cloth()
        return box

    def _pattern_box(self) -> QGroupBox:
        box = QGroupBox("Pattern overlay")
        f = QFormLayout(box)
        self.pattern_image, row = _file_row("tileable PNG")
        self.pattern_region = QComboBox()
        self.pattern_region.setEditable(True)
        self.pattern_region.addItems(list(REGIONS))
        self.pattern_region.setToolTip("a region, or mask:<png> for your own mask")
        self.pattern_region.setCurrentText("cloth")
        self.pattern_blend = QComboBox()
        self.pattern_blend.addItems(list(BLENDS))
        self.pattern_blend.setCurrentText("multiply")
        self.pattern_opacity = _spin(0, 1, 1.0)
        self.pattern_scale = _spin(0.05, 16, 1.0, 0.25)
        f.addRow("Image", row)
        f.addRow("Region", self.pattern_region)
        f.addRow("Blend", self.pattern_blend)
        f.addRow("Opacity", self.pattern_opacity)
        f.addRow("Scale", self.pattern_scale)
        return box

    def _spec_box(self) -> QGroupBox:
        box = QGroupBox("Country-colour masks (white adds that country colour; PdxMeshColor bases only)")
        f = QFormLayout(box)
        self.spec_masks = {}
        for channel in COUNTRY_REGIONS:
            edit, row = _file_row(f"PNG mask for the {TARGET_LABELS[channel]} colour")
            self.spec_masks[channel] = edit
            f.addRow(TARGET_LABELS[channel].capitalize(), row)
        return box

    def _gloss_box(self) -> QGroupBox:
        box = QGroupBox("Gloss")
        f = QFormLayout(box)
        self.gloss_preset = QComboBox()
        self.gloss_preset.addItems(["(unchanged)", *(f"{k} ({v}%)" for k, v in GLOSS_PRESETS.items())])
        self.gloss_region = QComboBox()
        self.gloss_region.addItems(list(REGIONS))
        self.gloss_region.setCurrentText("metal")
        self.gloss_mask, row = _file_row("optional PNG mask instead of a region")
        f.addRow("Preset", self.gloss_preset)
        f.addRow("Region", self.gloss_region)
        f.addRow("Mask", row)
        return box

    def _paint_cloth(self) -> None:
        c = self._cloth
        self.cloth_btn.setText(f"{c.red()} {c.green()} {c.blue()}")
        self.cloth_btn.setStyleSheet(f"background:{c.name()}; color:{'#fff' if c.lightness() < 128 else '#000'}")

    def _pick_cloth(self) -> None:
        c = QColorDialog.getColor(self._cloth, self, "Cloth colour")
        if c.isValid():
            self._cloth = c
            self.cloth_on.setChecked(True)
            self._paint_cloth()

    # ------------------------------------------------------------------ form <-> recipe
    def form(self) -> SimpleSlot:
        def text(edit: QLineEdit) -> str | None:
            return edit.text().strip() or None
        preset = self.gloss_preset.currentText().split(" ")[0]
        c = self._cloth
        return SimpleSlot(
            base=text(self.base), protect_skin=self.protect_skin.isChecked(),
            protect_metal=self.protect_metal.isChecked(), protect_mask=text(self.protect_mask),
            palette_image=text(self.palette_image),
            palette_targets=[r for r, cb in self.palette_targets.items() if cb.isChecked()],
            palette_fixed=self.palette_fixed.isChecked(),
            cloth_colour=(c.red(), c.green(), c.blue()) if self.cloth_on.isChecked() else None,
            cloth_strength=self.cloth_strength.value(),
            pattern_image=text(self.pattern_image), pattern_region=self.pattern_region.currentText().strip(),
            pattern_blend=self.pattern_blend.currentText(), pattern_opacity=self.pattern_opacity.value(),
            pattern_scale=self.pattern_scale.value(),
            spec_masks={ch: e.text().strip() for ch, e in self.spec_masks.items() if e.text().strip()},
            gloss_preset=preset if preset in GLOSS_PRESETS else None,
            gloss_region=self.gloss_region.currentText(), gloss_mask=text(self.gloss_mask),
            extra=list(getattr(self, "_extra", [])))

    def set_form(self, f: SimpleSlot) -> None:
        self.base.setText(f.base or "")
        self.protect_skin.setChecked(f.protect_skin)
        self.protect_metal.setChecked(f.protect_metal)
        self.protect_mask.setText(f.protect_mask or "")
        self.palette_image.setText(f.palette_image or "")
        for region, cb in self.palette_targets.items():
            cb.setChecked(region in f.palette_targets)
        self.palette_fixed.setChecked(f.palette_fixed)
        self.cloth_on.setChecked(f.cloth_colour is not None)
        if f.cloth_colour is not None:
            self._cloth = QColor(*f.cloth_colour)
        self._paint_cloth()
        self.cloth_strength.setValue(f.cloth_strength)
        self.pattern_image.setText(f.pattern_image or "")
        self.pattern_region.setCurrentText(f.pattern_region)
        self.pattern_blend.setCurrentText(f.pattern_blend)
        self.pattern_opacity.setValue(f.pattern_opacity)
        self.pattern_scale.setValue(f.pattern_scale)
        for ch, edit in self.spec_masks.items():
            edit.setText(f.spec_masks.get(ch, ""))
        i = self.gloss_preset.findText(f"{f.gloss_preset} ({GLOSS_PRESETS[f.gloss_preset]}%)") if f.gloss_preset else 0
        self.gloss_preset.setCurrentIndex(max(0, i))
        self.gloss_region.setCurrentText(f.gloss_region)
        self.gloss_mask.setText(f.gloss_mask or "")
        self._extra = list(f.extra)
        if f.extra:
            self.show_message(f"This slot also has {len(f.extra)} operation(s) the form doesn't show "
                              f"({', '.join(op['op'] for op in f.extra)}); they're kept and run last.")

    def slot_key(self) -> tuple[int, str]:
        return int(self.tier.currentText()), self.role.currentText()

    def current_slot(self) -> SlotRecipe:
        """The form as a recipe slot. Raises SpecError on a bad form."""
        try:
            return self.form().to_slot()
        except RecipeError as exc:
            raise SpecError(f"slot {self._slot_key[0]}/{self._slot_key[1]}: {exc}") from exc

    def store(self) -> None:
        slot = self.current_slot()
        if slot.ops or slot.base or slot.protect_mask:
            self.recipe.slots[self._slot_key] = slot
        else:
            self.recipe.slots.pop(self._slot_key, None)
        self._refresh_slots()

    def _switch_slot(self) -> None:
        try:
            self.store()
        except SpecError as exc:
            self.show_message(str(exc), error=True)
        self._slot_key = self.slot_key()
        self._extra = []
        self.show_message("")
        self.set_form(SimpleSlot.from_slot(self.recipe.slots.get(self._slot_key, SlotRecipe())))

    def _clear_slot(self) -> None:
        self.recipe.slots.pop(self._slot_key, None)
        self._extra = []
        self.set_form(SimpleSlot())
        self._refresh_slots()

    def _refresh_slots(self) -> None:
        active = [f"{t}/{r} ({', '.join(op['op'] for op in s.ops)})" for (t, r), s in sorted(self.recipe.slots.items())
                  if s.ops]
        self.slots_label.setText("Slots with changes: " + ("; ".join(active) if active else "none yet"))

    def recipe_for_spec(self) -> TextureRecipe | None:
        """The recipe the set uses (None when unticked). Raises SpecError on a bad form."""
        if not self.use.isChecked():
            return None
        self.store()
        return self.recipe

    # ------------------------------------------------------------------ files
    def _load(self) -> None:
        path = self.recipe_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load texture recipe", str(Path.cwd()), "Recipe (*.json)")
        if not path:
            return
        try:
            self.load_file(Path(path))
        except RecipeError as exc:
            self.show_message(str(exc), error=True)

    def load_file(self, path: Path) -> None:
        self.recipe = TextureRecipe.load(path)
        self.recipe_path.setText(str(path))
        self.use.setChecked(True)
        self._slot_key = self.slot_key()
        self.set_form(SimpleSlot.from_slot(self.recipe.slots.get(self._slot_key, SlotRecipe())))
        self._refresh_slots()
        self.show_message(f"Loaded {path}: {len(self.recipe.active())} slot(s) with changes.")

    def _save(self) -> None:
        path = self.recipe_path.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save texture recipe", str(Path.cwd() / "recipe.json"),
                                                  "Recipe (*.json)")
        if not path:
            return
        try:
            self.save_file(Path(path))
        except (SpecError, OSError) as exc:
            self.show_message(str(exc), error=True)

    def save_file(self, path: Path) -> None:
        self.store()
        self.recipe.root = path.parent
        self.recipe.save(path)
        self.recipe_path.setText(str(path))
        self.show_message(f"Saved {path}. Image paths in it are kept as you entered them.")

    # ------------------------------------------------------------------ results
    def show_preview(self, preview) -> None:
        for label, img in zip(self.previews, (preview.guide, preview.before, preview.after)):
            label.setPixmap(_pixmap(img))
        notes = "; ".join(preview.notes)
        self.show_message(f"Base {preview.base} ({preview.shader})." + (f" {notes}" if notes else "")
                          + ("" if preview.shader == "PdxMeshColor" else " This base takes no country colours."))

    def show_guide(self, paths: list[Path]) -> None:
        folder = paths[0].parent if paths else ""
        self.show_message(f"Wrote {len(paths)} PNGs to {folder}: paint masks over them in GIMP/Krita "
                          "(white = selected), then pick them in the form.")

