"""The Parts screen (Phase 3): a parts loadout, one slot (tier + body) at a time, with a fit
preview rendered in background Blender.

The form maps to core.loadout.RoleLoadout: one box per attach bone (right hand, left hand,
head), each with a part kind, its params ("auto" = the kind's default, fitted to the body) and
zone colours. Previews run through export.pipeline.Session (via the main window).
"""
from __future__ import annotations

import io
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from eu4unitgen.core.loadout import STANCE_ROLES, STANCES, Loadout, LoadoutError, RoleLoadout, check_slot
from eu4unitgen.core.naming import LOADOUT_ROLES, TIERS
from eu4unitgen.core.parts import KINDS, Choice, PartSpec
from eu4unitgen.export.pipeline import SpecError
from eu4unitgen.gui.pages import Page, _path_row
from eu4unitgen.textures.procedural import COUNTRY_CHANNELS

BONES = (("Right_hand_node", "Right hand (weapon)"), ("Left_hand_node", "Left hand (shield)"), ("head", "Head"))
NONE = "(nothing: keep the base's)"
ZONE_MODES = ("default", "fixed colour", *COUNTRY_CHANNELS)
CHANNEL_LABEL = {"country1": "primary", "country2": "secondary", "country3": "tertiary"}
ROLE_TIP = {"infantry": "The infantry entity; also the gun crew unless the crew is dressed apart.",
            "rider": "The cavalry rider (same body mesh as the infantry; horse_* animations).",
            "crew": "The artillery gun crew. Dressing it apart gives it its own entity (TAG_ARTILLERY_n_crew)."}


def _pixmap(img, width: int) -> QPixmap:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    pm = QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    return pm.scaledToWidth(width, Qt.SmoothTransformation)


class _ZoneRow(QWidget):
    """One colour zone: default / fixed colour (button) / a country colour."""

    def __init__(self, zone):
        super().__init__()
        self.zone = zone
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        default = (f"country colour ({CHANNEL_LABEL[zone.channel]})" if zone.channel
                   else "rgb " + " ".join(map(str, zone.colour)))
        self.mode.addItems([f"default: {default}", "fixed colour",
                            *(f"country colour ({CHANNEL_LABEL[c]})" for c in COUNTRY_CHANNELS)])
        self.button = QPushButton()
        self.colour = QColor(*zone.colour)
        self.button.clicked.connect(self._pick)
        self.mode.currentIndexChanged.connect(lambda _: self.button.setEnabled(self.mode.currentIndex() == 1))
        self.button.setEnabled(False)
        lay.addWidget(self.mode, 1)
        lay.addWidget(self.button)
        self._paint()

    def _paint(self) -> None:
        c = self.colour
        self.button.setText(f"{c.red()} {c.green()} {c.blue()}")
        self.button.setStyleSheet(f"background:{c.name()}; color:{'#fff' if c.lightness() < 128 else '#000'}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(self.colour, self, f"{self.zone.label} colour")
        if c.isValid():
            self.colour = c
            self._paint()

    def value(self):
        i = self.mode.currentIndex()
        if i == 0:
            return None
        if i == 1:
            return [self.colour.red(), self.colour.green(), self.colour.blue()]
        return COUNTRY_CHANNELS[i - 2]

    def set_value(self, value) -> None:
        if value is None:
            self.mode.setCurrentIndex(0)
        elif isinstance(value, str):
            self.mode.setCurrentIndex(2 + COUNTRY_CHANNELS.index(value))
        else:
            self.colour = QColor(*value)
            self._paint()
            self.mode.setCurrentIndex(1)


class _BoneBox(QGroupBox):
    """The part on one bone: kind, params, colours, blazon."""

    def __init__(self, bone: str, label: str):
        super().__init__(label)
        self.bone = bone
        self.lay = QVBoxLayout(self)
        self.kind = QComboBox()
        self.lay.addWidget(self.kind)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.lay.addWidget(self.body)
        self.params: dict[str, QWidget] = {}
        self.zones: dict[str, _ZoneRow] = {}
        self.blazon: QLineEdit | None = None
        self.kind.currentIndexChanged.connect(lambda _: self._rebuild())

    def fill_kinds(self, role: str) -> None:
        keep = self.kind_name()
        self.kind.blockSignals(True)
        self.kind.clear()
        self.kind.addItem(NONE, None)
        for k in KINDS.values():
            if k.bone == self.bone and role in k.roles:
                self.kind.addItem(k.label, k.name)
        i = self.kind.findData(keep)
        self.kind.setCurrentIndex(max(0, i))
        self.kind.blockSignals(False)
        self._rebuild()

    def kind_name(self) -> str | None:
        return self.kind.currentData()

    def _rebuild(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.params, self.zones, self.blazon = {}, {}, None
        name = self.kind_name()
        if name is None:
            return
        k = KINDS[name]
        for pname, p in k.params.items():
            if isinstance(p, Choice):
                w = QComboBox()
                w.addItems(list(p.options))
                w.setCurrentText(p.default)
                w.setToolTip(p.tip)
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setSingleStep(p.step)
                w.setRange(p.lo - p.step, p.hi)  # the bottom value reads "auto": default, fitted to the body
                w.setSpecialValueText("auto")
                w.setValue(w.minimum())
                fitted = " (fitted to the body)" if pname in k.scaled or (name == "crest" and pname in ("base", "radius")) else ""
                w.setToolTip((f"{p.tip}. " if p.tip else "") + f"auto = {p.default}{fitted}; range {p.lo} to {p.hi}")
            self.params[pname] = w
            self.form.addRow(p.label, w)
        for z in k.zones:
            row = _ZoneRow(z)
            self.zones[z.name] = row
            self.form.addRow(f"{z.label} colour", row)
        if k.blazon:
            self.blazon = QLineEdit()
            self.blazon.setPlaceholderText("optional PNG painted on the face (transparent = no paint)")
            self.form.addRow("Blazon", _path_row(self.blazon, pick_dir=False))

    def spec(self) -> PartSpec | None:
        name = self.kind_name()
        if name is None:
            return None
        k = KINDS[name]
        params = {}
        for pname, w in self.params.items():
            p = k.params[pname]
            if isinstance(w, QComboBox):
                if w.currentText() != p.default:
                    params[pname] = w.currentText()
            elif w.value() > w.minimum():
                params[pname] = round(w.value(), 3)
        colours = {z: row.value() for z, row in self.zones.items() if row.value() is not None}
        blazon = self.blazon.text().strip() or None if self.blazon is not None else None
        return PartSpec(name, params, colours, blazon)

    def set_spec(self, spec: PartSpec | None) -> None:
        i = self.kind.findData(spec.kind if spec else None)
        self.kind.setCurrentIndex(max(0, i))
        self._rebuild()
        if spec is None:
            return
        for pname, value in spec.params.items():
            w = self.params.get(pname)
            if isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif w is not None:
                w.setValue(float(value))
        for z, row in self.zones.items():
            row.set_value(spec.colours.get(z))
        if self.blazon is not None:
            self.blazon.setText(spec.blazon or "")


class PartsPage(Page):
    title = "4. Parts"

    def __init__(self, window):
        super().__init__(window)
        self.loadout = Loadout(root=Path.cwd())
        self._slot_key: tuple[int, str] = (1, "infantry")

        top = QHBoxLayout()
        self.use = QCheckBox("Use a parts loadout for this set")
        self.loadout_path = QLineEdit()
        self.loadout_path.setPlaceholderText("loadout .json (blazon paths inside it are relative to its folder)")
        load, save = QPushButton("Load"), QPushButton("Save")
        load.clicked.connect(self._load)
        save.clicked.connect(self._save)
        top.addWidget(self.use)
        top.addWidget(_path_row(self.loadout_path, pick_dir=False), 1)
        top.addWidget(load)
        top.addWidget(save)
        self.layout_.addLayout(top)

        slot_row = QHBoxLayout()
        self.tier = QComboBox()
        self.tier.addItems([str(t) for t in TIERS])
        self.role = QComboBox()
        for r in LOADOUT_ROLES:
            self.role.addItem(r)
            self.role.setItemData(self.role.count() - 1, ROLE_TIP[r], Qt.ToolTipRole)
        self.stance = QComboBox()
        self.stance.addItems(list(STANCES))
        self.stance.setToolTip("Animation set for idle/walk/attack. auto: spear_shield with a hoplon, "
                               "polearm_twohanded with a dory or sarissa, polearm_onehanded with a xiphos, else the "
                               "base culture's own. Riders always use the horse_* set.")
        self.crew_apart = QCheckBox("Dress the gun crew apart from the infantry")
        self.crew_apart.setToolTip(ROLE_TIP["crew"])
        for w, label in ((self.tier, "Tier"), (self.role, "Body"), (self.stance, "Stance")):
            slot_row.addWidget(QLabel(label))
            slot_row.addWidget(w)
        slot_row.addWidget(self.crew_apart)
        slot_row.addStretch(1)
        self.tier.currentIndexChanged.connect(self._switch_slot)
        self.role.currentIndexChanged.connect(self._switch_slot)
        self.crew_apart.toggled.connect(self._crew_toggled)
        self.layout_.addLayout(slot_row)

        body = QWidget()
        bl = QVBoxLayout(body)
        self.boxes = {bone: _BoneBox(bone, label) for bone, label in BONES}
        for box in self.boxes.values():
            bl.addWidget(box)
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        scroll.setMinimumWidth(360)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview slot (Blender)")
        self.preview_btn.clicked.connect(window.parts_preview)
        clear = QPushButton("Clear slot")
        clear.clicked.connect(self._clear_slot)
        for b in (self.preview_btn, clear):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.slots_label = QLabel()
        self.slots_label.setWordWrap(True)
        self.preview = QLabel("The preview shows the tier's body in its bind pose with the slot's parts on their "
                              "bones, front and side, then each part alone.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumWidth(300)
        self.preview.setStyleSheet("background:#222; color:#aaa")
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll)
        split.addWidget(self.preview)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        self.layout_.addWidget(split, 1)
        self.layout_.addLayout(buttons)
        self.layout_.addWidget(self.slots_label)
        self.layout_.addWidget(self.message)
        self._show_slot()

    # ------------------------------------------------------------------ form <-> loadout
    def slot_key(self) -> tuple[int, str]:
        return int(self.tier.currentText()), self.role.currentText()

    def form(self) -> RoleLoadout:
        parts = [s for s in (box.spec() for box in self.boxes.values()) if s is not None]
        return RoleLoadout(parts, self.stance.currentText())

    def current_slot(self) -> RoleLoadout:
        """The form as a slot. Raises SpecError on a bad form."""
        slot = self.form()
        tier, role = self._slot_key
        try:
            check_slot(tier, role, slot)
        except LoadoutError as exc:
            raise SpecError(str(exc)) from exc
        return slot

    def store(self) -> None:
        tier, role = self._slot_key
        slot = self.current_slot()
        keep = (role == "crew" and self.crew_apart.isChecked()) or slot.parts or slot.stance != "auto"
        if keep:
            self.loadout.slots[self._slot_key] = slot
        else:
            self.loadout.slots.pop(self._slot_key, None)
        self._refresh_slots()

    def _show_slot(self) -> None:
        tier, role = self._slot_key
        slot = self.loadout.slots.get(self._slot_key)
        crew = role == "crew"
        self.crew_apart.setVisible(crew)
        self.crew_apart.blockSignals(True)
        self.crew_apart.setChecked(crew and slot is not None)
        self.crew_apart.blockSignals(False)
        editable = not crew or slot is not None
        self.stance.setEnabled(role in STANCE_ROLES and editable)
        self.stance.setCurrentText(slot.stance if slot else "auto")
        by_bone = {KINDS[p.kind].bone: p for p in (slot.parts if slot else [])}
        for bone, box in self.boxes.items():
            box.fill_kinds(role)
            box.set_spec(by_bone.get(bone))
            box.setEnabled(editable)
        if crew and slot is None:
            self.show_message("The gun crew is the infantry entity (it carries the infantry's parts). Tick "
                              "“Dress the gun crew apart” to give it parts of its own.")

    def _switch_slot(self) -> None:
        try:
            self.store()
        except SpecError as exc:
            self.show_message(str(exc), error=True)
        self._slot_key = self.slot_key()
        self.show_message("")
        self._show_slot()

    def _crew_toggled(self, on: bool) -> None:
        if on:
            self.loadout.slots.setdefault(self._slot_key, RoleLoadout())
        else:
            self.loadout.slots.pop(self._slot_key, None)
        self.show_message("")
        self._show_slot()
        self._refresh_slots()

    def _clear_slot(self) -> None:
        self.loadout.slots.pop(self._slot_key, None)
        self._show_slot()
        self._refresh_slots()

    def _refresh_slots(self) -> None:
        def describe(s: RoleLoadout) -> str:
            bits = [p.kind for p in s.parts] or ["no parts"]
            return ", ".join(bits) + (f"; {s.stance}" if s.stance != "auto" else "")
        active = [f"{t}/{r} ({describe(s)})" for (t, r), s in sorted(self.loadout.slots.items())]
        self.slots_label.setText("Slots: " + ("; ".join(active) if active else "none yet"))

    def loadout_for_spec(self) -> Loadout | None:
        """The loadout the set uses (None when unticked). Raises SpecError on a bad form."""
        if not self.use.isChecked():
            return None
        self.store()
        return self.loadout

    def loadout_for_preview(self) -> Loadout:
        self.store()
        return self.loadout

    # ------------------------------------------------------------------ files
    def _load(self) -> None:
        path = self.loadout_path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load parts loadout", str(Path.cwd()), "Loadout (*.json)")
        if not path:
            return
        try:
            self.load_file(Path(path))
        except LoadoutError as exc:
            self.show_message(str(exc), error=True)

    def load_file(self, path: Path) -> None:
        self.loadout = Loadout.load(path)
        self.loadout_path.setText(str(path))
        self.use.setChecked(True)
        self._slot_key = self.slot_key()
        self._show_slot()
        self._refresh_slots()
        self.show_message(f"Loaded {path}: {len(self.loadout.active())} slot(s) with changes.")

    def _save(self) -> None:
        path = self.loadout_path.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save parts loadout", str(Path.cwd() / "loadout.json"),
                                                  "Loadout (*.json)")
        if not path:
            return
        try:
            self.save_file(Path(path))
        except (SpecError, OSError) as exc:
            self.show_message(str(exc), error=True)

    def save_file(self, path: Path) -> None:
        self.store()
        self.loadout.root = path.parent
        self.loadout.save(path)
        self.loadout_path.setText(str(path))
        self.show_message(f"Saved {path}. Blazon paths in it are kept as you entered them.")

    # ------------------------------------------------------------------ results
    def show_preview(self, preview) -> None:
        self.preview.setPixmap(_pixmap(preview.image, max(300, min(900, self.preview.width() - 10))))
        lines = [f"Body {preview.body}. Saved to {preview.path}."]
        for p in preview.parts:
            fitted = ", ".join(f"{k} {v}" for k, v in sorted(p["fitted"].items()))
            lines.append(f"{p['entity']} on {p['bone']} ({p['shader']})" + (f"; fitted: {fitted}" if fitted else ""))
        lines += preview.notes
        self.show_message("<br>".join(lines))
