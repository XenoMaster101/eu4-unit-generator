"""The Result screen (SPEC Track F, F10): the whole army as 3D previews (infantry, cavalry,
artillery x tiers 1-4), the Understood strip, in-game unit names, and two tabs: Parts (picture
cards, "Apply to: all tiers / this tier only") and Colours (unit colours, materials, crest,
shield emblem). The bottom bar shows only problems, and holds Test in game and Export.

Thin by design: the army (core.army) holds the description and the edits; the window turns it
into a spec, validates it and renders the gallery (export.gallery) in the background, then calls
``show_*`` here. Clicking a tile picks the unit type and tier the Parts tab shows and edits.
"""
from __future__ import annotations

import io

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QFileDialog, QFrame, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMenu,
    QPushButton, QRadioButton, QScrollArea, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from eu4unitgen.core.army import Edit
from eu4unitgen.core.emblems import EMBLEMS, draw
from eu4unitgen.core.finishes import default_colour, finishes
from eu4unitgen.core.naming import TIERS, UNIT_TYPES
from eu4unitgen.gui.home_page import Chip, css
from eu4unitgen.gui.pages import Page
from eu4unitgen.gui.theme import tokens
from eu4unitgen.gui.widgets import FlowLayout

TILE = 168
TYPE_LABEL = {"infantry": "Infantry", "cavalry": "Cavalry", "artillery": "Artillery"}
WEAPONS = {"infantry": ("dory", "sarissa", "xiphos"), "cavalry": ("xyston", "dory", "xiphos"),
           "artillery": ("xiphos", "dory")}
WEAPON_LABEL = {"dory": "Spear (dory)", "sarissa": "Long pike (sarissa)", "xiphos": "Sword (xiphos)",
                "xyston": "Lance (xyston)"}
BODY_AREAS = (("torso", "Torso (cuirass, tunic)"), ("upper_torso", "Chest"), ("lower_torso", "Skirt"),
              ("belt", "Belt"), ("collar", "Collar"), ("arms", "Arms"), ("legs", "Legs"), ("shins", "Shins (greaves)"),
              ("feet", "Feet"))
MATERIALS = ("bronze", "iron", "linen", "leather", "skin")


def _pixmap(img: Image.Image, size: int) -> QPixmap:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _swatch(colour, size: int = 16) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor(*colour))
    return QIcon(pix)


class Tile(QLabel):
    """One unit's preview; a click picks it (its unit type and tier). A stale tile (its unit is being
    checked or rendered again) keeps its old picture, dimmed."""
    clicked = Signal()
    STALE_OPACITY = 0.35

    def __init__(self):
        super().__init__()
        self.setFixedSize(TILE, TILE)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("tile")
        self.setCursor(Qt.PointingHandCursor)
        self.setWordWrap(True)

    def set_stale(self, stale: bool) -> None:
        if stale:
            effect = QGraphicsOpacityEffect(self)
            effect.setOpacity(self.STALE_OPACITY)
            self.setGraphicsEffect(effect)
        else:
            self.setGraphicsEffect(None)

    def is_stale(self) -> bool:
        return self.graphicsEffect() is not None

    def mousePressEvent(self, e) -> None:  # noqa: N802 - Qt name
        self.clicked.emit()


class Card(QFrame):
    """A picture card: the part, its name and what it is, and a Change menu."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.picture = QLabel()
        self.picture.setFixedSize(132, 104)
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setObjectName("cardpic")
        lay.addWidget(self.picture, 0, Qt.AlignHCenter)
        self.title = QLabel(f"<b>{title}</b>")
        lay.addWidget(self.title)
        self.caption = QLabel()
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)
        self.change = QToolButton()
        self.change.setText("Change  ▾")
        self.change.setPopupMode(QToolButton.InstantPopup)
        self.change.setObjectName("change")
        lay.addWidget(self.change)


class ResultPage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.layout_.itemAt(0).widget().hide()
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.selected = ("infantry", 1)
        self.look = None  # the army's current look (core.look.Look)
        self.gallery_parts: dict = {}  # (unit type, tier) -> {bone: part entity}
        self.thumbs: dict = {}  # part entity -> picture
        self.images: dict = {}  # (unit type, tier) -> picture

        top = QHBoxLayout()
        brand = QLabel("eu4unitgen")
        brand.setObjectName("brand")
        top.addWidget(brand)
        top.addStretch(1)
        adv = QPushButton("Advanced editor")
        adv.setObjectName("link")
        adv.setCursor(Qt.PointingHandCursor)
        adv.setToolTip("Every setting, page by page: textures, parts, validation, test mod")
        adv.clicked.connect(lambda: window.show_advanced())
        top.addWidget(adv)
        self.layout_.addLayout(top)

        bar = QHBoxLayout()
        new = QPushButton("← New army")
        new.setObjectName("ghost")
        new.setToolTip("Back to the description (it's kept)")
        new.clicked.connect(lambda: window.show_home())
        bar.addWidget(new, 0, Qt.AlignTop)
        self.understood_box = QWidget()
        self.understood_flow = FlowLayout(self.understood_box)
        bar.addWidget(self.understood_box, 1)
        self.layout_.addLayout(bar)

        names = QHBoxLayout()
        names.addWidget(QLabel("Names in game:"))
        self.names: dict[str, QLineEdit] = {}
        for utype in UNIT_TYPES:
            edit = QLineEdit()
            edit.setPlaceholderText(f"{TYPE_LABEL[utype]} name")
            edit.setToolTip(f"What the game calls your {utype} (tiers add I-IV). Empty: \"<Adjective> "
                            f"{TYPE_LABEL[utype]}\".")
            edit.editingFinished.connect(lambda u=utype: self._name_changed(u))
            names.addWidget(QLabel(TYPE_LABEL[utype]))
            names.addWidget(edit, 1)
            self.names[utype] = edit
        self.layout_.addLayout(names)

        body = QHBoxLayout()
        grid_box = QWidget()
        grid = QGridLayout(grid_box)
        grid.setSpacing(8)
        self.tiles: dict[tuple[str, int], Tile] = {}
        for c, tier in enumerate(TIERS):
            h = QLabel(f"<b>Tier {tier}</b>")
            h.setAlignment(Qt.AlignCenter)
            grid.addWidget(h, 0, c + 1)
        for r, utype in enumerate(UNIT_TYPES):
            grid.addWidget(QLabel(f"<b>{TYPE_LABEL[utype]}</b>"), r + 1, 0)
            for c, tier in enumerate(TIERS):
                tile = Tile()
                tile.setToolTip(f"{TYPE_LABEL[utype]}, tier {tier}. Click to change its parts.")
                tile.clicked.connect(lambda u=utype, t=tier: self.select(u, t))
                grid.addWidget(tile, r + 1, c + 1)
                self.tiles[(utype, tier)] = tile
        body.addWidget(grid_box, 0, Qt.AlignTop)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(400)
        self.tabs.addTab(self._parts_tab(), "Parts")
        self.tabs.addTab(self._colours_tab(), "Colours")
        body.addWidget(self.tabs, 1)
        self.layout_.addLayout(body, 1)

        bottom = QHBoxLayout()
        self.status = QLabel("Not checked yet.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.test = QPushButton("Test in game")
        self.test.setObjectName("ghost")
        self.test.setToolTip("Build a test mod with this army, ready to copy into EU4's mod folder")
        self.test.clicked.connect(lambda: window.result_test())
        self.export = QPushButton("Export  ⇩")
        self.export.setObjectName("generate")
        self.export.setToolTip("Write the finished files for your mod (you'll see the list first)")
        self.export.clicked.connect(lambda: window.result_export())
        bottom.addWidget(self.test)
        bottom.addWidget(self.export)
        self.layout_.addLayout(bottom)
        self.layout_.addWidget(self.message)
        self.restyle()
        self.select(*self.selected)

    def restyle(self) -> None:
        t = tokens()
        self.setStyleSheet(css(t) + f"""
QLabel#tile {{ background: {t['surface']}; border: 2px solid {t['border']}; border-radius: 8px; color: {t['muted']}; }}
QLabel#tile[selected="true"] {{ border: 3px solid {t['accent']}; }}
QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QLabel#cardpic {{ background: {t['alt']}; border-radius: 6px; color: {t['muted']}; }}
QToolButton#change {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
                     border-radius: 6px; padding: 4px 10px; }}
QToolButton#change::menu-indicator {{ image: none; width: 0; }}
QTabWidget::pane {{ border: 1px solid {t['border']}; background: {t['window']}; }}
QTabBar::tab {{ background: {t['alt']}; color: {t['text']}; padding: 6px 14px; border: 1px solid {t['border']}; }}
QTabBar::tab:selected {{ background: {t['surface']}; font-weight: 600; }}
QRadioButton {{ color: {t['text']}; }}
QPushButton#swatch {{ border: 1px solid {t['border']}; border-radius: 8px; padding: 6px 10px; }}
QLabel#bad {{ color: {t['error']}; font-weight: 600; }}
QLabel#good {{ color: {t['ok']}; }}
""")

    # ------------------------------------------------------------------ building the tabs
    def _parts_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.showing = QLabel()
        v.addWidget(self.showing)
        row = QHBoxLayout()
        row.addWidget(QLabel("Apply to:"))
        self.scope_all, self.scope_one = QRadioButton("All tiers"), QRadioButton("This tier only")
        self.scope_all.setChecked(True)
        self.scope_all.setToolTip("A change here changes every tier of this unit type")
        self.scope_one.setToolTip("A change here changes only the tier picked in the gallery")
        group = QButtonGroup(w)
        for b in (self.scope_all, self.scope_one):
            group.addButton(b)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        cards = QGridLayout()
        self.cards = {k: Card(t) for k, t in (("helmet", "Helmet"), ("weapon", "Weapon"), ("shield", "Shield"),
                                              ("body", "Body look"))}
        for i, k in enumerate(("helmet", "weapon", "shield", "body")):
            cards.addWidget(self.cards[k], i // 2, i % 2)
        v.addLayout(cards)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _colours_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("<b>Unit colours</b> (what the game tints the unit with)"))
        row = QHBoxLayout()
        self.unit_swatches = []
        for i, label in enumerate(("Main", "Second", "Third")):
            b = QPushButton(label)
            b.setObjectName("swatch")
            b.setToolTip(f"The {label.lower()} unit colour; click to pick another")
            b.clicked.connect(lambda _=False, i=i: self._pick_unit_colour(i))
            row.addWidget(b)
            self.unit_swatches.append(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel("<b>Materials</b> (click one to tint it everywhere)"))
        row = QHBoxLayout()
        self.material_swatches = {}
        for m in MATERIALS:
            b = QPushButton(finishes()[m]["label"].split(" (")[0].capitalize())
            b.setObjectName("swatch")
            b.setToolTip(f"Tint {m} on every body area that uses it")
            b.clicked.connect(lambda _=False, m=m: self._pick_material(m))
            row.addWidget(b)
            self.material_swatches[m] = b
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.crest_swatch = QPushButton("Crest colour")
        self.crest_swatch.setObjectName("swatch")
        self.crest_swatch.setToolTip("The horsehair crest's colour (every unit type)")
        self.crest_swatch.clicked.connect(self._pick_crest)
        row.addWidget(self.crest_swatch)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel("<b>Shield emblem</b>"))
        erow = QHBoxLayout()
        self.emblem_preview = QLabel()
        self.emblem_preview.setFixedSize(84, 84)
        erow.addWidget(self.emblem_preview)
        ev = QVBoxLayout()
        pick = QToolButton()
        pick.setText("Choose emblem  ▾")
        pick.setObjectName("change")
        pick.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        for name, label in EMBLEMS.items():
            act = QAction(_swatch((128, 128, 128)), label, menu)
            act.setIcon(QIcon(_pixmap(draw(name, (40, 40, 40)), 24)))
            act.triggered.connect(lambda _=False, n=name: self.window_.result_emblem(n))
            menu.addAction(act)
        pick.setMenu(menu)
        own = QPushButton("Your own (PNG)…")
        own.setObjectName("ghost")
        own.setToolTip("A PNG you made or may use; transparent where the shield shows through")
        own.clicked.connect(self._own_emblem)
        none = QPushButton("No emblem")
        none.setObjectName("ghost")
        none.clicked.connect(lambda: self.window_.result_edit(Edit("emblem", (), None)))
        for b in (pick, own, none):
            ev.addWidget(b)
        erow.addLayout(ev)
        erow.addStretch(1)
        v.addLayout(erow)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ showing the army
    def show_army(self, army, look, interpretation, skin) -> None:
        """Chips, names, cards and swatches from the army as it is now (gallery comes separately)."""
        self.look, self.skin = look, skin
        self._fill_chips(army, interpretation)
        for utype, edit in self.names.items():
            edit.setText((look.names or {}).get(utype, ""))
        self._fill_cards()
        self._fill_colours(army)

    def _fill_chips(self, army, interp) -> None:
        self.understood_flow.clear()
        n_text = len(interp.items) - len(army.image_items)
        for n, item in enumerate(interp.items):
            from_image = n >= n_text
            swatch = item.colour or ((item.params or {}).get("swatch") if from_image else None)
            chip = Chip(item.text(), "image" if from_image else "item", tip="Click to change it, x to remove it.",
                        swatch=tuple(swatch) if swatch else None)
            chip.closed.connect(lambda n=n, im=from_image: self.window_.result_chip_remove(n, im))
            menu = self.window_.home._swap_menu(n, item, from_image, self.window_.result_chip_swap, self)
            if menu is not None:
                chip.set_menu(menu)
            self.understood_flow.addWidget(chip)

    def show_status(self, text: str, good: bool | None = None) -> None:
        self.status.setText(text)
        self.status.setObjectName("good" if good else ("bad" if good is False else "muted"))
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def show_report(self, report) -> None:
        """Validation runs silently: only problems are shown (errors stop Export)."""
        errors = list(report.errors)
        if errors:
            self.show_status(f"✗ {len(errors)} problem(s) to fix before export: " + "; ".join(i.message for i in errors[:3])
                             + (" …" if len(errors) > 3 else ""), False)
            self.status.setToolTip("\n".join(str(i) for i in errors))
        else:
            self.show_status("✓ Checked: no problems", True)
            self.status.setToolTip("")
        self.export.setEnabled(not errors)

    def show_waiting(self, text: str, units=None) -> None:
        for key in units or self.tiles:
            tile = self.tiles[key]
            if key not in self.images:
                tile.setText(text)

    def mark_stale(self, keys, stale: bool = True) -> None:
        """Dim the tiles of ``keys`` while their units are redone (or undim them: unchanged)."""
        for key in keys:
            self.tiles[key].set_stale(stale and key in self.images)

    def show_gallery(self, gallery, units=None) -> None:
        """The rendered tiles; a unit of ``units`` that rendered no picture loses its old one."""
        for key in units or ():
            if key not in gallery.images:
                self.images.pop(key, None)
                self.tiles[key].clear()
                self.tiles[key].set_stale(False)
        for key, img in gallery.images.items():
            self.images[key] = img
            self.tiles[key].setPixmap(_pixmap(img, TILE - 6))
            self.tiles[key].set_stale(False)
        self.gallery_parts.update(gallery.parts)
        self.thumbs.update(gallery.thumbs)
        self._fill_cards()

    def clear_gallery(self) -> None:
        self.images.clear()
        for tile in self.tiles.values():
            tile.clear()
            tile.set_stale(False)

    def select(self, utype: str, tier: int) -> None:
        self.selected = (utype, tier)
        for key, tile in self.tiles.items():
            tile.setProperty("selected", "true" if key == self.selected else "false")
            tile.style().unpolish(tile)
            tile.style().polish(tile)
        self.scope_one.setText(f"Tier {tier} only")
        self._fill_cards()

    # ------------------------------------------------------------------ Parts cards
    def tier_scope(self) -> int | None:
        return self.selected[1] if self.scope_one.isChecked() else None

    def _fill_cards(self) -> None:
        utype, tier = self.selected
        self.showing.setText(f"Showing: <b>{TYPE_LABEL[utype]}, tier {tier}</b> (click a picture on the left to "
                             "pick another)")
        if self.look is None:
            return
        tl = self.look.tier(tier)
        role = tl["roles"].get(utype) or {}
        parts = self.gallery_parts.get((utype, tier), {})

        def picture(card: Card, bone: str | None, fallback: str) -> None:
            entity = parts.get(bone) if bone else None
            if entity and entity in self.thumbs:
                card.picture.setPixmap(_pixmap(self.thumbs[entity], 100))
            else:
                card.picture.clear()
                card.picture.setText(fallback)

        helmet = self.cards["helmet"]
        helmet.title.setText("<b>Helmet</b>" if utype == "infantry" else "<b>Crest</b>")
        if utype == "infantry":
            h = role.get("helmet")
            crest = {"fore_aft": "front to back", "transverse": "side to side", "none": "none"}.get(
                (h or {}).get("crest", "fore_aft"), "")
            helmet.caption.setText(f"{h.get('style', 'attic').capitalize()}, crest {crest}" if h
                                   else "the base body's own helmet")
        else:
            c = role.get("crest")
            helmet.caption.setText({"fore_aft": "Crest front to back", "transverse": "Crest side to side"}.get(
                c["orientation"], "Crest") if c else "none (base helmet)")
        picture(helmet, "head", "base helmet")
        helmet.change.setMenu(self._helmet_menu(utype, role))

        weapon = self.cards["weapon"]
        weapon.caption.setText(WEAPON_LABEL.get(role.get("weapon"), "the base weapon"))
        picture(weapon, "Right_hand_node", "base weapon")
        weapon.change.setMenu(self._choice_menu(utype, "weapon", [(w, WEAPON_LABEL[w]) for w in WEAPONS[utype]]
                                                + [(None, "Keep the base weapon")]))

        shield = self.cards["shield"]
        small = (role.get("shield_params") or {}).get("radius")
        shield.caption.setText(("Small round shield" if small else "Round shield") if role.get("shield") else "no shield")
        picture(shield, "Left_hand_node", "no shield")
        menu = QMenu(self)
        for label, value, params in (("Round shield", "hoplon", None), ("Small round shield", "hoplon", {"radius": 2.6}),
                                     ("No shield", None, None)):
            act = QAction(label, menu)
            act.triggered.connect(lambda _=False, v=value, p=params, u=utype: self._shield(u, v, p))
            menu.addAction(act)
        shield.change.setMenu(menu)

        body = self.cards["body"]
        picked = ", ".join(f"{a.replace('_', ' ')} {f[0] if isinstance(f, list) else f}" for a, f in tl["body"].items())
        body.caption.setText(picked or "the base texture")
        if (utype, tier) in self.images:
            body.picture.setPixmap(_pixmap(self.images[(utype, tier)], 100))
        else:
            body.picture.setText("rendering…")
        menu = QMenu(self)
        for area, label in BODY_AREAS:
            sub = menu.addMenu(label)
            for fname, f in finishes().items():
                if fname == "colour":
                    continue
                act = QAction(f["label"], sub)
                act.triggered.connect(lambda _=False, a=area, fn=fname: self._finish(a, fn))
                sub.addAction(act)
            act = QAction("A colour of your choice…", sub)
            act.triggered.connect(lambda _=False, a=area: self._finish_colour(a))
            sub.addAction(act)
            act = QAction("Keep the base texture", sub)
            act.triggered.connect(lambda _=False, a=area: self._finish(a, None))
            sub.addAction(act)
        body.change.setMenu(menu)

    def _helmet_menu(self, utype: str, role: dict) -> QMenu:
        menu = QMenu(self)
        if utype == "infantry":
            h = dict(role.get("helmet") or {"style": "attic", "crest": "fore_aft"})
            for style in ("corinthian", "attic", "pilos"):
                act = QAction(f"{style.capitalize()} helmet", menu)
                act.triggered.connect(lambda _=False, s=style: self._part("infantry", "helmet", {**h, "style": s}))
                menu.addAction(act)
            menu.addSeparator()
            for crest, label in (("fore_aft", "Crest front to back"), ("transverse", "Crest side to side"),
                                 ("none", "No crest")):
                act = QAction(label, menu)
                act.triggered.connect(lambda _=False, c=crest: self._part("infantry", "helmet", {**h, "crest": c}))
                menu.addAction(act)
            menu.addSeparator()
            act = QAction("No Greek helmet (the base one)", menu)
            act.triggered.connect(lambda: self._part("infantry", "helmet", None))
            menu.addAction(act)
        else:
            for value, label in (({"orientation": "fore_aft"}, "Crest front to back"),
                                 ({"orientation": "transverse"}, "Crest side to side"), (None, "No crest")):
                act = QAction(label, menu)
                act.triggered.connect(lambda _=False, v=value: self._part(utype, "crest", v))
                menu.addAction(act)
        return menu

    def _choice_menu(self, utype: str, slot: str, choices) -> QMenu:
        menu = QMenu(self)
        for value, label in choices:
            act = QAction(label, menu)
            act.triggered.connect(lambda _=False, v=value: self._part(utype, slot, v))
            menu.addAction(act)
        return menu

    def _part(self, utype: str, slot: str, value) -> None:
        self.window_.result_edit(Edit("part", (utype, slot), value, self.tier_scope()))

    def _shield(self, utype: str, value, params) -> None:
        tier = self.tier_scope()
        self.window_.result_edit(Edit("part", (utype, "shield"), value, tier), Edit("part", (utype, "shield_params"),
                                                                                    params, tier))

    def _finish(self, area: str, finish) -> None:
        self.window_.result_edit(Edit("finish", (area,), finish, self.tier_scope()))

    def _finish_colour(self, area: str) -> None:
        c = QColorDialog.getColor(QColor(150, 60, 40), self, f"Colour for {area.replace('_', ' ')}")
        if c.isValid():
            self.window_.result_edit(Edit("finish", (area,), ["colour", [c.red(), c.green(), c.blue()]],
                                          self.tier_scope()))

    def _name_changed(self, utype: str) -> None:
        text = self.names[utype].text().strip()
        if self.look is not None and text != (self.look.names or {}).get(utype, ""):
            self.window_.result_edit(Edit("name", (utype,), text))

    # ------------------------------------------------------------------ Colours tab
    def _fill_colours(self, army) -> None:
        look = self.look
        colours = look.colours or army.base_colours or [[150, 28, 32], [195, 150, 60], [230, 222, 200]]
        for b, c in zip(self.unit_swatches, colours):
            b.setIcon(_swatch(c, 22))
        for m, b in self.material_swatches.items():
            b.setIcon(_swatch(default_colour(m, self.skin) or (128, 128, 128), 22))
        self.crest_swatch.setIcon(_swatch(look.crest_colour or (170, 20, 25), 22))
        face = colours[0]
        img = Image.new("RGBA", (84, 84), (0, 0, 0, 0))
        from PIL import ImageDraw

        ImageDraw.Draw(img).ellipse((2, 2, 82, 82), fill=tuple(face) + (255,), outline=(125, 85, 38, 255), width=4)
        if look.emblem:
            try:
                with Image.open(look.emblem) as em:
                    img.alpha_composite(em.convert("RGBA").resize((60, 60)), (12, 12))
            except OSError:
                pass
        buf = io.BytesIO()
        img.save(buf, "PNG")
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        self.emblem_preview.setPixmap(pix)

    def _pick_unit_colour(self, i: int) -> None:
        c = QColorDialog.getColor(QColor(128, 128, 128), self, "Unit colour")
        if c.isValid():
            self.window_.result_edit(Edit("unit_colour", (i,), [c.red(), c.green(), c.blue()]))

    def _pick_material(self, m: str) -> None:
        start = default_colour(m, self.skin) or (128, 128, 128)
        c = QColorDialog.getColor(QColor(*start), self, f"Tint {m}")
        if c.isValid():
            self.window_.result_edit(Edit("material_colour", (m,), [c.red(), c.green(), c.blue()]))

    def _pick_crest(self) -> None:
        c = QColorDialog.getColor(QColor(170, 20, 25), self, "Crest colour")
        if c.isValid():
            self.window_.result_edit(Edit("crest_colour", (), [c.red(), c.green(), c.blue()]))

    def _own_emblem(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Shield emblem (a PNG you made or may use)", "", "PNG (*.png)")
        if path:
            self.window_.result_edit(Edit("emblem", (), path))
