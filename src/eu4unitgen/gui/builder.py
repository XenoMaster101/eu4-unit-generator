"""Build a unit (the top bar's Build a unit; SPEC Track L): a unit put together from library parts, on one screen.

  1 the type (infantry = a soldier; cavalry = a horse + a rider; artillery = a gun + its crew)
  2 the country or graphical culture (optional: it marks the suggested parts ★, names the unit, and is Add to mod's
    default)
  3 the parts per tier: four compact columns, each with that type's slots and one preview of the whole unit at that
    tier; "the game's own" keeps the game's look there
  a bar always in view at the bottom: the name (it follows the type and country until you type one) and Build it

No AI is needed: assembling uses parts already made. Thin by design: the parts, suggestions, previews and the unit's
record are ``core.library``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea, QVBoxLayout, QWidget,
)

from eu4unitgen.gui.base import Page, card, framed, ghost, label, primary, thumb

TYPES = (("infantry", "soldier", "Infantry", "A soldier on foot. Also crews your guns."),
         ("cavalry", "horse", "Cavalry", "A horse with a soldier riding it."),
         ("artillery", "gun", "Artillery", "A gun with a soldier as its crew."))
GAME = "— the game's own —"
PREVIEW = (168, 120)


class BuildUnitScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        self.utype = "cavalry"
        self.found = None  # core.library.Country
        self.auto_name = ""  # the name the app suggested (replaced as the type and country change)
        self.slots: dict[tuple[int, str], QComboBox] = {}
        self.previews: dict[int, object] = {}
        self.captions: dict[int, object] = {}
        self.layout_.setContentsMargins(24, 12, 24, 0)
        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("BUILD A UNIT", "step"))
        head.addWidget(label("Put a unit together from your library", "h1"))
        head.addWidget(label("Pick its parts tier by tier. No AI needed; building takes a few minutes.", "explain"))
        self.layout_.addLayout(head)
        body = QWidget()
        col = QVBoxLayout(body)
        col.setSpacing(8)
        col.setContentsMargins(0, 4, 8, 8)
        # 1 the type, 2 the country: one row
        top = QHBoxLayout()
        top.setSpacing(18)
        left = QVBoxLayout()
        left.addWidget(label("1  What kind of unit?", "h2"))
        row = QHBoxLayout()
        self.type_group = QButtonGroup(self)
        self.type_cards = {}
        for key, ic, title, text in TYPES:
            c = card(title, text, text, checkable=True, icon=ic)
            c.setMinimumHeight(64)
            c.setMaximumHeight(78)
            c.clicked.connect(lambda _=False, k=key: self.set_type(k))
            self.type_group.addButton(c)
            self.type_cards[key] = c
            row.addWidget(c)
        left.addLayout(row)
        self.on_foot = QCheckBox("On foot: the soldier is the cavalry, no horse (a weapon per tier)")
        self.on_foot.setToolTip("Cavalry without a horse: each tier's soldier stands in as the cavalry, playing the "
                                "infantry animations, with the base-game weapon you pick (Tracen Academy's way)")
        self.on_foot.toggled.connect(lambda *_: self.refill())
        left.addWidget(self.on_foot)
        top.addLayout(left, 3)
        right = QVBoxLayout()
        right.addWidget(label("2  For which country?  (optional)", "h2"))
        row = QHBoxLayout()
        self.tag = QLineEdit()
        self.tag.setPlaceholderText("tag, e.g. BYZ")
        self.tag.setMaximumWidth(110)
        self.tag.setToolTip("The country's three-letter tag (BYZ Byzantium, ENG England, KAZ Kazan…). The app reads its "
                            "graphical culture, technology group and culture from your EU4 install to suggest parts.")
        self.tag.returnPressed.connect(self.look_up)
        row.addWidget(self.tag)
        row.addWidget(ghost("Look up", self.look_up, "Read the country from your EU4 install"))
        self.gfx = QComboBox()
        self.gfx.setToolTip("Or a graphical culture: the family of looks the game gives countries. Suggests parts that "
                            "fit it; Add to mod uses it for achievement-safe mods.")
        self.gfx.currentIndexChanged.connect(lambda *_: (self._suggest_name(), self.refill()))
        row.addWidget(self.gfx, 1)
        right.addLayout(row)
        self.country_line = label("", "explain")
        right.addWidget(self.country_line)
        right.addStretch(1)
        top.addLayout(right, 2)
        col.addLayout(top)
        # 3 the tiers
        row = QHBoxLayout()
        row.addWidget(label("3  Which parts at each tier?", "h2", wrap=False))
        row.addStretch(1)
        self.copy_btn = ghost("Use tier 1's parts for every tier", self.copy_first,
                              "Fill tiers 2-4 with tier 1's parts (change any of them after)")
        self.clear_btn = ghost("Clear tiers 2-4", self.clear_rest, "Tiers 2-4 back to the game's own look")
        row.addWidget(self.copy_btn)
        row.addWidget(self.clear_btn)
        col.addLayout(row)
        col.addWidget(label("The army changes its look four times as military technology grows. ★ marks the parts that "
                            "suit the country; “the game's own” keeps the game's look at that tier.", "explain"))
        self.tiers_box = QGridLayout()
        self.tiers_box.setHorizontalSpacing(10)
        col.addLayout(self.tiers_box)
        self.no_part = label("", "explain")
        self.no_part.setTextFormat(Qt.RichText)
        col.addWidget(self.no_part)
        col.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(body)
        self.layout_.addWidget(sc, 1)
        # the bar at the bottom, always in view
        bar = QFrame()
        bar.setObjectName("sticky")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 10, 16, 10)
        bl.addWidget(label("Name", "h3", wrap=False))
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. byzantine_cavalry")
        self.name.setMaximumWidth(300)
        self.name.setToolTip("Lower-case letters, digits and _ only. Its name in game is made from it "
                             "(byzantine_cavalry → “Byzantine Cavalry”). It follows the type and country until you "
                             "type your own.")
        self.name.textEdited.connect(lambda *_: self._ready())
        self.name.textChanged.connect(lambda *_: self._ready())
        bl.addWidget(self.name)
        self.why = label("", "hint")
        self.own_colours = QCheckBox("Let each country paint its cloth")
        self.own_colours.setChecked(True)
        self.own_colours.setToolTip("On: the cloth takes each country's unit colour in game, like the game's own units. "
                                    "Off: it keeps your pictures' colours everywhere.")
        bl.addWidget(self.own_colours)
        bl.addWidget(self.why, 1)
        bl.addWidget(ghost("Make a part", lambda: self.window_.go("new"),
                           "Make a horse, a soldier or a gun from a picture (needs the AI setup)"))
        self.build_btn = primary("Build it", self.build, "Put the parts together, check them and draw the preview")
        bl.addWidget(self.build_btn)
        self.layout_.addWidget(bar)
        self.layout_.addWidget(self.message)
        self.restyle()

    # ------------------------------------------------------------------ entering
    def enter(self, utype: str | None = None, pick: str | None = None) -> None:
        """Open with a type, and optionally a part already picked for tier 1 (from the Library's Use in a unit)."""
        from eu4unitgen.core import config, library
        if self.gfx.count() == 0:
            self.gfx.blockSignals(True)
            self.gfx.addItem("any graphical culture", "")
            for g in library.graphical_cultures(config.load().eu4()):
                self.gfx.addItem(g, g)
            self.gfx.blockSignals(False)
        p = library.part(pick) if pick else None
        if utype is None and p is not None:
            utype = {"horse": "cavalry", "gun": "artillery"}.get(p.kind, "infantry")
        self.set_type(utype or self.utype)
        if p is not None:
            box = self.slots.get((1, p.kind))
            if box is not None:
                i = box.findData(p.ref)
                if i >= 0:
                    box.setCurrentIndex(i)
        self.show_message("")

    def set_type(self, utype: str) -> None:
        self.utype = utype
        self.type_cards[utype].setChecked(True)
        self.on_foot.setVisible(utype == "cavalry")
        self._suggest_name()
        self.refill()

    # ------------------------------------------------------------------ the country and the name
    def look_up(self) -> None:
        from eu4unitgen.core import config, library
        tag = self.tag.text().strip().upper()
        if not tag:
            self.found = None
            self.country_line.setText("")
            self._suggest_name()
            return self.refill()
        try:
            c = library.country(tag, config.load().eu4())
        except OSError as exc:
            return self.show_message(f"Couldn't read the EU4 install ({exc}). Check Setup.", error=True)
        self.found = c
        if c is None:
            self.country_line.setText(f"{tag} isn't a country in your game. Check the tag (three letters).")
        else:
            self.gfx.blockSignals(True)
            self.gfx.setCurrentIndex(max(0, self.gfx.findData(c.gfx or "")))
            self.gfx.blockSignals(False)
            self.country_line.setText(f"{pretty(c.name)} ({c.tag}): {c.gfx}, {c.tech_group} technology, {c.culture} "
                                      f"culture ({c.culture_group}). ★ marks the parts that suit it.")
        self._suggest_name()
        self.refill()

    def _suggest_name(self) -> None:
        """The name follows the type and the country (byzantine_cavalry, eastern_artillery, my_infantry) until the
        user types one of their own."""
        if not hasattr(self, "name"):
            return
        c = self.found
        stem = ""
        if c is not None:
            stem = (c.culture_group or "").removesuffix("_group") or pretty(c.name).split(" ")[0].lower()
        elif self.gfx.currentData():
            stem = self.gfx.currentData().removesuffix("gfx")
        new = f"{stem or 'my'}_{self.utype}"
        if not self.name.text().strip() or self.name.text().strip() == self.auto_name:
            self.name.blockSignals(True)
            self.name.setText(new)
            self.name.blockSignals(False)
        self.auto_name = new

    # ------------------------------------------------------------------ the tiers
    def refill(self) -> None:
        """The tier columns for this type, each slot's list ranked for the country (★ the suggested ones)."""
        from eu4unitgen.core import library
        keep = {k: b.currentData() for k, b in self.slots.items()}
        while self.tiers_box.count():
            w = self.tiers_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.slots, self.previews, self.captions = {}, {}, {}
        c = self.found
        gfx = self.gfx.currentData() or (c.gfx if c else None)
        foot = self.foot()
        kinds = ("soldier", "weapon") if foot else library.UNIT_TYPES[self.utype]
        ranked = {k: library.suggest(k, gfx, c.tech_group if c else None, c.culture if c else None,
                                     c.culture_group if c else None) for k in kinds if k != "weapon"}
        for t in range(1, 5):
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(2, 2, 2, 2)
            lay.setSpacing(4)
            lay.addWidget(label(library.TIER_TEXT[t], "h3", wrap=False))
            pv = label("")
            pv.setObjectName("cardpic")
            pv.setAlignment(Qt.AlignCenter)
            pv.setFixedHeight(PREVIEW[1])
            lay.addWidget(pv)
            cap = label("", "explain")
            cap.setAlignment(Qt.AlignCenter)
            lay.addWidget(cap)
            self.previews[t] = pv
            self.captions[t] = cap
            for k in kinds:
                lay.addWidget(label(library.SLOT_TEXT[(self.utype, k)], "muted", wrap=False))
                box = QComboBox()
                if k == "weapon":  # the base game's weapons, by name (none from DLC)
                    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                    box.setMinimumContentsLength(12)
                    for entity, node, stance, text in library.WEAPONS:
                        box.addItem(text, entity or None)
                        box.setItemData(box.count() - 1, f"{entity or 'nothing'} in the {'right' if node == 'Right_hand_node' else 'left'} "
                                        f"hand; {stance} animations" if entity else "its own animations, nothing held",
                                        Qt.ToolTipRole)
                    box.setToolTip(f"What the cavalry on foot holds at tier {t}: a base-game weapon, with the "
                                   "animations that go with it")
                    old = keep.get((t, k))
                    if old:
                        box.setCurrentIndex(max(0, box.findData(old)))
                    lay.addWidget(box)
                    self.slots[(t, k)] = box
                    continue
                # long part names mustn't widen the page: four columns fit the window, the full name is in the list
                box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                box.setMinimumContentsLength(12)
                box.addItem(GAME, None)
                for p, why in ranked[k]:
                    star = "★ " if why else ""
                    box.addItem(f"{star}{p.title}{'  · starter' if p.starter else ''}", p.ref)
                    box.setItemData(box.count() - 1, (why or "No tag of this part speaks for or against this "
                                                             "country.") + (f" Tags: {p.tag_text()}." if p.tag_text() else ""),
                                    Qt.ToolTipRole)
                box.setToolTip(f"The {library.SLOT_TEXT[(self.utype, k)].lower()} at tier {t}: any part of your library, "
                               "★ the ones that suit the country")
                old = keep.get((t, k))
                if old:
                    box.setCurrentIndex(max(0, box.findData(old)))
                box.currentIndexChanged.connect(lambda *_, tt=t: self._picked(tt))
                lay.addWidget(box)
                self.slots[(t, k)] = box
            self.tiers_box.addWidget(framed(inner), 0, t - 1)
            self._picked(t)
        missing = [library.SLOT_TEXT[(self.utype, k)].lower() for k in kinds if k != "weapon" and not ranked[k]]
        ok, why = library.ai_ready()
        self.no_part.setText((f"Your library has no {' or '.join(missing)} yet. Make one with Make a part"
                              + ("." if ok else f": {why}") if missing else "") or ("" if ok else why))
        self._ready()

    def _picked(self, t: int) -> None:
        """Tier ``t``'s preview: the whole unit as built before, else its main part with a note; a small mark for the
        game's own."""
        from eu4unitgen.core import library
        from eu4unitgen.gui.icons import pixmap as icon_pixmap
        from eu4unitgen.gui.theme import tokens
        pv, cap = self.previews.get(t), self.captions.get(t)
        if pv is None:
            return
        slots = {k: b.currentData() for (tt, k), b in self.slots.items()
                 if tt == t and b.currentData() and k != "weapon"}
        render, whole, shown = library.unit_preview(self.utype, slots)
        if not slots:
            pv.setPixmap(icon_pixmap({"infantry": "soldier", "cavalry": "horse", "artillery": "gun"}[self.utype],
                                     tokens()["card_muted"], 36, 1.3))
            pv.setToolTip("The game's own look at this tier")
            cap.setText("the game's own look")
        else:
            pix = thumb(render, *PREVIEW)
            pv.setPixmap(pix)
            pv.setToolTip("The whole unit at this tier, as the game draws it" if whole else
                          "The full preview (the whole unit together) is drawn when you build it")
            what = library.SLOT_TEXT[(self.utype, shown)].lower() if shown else ""
            gun = " with the game's own gun" if self.utype == "artillery" and "gun" not in slots else ""
            cap.setText("the whole unit" if whole or self.utype == "infantry" else
                        f"the {what}{gun}; the whole unit is drawn when you build it")
        self._ready()

    def copy_first(self) -> None:
        for (t, k), box in self.slots.items():
            if t > 1:
                box.setCurrentIndex(max(0, box.findData(self.slots[(1, k)].currentData())))

    def clear_rest(self) -> None:
        for (t, _), box in self.slots.items():
            if t > 1:
                box.setCurrentIndex(0)

    def foot(self) -> bool:
        return self.utype == "cavalry" and hasattr(self, "on_foot") and self.on_foot.isChecked()

    def tiers(self) -> dict[int, dict[str, str]]:
        out: dict[int, dict[str, str]] = {}
        for (t, k), box in self.slots.items():
            if box.currentData():
                out.setdefault(t, {})[k] = box.currentData()
        return out

    # ------------------------------------------------------------------ building
    def _problem(self) -> str:
        from eu4unitgen.core import library
        from eu4unitgen.gui.new_unit import name_problem
        tiers = self.tiers()
        if not tiers:
            return "Pick the parts for at least one tier."
        for t, s in sorted(tiers.items()):
            missing = [library.SLOT_TEXT[(self.utype, k)].lower() for k in library.needed(self.utype, self.foot())
                       if k not in s]
            if missing:
                return f"Tier {t} needs a {' and a '.join(missing)} too."
        return name_problem(self.name.text().strip())

    def _ready(self) -> None:
        if not hasattr(self, "build_btn"):
            return
        why = self._problem()
        self.why.setText(why)
        self.build_btn.setEnabled(not why)

    def build(self) -> None:
        from eu4unitgen.core import library
        if self._problem():
            return self.show_message(self._problem(), error=True)
        try:
            f = library.new_unit(self.name.text().strip(), self.utype, self.tiers(), self.tag.text().strip(),
                                 self.gfx.currentData() or "", country_colours=self.own_colours.isChecked(),
                                 on_foot=self.foot())
        except (ValueError, FileExistsError, OSError) as exc:
            return self.show_message(str(exc), error=True)
        from eu4unitgen.ref3d import flow
        flow.start(f)  # assembling needs no AI: a few minutes, in its own process
        self.show_message(f"Building {f.name}. Opening it in the Library.")
        self.window_.go("units", folder=str(f))


def pretty(name: str) -> str:
    """"ByzantineEmpire" -> "Byzantine Empire"."""
    return "".join(" " + ch if ch.isupper() and k else ch for k, ch in enumerate(name or "")).strip()
