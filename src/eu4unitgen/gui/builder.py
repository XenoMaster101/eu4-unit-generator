"""Build a unit (the top bar's New unit; SPEC Track L): a unit put together from library parts, one page.

  1 the type (infantry = a soldier; cavalry = a horse + a rider; artillery = a gun + its crew)
  2 the country or graphical culture (optional: it marks the suggested parts ★ and is Add to mod's default)
  3 the parts per tier: four columns, each with that type's slots; "the game's own" leaves a tier as the game draws it
  4 the name, then Build it

No AI is needed: assembling uses parts already made. Thin by design: the parts, suggestions and the unit's record are
``core.library``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea, QVBoxLayout, QWidget,
)

from eu4unitgen.gui.base import Page, card, framed, ghost, label, pixmap, primary

TYPES = (("infantry", "🛡  Infantry", "A soldier on foot. Also crews your guns."),
         ("cavalry", "🐎  Cavalry", "A horse with a soldier riding it."),
         ("artillery", "💣  Artillery", "A gun with a soldier as its crew."))
GAME = "— the game's own —"


class BuildUnitScreen(Page):
    def __init__(self, window):
        super().__init__(window)
        self.utype = "cavalry"
        self.found = None  # core.library.Country
        self.slots: dict[tuple[int, str], QComboBox] = {}
        self.pics: dict[tuple[int, str], object] = {}
        head = QVBoxLayout()
        head.addWidget(label("NEW UNIT", "step"))
        head.addWidget(label("Build a unit from your library", "h1"))
        head.addWidget(label("Pick the parts it's made of, tier by tier. Parts are made once from a picture (New part) "
                             "and can be used in as many units as you like. Building needs no AI and takes a few "
                             "minutes.", "explain"))
        self.layout_.addLayout(head)
        body = QWidget()
        col = QVBoxLayout(body)
        col.setSpacing(10)
        # 1 the type
        col.addWidget(label("1  What kind of unit?", "h2"))
        row = QHBoxLayout()
        self.type_group = QButtonGroup(self)
        self.type_cards = {}
        for key, title, text in TYPES:
            c = card(title, text, text, checkable=True)
            c.setMinimumHeight(70)
            c.clicked.connect(lambda _=False, k=key: self.set_type(k))
            self.type_group.addButton(c)
            self.type_cards[key] = c
            row.addWidget(c)
        col.addLayout(row)
        # 2 the country
        col.addWidget(label("2  For which country?  (optional)", "h2"))
        row = QHBoxLayout()
        self.tag = QLineEdit()
        self.tag.setPlaceholderText("tag, e.g. BYZ")
        self.tag.setMaximumWidth(140)
        self.tag.setToolTip("The country's three-letter tag (BYZ Byzantium, FRA France, KAZ Kazan…). The app reads its "
                            "graphical culture, technology group and culture from your EU4 install to suggest parts.")
        self.tag.returnPressed.connect(self.look_up)
        row.addWidget(self.tag)
        row.addWidget(ghost("Look up", self.look_up, "Read the country from your EU4 install"))
        row.addSpacing(16)
        row.addWidget(label("or a graphical culture", wrap=False))
        self.gfx = QComboBox()
        self.gfx.setToolTip("The family of looks the game gives countries. Suggests parts that fit it; Add to mod uses "
                            "it for achievement-safe mods.")
        self.gfx.currentIndexChanged.connect(lambda *_: self.refill())
        row.addWidget(self.gfx)
        row.addStretch(1)
        col.addLayout(row)
        self.country_line = label("", "explain")
        col.addWidget(self.country_line)
        # 3 the tiers
        col.addWidget(label("3  Which parts at each tier?", "h2"))
        col.addWidget(label("A country's army changes its look four times as its military technology grows. Pick the "
                            "parts for the tiers you want; leave a tier on “the game's own” to keep the game's look "
                            "there. ★ marks the parts that suit the country.", "explain"))
        self.tiers_box = QGridLayout()
        self.tiers_box.setHorizontalSpacing(12)
        col.addLayout(self.tiers_box)
        tools = QHBoxLayout()
        tools.addWidget(ghost("Use tier 1's parts for every tier", self.copy_first,
                              "Fill tiers 2-4 with tier 1's parts (change any of them after)"))
        tools.addWidget(ghost("Clear tiers 2-4", self.clear_rest, "Tiers 2-4 back to the game's own look"))
        tools.addStretch(1)
        col.addLayout(tools)
        # 4 the name
        col.addWidget(label("4  Name it and build it", "h2"))
        row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. byzantine_cavalry")
        self.name.setMaximumWidth(320)
        self.name.setToolTip("Lower-case letters, digits and _ only. Its name in game is made from it "
                             "(byzantine_cavalry → “Byzantine Cavalry”).")
        self.name.textChanged.connect(lambda *_: self._ready())
        row.addWidget(self.name)
        self.build_btn = primary("Build it", self.build, "Put the parts together, check them and draw the preview")
        row.addWidget(self.build_btn)
        self.why = label("", "hint")
        row.addWidget(self.why)
        row.addStretch(1)
        col.addLayout(row)
        self.no_part = label("", "explain")
        self.no_part.setTextFormat(Qt.RichText)
        col.addWidget(self.no_part)
        more = QHBoxLayout()
        more.addWidget(ghost("＋  New part from a picture", lambda: self.window_.go("new"),
                             "Make a horse, a soldier or a gun from a picture (needs the AI setup)"))
        more.addStretch(1)
        col.addLayout(more)
        col.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(body)
        self.layout_.addWidget(sc, 1)
        self.layout_.addWidget(self.message)
        self.restyle()

    # ------------------------------------------------------------------ entering
    def enter(self, utype: str | None = None, pick: str | None = None) -> None:
        """Open with a type, and optionally a part already picked for tier 1 (from the Library's Use in a unit)."""
        from eu4unitgen.core import config, library
        if self.gfx.count() == 0:
            self.gfx.blockSignals(True)
            self.gfx.addItem("any", "")
            for g in library.graphical_cultures(config.load().eu4()):
                self.gfx.addItem(g, g)
            self.gfx.blockSignals(False)
        p = library.part(pick) if pick else None
        if utype is None and p is not None:
            utype = {"horse": "cavalry", "gun": "artillery"}.get(p.kind, "infantry")
        self.set_type(utype or self.utype)
        if p is not None:
            slot = p.kind
            box = self.slots.get((1, slot))
            if box is not None:
                i = box.findData(p.ref)
                if i >= 0:
                    box.setCurrentIndex(i)
        self.show_message("")

    def set_type(self, utype: str) -> None:
        self.utype = utype
        self.type_cards[utype].setChecked(True)
        self.refill()

    # ------------------------------------------------------------------ the country
    def look_up(self) -> None:
        from eu4unitgen.core import config, library
        tag = self.tag.text().strip().upper()
        if not tag:
            self.found = None
            self.country_line.setText("")
            return self.refill()
        try:
            c = library.country(tag, config.load().eu4())
        except OSError as exc:
            return self.show_message(f"Couldn't read the EU4 install ({exc}). Check Setup.", error=True)
        self.found = c
        if c is None:
            self.country_line.setText(f"{tag} isn't a country in your game. Check the tag (three letters).")
            return self.refill()
        self.gfx.blockSignals(True)
        i = self.gfx.findData(c.gfx or "")
        self.gfx.setCurrentIndex(max(0, i))
        self.gfx.blockSignals(False)
        pretty = "".join(" " + ch if ch.isupper() and k else ch for k, ch in enumerate(c.name)).strip()
        self.country_line.setText(f"{pretty} ({c.tag}): graphical culture {c.gfx}, technology {c.tech_group}, "
                                  f"culture {c.culture} ({c.culture_group}). The parts that suit it are marked ★.")
        self.refill()

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
        self.slots, self.pics = {}, {}
        c = self.found
        gfx = self.gfx.currentData() or (c.gfx if c else None)
        kinds = library.UNIT_TYPES[self.utype]
        ranked = {k: library.suggest(k, gfx, c.tech_group if c else None, c.culture if c else None,
                                     c.culture_group if c else None) for k in kinds}
        for t in range(1, 5):
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(label(library.TIER_TEXT[t], "h3", wrap=False))
            for k in kinds:
                lay.addWidget(label(library.SLOT_TEXT[(self.utype, k)], "muted", wrap=False))
                box = QComboBox()
                # long part names mustn't widen the page: four columns fit the window, the full name is in the list
                box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                box.setMinimumContentsLength(14)
                box.addItem(GAME, None)
                for p, why in ranked[k]:
                    star = "★ " if why else ""
                    tags = f"  ({p.tag_text()})" if p.tag_text() else ""
                    box.addItem(f"{star}{p.title}{tags}{'  · starter' if p.starter else ''}", p.ref)
                    box.setItemData(box.count() - 1, why or "No tag of this part speaks for or against this country.",
                                    Qt.ToolTipRole)
                box.setToolTip(f"The {library.SLOT_TEXT[(self.utype, k)].lower()} at tier {t}: any part of your library, "
                               "★ the ones that suit the country")
                old = keep.get((t, k))
                if old:
                    i = box.findData(old)
                    box.setCurrentIndex(max(0, i))
                box.currentIndexChanged.connect(lambda *_, tt=t, kk=k: self._picked(tt, kk))
                lay.addWidget(box)
                pic = label("")
                pic.setAlignment(Qt.AlignCenter)
                pic.setMinimumHeight(120)
                lay.addWidget(pic)
                self.slots[(t, k)] = box
                self.pics[(t, k)] = pic
                self._picked(t, k)
            lay.addStretch(1)
            self.tiers_box.addWidget(framed(inner), 0, t - 1)
        missing = [library.SLOT_TEXT[(self.utype, k)].lower() for k in kinds if not ranked[k]]
        ok, why = library.ai_ready()
        self.no_part.setText(
            (f"<p>Your library has no {' or '.join(missing)} yet. Make one with New part"
             + (". " if ok else f": {why}</p>") if missing else "") or ("" if ok else f"<p>{why}</p>"))
        self._ready()

    def _picked(self, t: int, k: str) -> None:
        from eu4unitgen.core import library
        box, pic = self.slots[(t, k)], self.pics[(t, k)]
        p = library.part(box.currentData())
        if p is None:
            pic.setPixmap(pixmap(None, 1, 1))
            pic.setText("the game's own look")
        else:
            pic.setText("")
            pic.setPixmap(pixmap(p.thumb or p.image, 150, 120))
        self._ready()

    def copy_first(self) -> None:
        for (t, k), box in self.slots.items():
            if t > 1:
                box.setCurrentIndex(max(0, box.findData(self.slots[(1, k)].currentData())))

    def clear_rest(self) -> None:
        for (t, _), box in self.slots.items():
            if t > 1:
                box.setCurrentIndex(0)

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
        kinds = library.UNIT_TYPES[self.utype]
        for t, s in sorted(tiers.items()):
            missing = [library.SLOT_TEXT[(self.utype, k)].lower() for k in kinds if k not in s]
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
                                 self.gfx.currentData() or "")
        except (ValueError, FileExistsError, OSError) as exc:
            return self.show_message(str(exc), error=True)
        self.show_message(f"Saved {f.name}. Opening it in the Library.")
        self.window_.go("units", folder=str(f))
