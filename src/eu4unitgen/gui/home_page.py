"""The Home screen (SPEC Track F, F8): one box to describe the soldiers, chips for what was
understood, a nation picker and one Generate button. Everything else is behind "Advanced editor".

Thin by design: reading the text is core.prompt; a reference image is core.reference; the look
is core.look. The text box is the one source of truth: a chip's x or swap rewrites the text,
which is read again (chips from a reference image are kept apart, since they have no words).
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QCompleter, QFileDialog, QFrame, QHBoxLayout, QLabel, QMenu, QPlainTextEdit, QPushButton, QToolButton,
    QVBoxLayout, QWidget,
)

from eu4unitgen.core.look import templates
from eu4unitgen.core.prompt import Item, KeywordInterpreter, VocabularyError, text_with_template
from eu4unitgen.gui.pages import Page
from eu4unitgen.gui.theme import tokens
from eu4unitgen.gui.widgets import FlowLayout

TIER_CHOICES = (("heavier", "Gets heavier with tech"), ("same", "Same look on all tiers"))
def arrow(colour: str) -> str:
    """A small down-arrow PNG in ``colour`` (a stylesheet can't draw one; Fusion's own arrow
    disappears once a combo box has a styled border). Written once per colour to the temp folder."""
    import tempfile

    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPainter, QPolygonF

    path = Path(tempfile.gettempdir()) / f"eu4unitgen_arrow_{colour.lstrip('#')}.png"
    if not path.is_file():
        pix = QPixmap(20, 12)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(colour))
        p.drawPolygon(QPolygonF([QPointF(1, 1), QPointF(19, 1), QPointF(10, 11)]))
        p.end()
        pix.save(str(path))
    return path.as_posix()


def css(t: dict[str, str]) -> str:
    """The Home screen's stylesheet: every colour from the theme's tokens (gui.theme)."""
    return f"""
QWidget#home {{ background: {t['window']}; color: {t['text']}; }}
QWidget#home QLabel {{ color: {t['text']}; background: transparent; }}
QWidget#home QLabel#brand {{ font-weight: 700; color: {t['text']}; }}
QWidget#home QLabel#h1 {{ font-size: 24px; font-weight: 600; color: {t['text']}; }}
QWidget#home QLabel#muted {{ color: {t['muted']}; }}
QWidget#home QLabel#status {{ color: {t['ok']}; }}
QWidget#home QLabel#warn {{ color: {t['warn']}; }}
QWidget#home QLabel#hint {{ color: {t['error']}; font-weight: 600; }}
QPlainTextEdit#prompt {{ font-size: 15px; border: 2px solid {t['border']}; border-radius: 12px; padding: 10px;
                        background: {t['surface']}; color: {t['text']}; }}
QComboBox, QLineEdit {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
                       border-radius: 6px; padding: 4px 8px; min-height: 22px; }}
QComboBox[missing="true"] {{ border: 2px solid {t['error']}; }}
QComboBox QLineEdit {{ border: none; padding: 0; background: transparent; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; }}
QComboBox::down-arrow {{ image: url({arrow(t['text'])}); width: 10px; height: 6px; }}
QPushButton#generate {{ background: {t['accent']}; color: {t['accent_text']}; font-size: 16px; font-weight: 600;
                       border: none; border-radius: 10px; padding: 9px 26px; }}
QPushButton#generate:disabled {{ background: {t['accent_off']}; color: {t['accent_text']}; }}
QPushButton#ghost, QPushButton#template {{ background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 8px; padding: 6px 14px; }}
QPushButton#template {{ font-weight: 600; }}
QPushButton#ghost:hover, QPushButton#template:hover {{ border-color: {t['accent']}; }}
QPushButton#link {{ border: none; color: {t['link']}; text-decoration: underline; background: transparent; }}
QFrame#chip {{ background: {t['chip']}; border: 1px solid {t['chip_border']}; border-radius: 13px; }}
QFrame#chip[kind="image"] {{ background: {t['image_chip']}; border-color: {t['image_border']}; }}
QFrame#chip[kind="left"] {{ background: {t['left_chip']}; border-color: {t['left_border']}; }}
QFrame#chip[kind="suggest"] {{ background: {t['surface']}; border-style: dashed; }}
QFrame#chip QToolButton {{ border: none; background: transparent; color: {t['text']}; padding: 3px 6px; }}
QFrame#chip QToolButton::menu-indicator {{ image: none; width: 0; }}
"""


class Chip(QFrame):
    """A rounded label; clicking it opens ``menu`` (or emits ``clicked``); an optional x."""
    clicked = Signal()
    closed = Signal()

    def __init__(self, text: str, kind: str = "item", closable: bool = True, tip: str = "",
                 swatch: tuple | None = None):
        super().__init__()
        self.setObjectName("chip")
        self.setProperty("kind", kind)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 4, 0)
        lay.setSpacing(0)
        self.button = QToolButton()
        self.button.setText(text)
        self.button.setToolTip(tip)
        if swatch:
            pix = QPixmap(14, 14)
            pix.fill(QColor(*swatch))
            from PySide6.QtGui import QPainter

            p = QPainter(pix)  # an outline, so white on a light chip (or black on a dark one) still shows
            p.setPen(QColor(tokens()["muted"]))
            p.drawRect(0, 0, 13, 13)
            p.end()
            self.button.setIcon(QIcon(pix))
            self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.clicked.connect(self.clicked)
        lay.addWidget(self.button)
        if closable:
            x = QToolButton()
            x.setText("×")
            x.setToolTip("Remove this")
            x.clicked.connect(self.closed)
            lay.addWidget(x)
            self.x = x

    def set_menu(self, menu: QMenu) -> None:
        """Clicking opens ``menu``; the arrow is part of the text, so it can't overlap it."""
        self.button.setMenu(menu)
        self.button.setPopupMode(QToolButton.InstantPopup)
        if not self.button.text().endswith("▾"):
            self.button.setText(self.button.text() + "  ▾")


def _chip_row(title: str) -> tuple[QWidget, FlowLayout, QLabel]:
    w = QWidget()
    outer = QHBoxLayout(w)
    outer.setContentsMargins(0, 0, 0, 0)
    label = QLabel(title)
    label.setObjectName("muted")
    label.setFixedWidth(118)
    label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    outer.addWidget(label)
    box = QWidget()
    flow = FlowLayout(box)
    outer.addWidget(box, 1)
    return w, flow, label


class HomePage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.layout_.itemAt(0).widget().hide()  # Page's heading: this screen has its own
        self.setObjectName("home")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.restyle()
        self.interp = KeywordInterpreter()
        self.result = self.interp.interpret("")
        self.image_items: list[Item] = []
        self.image_path: Path | None = None

        top = QHBoxLayout()
        brand = QLabel("eu4unitgen")
        brand.setObjectName("brand")
        top.addWidget(brand)
        top.addStretch(1)
        self.status = QLabel("Looking for EU4, DLC and Blender…")
        self.status.setObjectName("muted")
        self.status.setToolTip("Found on this computer at start-up; change them in the Advanced editor's Settings")
        top.addWidget(self.status)
        top.addSpacing(16)
        self.advanced = QPushButton("Advanced editor")
        self.advanced.setObjectName("link")
        self.advanced.setCursor(Qt.PointingHandCursor)
        self.advanced.setToolTip("Every setting, page by page: textures, parts, validation, test mod")
        self.advanced.clicked.connect(lambda: window.show_advanced())
        top.addWidget(self.advanced)
        self.layout_.addLayout(top)
        self.layout_.addStretch(1)

        col = QVBoxLayout()
        col.setSpacing(10)
        h1 = QLabel("What should your soldiers look like?")
        h1.setObjectName("h1")
        h1.setAlignment(Qt.AlignCenter)
        sub = QLabel("Describe them in your own words. You can change every detail afterwards.")
        sub.setObjectName("muted")
        sub.setAlignment(Qt.AlignCenter)
        col.addWidget(h1)
        col.addWidget(sub)
        self.prompt = QPlainTextEdit()
        self.prompt.setObjectName("prompt")
        self.prompt.setPlaceholderText("Describe your soldiers… e.g. \"Bronze-armoured hoplites with red horsehair "
                                       "crests, round shields and long spears; linen cuirass, bare arms\"")
        self.prompt.setFixedHeight(110)
        self.prompt.setToolTip("Words for weapons, shields, helmets, crests, materials, garments and colours are "
                               "understood; the chips below show what was")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.refresh)
        self.prompt.textChanged.connect(self._timer.start)
        col.addWidget(self.prompt)

        rows = {}
        for key, title in (("suggest", "Suggestions"), ("understood", "Understood"), ("left", "Not understood")):
            w, flow, _ = _chip_row(title + ":")
            rows[key] = (w, flow)
            col.addWidget(w)
        self.suggest_row, self.suggest_flow = rows["suggest"]
        self.understood_row, self.understood_flow = rows["understood"]
        self.left_row, self.left_flow = rows["left"]
        self.notes = QLabel()
        self.notes.setObjectName("warn")
        self.notes.setWordWrap(True)
        col.addWidget(self.notes)

        row = QHBoxLayout()
        self.upload = QPushButton("Upload reference image")
        self.upload.setObjectName("ghost")
        self.upload.setToolTip("A front view of one soldier. Use only pictures you made or are allowed to use. "
                               "Its colours become suggestions you can remove.")
        self.upload.clicked.connect(self._pick_image)
        row.addWidget(self.upload)
        self.thumb = QLabel()
        self.thumb.setFixedSize(40, 40)
        self.thumb.setVisible(False)
        row.addWidget(self.thumb)
        row.addSpacing(10)
        row.addWidget(QLabel("For:"))
        self.nation = QComboBox()
        self.nation.setEditable(True)
        self.nation.setInsertPolicy(QComboBox.NoInsert)
        self.nation.setFixedWidth(250)
        self.nation.lineEdit().setPlaceholderText("type a country name or tag")
        self.nation.setToolTip("The country whose army gets this look (from the game and your mod)")
        completer = QCompleter(self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self._names = QStringListModel(self)
        completer.setModel(self._names)
        self.nation.setCompleter(completer)
        self.nation.editTextChanged.connect(lambda _: self._check_ready())
        self.nation.currentIndexChanged.connect(lambda _: self._check_ready())
        row.addWidget(self.nation)
        row.addStretch(1)
        self.generate = QPushButton("Generate  ▶")
        self.generate.setObjectName("generate")
        self.generate.setToolTip("Make the whole army (infantry, cavalry, artillery, tiers 1-4) from this description")
        self.generate.clicked.connect(lambda: window.home_generate())
        row.addWidget(self.generate)
        col.addLayout(row)
        self.ready_hint = QLabel()
        self.ready_hint.setObjectName("hint")
        col.addWidget(self.ready_hint)

        tw, tflow, _ = _chip_row("Start from:")
        for name, t in templates().items():
            b = QPushButton(t["label"])
            b.setObjectName("template")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(t["blurb"] + " Click to start from this.")
            b.clicked.connect(lambda _=False, n=name: self.use_template(n))
            tflow.addWidget(b)
        trow = QHBoxLayout()  # the templates, and on the right how the tiers differ
        trow.addWidget(tw, 1)
        trow.addWidget(QLabel("Tiers:"), 0, Qt.AlignTop)
        self.tiers = QComboBox()
        for value, label in TIER_CHOICES:
            self.tiers.addItem(label, value)
        self.tiers.setToolTip("Whether the look changes as military technology improves (tiers 1-4)")
        trow.addWidget(self.tiers, 0, Qt.AlignTop)
        col.addLayout(trow)
        foot = QLabel("Read offline by keywords: nothing leaves this computer.")
        foot.setObjectName("muted")
        col.addWidget(foot)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        box = QWidget()
        box.setLayout(col)
        box.setMaximumWidth(900)
        box.setMinimumWidth(640)
        wrap.addWidget(box, 4)
        wrap.addStretch(1)
        self.layout_.addLayout(wrap)
        self.layout_.addStretch(2)
        self.layout_.addWidget(self.message)
        self.refresh()
        self._check_ready()

    def restyle(self) -> None:
        """Take the theme's colours (gui.theme) again: at start and when the theme changes."""
        self.setStyleSheet(css(tokens()))

    def _check_ready(self) -> None:
        """Generate needs a country: until one is picked, the For: field is outlined and Generate is off."""
        missing = self.tag() is None
        self.nation.setProperty("missing", "true" if missing else "false")
        self.nation.style().unpolish(self.nation)
        self.nation.style().polish(self.nation)
        self.generate.setEnabled(not missing and not self._busy)
        self.generate.setToolTip("Pick a country first (For:)" if missing else
                                 "Make the whole army (infantry, cavalry, artillery, tiers 1-4) from this description")
        self.ready_hint.setText("Pick a country first: type its name or tag in For:" if missing else "")
        self.ready_hint.setVisible(missing)

    _busy = False

    def set_busy(self, busy: bool) -> None:
        super().set_busy(busy)
        self._busy = busy
        if not busy:
            self._check_ready()

    # ------------------------------------------------------------------ status and nations
    def set_status(self, parts: list[tuple[bool, str]]) -> None:
        """``parts``: (ok, text) for EU4, DLC and Blender."""
        self.status.setText("   ".join(f"{'✓' if ok else '✗'} {text}" for ok, text in parts))
        self.status.setObjectName("status" if all(ok for ok, _ in parts) else "warn")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def fill_nations(self, rows: list[tuple[str, str, bool]]) -> None:
        """(tag, name, mod-only) from Session.country_list."""
        current = self.tag()
        self.nation.blockSignals(True)
        self.nation.clear()
        labels = []
        for tag, name, mod_only in rows:
            label = f"{name} ({tag})" + ("  · mod" if mod_only else "")
            self.nation.addItem(label, tag)
            labels.append(label)
        self._names.setStringList(labels)
        self.nation.setCurrentIndex(-1)
        self.nation.setEditText("")
        self.nation.blockSignals(False)
        if current:
            self.set_nation(current)

    def set_nation(self, tag: str) -> None:
        i = self.nation.findData(tag.upper())
        if i >= 0:
            self.nation.setCurrentIndex(i)
        else:
            self.nation.setEditText(tag.upper())

    def tag(self) -> str | None:
        """The chosen country's tag: the picked row, else "(TAG)" or a bare tag typed in."""
        text = self.nation.currentText().strip()
        i = self.nation.findText(text)
        if i >= 0 and self.nation.itemData(i):
            return self.nation.itemData(i)
        m = re.search(r"\(([A-Za-z][A-Za-z0-9]{2})\)", text) or re.fullmatch(r"([A-Za-z][A-Za-z0-9]{2})", text)
        return m.group(1).upper() if m else None

    # ------------------------------------------------------------------ reading
    def text(self) -> str:
        return self.prompt.toPlainText()

    def set_text(self, text: str) -> None:
        self.prompt.setPlainText(text)
        cur = self.prompt.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        self.prompt.setTextCursor(cur)
        self.refresh()

    def refresh(self) -> None:
        self._timer.stop()
        self.result = self.interp.interpret(self.text(), self.image_items)
        prog = next((i.value for i in self.result.items if i.kind == "progression"), None)
        if prog:
            self.tiers.setCurrentIndex(self.tiers.findData(prog))
        self._fill_understood()
        self._fill_leftovers()
        self._fill_suggestions()
        self.notes.setText("; ".join(self.result.notes))
        self.notes.setVisible(bool(self.result.notes))

    def look(self):
        """The look Generate makes: the chips' look, with the Tiers choice."""
        look = self.result.look()
        look.progression = self.tiers.currentData()
        return look

    def _fill_understood(self) -> None:
        self.understood_flow.clear()
        n_text = len(self.result.items) - len(self.image_items)
        for n, item in enumerate(self.result.items):
            from_image = n >= n_text
            swatch = item.colour or ((item.params or {}).get("swatch") if from_image else None)
            chip = Chip(item.text(), "image" if from_image else "item",
                        tip=("From the reference image" if from_image else f"From: \"{item.words}\"")
                        + ". Click to change it, x to remove it.", swatch=tuple(swatch) if swatch else None)
            chip.closed.connect(lambda n=n, im=from_image: self._remove(n, im))
            menu = self._swap_menu(n, item, from_image)
            if menu is not None:
                chip.set_menu(menu)
            self.understood_flow.addWidget(chip)
        if not self.result.items:
            hint = QLabel("nothing yet: try words like spear, round shield, bronze greaves, red crest")
            hint.setObjectName("muted")
            self.understood_flow.addWidget(hint)

    def _swap_kind(self, item: Item) -> str | None:
        return {"finish": "material", "crest_colour": "colour", "unit_colour": "colour"}.get(
            item.kind, item.kind if item.kind in ("template", "weapon", "shield", "helmet", "crest", "progression")
            else None)

    def _swap_menu(self, n: int, item: Item, from_image: bool) -> QMenu | None:
        kind = self._swap_kind(item)
        if kind is None or (kind == "material" and item.value is None):
            return None
        menu = QMenu(self)
        for entry in self.interp.vocab.alternatives(kind):
            act = QAction(entry["label"], menu)
            act.triggered.connect(lambda _=False, n=n, e=entry, im=from_image: self._swap(n, e, im))
            menu.addAction(act)
        return menu

    def _remove(self, n: int, from_image: bool) -> None:
        if from_image:
            del self.image_items[n - (len(self.result.items) - len(self.image_items))]
            return self.refresh()
        self.set_text(self.result.text_without(n))

    def _swap(self, n: int, entry: dict, from_image: bool) -> None:
        if from_image:
            k = n - (len(self.result.items) - len(self.image_items))
            self.result.swap(n, entry)
            self.image_items[k] = self.result.items[n]
            return self.refresh()
        self.set_text(self.result.text_swapped(n, entry))

    def _fill_leftovers(self) -> None:
        self.left_flow.clear()
        words = list(self.result.leftovers)
        text = self.text()
        if self.prompt.hasFocus() and text[-1:].isalpha() and words:  # still typing it
            last = re.findall(r"[A-Za-z]+", text)[-1].lower()
            words = [w for w in words if w != last]
        for word in words:
            chip = Chip(word, "left", tip="Not a word the app knows. Click to teach it, x to take it out.")
            chip.closed.connect(lambda w=word: self._drop_word(w))
            menu = QMenu(self)
            teach = menu.addMenu(f"Add \"{word}\" as…")
            by_kind: dict[str, QMenu] = {}
            for eid, label in self.interp.vocab.meanings():
                kind = label.rsplit("(", 1)[-1].rstrip(")")
                sub = by_kind.get(kind) or teach.addMenu(kind.capitalize())
                by_kind[kind] = sub
                act = QAction(label.rsplit(" (", 1)[0], sub)
                act.triggered.connect(lambda _=False, w=word, e=eid: self.teach(w, e))
                sub.addAction(act)
            drop = QAction("Take it out of the text", menu)
            drop.triggered.connect(lambda _=False, w=word: self._drop_word(w))
            menu.addAction(drop)
            chip.set_menu(menu)
            self.left_flow.addWidget(chip)
        self.left_row.setVisible(bool(words))

    def teach(self, word: str, eid: str) -> None:
        """Save ``word`` as meaning ``eid`` in the user's vocabulary, then read the text again."""
        try:
            self.interp.vocab.add_synonym(word, eid)
        except (VocabularyError, OSError) as exc:
            return self.show_message(str(exc), error=True)
        self.show_message(f"\"{word}\" now means {self.interp.vocab.by_id[eid]['label']} "
                          f"(saved in {self.interp.vocab.user_file}).")
        self.refresh()

    def _drop_word(self, word: str) -> None:
        text = re.sub(rf"(?i)(?<![a-z]){re.escape(word)}(?![a-z])", "", self.text(), count=1)
        self.set_text(re.sub(r"[ \t]{2,}", " ", text).strip())

    def _fill_suggestions(self) -> None:
        self.suggest_flow.clear()
        cursor = self.prompt.textCursor().position()
        words = self.interp.vocab.suggest(self.text()[:cursor] if self.prompt.hasFocus() else self.text())
        for w in words:
            chip = Chip(w, "suggest", closable=False, tip="Click to use this word")
            chip.clicked.connect(lambda w=w: self.complete(w))
            self.suggest_flow.addWidget(chip)
        self.suggest_row.setVisible(bool(words))

    def complete(self, word: str) -> None:
        """Put a suggested word in: finish the half-typed word, or add it after a space."""
        text = self.text()
        m = re.search(r"[A-Za-z]+$", text)
        stem = m.group(0).lower() if m else ""
        if stem and word.lower().startswith(stem) and not text.endswith(" "):
            text = text[: m.start()] + word
        else:
            text = (text + ("" if not text or text.endswith(" ") else " ") + word)
        self.set_text(text + " ")
        self.prompt.setFocus()

    def use_template(self, name: str) -> None:
        self.set_text(text_with_template(self.text(), name, self.interp))

    # ------------------------------------------------------------------ image
    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Reference image (a front view; your own or licensed)",
                                              str(Path.home()), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.use_image(Path(path))

    def use_image(self, path: Path) -> None:
        from eu4unitgen.core.reference import suggestions

        try:
            with Image.open(path) as im:
                im.load()
                items = suggestions(im)
        except OSError as exc:
            return self.show_message(f"Can't open {path}: {exc}", error=True)
        self.image_path = Path(path)
        self.image_items = items
        pix = QPixmap(str(path))
        self.thumb.setPixmap(pix.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.thumb.setToolTip(str(path))
        self.thumb.setVisible(True)
        self.show_message(f"{len(items)} suggestion(s) from the image, in blue below. Remove any you don't want."
                          if items else "Couldn't tell the soldier from the background in that image.")
        self.refresh()
