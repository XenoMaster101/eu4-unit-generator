"""What every screen shares: the page base, the stylesheet (every colour from ``gui.theme``'s tokens), the picture
well, a turntable player and small helpers. Thin by design: screens collect input and show results; the work is in
``core``, ``ref3d`` and ``export``."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from eu4unitgen.gui.theme import tokens


def open_folder(path: Path) -> bool:
    """Open ``path`` in the system file manager (Explorer, Files, ...). False if it doesn't exist."""
    if not Path(path).is_dir():
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))


def open_file(path: Path) -> bool:
    return Path(path).is_file() and QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))


def minutes(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{sec:.0f} s" if sec < 90 else f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


class Page(QWidget):
    """A screen: ``layout_`` to fill, ``message`` (a line for results and errors), ``window_`` (the main window)."""

    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(24, 16, 24, 16)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def show_message(self, text: str, error: bool = False) -> None:
        self.message.setText(f'<span style="color:{tokens()["error"]}">{text}</span>' if text and error else text)

    def set_busy(self, busy: bool) -> None:
        for b in self.findChildren(QPushButton):
            b.setEnabled(not busy)

    def restyle(self) -> None:
        self.setStyleSheet(css(tokens()))


def arrow(colour: str) -> str:
    """A small down-arrow PNG in ``colour`` (a stylesheet can't draw one; Fusion's own arrow disappears once a combo
    box has a styled border). Written once per colour to the temp folder."""
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
    """The app's stylesheet: every colour from the design's tokens (gui.theme). Buttons: ``generate`` the one primary
    action of a screen (filled), ``ghost`` secondary (outlined), ``danger`` (outlined red), ``link`` (text in the
    accent, never the default blue). Cards and framed panels are parchment."""
    from eu4unitgen.gui.theme import backdrop, serif
    sf = serif(t)
    return f"""
QWidget#front {{ background-color: {t['window']}; border-image: url({backdrop(t)}) 0 0 0 0 stretch stretch;
                 color: {t['text']}; }}
QWidget#front QLabel {{ color: {t['text']}; background: transparent; border-image: none; }}
QWidget#front QLabel#brand {{ font-family: "{sf}"; font-weight: 700; font-size: 17px; color: {t['accent_fg']}; }}
QWidget#front QLabel#h1 {{ font-family: "{sf}"; font-size: 27px; font-weight: 600; color: {t['text']}; }}
QWidget#front QLabel#h2 {{ font-family: "{sf}"; font-size: 18px; font-weight: 600; color: {t['accent_fg']}; }}
QWidget#front QLabel#h3 {{ font-family: "{sf}"; font-size: 15px; font-weight: 600; }}
QWidget#front QLabel#step {{ color: {t['accent_fg']}; font-weight: 600; letter-spacing: 2px; font-size: 12px; }}
QWidget#front QLabel#muted {{ color: {t['muted']}; }}
QWidget#front QLabel#explain {{ color: {t['muted']}; font-size: 13px; }}
QWidget#front QLabel#status, QWidget#front QLabel#good {{ color: {t['ok']}; }}
QWidget#front QLabel#warn {{ color: {t['warn']}; }}
QWidget#front QLabel#bad, QWidget#front QLabel#hint {{ color: {t['error']}; font-weight: 600; }}
QPlainTextEdit {{ font-size: 14px; border: 1px solid {t['border']}; border-radius: 10px; padding: 10px;
                  background: {t['surface']}; color: {t['text']}; }}
QPlainTextEdit:disabled {{ background: {t['alt']}; color: {t['muted']}; border-style: dashed; }}
QComboBox, QLineEdit {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
                       border-radius: 6px; padding: 4px 8px; min-height: 24px; }}
QComboBox:hover, QLineEdit:hover {{ border-color: {t['accent_off']}; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {t['accent']}; }}
QLineEdit[bad="true"] {{ border: 2px solid {t['error']}; }}
QComboBox QLineEdit {{ border: none; padding: 0; background: transparent; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; }}
QComboBox::down-arrow {{ image: url({arrow(t['accent_fg'])}); width: 10px; height: 6px; }}
QPushButton#generate {{ background: {t['accent']}; color: {t['accent_text']}; font-size: 15px; font-weight: 700;
                       border: 1px solid {t['accent_hover']}; border-radius: 8px; padding: 9px 26px; }}
QPushButton#generate:hover {{ background: {t['accent_hover']}; }}
QPushButton#generate:disabled {{ background: {t['accent_off']}; color: {t['disabled']}; border-color: {t['accent_off']}; }}
QPushButton#ghost {{ background: transparent; color: {t['secondary']}; border: 1px solid {t['accent']};
                    border-radius: 8px; padding: 6px 14px; }}
QPushButton#ghost:hover {{ background: {t['alt']}; }}
QPushButton#ghost:checked {{ background: {t['accent']}; color: {t['accent_text']}; }}
QPushButton#ghost:disabled {{ color: {t['disabled']}; border-color: {t['border']}; }}
QPushButton#danger {{ background: transparent; color: {t['danger']}; border: 1px solid {t['danger']};
                     border-radius: 8px; padding: 6px 14px; }}
QPushButton#danger:hover {{ background: {t['alt']}; }}
QPushButton#link {{ border: none; color: {t['link']}; background: transparent; text-decoration: none; padding: 2px 4px; }}
QPushButton#link:hover {{ color: {t['accent_hover']}; }}
QPushButton#nav {{ border: none; border-bottom: 2px solid transparent; background: transparent; color: {t['text']};
                  padding: 9px 12px 7px 10px; font-size: 14px; }}
QPushButton#nav:hover {{ color: {t['accent_fg']}; }}
QPushButton#nav:checked {{ border-bottom-color: {t['accent']}; color: {t['accent_fg']}; font-weight: 600; }}
QFrame#topbar {{ background: {t['topbar']}; border-bottom: 1px solid {t['topbar_line']}; }}
QFrame#topbar QLabel {{ background: transparent; }}
QFrame#topbar QLabel#muted {{ color: {t['muted']}; }}
QFrame#topbar QLabel#brand {{ font-family: "{sf}"; font-weight: 700; font-size: 17px; color: {t['accent_fg']}; }}
QFrame#card, QPushButton#card {{ background: {t['card']}; border: 1px solid {t['card_border']}; border-radius: 10px;
                                 border-image: none; }}
QPushButton#card {{ text-align: left; padding: 12px; color: {t['card_text']}; }}
QPushButton#card:hover {{ background: {t['card_hover']}; border: 2px solid {t['accent']}; }}
QPushButton#card:checked {{ background: {t['card_selected']}; border: 2px solid {t['accent']}; }}
QPushButton#card:disabled {{ background: {t['alt']}; border-color: {t['border']}; }}
QWidget#front QFrame#card QLabel, QWidget#front QPushButton#card QLabel {{ color: {t['card_text']}; }}
QWidget#front QFrame#card QLabel#muted, QWidget#front QPushButton#card QLabel#muted,
QWidget#front QFrame#card QLabel#explain {{ color: {t['card_muted']}; }}
QWidget#front QFrame#card QLabel#h3, QWidget#front QPushButton#card QLabel#cardtitle {{
    font-family: "{sf}"; font-size: 16px; font-weight: 700; color: {t['card_text']}; }}
QWidget#front QPushButton#card:disabled QLabel {{ color: {t['disabled']}; }}
QWidget#front QLabel#chip, QWidget#front QPushButton#card QLabel#chip {{ background: {t['chip']}; color: {t['chip_text']}; border: 1px solid {t['chip_border']};
                             border-radius: 9px; padding: 1px 8px; font-size: 12px; }}
QWidget#front QPushButton#card QLabel#dot_ok {{ color: {t['card_ok']}; font-size: 12px; }}
QWidget#front QPushButton#card QLabel#dot_warn {{ color: {t['card_warn']}; font-size: 12px; }}
QWidget#front QPushButton#card QLabel#dot_error {{ color: {t['card_error']}; font-size: 12px; }}
QWidget#front QPushButton#card QLabel#dot_none {{ color: {t['card_muted']}; font-size: 12px; }}
QFrame#card QComboBox, QFrame#card QLineEdit {{ background: #fffaf0; color: {t['card_text']}; border-color: {t['card_border']}; }}
QFrame#card QComboBox::down-arrow {{ image: url({arrow(t['card_muted'])}); }}
QWidget#front QPushButton#card QLabel#badge {{ background: {t['accent']}; color: {t['accent_text']}; border-radius: 8px;
                                               padding: 1px 8px; font-size: 11px; font-weight: 700; }}
QPushButton#tileaction {{ background: transparent; color: {t['card_text']}; border: 1px solid {t['card_border']};
                         border-radius: 6px; padding: 3px 12px; font-size: 12px; }}
QPushButton#tileaction:hover {{ border-color: {t['accent']}; background: {t['card_selected']}; }}
QFrame#sticky {{ background: {t['topbar']}; border-top: 1px solid {t['topbar_line']}; border-image: none; }}
QLabel#drop {{ background: {t['surface']}; color: {t['muted']}; border: 2px dashed {t['accent_off']}; border-radius: 12px; }}
QLabel#cardpic, QWidget#front QLabel#cardpic {{ background: {t['thumb_inner']}; border-radius: 8px;
                                               color: {t['card_muted']}; }}
QPushButton#filter {{ background: transparent; color: {t['muted']}; border: 1px solid {t['border']}; border-radius: 14px;
                     padding: 4px 14px; }}
QPushButton#filter:hover {{ color: {t['accent_fg']}; border-color: {t['accent_off']}; }}
QPushButton#filter:checked {{ color: {t['accent_fg']}; border-color: {t['accent']}; background: {t['alt']}; font-weight: 600; }}
QGroupBox {{ color: {t['accent_fg']}; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 8px;
            font-family: "{sf}"; font-size: 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 6px; background: {t['alt']}; color: {t['text']};
               text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
QScrollArea {{ background: transparent; border: none; border-image: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; border-image: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t['accent_off']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QCheckBox {{ color: {t['text']}; background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['accent']}; width: 14px; margin: -6px 0; border-radius: 7px; }}
"""


def pixmap(path: Path | str | None, w: int, h: int) -> QPixmap:
    """An image scaled to fit ``w`` x ``h`` (an empty pixmap when there is none)."""
    if not path or not Path(path).is_file():
        return QPixmap()
    pix = QPixmap(str(path))
    return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pix.isNull() else pix


def thumb(path: Path | str | None, w: int, h: int) -> QPixmap:
    """A render as a thumbnail: lifted onto the design's lit backdrop (core.thumbs), made once and cached in the
    projects folder; the plain picture if that fails."""
    from eu4unitgen.core import thumbs
    t = tokens()
    try:
        from eu4unitgen.ref3d import flow
        p = thumbs.cache(path, (w * 2, h * 2), t["thumb_inner"], t["thumb_outer"], flow.root() / ".thumbs")
    except Exception:  # noqa: BLE001 - a thumbnail is a nicety: the picture as it is
        p = path
    pix = pixmap(p, w * 2, h * 2)
    pix.setDevicePixelRatio(2.0)
    return pix


def pil_pixmap(im, w: int, h: int) -> QPixmap:
    """A Pillow image as a pixmap fitted to ``w`` x ``h``."""
    from PySide6.QtGui import QImage
    im = im.convert("RGBA")
    q = QImage(im.tobytes("raw", "RGBA"), im.width, im.height, im.width * 4, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(q).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class DropImage(QLabel):
    """The picture well: drop a picture on it, or click to choose one."""
    picked = Signal(object)

    def __init__(self, text: str = "Drop your picture here\n(or click to choose one)", w: int = 360, h: int = 300):
        super().__init__(text)
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(w, h)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Drag a picture file here from Explorer, or click to pick one (PNG, JPG or WEBP)")
        self.empty_text = text

    def dragEnterEvent(self, e):  # noqa: N802 - Qt name
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802 - Qt name
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.picked.emit(Path(urls[0]))

    def mousePressEvent(self, e):  # noqa: N802 - Qt name
        path, _ = QFileDialog.getOpenFileName(self, "Choose your picture", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.picked.emit(Path(path))

    def show_image(self, path: Path | None) -> None:
        pix = pixmap(path, self.width() - 12, self.height() - 12)
        if pix.isNull():
            self.setPixmap(QPixmap())
            self.setText(self.empty_text)
        else:
            self.setPixmap(pix)


def clear_layout(lay) -> None:
    """Every widget out of ``lay`` at once: hidden and unparented, then deleted (deleteLater alone leaves it drawn
    where it was until the event loop runs)."""
    while lay.count():
        w = lay.takeAt(0).widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()


class Turntable(QLabel):
    """A unit turning round: its turntable frames played in a loop (click to pause); a still image when there are
    none."""

    def __init__(self, size: int = 300, fps: float = 6.0):
        super().__init__()
        self.setObjectName("cardpic")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(size, size)
        self.size_ = size
        self.frames: list[QPixmap] = []
        self.i = 0
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / fps))
        self.timer.timeout.connect(self._next)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("The unit turning round, as the game draws it. Click to pause or play.")

    def set_frames(self, paths: list, still: Path | None = None) -> None:
        self.timer.stop()
        self.frames = [p for p in (pixmap(x, self.size_ - 8, self.size_ - 8) for x in paths) if not p.isNull()]
        self.i = 0
        if self.frames:
            self.setPixmap(self.frames[0])
            self.timer.start()
        else:
            pix = thumb(still, self.size_ - 8, self.size_ - 8)
            self.setPixmap(pix)
            if pix.isNull():
                self.setText("No preview yet")

    def _next(self) -> None:
        if self.frames:
            self.i = (self.i + 1) % len(self.frames)
            self.setPixmap(self.frames[self.i])

    def mousePressEvent(self, e):  # noqa: N802 - Qt name
        if self.timer.isActive():
            self.timer.stop()
        elif self.frames:
            self.timer.start()


class Card(QPushButton):
    """A large choice button on parchment: an optional picture (or an icon), a serif title, tags as small chips, a
    wrapped plain sentence and a status dot ("ok", "warn", "error" or "none" with its words)."""

    def __init__(self, title: str, text: str, tip: str = "", checkable: bool = False, picture: QPixmap | None = None,
                 chips: list[str] | None = None, status: tuple[str, str] | None = None, icon: str | None = None,
                 vertical: bool = False):
        from PySide6.QtWidgets import QHBoxLayout
        super().__init__()
        self.setObjectName("card")
        self.setCheckable(checkable)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip or text)
        self.setMinimumHeight(84)
        row = QVBoxLayout(self) if vertical else QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(12)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignCenter)
        if icon and (picture is None or picture.isNull()):
            from eu4unitgen.gui.icons import pixmap as icon_pixmap
            picture = icon_pixmap(icon, tokens()["card_text"], 40, 1.5)
        else:
            self.pic.setObjectName("cardpic")
        self.pic.setVisible(picture is not None and not picture.isNull())
        if picture is not None:
            self.pic.setPixmap(picture)
        row.addWidget(self.pic, 0, Qt.AlignTop | (Qt.AlignHCenter if vertical else Qt.AlignLeft))
        col = QVBoxLayout()
        col.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardtitle")
        self.title_label.setWordWrap(True)
        col.addWidget(self.title_label)
        from eu4unitgen.gui.widgets import FlowLayout
        chip_box = QWidget()
        chip_box.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.chip_row = FlowLayout(chip_box, spacing=4)
        self.chips = []
        for c in chips or []:
            ch = QLabel(c)
            ch.setObjectName("chip")
            ch.setMinimumWidth(ch.fontMetrics().horizontalAdvance(c) + 20)  # its padding counted: never clipped
            self.chips.append(ch)
            self.chip_row.addWidget(ch)
        if chips:
            col.addWidget(chip_box)
        self.text_label = QLabel(text)
        self.text_label.setObjectName("muted")
        self.text_label.setWordWrap(True)
        self.text_label.setVisible(bool(text))
        col.addWidget(self.text_label)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        if status:
            kind, words = status
            self.status_label.setObjectName(f"dot_{kind}")
            self.status_label.setText(f"●  {words}")
        self.status_label.setVisible(bool(status))
        col.addWidget(self.status_label)
        col.addStretch(1)
        for lab in (self.title_label, self.text_label, self.status_label, self.pic, *self.chips):
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addLayout(col, 1)

    def set_text(self, title: str, text: str) -> None:
        self.title_label.setText(title)
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))

    def changeEvent(self, e):  # noqa: N802 - Qt name: the labels follow the button's enabled state
        super().changeEvent(e)
        for lab in (self.title_label, self.text_label, self.status_label):
            lab.setEnabled(self.isEnabled())

    def sizeHint(self):  # noqa: N802 - Qt name
        return self.layout().sizeHint()


def card(title: str, text: str, tip: str = "", checkable: bool = False, picture: QPixmap | None = None,
         **kw) -> Card:
    return Card(title, text, tip, checkable, picture, **kw)


class Tile(QPushButton):
    """A gallery tile (the Library): a big picture on top, the name, one line of tag chips (the rest as "+2"), a
    "Starter" badge, a status dot with its words and one action button (``action``: (text, fn)). Click anywhere to
    open it (``clicked``)."""

    PIC_H = 168

    def __init__(self, title: str, picture: QPixmap | None, chips: list[str] | None = None,
                 status: tuple[str, str] | None = None, starter: bool = False, action: tuple | None = None,
                 tip: str = ""):
        from PySide6.QtWidgets import QHBoxLayout
        super().__init__()
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip)
        col = QVBoxLayout(self)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(6)
        self.pic = QLabel()
        self.pic.setObjectName("cardpic")
        self.pic.setAlignment(Qt.AlignCenter)
        self.pic.setFixedHeight(self.PIC_H)
        self.picture = picture
        col.addWidget(self.pic)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardtitle")
        col.addWidget(self.title_label)
        self.chip_box = QWidget()
        self.chip_row = QHBoxLayout(self.chip_box)
        self.chip_row.setContentsMargins(0, 0, 0, 0)
        self.chip_row.setSpacing(4)
        self.chips = []
        for c in chips or []:
            ch = QLabel(c)
            ch.setObjectName("chip")
            ch.setMinimumWidth(ch.fontMetrics().horizontalAdvance(c) + 20)
            self.chips.append(ch)
            self.chip_row.addWidget(ch)
        self.more = QLabel()
        self.more.setObjectName("chip")
        self.chip_row.addWidget(self.more)
        self.chip_row.addStretch(1)
        col.addWidget(self.chip_box)
        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.status_label = QLabel()
        if status:
            self.status_label.setObjectName(f"dot_{status[0]}")
            self.status_label.setText(f"●  {status[1]}")
        foot.addWidget(self.status_label, 1)
        self.badge = QLabel("Starter")
        self.badge.setObjectName("badge")
        self.badge.setVisible(starter)
        foot.addWidget(self.badge)
        self.action = None
        if action:
            self.action = QPushButton(action[0])
            self.action.setObjectName("tileaction")
            self.action.setCursor(Qt.PointingHandCursor)
            self.action.clicked.connect(lambda _=False: action[1]())
            foot.addWidget(self.action)
        col.addLayout(foot)
        for lab in (self.pic, self.title_label, self.status_label, self.badge, self.chip_box, self.more, *self.chips):
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedHeight(self.PIC_H + 118)

    def resizeEvent(self, e):  # noqa: N802 - Qt name: the picture and the chips fit the tile's width
        super().resizeEvent(e)
        w = max(40, self.width() - 22)
        if self.picture is not None and not self.picture.isNull():
            self.pic.setPixmap(self.picture.scaled(int(w * self.picture.devicePixelRatio()),
                                                   int(self.PIC_H * self.picture.devicePixelRatio()),
                                                   Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.fit_chips(w)

    def fit_chips(self, width: int) -> None:
        """One line of chips: as many as fit, the rest counted in "+N"."""
        used, shown = 0, 0
        more_w = self.more.fontMetrics().horizontalAdvance("+99") + 22
        for i, ch in enumerate(self.chips):
            need = ch.minimumWidth() + 4
            left = len(self.chips) - i - 1
            if used + need + (more_w if left else 0) <= width:
                ch.show()
                used += need
                shown += 1
            else:
                ch.hide()
        hidden = len(self.chips) - shown
        self.more.setText(f"+{hidden}")
        self.more.setMinimumWidth(self.more.fontMetrics().horizontalAdvance(f"+{hidden}") + 20)
        self.more.setVisible(hidden > 0)


def danger(text: str, fn=None, tip: str = "") -> QPushButton:
    b = ghost(text, fn, tip)
    b.setObjectName("danger")
    return b


def framed(*widgets, name: str = "card") -> QFrame:
    f = QFrame()
    f.setObjectName(name)
    lay = QVBoxLayout(f)
    for w in widgets:
        lay.addWidget(w)
    return f


def label(text: str, name: str = "", wrap: bool = True, rich: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    if rich:
        lab.setTextFormat(Qt.RichText)
        lab.setOpenExternalLinks(True)
    return lab


def ghost(text: str, fn=None, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("ghost")
    b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    if fn is not None:
        b.clicked.connect(lambda _=False: fn())
    return b


def primary(text: str, fn=None, tip: str = "") -> QPushButton:
    b = ghost(text, fn, tip)
    b.setObjectName("generate")
    return b


def spacer() -> QWidget:
    w = QWidget()
    w.setFixedHeight(8)
    return w
