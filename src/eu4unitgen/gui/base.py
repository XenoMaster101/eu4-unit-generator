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
    """The app's stylesheet: every colour from the theme's tokens (gui.theme)."""
    return f"""
QWidget#front {{ background: {t['window']}; color: {t['text']}; }}
QWidget#front QLabel {{ color: {t['text']}; background: transparent; }}
QWidget#front QLabel#brand {{ font-weight: 700; font-size: 15px; }}
QWidget#front QLabel#h1 {{ font-size: 24px; font-weight: 600; }}
QWidget#front QLabel#h2 {{ font-size: 17px; font-weight: 600; }}
QWidget#front QLabel#step {{ color: {t['accent_fg']}; font-weight: 600; letter-spacing: 1px; }}
QWidget#front QLabel#muted {{ color: {t['muted']}; }}
QWidget#front QLabel#explain {{ color: {t['muted']}; font-size: 13px; }}
QWidget#front QLabel#status {{ color: {t['ok']}; }}
QWidget#front QLabel#good {{ color: {t['ok']}; }}
QWidget#front QLabel#warn {{ color: {t['warn']}; }}
QWidget#front QLabel#bad, QWidget#front QLabel#hint {{ color: {t['error']}; font-weight: 600; }}
QPlainTextEdit {{ font-size: 14px; border: 2px solid {t['border']}; border-radius: 12px; padding: 10px;
                  background: {t['surface']}; color: {t['text']}; }}
QPlainTextEdit:disabled {{ background: {t['alt']}; color: {t['muted']}; border-style: dashed; }}
QComboBox, QLineEdit {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
                       border-radius: 6px; padding: 4px 8px; min-height: 24px; }}
QLineEdit[bad="true"] {{ border: 2px solid {t['error']}; }}
QComboBox QLineEdit {{ border: none; padding: 0; background: transparent; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; }}
QComboBox::down-arrow {{ image: url({arrow(t['text'])}); width: 10px; height: 6px; }}
QPushButton#generate {{ background: {t['accent']}; color: {t['accent_text']}; font-size: 15px; font-weight: 600;
                       border: none; border-radius: 10px; padding: 9px 26px; }}
QPushButton#generate:disabled {{ background: {t['accent_off']}; color: {t['accent_text']}; }}
QPushButton#ghost {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
                    border-radius: 8px; padding: 6px 14px; }}
QPushButton#ghost:hover {{ border-color: {t['accent']}; }}
QPushButton#ghost:disabled {{ color: {t['disabled']}; }}
QPushButton#danger {{ background: {t['surface']}; color: {t['error']}; border: 1px solid {t['error']};
                     border-radius: 8px; padding: 6px 14px; }}
QPushButton#link {{ border: none; color: {t['link']}; text-decoration: underline; background: transparent; }}
QPushButton#nav {{ border: none; border-bottom: 3px solid transparent; background: transparent; color: {t['text']};
                  padding: 8px 12px; font-size: 14px; }}
QPushButton#nav:hover {{ color: {t['accent_fg']}; }}
QPushButton#nav:checked {{ border-bottom-color: {t['accent']}; font-weight: 600; }}
QFrame#topbar {{ background: {t['surface']}; border-bottom: 1px solid {t['border']}; }}
QFrame#topbar QLabel {{ background: transparent; }}
QFrame#card, QPushButton#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; }}
QPushButton#card {{ text-align: left; padding: 14px; color: {t['text']}; }}
QPushButton#card:hover {{ border: 2px solid {t['accent']}; }}
QPushButton#card:checked {{ border: 2px solid {t['accent']}; background: {t['alt']}; }}
QPushButton#card:disabled {{ color: {t['disabled']}; background: {t['alt']}; }}
QLabel#drop {{ background: {t['surface']}; color: {t['muted']}; border: 2px dashed {t['border']}; border-radius: 12px; }}
QLabel#cardpic {{ background: {t['alt']}; border-radius: 6px; color: {t['muted']}; }}
QGroupBox {{ color: {t['text']}; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 6px; background: {t['alt']}; color: {t['text']};
               text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
"""


def pixmap(path: Path | str | None, w: int, h: int) -> QPixmap:
    """An image scaled to fit ``w`` x ``h`` (an empty pixmap when there is none)."""
    if not path or not Path(path).is_file():
        return QPixmap()
    pix = QPixmap(str(path))
    return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pix.isNull() else pix


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
            pix = pixmap(still, self.size_ - 8, self.size_ - 8)
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
    """A large choice button: an optional picture, a bold title and a wrapped plain sentence under it."""

    def __init__(self, title: str, text: str, tip: str = "", checkable: bool = False, picture: QPixmap | None = None):
        from PySide6.QtWidgets import QHBoxLayout
        super().__init__()
        self.setObjectName("card")
        self.setCheckable(checkable)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip or text)
        self.setMinimumHeight(84)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 14, 10)
        self.pic = QLabel()
        self.pic.setVisible(picture is not None and not picture.isNull())
        if picture is not None:
            self.pic.setPixmap(picture)
        row.addWidget(self.pic, 0, Qt.AlignTop)
        col = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 14px; background: transparent;")
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("background: transparent;")
        for lab in (self.title_label, self.text_label):
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
            col.addWidget(lab)
        col.addStretch(1)
        self.pic.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addLayout(col, 1)

    def set_text(self, title: str, text: str) -> None:
        self.title_label.setText(title)
        self.text_label.setText(text)

    def changeEvent(self, e):  # noqa: N802 - Qt name: the labels follow the button's enabled state
        super().changeEvent(e)
        for lab in (self.title_label, self.text_label):
            lab.setEnabled(self.isEnabled())

    def sizeHint(self):  # noqa: N802 - Qt name
        return self.layout().sizeHint()


def card(title: str, text: str, tip: str = "", checkable: bool = False, picture: QPixmap | None = None) -> Card:
    return Card(title, text, tip, checkable, picture)


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
