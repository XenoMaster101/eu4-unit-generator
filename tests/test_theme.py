"""The app's own theme (gui.theme): with a dark Windows palette already on the app (what dark mode
does), every text on the Home screen, in light and dark themes, still reads: rendered, and its text
pixels checked against its background (WCAG contrast, 4.5:1 for normal text)."""
from __future__ import annotations

import os
from collections import Counter

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QPlainTextEdit, QStyleFactory,
)

from eu4unitgen.export.pipeline import Session, Settings  # noqa: E402
from eu4unitgen.gui import theme  # noqa: E402
from eu4unitgen.gui.app import MainWindow  # noqa: E402

MIN = 4.5


def _system_dark(app: QApplication) -> None:
    """What Windows dark mode hands a Qt app: the native style, white text, dark windows."""
    app.setStyle(QStyleFactory.create("Windows"))
    p = QPalette()
    for role, c in ((QPalette.Window, "#1e1e1e"), (QPalette.WindowText, "#ffffff"), (QPalette.Base, "#2d2d2d"),
                    (QPalette.Text, "#ffffff"), (QPalette.Button, "#2d2d2d"), (QPalette.ButtonText, "#ffffff"),
                    (QPalette.PlaceholderText, "#a0a0a0")):
        p.setColor(role, QColor(c))
    app.setPalette(p)
    app.setStyleSheet("")
    app._eu4unitgen_theme = None


def _window_image(win):
    """The whole window, rendered once, as a Pillow image (device pixels)."""
    from PIL import Image
    from PySide6.QtGui import QImage

    q = win.grab().toImage().convertToFormat(QImage.Format_RGB888)
    return Image.frombytes("RGB", (q.width(), q.height()), bytes(q.constBits()), "raw", "RGB", q.bytesPerLine()),         q.width() / max(1, win.width())


def _text_contrast(widget, shot=None, inset: int = 3) -> float:
    """``widget`` as the user sees it (cut from a render of its whole window, so whatever shows
    through behind it counts); background = its commonest colour inside the frame, text = the
    colour that differs from it most. Returns their contrast ratio."""
    img, scale = shot or _window_image(widget.window())
    top = widget.mapTo(widget.window(), widget.rect().topLeft())
    box = tuple(int(v * scale) for v in (top.x() + inset, top.y() + inset, top.x() + widget.width() - inset,
                                          top.y() + widget.height() - inset))
    colours = img.crop(box).getcolors(1 << 20)
    bg = max(colours)[1]
    return max(theme.contrast(bg, c) for _, c in colours)


def _texts(page):
    for w in page.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(page) or w.width() < 8 or w.height() < 8:
            continue
        if isinstance(w, QLabel) and w.text().strip() and w.pixmap().isNull():
            yield w, w.text()
        elif isinstance(w, QComboBox):
            yield w, w.currentText() or w.lineEdit().placeholderText() if w.isEditable() else w.currentText()
        elif isinstance(w, QAbstractButton) and w.text().strip():
            yield w, w.text()
        elif isinstance(w, QPlainTextEdit) and w.toPlainText():
            yield w, w.toPlainText()[:20]
        elif isinstance(w, QLineEdit) and not isinstance(w.parent(), QComboBox) and (w.text() or w.placeholderText()):
            yield w, w.text() or w.placeholderText()


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("choice", ["light", "dark", "system"])
def test_home_is_readable_whatever_windows_says(qapp, fake_install, tmp_path, monkeypatch, choice):
    _system_dark(qapp)
    monkeypatch.setattr(qapp.styleHints(), "colorScheme", lambda: theme.Qt.ColorScheme.Dark)
    w = MainWindow(Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out")), run_async=False)
    w.set_theme(choice)
    want = theme.DARK if choice in ("dark", "system") else theme.LIGHT
    assert qapp.palette().color(QPalette.WindowText).name() == want["text"]  # ours, not the system's
    assert w.theme_mode == ("dark" if choice in ("dark", "system") else "light")
    home = w.home
    home.set_text("Hoplites with red crests, no shield, long spears, javelins, bronze greaves")
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    bad = []
    checked = 0
    shot = _window_image(w)
    for widget, text in _texts(home):
        if not widget.isEnabled() and widget is home.generate:
            continue  # off until a country is picked; the hint beside it says so (checked below)
        ratio = _text_contrast(widget, shot)
        checked += 1
        if ratio < MIN:
            bad.append(f"{type(widget).__name__} {text!r}: {ratio:.1f}")
    assert checked >= 15 and not bad, bad
    assert home.ready_hint.isVisible() and not home.generate.isEnabled()  # no country yet: said where it matters
    assert home.nation.property("missing") == "true"
    home.set_nation("MAC")
    assert home.generate.isEnabled() and not home.ready_hint.isVisible()
    assert _text_contrast(home.generate) >= MIN
    w.hide()
    w.session.close()
    import shiboken6

    shiboken6.delete(w)


def test_contrast_maths():
    assert round(theme.contrast((0, 0, 0), (255, 255, 255)), 1) == 21.0
    assert theme.contrast((246, 245, 242), (255, 255, 255)) < 1.1  # the bug: white text on cream
    for t in (theme.LIGHT, theme.DARK):
        for fg, bg in (("text", "window"), ("text", "surface"), ("muted", "window"), ("placeholder", "surface"),
                       ("text", "chip"), ("accent_text", "accent"), ("link", "window"), ("error", "window")):
            assert theme.contrast(QColor(t[fg]), QColor(t[bg])) >= MIN, (fg, bg, t is theme.DARK)
