"""The app's own look (Track F): Qt's Fusion style with a full palette from named colour tokens,
light or dark, so the Windows theme (dark mode, accent colours) can't leak into a widget. Every
stylesheet in the GUI takes its colours from ``tokens()``; nothing relies on the system's
defaults. "system" picks light or dark once, at start-up, from the OS colour scheme.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

THEMES = ("system", "light", "dark")
LIGHT = {
    "window": "#f6f5f2", "surface": "#ffffff", "alt": "#efece6", "text": "#1f1f1f", "muted": "#555555",
    "placeholder": "#6f6f6f", "border": "#bdb6a8", "chip": "#ece7dc", "chip_border": "#cfc6b3",
    "image_chip": "#dfeaf6", "image_border": "#a9c1dc", "left_chip": "#fbe4dd", "left_border": "#e2b2a4",
    "accent": "#b0412b", "accent_text": "#ffffff", "accent_off": "#c8a59d", "accent_fg": "#a33a25", "link": "#1f56a8",
    "ok": "#1f6b2c", "warn": "#9a4d00", "error": "#b3261e", "highlight": "#2a5fb0", "highlight_text": "#ffffff",
    "disabled": "#8e8e8e", "tooltip": "#fffbe6",
}
DARK = {
    "window": "#1f2023", "surface": "#2a2b2f", "alt": "#323338", "text": "#ececec", "muted": "#b4b4b4",
    "placeholder": "#ababab", "border": "#55565c", "chip": "#3a3b40", "chip_border": "#5a5b62",
    "image_chip": "#23384f", "image_border": "#3f6390", "left_chip": "#4a2b25", "left_border": "#83493c",
    "accent": "#b84a32", "accent_text": "#ffffff", "accent_off": "#6e4d46", "accent_fg": "#ff9a80", "link": "#8ab4f8",
    "ok": "#7fd18b", "warn": "#f0b45a", "error": "#ff8a80", "highlight": "#3d6fc4", "highlight_text": "#ffffff",
    "disabled": "#7c7c7c", "tooltip": "#3a3b40",
}
_current = dict(LIGHT)


def tokens() -> dict[str, str]:
    """The colours in use now (stylesheets read these)."""
    return _current


def resolve(choice: str) -> str:
    """"light" or "dark": a choice of "system" asks the OS once."""
    if choice in ("light", "dark"):
        return choice
    app = QApplication.instance()
    try:
        return "dark" if app is not None and app.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"
    except AttributeError:  # Qt before 6.5: no colour scheme
        return "light"


def palette(t: dict[str, str]) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.Window: t["window"], QPalette.WindowText: t["text"], QPalette.Base: t["surface"],
        QPalette.AlternateBase: t["alt"], QPalette.Text: t["text"], QPalette.Button: t["surface"],
        QPalette.ButtonText: t["text"], QPalette.BrightText: t["error"], QPalette.Highlight: t["highlight"],
        QPalette.HighlightedText: t["highlight_text"], QPalette.Link: t["link"], QPalette.LinkVisited: t["link"],
        QPalette.ToolTipBase: t["tooltip"], QPalette.ToolTipText: t["text"], QPalette.PlaceholderText: t["placeholder"],
        QPalette.Light: t["surface"], QPalette.Midlight: t["alt"], QPalette.Mid: t["border"],
        QPalette.Dark: t["border"], QPalette.Shadow: "#000000",
    }
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, colour in roles.items():
            p.setColor(group, role, QColor(colour))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.PlaceholderText):
        p.setColor(QPalette.Disabled, role, QColor(t["disabled"]))
    return p


def base_css(t: dict[str, str]) -> str:
    """App-wide: popups, menus and tooltips (drawn outside any page's own stylesheet)."""
    return f"""
QToolTip {{ background: {t['tooltip']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px; }}
QMenu {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; }}
QMenu::item:selected {{ background: {t['highlight']}; color: {t['highlight_text']}; }}
QMenu::item:disabled {{ color: {t['disabled']}; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['text']};
    selection-background-color: {t['highlight']}; selection-color: {t['highlight_text']}; }}
"""


def apply_theme(app: QApplication, choice: str = "system") -> str:
    """Fusion + the chosen palette, app-wide. Returns "light" or "dark"."""
    global _current
    mode = resolve(choice)
    if getattr(app, "_eu4unitgen_theme", None) == mode:  # already on: re-styling every live widget is slow
        return mode
    app._eu4unitgen_theme = mode
    _current = dict(DARK if mode == "dark" else LIGHT)
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if mode == "dark" else Qt.ColorScheme.Light)
    except AttributeError:  # Qt before 6.8
        pass
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(palette(_current))
    app.setStyleSheet(base_css(_current))
    return mode


def contrast(a: QColor | tuple, b: QColor | tuple) -> float:
    """WCAG contrast ratio of two colours (1 to 21)."""
    def lum(c) -> float:
        rgb = (c.red(), c.green(), c.blue()) if isinstance(c, QColor) else c
        ch = [v / 255 for v in rgb]
        ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
