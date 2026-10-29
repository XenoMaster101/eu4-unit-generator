"""The app's own look: Qt's Fusion style with a full palette from named colour tokens, so the Windows theme (dark
mode, accent colours) can't leak into a widget. Every stylesheet in the GUI takes its colours from ``tokens()``;
nothing relies on the system's defaults.

Two designs, both dark and warm, for a game about historical warfare (the user chose admiralty as the default, 2026-10-28):
  campaign   deep charcoal-brown with a faint grain, gold accents, parchment cards, Palatino headings
  admiralty  deep navy with a faint grain, bronze accents, parchment cards, Georgia headings
Each has its own status colours for the dark ground and for the parchment cards (``card_ok`` etc.), every pair
checked for contrast (tests/test_theme.py). "system", "light" and "dark" (older settings) mean the default design.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

DESIGNS = ("campaign", "admiralty")
DEFAULT = "admiralty"  # the user, 2026-10-28; Campaign stays in Help → Design
THEMES = DESIGNS  # what Help offers

CAMPAIGN = {
    # the ground: charcoal-brown, a top-to-bottom gradient with grain (``backdrop``)
    "window": "#221c17", "window_top": "#2e251e", "window_bottom": "#17120e", "grain": "#000000",
    "surface": "#2c241d", "alt": "#372d24", "text": "#efe6d2", "muted": "#c2b296", "placeholder": "#a8987c",
    "border": "#5e4c38", "topbar": "#15110d", "topbar_line": "#8a6d3b",
    # gold: the one primary action, the selected item, focus
    "accent": "#c9a14a", "accent_hover": "#dcb55e", "accent_text": "#1d160e", "accent_off": "#6b5a3a",
    "accent_fg": "#e3c172", "link": "#e3c172", "secondary": "#e3c172",
    "ok": "#93cf7a", "warn": "#f0b85a", "error": "#f08a7a", "danger": "#e7826f",
    "highlight": "#8a6d3b", "highlight_text": "#fff8e8", "disabled": "#8c7f6a", "tooltip": "#372d24",
    # parchment panels and cards
    "card": "#ecdfc4", "card_hover": "#f3e8d1", "card_text": "#2a2016", "card_muted": "#5c4a35",
    "card_border": "#b99c69", "card_selected": "#e2cf9f",
    "card_ok": "#2f6a26", "card_warn": "#8a5400", "card_error": "#9c2b1f",
    "chip": "#d9c69e", "chip_text": "#3a2c1c", "chip_border": "#b99c69",
    "image_chip": "#d2dbc2", "image_border": "#a8b48c", "left_chip": "#e8c7b4", "left_border": "#c59577",
    # thumbnails: a lit parchment well
    "thumb_inner": "#f6eedd", "thumb_outer": "#c9b38a",
    "serif": "Palatino Linotype|Book Antiqua|Palatino|URW Palladio L|Georgia|DejaVu Serif|serif",
}
ADMIRALTY = {
    "window": "#141c29", "window_top": "#1c2738", "window_bottom": "#0d131c", "grain": "#000000",
    "surface": "#1b2535", "alt": "#233046", "text": "#e9edf3", "muted": "#b5bfcd", "placeholder": "#93a0b3",
    "border": "#3e4e66", "topbar": "#0b1018", "topbar_line": "#9a6b3c",
    "accent": "#c48a4c", "accent_hover": "#d69e60", "accent_text": "#1a1009", "accent_off": "#5d4a38",
    "accent_fg": "#e2ad72", "link": "#e2ad72", "secondary": "#e2ad72",
    "ok": "#8fd19a", "warn": "#f0bc66", "error": "#f2958a", "danger": "#ee8a7d",
    "highlight": "#855a30", "highlight_text": "#fff7ee", "disabled": "#7c8799", "tooltip": "#233046",
    "card": "#ebe3d1", "card_hover": "#f3ede0", "card_text": "#1f2530", "card_muted": "#4c5566",
    "card_border": "#b39a74", "card_selected": "#e0d0ad",
    "card_ok": "#2a6630", "card_warn": "#855400", "card_error": "#9a2c24",
    "chip": "#d7cbb1", "chip_text": "#2b3240", "chip_border": "#b39a74",
    "image_chip": "#cdd8d0", "image_border": "#9fb2a6", "left_chip": "#e6c9b8", "left_border": "#c4977b",
    "thumb_inner": "#f4eee2", "thumb_outer": "#bdae90",
    "serif": "Georgia|Constantia|Cambria|DejaVu Serif|Liberation Serif|serif",
}
_DESIGN = {"campaign": CAMPAIGN, "admiralty": ADMIRALTY}
# the older names, kept so tests and settings that ask for them still work: both mean the default design
LIGHT = ADMIRALTY
DARK = ADMIRALTY
_current = dict(ADMIRALTY)
_name = "admiralty"


def tokens() -> dict[str, str]:
    """The colours in use now (stylesheets read these)."""
    return _current


def design() -> str:
    return _name


def resolve(choice: str) -> str:
    """The design for a choice: a design's name, else the default."""
    return choice if choice in DESIGNS else DEFAULT


def serif(t: dict[str, str] | None = None) -> str:
    """The first heading font of the design's list that this computer has."""
    from PySide6.QtGui import QFontDatabase
    have = set(QFontDatabase.families())
    names = (t or _current)["serif"].split("|")
    return next((n for n in names if n in have), "serif")


def backdrop(t: dict[str, str] | None = None) -> str:
    """The pages' ground: the design's vertical gradient with a faint grain and a soft vignette, drawn once per design
    into the temp folder (a stylesheet stretches it to the page)."""
    t = t or _current
    key = t["window_top"].lstrip("#") + t["window_bottom"].lstrip("#")
    path = Path(tempfile.gettempdir()) / f"eu4unitgen_ground_{key}.png"
    if not path.is_file():
        import random

        from PIL import Image, ImageFilter

        w, h = 640, 400
        top, bot = QColor(t["window_top"]), QColor(t["window_bottom"])
        im = Image.new("RGB", (w, h))
        px = im.load()
        rnd = random.Random(7)
        for y in range(h):
            f = y / (h - 1)
            base = [top.red() + (bot.red() - top.red()) * f, top.green() + (bot.green() - top.green()) * f,
                    top.blue() + (bot.blue() - top.blue()) * f]
            for x in range(w):
                dx, dy = (x - w / 2) / (w / 2), (y - h * 0.4) / (h / 2)
                v = max(0.0, 1.0 - 0.18 * (dx * dx + dy * dy))  # the vignette
                n = rnd.uniform(-4.0, 4.0)  # the grain
                px[x, y] = tuple(max(0, min(255, int(c * v + n))) for c in base)
        im = im.filter(ImageFilter.GaussianBlur(0.4))
        im.save(path)
    return path.as_posix()


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
    """App-wide: popups, menus, tooltips and dialogs (drawn outside any page's own stylesheet)."""
    return f"""
QToolTip {{ background: {t['tooltip']}; color: {t['text']}; border: 1px solid {t['accent']}; padding: 5px; }}
QMenu {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; }}
QMenu::item:selected {{ background: {t['highlight']}; color: {t['highlight_text']}; }}
QMenu::item:disabled {{ color: {t['disabled']}; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['accent']};
    selection-background-color: {t['highlight']}; selection-color: {t['highlight_text']}; outline: 0; }}
QDialog, QMessageBox {{ background: {t['window']}; color: {t['text']}; }}
"""


def apply_theme(app: QApplication, choice: str = DEFAULT) -> str:
    """Fusion + the chosen design's palette, app-wide. Returns the design's name."""
    global _current, _name
    mode = resolve(choice)
    if getattr(app, "_eu4unitgen_theme", None) == mode:  # already on: re-styling every live widget is slow
        return mode
    app._eu4unitgen_theme = mode
    _name = mode
    _current = dict(_DESIGN[mode])
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
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
