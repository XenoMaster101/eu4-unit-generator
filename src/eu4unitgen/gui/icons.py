"""The app's icons: one set of line drawings (24 x 24, round strokes) made for this app, drawn in a theme colour.

``icon(name, colour)`` -> a QIcon; ``pixmap(name, colour, size)`` -> a QPixmap. Names: home, build (a unit's
banner), part (a picture becoming a part), library, setup, help, soldier, horse, gun, example, tour, check, dot.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

# stroke paths only (fill="none"); <circle>/<path> elements inside one <g>
SHAPES = {
    "home": '<path d="M3.5 11 12 4l8.5 7"/><path d="M5.5 9.5V20h13V9.5"/><path d="M10 20v-5.5h4V20"/>',
    "build": '<path d="M6 21V3"/><path d="M6 4h12l-3 4 3 4H6"/><path d="M4 21h4"/>',
    "part": '<rect x="3" y="4" width="13" height="11" rx="1.5"/><path d="m3.5 13 4-4 3.5 3 2-2 3 3"/>'
            '<circle cx="12" cy="7.5" r="1"/><path d="M19 14v7M15.5 17.5h7"/>',
    "library": '<path d="M4 4h3v16H4z"/><path d="M8.5 4h3v16h-3z"/><path d="m14 5.2 2.9-.8 4 15.4-2.9.8z"/>'
               '<path d="M4 8h3M8.5 8h3"/>',
    "setup": '<path d="M4 6.5h9M17 6.5h3M4 12h3M11 12h9M4 17.5h11M19 17.5h1"/>'
             '<circle cx="15" cy="6.5" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17.5" r="2"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.7.3-1 .8-1 1.6v.4"/>'
            '<path d="M12 17h.01"/>',
    # a kettle hat over a face, on shoulders
    "soldier": '<path d="M4 10.5h16"/><path d="M7 10.5C7 6.5 9.2 4.5 12 4.5s5 2 5 6"/>'
               '<path d="M9 10.5V13a3 3 0 0 0 6 0v-2.5"/><path d="M5 21c.4-3.2 3.2-5 7-5s6.6 1.8 7 5"/>',
    # a horse's head in profile
    "horse": '<path d="M9 21c0-3 .6-5.2 2.2-6.6-2.2.4-4.5-.3-5.9-1.9l1.1-2.3c1.6.6 3.1.4 4.1-.6L9.8 7.2 10.2 3.5'
             'l2.3 2.1c4.4.6 7.5 4.4 7.5 9.4V21"/><path d="M12.6 9h.01"/><path d="M14.2 6.6c1.5 1.8 2.3 4 2.3 6.4"/>',
    # a cannon on its wheel, the trail behind
    "gun": '<path d="M5.4 14.1 4.3 10.9l11.4-4.1c.9-.3 1.9.2 2.2 1.1s-.2 1.9-1.1 2.2z"/><path d="M3.4 10.6l1.5 4"/>'
           '<circle cx="12.5" cy="16" r="4.2"/><path d="m9.5 13 6 6M15.5 13l-6 6"/><path d="m16.5 18.3 5 2.7"/>',
    "example": '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l5.5-3.5z"/>',
    "tour": '<rect x="3" y="5.5" width="18" height="13" rx="1.5"/><path d="M7 5.5v13M17 5.5v13M3 10h4M3 14h4M17 10h4'
            'M17 14h4"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "refresh": '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4.5v4h-4"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.5h.01"/>',
    "folder": '<path d="M3.5 7V18a1 1 0 0 0 1 1h15a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1h-7.5l-2-2.5h-5.5a1 1 0 0 0-1 1z"/>',
    "dot": '<circle cx="12" cy="12" r="5" fill="currentColor"/>',
}


def svg(name: str, colour: str, width: float = 1.7) -> bytes:
    body = SHAPES[name].replace("currentColor", colour)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"><g>{body}</g></svg>').encode()


@lru_cache(maxsize=256)
def pixmap(name: str, colour: str, size: int = 24, width: float = 1.7) -> QPixmap:
    from PySide6.QtSvg import QSvgRenderer
    r = QSvgRenderer(QByteArray(svg(name, colour, width)))
    pix = QPixmap(size * 2, size * 2)  # drawn at twice the size, for high-DPI screens
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, size * 2, size * 2))
    p.end()
    pix.setDevicePixelRatio(2.0)
    return pix


def icon(name: str, colour: str, size: int = 24) -> QIcon:
    return QIcon(pixmap(name, colour, size))
