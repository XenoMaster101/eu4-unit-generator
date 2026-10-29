"""Small GUI helpers shared by the pages."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLayout


class WheelGuard(QObject):
    """App-wide: dropdowns and number boxes ignore the mouse wheel unless clicked first (focused),
    so scrolling a page never changes a value by accident. The wheel goes on to the page instead."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt name
        if isinstance(obj, (QComboBox, QAbstractSpinBox)):
            if event.type() == QEvent.Polish:
                obj.setFocusPolicy(Qt.StrongFocus)
            elif event.type() == QEvent.Wheel and not obj.hasFocus():
                parent = obj.parentWidget()
                if parent is not None:
                    QApplication.sendEvent(parent, event)
                return True
        return False


def install_wheel_guard(app: QApplication) -> WheelGuard:
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard


class FlowLayout(QLayout):
    """Lays widgets out left to right and wraps them (Qt's flow layout example): chip rows."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 - Qt names
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._place(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def clear(self) -> None:
        while self._items:
            w = self._items.pop().widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _place(self, rect, move: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() and line > 0:
                x, y, line = rect.x(), y + line + self.spacing(), 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            line = max(line, hint.height())
        return y + line - rect.y()


class TileLayout(FlowLayout):
    """A gallery grid: tiles all the same size, as many columns as fit (each at least ``min_width``), the row filling
    the width."""

    def __init__(self, parent=None, min_width: int = 240, spacing: int = 14):
        super().__init__(parent, spacing)
        self.min_width = min_width

    def _place(self, rect, move: bool) -> int:
        if not self._items:
            return 0
        sp = self.spacing()
        cols = max(1, (rect.width() + sp) // (self.min_width + sp))
        w = max(1, (rect.width() - sp * (cols - 1)) // cols)
        h = max(item.sizeHint().height() for item in self._items)
        for i, item in enumerate(self._items):
            r, c = divmod(i, cols)
            if move:
                item.setGeometry(QRect(rect.x() + c * (w + sp), rect.y() + r * (h + sp), w, h))
        rows = (len(self._items) + cols - 1) // cols
        return rows * h + (rows - 1) * sp
