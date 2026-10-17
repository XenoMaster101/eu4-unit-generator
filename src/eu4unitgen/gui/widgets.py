"""Small GUI helpers shared by the pages."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox


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
