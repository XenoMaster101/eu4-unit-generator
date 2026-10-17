"""Edit history for the Character page (GUI usability pass U3): undo, redo, reset to saved, and
whether anything changed since the file was loaded or saved.

A step is a whole ``Snapshot`` (the character as the pages show it), not a diff: a character is
small JSON, and the painted masks it holds are shared images that a new stroke replaces rather
than edits, so a step costs a JSON string and a few references.
"""
from __future__ import annotations

import json
from typing import Any

LIMIT = 200  # steps kept; the oldest go first


class Snapshot:
    """``doc``: JSON-able state; ``images``: name -> image (compared by identity: a stroke makes a
    new image, so the same object means the same pixels)."""

    def __init__(self, doc: dict, images: dict[str, Any] | None = None):
        self.doc = doc
        self.images = dict(images or {})
        self._key = json.dumps(doc, sort_keys=True)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Snapshot) and self._key == other._key and self.images.keys() == other.images.keys()
                and all(self.images[k] is other.images[k] for k in self.images))

    def __hash__(self) -> int:  # pragma: no cover - snapshots aren't dict keys
        return hash(self._key)


class History:
    def __init__(self, state: Snapshot | None = None, limit: int = LIMIT):
        self.limit = limit
        self.reset(state)

    def reset(self, state: Snapshot | None) -> None:
        """A fresh history whose only step, ``state``, is also the saved one (a load or a new file)."""
        self._undo = [state] if state is not None else []
        self._redo: list[Snapshot] = []
        self.saved = state

    @property
    def current(self) -> Snapshot | None:
        return self._undo[-1] if self._undo else None

    def push(self, state: Snapshot | None) -> bool:
        """Record ``state`` as a new step (nothing if it equals the current one). Clears redo."""
        if state is None or state == self.current:
            return False
        self._undo.append(state)
        self._redo.clear()
        del self._undo[:-self.limit]
        return True

    def undo(self) -> Snapshot | None:
        """The step before the current one (None: nothing to undo)."""
        if len(self._undo) < 2:
            return None
        self._redo.append(self._undo.pop())
        return self.current

    def redo(self) -> Snapshot | None:
        if not self._redo:
            return None
        self._undo.append(self._redo.pop())
        return self.current

    def mark_saved(self, state: Snapshot | None = None) -> None:
        if state is not None:
            self.push(state)
        self.saved = self.current

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 1

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def dirty(self) -> bool:
        return self.current != self.saved
