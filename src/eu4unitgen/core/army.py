"""The army the Home and Result screens edit (SPEC Track F, F10).

Two kinds of input make one look:
- the description (text, plus chips from a reference image), read by the interpreter: the Home
  screen's box and the Result screen's Understood strip both edit it;
- edits made on the Result screen (a part swapped, a finish or colour picked, a unit name, "this
  tier only"), kept as a list and replayed on top, in order.

So changing a word never throws away a colour you picked, and changing a part never rewrites
your words. Everything here is data; the screens only call it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from eu4unitgen.core.look import Look
from eu4unitgen.core.naming import UNIT_TYPES

EDITS = ("part", "finish", "material_colour", "unit_colour", "crest_colour", "emblem", "name")
SLOTS = ("weapon", "shield", "shield_params", "helmet", "crest")


@dataclass
class Edit:
    """One change on the Result screen. ``kind`` in EDITS; ``tier``: that tier only (None: all)."""
    kind: str
    key: tuple = ()
    value: Any = None
    tier: int | None = None


@dataclass
class Army:
    text: str = ""
    image_items: list = field(default_factory=list)  # core.prompt.Item from a reference image
    tag: str | None = None
    progression: str = "heavier"
    edits: list[Edit] = field(default_factory=list)
    base_colours: list | None = None  # the country's own unit colours (what the look starts from)

    def interpretation(self, interpreter):
        return interpreter.interpret(self.text, self.image_items)

    def look(self, interpreter) -> Look:
        """The description's look with the Tiers choice, then every edit in order."""
        look = self.interpretation(interpreter).look()
        look.progression = self.progression
        for e in self.edits:
            apply_edit(look, e, self.base_colours)
        return look

    def add(self, edit: Edit) -> None:
        """Record ``edit``, dropping an earlier one it makes pointless (same thing, same tiers)."""
        if edit.kind not in EDITS:
            raise ValueError(f"edit kind {edit.kind!r} must be one of {EDITS}")
        self.edits = [e for e in self.edits if not (e.kind == edit.kind and e.key == edit.key and
                                                    (e.tier == edit.tier or edit.tier is None))]
        self.edits.append(edit)


def apply_edit(look: Look, e: Edit, base_colours=None) -> None:
    if e.kind == "part":  # key (unit type, slot)
        utype, slot = e.key
        if utype not in UNIT_TYPES or slot not in SLOTS:
            raise ValueError(f"part edit {e.key!r}: unit type and slot must be from {UNIT_TYPES} and {SLOTS}")
        look.edit(("roles", utype, slot), copy.deepcopy(e.value), e.tier)
    elif e.kind == "finish":  # key (area,), value finish | [finish, rgb] | None
        look.edit(("body", e.key[0]), copy.deepcopy(e.value), e.tier)
    elif e.kind == "material_colour":  # key (finish,), value rgb: every area with that finish takes it
        finish = e.key[0]

        def tint(body: dict) -> None:
            for area, f in list(body.items()):
                name = f[0] if isinstance(f, list) else f
                if name == finish:
                    body[area] = [finish, list(e.value)]
        tint(look.body)
        for d in (*look.steps.values(), *look.per_tier.values()):
            tint(d.get("body") or {})
    elif e.kind == "unit_colour":  # key (0 | 1 | 2,), value rgb
        colours = [list(c) for c in (look.colours or base_colours or [[150, 28, 32], [195, 150, 60], [230, 222, 200]])]
        colours[e.key[0]] = list(e.value)
        look.colours = colours
    elif e.kind == "crest_colour":
        look.crest_colour = list(e.value) if e.value else None
    elif e.kind == "emblem":
        look.emblem = e.value
    elif e.kind == "name":  # key (unit type,)
        look.names = {**look.names, e.key[0]: e.value}
