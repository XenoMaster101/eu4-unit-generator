"""Looks: what the prompt front end edits (SPEC Track F). A look says, in plain terms, what each
unit type carries and what each body area is made of; ``compile_look`` turns it into the
character spec's texture recipe and loadout, which the rest of the app (and the Advanced editor)
works on.

    {"template": "hoplite",
     "body_base": "western_generic_01_type1",           # every tier's infantry body
     "roles": {"infantry": {"weapon": "dory", "shield": "hoplon", "shield_params": {"radius": 3.8},
                            "helmet": {"style": "corinthian", "crest": "fore_aft"}},
               "cavalry": {"weapon": "xyston", "crest": {"orientation": "fore_aft"}},
               "artillery": {}},
     "body": {"torso": "linen", "shins": "bronze", "collar": ["colour", [120, 20, 20]]},
     "steps": {"1": {...}, ...},           # the template's "gets heavier with tech" looks per tier
     "progression": "heavier",             # or "same": every tier wears the main look
     "per_tier": {"4": {...}},             # "this tier only" edits, on top of either
     "colours": [[r, g, b], [r, g, b], [r, g, b]] | null,   # unit colours
     "crest_colour": [r, g, b] | null,     # every unit type's crest, unless its role sets "crest_colour"
     "emblem": "<png path>" | null,
     "names": {"infantry": "Pezhetairoi", ...}}

Tier overrides merge over the main look key by key; a null removes a key (an area goes back to
the base texture). Unit types map to loadout roles: infantry -> infantry, cavalry -> rider,
artillery -> crew (a crew slot, so the gun crew doesn't carry the infantry's shield).

Compiling merges into an existing spec: only the parts a look controls (weapons, shields, Greek
helmets, crests) and the finishes of ``area:`` regions are replaced. Other parts, painted
``mask:`` finishes and every other recipe operation stay.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from eu4unitgen.core.finishes import finishes, split, with_finishes
from eu4unitgen.core.naming import TIERS, UNIT_TYPES
from eu4unitgen.library.data import library

ROLE_OF = {"infantry": "infantry", "cavalry": "rider", "artillery": "crew"}
WEAPONS = ("dory", "sarissa", "xyston", "xiphos")
SHIELDS = ("hoplon",)
CONTROLLED = (*WEAPONS, *SHIELDS, "greek_helmet", "crest")  # part kinds a look replaces
PROGRESSIONS = ("same", "heavier")
DEFAULT_SKIN = (196, 150, 120)  # textures.areas.skin_colour's "no skin found" tone, until the body is read


class LookError(ValueError):
    pass


def merge(base: dict, over: dict) -> dict:
    """``over`` merged into a copy of ``base``: dicts key by key, a None removes the key."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def templates() -> dict[str, dict]:
    return library("templates")["templates"]


@dataclass
class Look:
    roles: dict = field(default_factory=lambda: {t: {} for t in UNIT_TYPES})
    body: dict = field(default_factory=dict)
    body_base: str = "western_generic_01_type1"
    steps: dict = field(default_factory=dict)  # tier (str) -> override
    progression: str = "heavier"
    per_tier: dict = field(default_factory=dict)  # tier (str) -> override
    colours: list | None = None
    crest_colour: list | None = None
    emblem: str | None = None
    names: dict = field(default_factory=dict)
    template: str = ""

    # ------------------------------------------------------------------ files
    @classmethod
    def from_template(cls, name: str) -> "Look":
        t = templates().get(name)
        if t is None:
            raise LookError(f"unknown template {name!r}; known: {', '.join(templates())}")
        return cls(roles=copy.deepcopy(t["roles"]), body=copy.deepcopy(t["body"]), body_base=t["body_base"],
                   steps=copy.deepcopy(t.get("steps") or {}), names=dict(t.get("names") or {}), template=name)

    @classmethod
    def from_json(cls, doc: dict) -> "Look":
        if not isinstance(doc, dict):
            raise LookError("a look must be an object")
        known = {f for f in cls.__dataclass_fields__}
        extra = sorted(set(doc) - known)
        if extra:
            raise LookError(f"look: unknown keys {extra}")
        look = cls(**copy.deepcopy(doc))
        look.check()
        return look

    def to_json(self) -> dict:
        return {k: copy.deepcopy(getattr(self, k)) for k in self.__dataclass_fields__}

    def check(self) -> None:
        if self.progression not in PROGRESSIONS:
            raise LookError(f"progression must be one of {PROGRESSIONS}")
        for t in range(1, 5):
            self.tier(t)  # raises on a bad part or finish

    # ------------------------------------------------------------------ use
    def tier(self, t: int) -> dict:
        """The look tier ``t`` wears: {"roles", "body"} (main look, then its step, then this-tier edits)."""
        out = {"roles": self.roles, "body": self.body}
        if self.progression == "heavier":
            out = merge(out, self.steps.get(str(t)) or {})
        out = merge(out, self.per_tier.get(str(t)) or {})
        for utype, r in out["roles"].items():
            if utype not in UNIT_TYPES:
                raise LookError(f"unit type {utype!r} must be one of {UNIT_TYPES}")
            if r.get("weapon") not in (None, *WEAPONS):
                raise LookError(f"{utype}: weapon {r['weapon']!r} must be one of {WEAPONS}")
            if r.get("shield") not in (None, *SHIELDS):
                raise LookError(f"{utype}: shield {r['shield']!r} must be one of {SHIELDS}")
            if r.get("helmet") and utype != "infantry":
                raise LookError(f"{utype}: only infantry can wear a Greek helmet (riders and crews keep the base "
                                "body's own helmet; give them a crest)")
        for area, f in out["body"].items():
            name = f[0] if isinstance(f, list) else f
            if name not in finishes():
                raise LookError(f"body {area}: finish {name!r} must be one of {', '.join(finishes())}")
        return out

    def edit(self, path: tuple, value: Any, tier: int | None = None) -> None:
        """Set ``path`` (e.g. ("roles", "infantry", "weapon") or ("body", "shins")) to ``value``
        (None removes it). ``tier``: that tier only; None: every tier, including tiers whose
        step sets the same thing."""
        def put(d: dict, v) -> None:
            for k in path[:-1]:
                if not isinstance(d.get(k), dict):  # missing, or set to nothing (a helmet of None)
                    d[k] = {}
                d = d[k]
            d[path[-1]] = v

        def has(d: dict) -> bool:
            for k in path[:-1]:
                d = d.get(k) if isinstance(d.get(k), dict) else {}
            return path[-1] in d

        if tier is not None:
            put(self.per_tier.setdefault(str(tier), {}), value)
            return
        base = {"roles": self.roles, "body": self.body}
        if value is None:
            node = base
            for k in path[:-1]:
                if not isinstance(node.get(k), dict):
                    node[k] = {}
                node = node[k]
            node.pop(path[-1], None)
        else:
            put(base, value)
        for t in TIERS:
            step = self.steps.get(str(t)) or {}
            if has(self.per_tier.get(str(t)) or {}) or (self.progression == "heavier" and has(step)):
                put(self.per_tier.setdefault(str(t), {}), value)


# --------------------------------------------------------------------------- compiling

def _parts(utype: str, r: dict, crest_colour, emblem) -> list[dict]:
    crest_colour = r.get("crest_colour") or crest_colour  # this unit type's own, else the army's
    out = []
    if r.get("weapon"):
        out.append({"kind": r["weapon"]})
    if r.get("shield"):
        shield = {"kind": r["shield"]}
        if r.get("shield_params"):
            shield["params"] = dict(r["shield_params"])
        if emblem:
            shield["blazon"] = emblem
        out.append(shield)
    if r.get("helmet"):
        helmet = {"kind": "greek_helmet", "params": dict(r["helmet"])}
        if crest_colour:
            helmet["colours"] = {"crest": list(crest_colour)}
        out.append(helmet)
    elif r.get("crest"):
        crest = {"kind": "crest", "params": dict(r["crest"])}
        if crest_colour:
            crest["colours"] = {"hair": list(crest_colour)}
        out.append(crest)
    return out


def compile_look(look: Look, doc: dict, skins: dict[int, tuple] | None = None) -> dict:
    """``doc`` (a character spec as JSON, with at least its "set") with the look applied to its
    loadout, texture recipe and unit colours. ``skins``: tier -> the body's own skin tone (for
    "bare skin"; Session.body_areas), else DEFAULT_SKIN."""
    from eu4unitgen.textures.areas import area_order

    out = copy.deepcopy(doc)
    loadout = out.get("loadout") or {"schema": 1}
    loadout.setdefault("slots", {})
    textures = out.get("textures") or {"schema": 1, "slots": {}}
    textures.setdefault("slots", {})
    order = [f"area:{a}" for a in area_order(f"gfx/models/units/{look.body_base}.mesh")]
    for t in TIERS:
        tl = look.tier(t)
        loadout.setdefault("bodies", {})[str(t)] = look.body_base
        for utype, role in ROLE_OF.items():
            key = f"{t}/{role}"
            slot = loadout["slots"].get(key) or {}
            kept = [p for p in slot.get("parts") or [] if p.get("kind") not in CONTROLLED]
            parts = kept + _parts(utype, tl["roles"].get(utype) or {}, look.crest_colour, look.emblem)
            if parts or role == "crew":  # a crew slot even when empty: the crew drops the infantry's shield
                loadout["slots"][key] = {**slot, "parts": parts}
            else:
                loadout["slots"].pop(key, None)
        derived = loadout.setdefault("derived", {})
        d = derived.get(str(t))
        d = {"add": list(d), "remove": []} if isinstance(d, list) else dict(d or {"add": [], "remove": []})
        removes = [x for x in d.get("remove") or [] if x != "helmet"]
        if (tl["roles"].get("infantry") or {}).get("helmet"):
            removes.append("helmet")  # the Greek helmet replaces type1's own
        d["remove"] = removes
        if d.get("add") or d["remove"]:
            derived[str(t)] = d
        else:
            derived.pop(str(t), None)

        tslot = textures["slots"].get(f"{t}/infantry") or {"base": None, "ops": []}
        _, existing = split(tslot.get("ops") or [])
        chosen = {r: c for r, c in existing.items() if not r.startswith("area:")}  # painted masks stay
        for area, f in tl["body"].items():
            name, colour = (f[0], tuple(f[1])) if isinstance(f, list) else (f, None)
            chosen[f"area:{area}"] = {"finish": name, "colour": colour}
        tslot["ops"] = with_finishes(tslot.get("ops") or [], chosen, tuple((skins or {}).get(t) or DEFAULT_SKIN),
                                     order)
        if tslot["ops"] or tslot.get("base"):
            textures["slots"][f"{t}/infantry"] = tslot
        else:
            textures["slots"].pop(f"{t}/infantry", None)
    if not loadout.get("derived"):
        loadout.pop("derived", None)
    out["loadout"] = loadout
    out["textures"] = textures
    if look.colours:
        out.setdefault("set", {})["unit_colours"] = [list(c) for c in look.colours]
    names = {k: v for k, v in (look.names or {}).items() if v}
    if names:
        out.setdefault("set", {})["unit_names"] = names
    else:
        out.get("set", {}).pop("unit_names", None)
    return out
