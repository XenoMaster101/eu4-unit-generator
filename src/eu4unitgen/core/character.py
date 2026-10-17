"""Character specs: one JSON file per character (SPEC Track R, data-driven characters).

Everything that makes a character lives in its spec; nothing character- or test-specific is in
code (the numbers code needs are in the data libraries, eu4unitgen/library/*.json):

    {"schema": 1,
     "name": "Hellenistic hoplite",                        # free text for the GUI
     "set": {"tags": ["HDA"], "cultures": [], "gfx": "hellenicgfx", "set_name": "hoplite_test",
             "base_gfx": "westerngfx", "name": "", "adjective": "", "colour": null, "tech_group": "",
             "unit_colours": [[r, g, b], [r, g, b], [r, g, b]]   # every tag; or {"TAG": [[...], [...], [...]]}
             },
     "textures": {...},     # a core.recipe texture recipe, inline (or null)
     "loadout": {...},      # a core.loadout loadout, inline (or null): slots, bodies, derived, accept_clipping
     "hair": {"colour": [r, g, b], "inner": [r, g, b], "gloss": n},   # the derived body's hair swatch (optional)
     "export": {"prefix": "hr_", "version": 20, "pack_order": 50},   # optional; app defaults otherwise
     "test": {"sprite_techs": [2, 3, 4], "base_mod": null, "base_name": null, "give_cores": false}}

The ``test`` section only feeds the test mod; a real export never reads it. Relative paths
(blazons, recipe images and masks) resolve against the spec file's folder.

``from_parts`` converts the old way of describing a set (CLI flags, a loadout.json, a recipe
file) into a spec, rebasing the files' relative paths onto the spec's folder.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1
SECTIONS = ("schema", "name", "set", "textures", "loadout", "hair", "export", "test")
SET_KEYS = ("tags", "cultures", "gfx", "set_name", "base_gfx", "name", "adjective", "colour", "tech_group",
            "unit_colours", "gfx_set_base")
EXPORT_KEYS = ("prefix", "version", "pack_order")
TEST_KEYS = ("sprite_techs", "base_mod", "base_name", "give_cores")
HAIR_KEYS = ("colour", "inner", "gloss")


class CharacterError(ValueError):
    pass


def _unknown(where: str, doc: dict, known) -> None:
    extra = sorted(set(doc) - set(known))
    if extra:
        raise CharacterError(f"{where}: unknown keys {extra}; known: {', '.join(known)}")


def _triple(v: Any, where: str) -> tuple:
    if not (isinstance(v, list) and len(v) == 3 and all(isinstance(c, list) and len(c) == 3 for c in v)
            and all(isinstance(x, int) and 0 <= x <= 255 for c in v for x in c)):
        raise CharacterError(f"{where}: unit colours are three [r, g, b] lists, 0-255")
    return tuple(tuple(c) for c in v)


@dataclass
class CharacterSpec:
    set: dict
    name: str = ""
    textures: dict | None = None
    loadout: dict | None = None
    hair: dict = field(default_factory=dict)
    export: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    root: Path = Path(".")

    # ------------------------------------------------------------------ files
    @classmethod
    def load(cls, path: Path) -> "CharacterSpec":
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CharacterError(f"can't read character spec {path}: {exc}") from exc
        return cls.from_json(doc, path.parent)

    @classmethod
    def from_json(cls, doc: dict, root: Path = Path(".")) -> "CharacterSpec":
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise CharacterError(f"a character spec must be an object with \"schema\": {SCHEMA}")
        _unknown("character spec", doc, SECTIONS)
        s = doc.get("set")
        if not isinstance(s, dict) or not s.get("gfx"):
            raise CharacterError("set: an object with at least \"gfx\" (and tags or cultures) is required")
        _unknown("set", s, SET_KEYS)
        for key, keys in (("export", EXPORT_KEYS), ("test", TEST_KEYS), ("hair", HAIR_KEYS)):
            if doc.get(key) is not None:
                if not isinstance(doc[key], dict):
                    raise CharacterError(f"{key}: must be an object")
                _unknown(key, doc[key], keys)
        spec = cls(set=dict(s), name=str(doc.get("name") or ""), textures=doc.get("textures"),
                   loadout=doc.get("loadout"), hair=dict(doc.get("hair") or {}), export=dict(doc.get("export") or {}),
                   test=dict(doc.get("test") or {}), root=Path(root))
        spec.set_spec()  # checks the set, the recipe and the loadout now, not at generation
        return spec

    def to_json(self) -> dict:
        out: dict = {"schema": SCHEMA}
        if self.name:
            out["name"] = self.name
        out["set"] = self.set
        for key in ("textures", "loadout"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        for key in ("hair", "export", "test"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ use
    def unit_colours(self):
        uc = self.set.get("unit_colours")
        if uc is None:
            return None
        if isinstance(uc, dict):
            return {str(t).upper(): _triple(v, f"set.unit_colours.{t}") for t, v in uc.items()}
        triple = _triple(uc, "set.unit_colours")
        return {t.upper(): triple for t in self.set.get("tags") or []} or None

    def set_spec(self):
        """The pipeline's SetSpec, with the recipe and loadout objects (paths against the spec's folder)."""
        from eu4unitgen.core.loadout import Loadout, LoadoutError
        from eu4unitgen.core.recipe import RecipeError, TextureRecipe
        from eu4unitgen.export.pipeline import SetSpec

        s = self.set
        try:
            recipe = TextureRecipe.from_json(self.textures, self.root) if self.textures is not None else None
            loadout = None
            if self.loadout is not None:
                loadout = Loadout.from_json(self.loadout, self.root)
                loadout.hair = dict(self.hair)
        except (RecipeError, LoadoutError) as exc:
            raise CharacterError(str(exc)) from exc
        colour = s.get("colour")
        return SetSpec(tags=[t.upper() for t in s.get("tags") or []], cultures=[c.lower() for c in s.get("cultures") or []],
                       gfx=s["gfx"], set_name=s.get("set_name") or "", base_gfx=s.get("base_gfx") or "westerngfx",
                       name=s.get("name") or "", adjective=s.get("adjective") or "",
                       colour=tuple(colour) if colour else None, tech_group=s.get("tech_group") or "",
                       gfx_set_base=s.get("gfx_set_base"), unit_colours=self.unit_colours(),
                       texture_recipe=recipe, loadout=loadout)

    def test_options(self) -> dict:
        """Keyword arguments for Session.build_test_mod (the ``test`` section)."""
        t = self.test
        techs = t.get("sprite_techs")
        base_mod = t.get("base_mod")
        return {"sprite_techs": dict(zip((2, 3, 4), techs)) if techs else None,
                "base_mod": (self.root / base_mod if not Path(base_mod).is_absolute() else Path(base_mod)) if base_mod else None,
                "base_name": t.get("base_name"), "give_cores": bool(t.get("give_cores"))}


# --------------------------------------------------------------------------- the converter

PATH_KEYS = ("image", "mask", "blazon")  # recipe ops' images and masks, loadout parts' blazons


def _rebase(value: str, src: Path, dst: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return value
    target = (src / p).resolve()
    try:
        return Path(os.path.relpath(target, dst.resolve())).as_posix()
    except ValueError:  # another drive: keep it absolute
        return target.as_posix()


def rebase_paths(doc: Any, src: Path, dst: Path, keep=()) -> Any:
    """``doc`` with every relative file path (recipe images and masks, ``mask:<png>`` regions,
    blazons, the recipe's protect mask) rebased from folder ``src`` onto folder ``dst``. Paths in
    ``keep`` stay as they are (files about to be written in ``dst`` itself)."""
    if isinstance(doc, list):
        return [rebase_paths(v, src, dst, keep) for v in doc]
    if not isinstance(doc, dict):
        return doc
    out = {}
    for k, v in doc.items():
        if isinstance(v, str) and k in PATH_KEYS:
            out[k] = v if v in keep else _rebase(v, src, dst)
        elif isinstance(v, str) and k == "region" and v.startswith("mask:"):
            out[k] = v if v[5:] in keep else "mask:" + _rebase(v[5:], src, dst)
        else:
            out[k] = rebase_paths(v, src, dst, keep)
    return out


def from_parts(spec_dir: Path, *, tags=(), cultures=(), gfx: str, set_name: str = "", base_gfx: str = "westerngfx",
               name: str = "", adjective: str = "", colour=None, tech_group: str = "", unit_colours=None,
               gfx_set_base=None, recipe: Path | None = None, loadout: Path | None = None, prefix: str | None = None,
               version: int | None = None, pack_order: int | None = None, sprite_techs=None, base_mod: Path | None = None,
               base_name: str | None = None, give_cores: bool = False, label: str = "") -> CharacterSpec:
    """A spec from the old way of describing a set: CLI-style values plus a loadout.json and a
    recipe file, both embedded with their relative paths rebased onto ``spec_dir``."""
    spec_dir = Path(spec_dir)
    s: dict = {"tags": [t.upper() for t in tags], "gfx": gfx}
    for k, v in (("cultures", list(cultures)), ("set_name", set_name), ("base_gfx", base_gfx if base_gfx != "westerngfx" else ""),
                 ("name", name), ("adjective", adjective), ("colour", list(colour) if colour else None),
                 ("tech_group", tech_group), ("gfx_set_base", gfx_set_base)):
        if v:
            s[k] = v
    if unit_colours:
        if isinstance(unit_colours, dict):
            triples = {t: [list(c) for c in v] for t, v in unit_colours.items()}
            same = {json.dumps(v) for v in triples.values()}
            s["unit_colours"] = next(iter(triples.values())) if len(same) == 1 and set(triples) == set(s["tags"]) else triples
        else:
            s["unit_colours"] = [list(c) for c in unit_colours]

    def embed(path: Path | None):
        if path is None:
            return None
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CharacterError(f"can't read {path}: {exc}") from exc
        return rebase_paths(doc, path.parent, spec_dir)

    export = {k: v for k, v in (("prefix", prefix), ("version", version), ("pack_order", pack_order)) if v is not None}
    test = {k: v for k, v in (("sprite_techs", list(sprite_techs) if sprite_techs else None),
                              ("base_mod", _rebase(str(base_mod), Path("."), spec_dir) if base_mod else None),
                              ("base_name", base_name), ("give_cores", give_cores or None)) if v}
    return CharacterSpec.from_json({"schema": SCHEMA, **({"name": label} if label else {}), "set": s,
                                    **({"textures": embed(recipe)} if recipe else {}),
                                    **({"loadout": embed(loadout)} if loadout else {}),
                                    **({"export": export} if export else {}), **({"test": test} if test else {})},
                                   spec_dir)
