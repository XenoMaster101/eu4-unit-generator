"""Reading a description of soldiers (SPEC Track F, F4): words -> understood items -> a look.

Only the offline keyword interpreter exists. It sits behind ``Interpreter`` so another one could
be added later; none is planned (no AI, no API keys).

The vocabulary is data (``library/vocabulary.json``) merged with the user's own synonyms
(``vocabulary.json`` in the user's config folder, never in the repo). Reading works clause by
clause (a comma, semicolon or full stop ends one): a scope word (cavalry, crew) says which unit
type the following parts are for; materials and colours wait for the next garment or part they
describe ("bronze greaves", "red crest"); "no" / "without" removes the next part. What isn't in
the vocabulary or the stopwords comes back as leftovers ("not understood").

Items are what the Home and Result screens show as chips: each can be removed or swapped for
another value of its kind, and ``Interpretation.look()`` rebuilds the look from them.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from eu4unitgen.core.look import Look
from eu4unitgen.library.data import library

DEFAULT_TEMPLATE = "hoplite"
SECOND, THIRD = (195, 150, 60), (230, 222, 200)  # unit colours 2 and 3 when only the main one is named
_TOKEN = re.compile(r"[a-z]+|[.,;:!?\n]")


class VocabularyError(ValueError):
    pass


def config_dir() -> Path:
    """The user's config folder: %APPDATA%\\eu4unitgen on Windows, else $XDG_CONFIG_HOME or ~/.config."""
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "eu4unitgen"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "eu4unitgen"


def entry_id(e: dict) -> str:
    return f"{e['kind']}:{e.get('id') or e['value']}"


def words_of(text: str) -> list[str]:
    return _TOKEN.findall(_flat(text))


def _flat(text: str) -> str:  # same length as text, so token spans index the original
    return text.lower().replace("-", " ").replace("'", " ")


class Vocabulary:
    """The library's entries plus the user's synonyms (``user_file``: word -> entry id)."""

    def __init__(self, user_file: Path | None = None):
        lib = library("vocabulary")
        self.user_file = user_file if user_file is not None else config_dir() / "vocabulary.json"
        self.stopwords = set(lib["stopwords"])
        self.entries = [dict(e, words=list(e["words"])) for e in lib["entries"]]
        self.by_id = {entry_id(e): e for e in self.entries}
        self.user: dict[str, str] = {}
        if self.user_file.is_file():
            try:
                doc = json.loads(self.user_file.read_text(encoding="utf-8"))
                self.user = {str(k): str(v) for k, v in (doc.get("synonyms") or {}).items()}
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                raise VocabularyError(f"can't read your vocabulary {self.user_file}: {exc}") from exc
        self.phrases: dict[tuple[str, ...], dict] = {}
        for e in self.entries:
            for w in e["words"]:
                self.phrases[tuple(words_of(w))] = e
        for w, eid in self.user.items():  # the user's words win
            if eid in self.by_id:
                self.phrases[tuple(words_of(w))] = self.by_id[eid]
        self.longest = max(len(p) for p in self.phrases)

    def add_synonym(self, word: str, eid: str) -> None:
        """Save ``word`` as meaning entry ``eid`` in the user's vocabulary file."""
        if not re.fullmatch(r"\s*[A-Za-z][A-Za-z' -]*\s*", word or ""):
            raise VocabularyError("a synonym is one or more words of letters")
        word = " ".join(words_of(word))
        if eid not in self.by_id:
            raise VocabularyError(f"unknown meaning {eid!r}")
        self.user[word] = eid
        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(json.dumps({"schema": 1, "synonyms": dict(sorted(self.user.items()))}, indent=2)
                                  + "\n", encoding="utf-8")
        self.phrases[tuple(word.split())] = self.by_id[eid]
        self.longest = max(self.longest, len(word.split()))

    def meanings(self, kinds=("weapon", "shield", "helmet", "crest", "material", "garment", "colour", "template",
                              "scope")) -> list[tuple[str, str]]:
        """(entry id, "Label (kind)") for "add this word as…"."""
        return [(entry_id(e), f"{e['label']} ({e['kind']})") for e in self.entries if e["kind"] in kinds]

    def alternatives(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e["kind"] == kind]

    def suggest(self, text: str, limit: int = 8) -> list[str]:
        """Words to offer while typing: completions of the last partial word, else (after a
        material or colour) the garments and parts it could describe."""
        if not text or text[-1].isspace() or text[-1] in ",.;":
            toks = [t for t in words_of(text) if t.isalpha()]
            if toks:
                e = self.phrases.get((toks[-1],))
                if e and e["kind"] in ("material", "colour"):  # what it could describe, likeliest first
                    kinds = ("garment",) if e["kind"] == "material" else ("crest", "shield", "garment")
                    nouns = [x["words"][0] for k in kinds for x in self.entries if x["kind"] == k and "id" not in x]
                    return nouns[:limit]
            return []
        toks = words_of(text)
        last = toks[-1] if toks else ""
        before = self.phrases.get((toks[-2],)) if len(toks) > 1 else None
        nouns_first = before is not None and before["kind"] in ("material", "colour")
        out: list[tuple[tuple, str]] = []
        for e in self.entries:
            for w in e["words"]:
                if w.startswith(last) and w != last and w not in (x[1] for x in out):
                    noun = e["kind"] in ("garment", "crest", "shield", "helmet")
                    out.append(((nouns_first and not noun, len(w.split()) > 1, len(w)), w))
        return [w for _, w in sorted(out)][:limit]


# --------------------------------------------------------------------------- items

KIND_LABEL = {"template": "Start from", "weapon": "Weapon", "shield": "Shield", "helmet": "Helmet", "crest": "Crest",
              "crest_colour": "Crest colour", "finish": "Finish", "unit_colour": "Main colour",
              "progression": "Tiers"}


@dataclass
class Item:
    """One understood thing. ``value``: the entry's value (None: "no ..."); ``utype``: the unit
    type it's for; ``area``/``colour``/``params`` where they apply; ``words``: the source text."""
    kind: str
    value: Any
    label: str
    utype: str = "infantry"
    area: str | None = None
    colour: tuple | None = None
    params: dict | None = None
    words: str = ""
    spans: dict = field(default_factory=dict)  # "noun"/"material"/"colour"/"negation" -> (start, end) in the text

    def text(self) -> str:
        """The chip's text: the label, and the unit type when it isn't infantry and the label
        doesn't already say it ("Xyston (cavalry lance)" stays as it is)."""
        other = self.kind in ("weapon", "shield", "helmet", "crest") and self.utype != "infantry"
        return f"{self.label} ({self.utype})" if other and self.utype not in self.label.lower() else self.label


@dataclass
class Interpretation:
    text: str
    items: list[Item] = field(default_factory=list)
    leftovers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def look(self) -> Look:
        """The look these items describe, from their template (default: the hoplite)."""
        tmpl = next((i.value for i in self.items if i.kind == "template"), DEFAULT_TEMPLATE)
        look = Look.from_template(tmpl)
        main = None
        for i in self.items:
            if i.kind == "weapon":
                look.edit(("roles", i.utype, "weapon"), i.value)
            elif i.kind == "shield":
                look.edit(("roles", i.utype, "shield"), i.value)
                look.edit(("roles", i.utype, "shield_params"), dict(i.params) if (i.value and i.params) else None)
            elif i.kind == "helmet":
                if i.value is None:
                    look.edit(("roles", "infantry", "helmet"), None)
                else:
                    look.edit(("roles", "infantry", "helmet", "style"), i.value)
            elif i.kind == "crest":
                if i.utype == "infantry":
                    look.edit(("roles", "infantry", "helmet", "crest"), i.value or "none")
                else:
                    look.edit(("roles", i.utype, "crest"), {"orientation": i.value} if i.value else None)
            elif i.kind == "crest_colour":
                look.crest_colour = list(i.colour)
            elif i.kind == "finish":
                look.edit(("body", i.area), None if i.value is None else ([i.value, list(i.colour)] if i.colour else i.value))
            elif i.kind == "unit_colour":
                main = i.colour
            elif i.kind == "progression":
                look.progression = i.value
        if main:
            look.colours = [list(main), list(SECOND), list(THIRD)]
        return look

    def remove(self, index: int) -> None:
        del self.items[index]

    def swap(self, index: int, entry: dict) -> None:
        """Replace item ``index``'s value with another entry of the same kind (a chip click)."""
        i = self.items[index]
        if entry["kind"] == "colour":
            self.items[index] = replace(i, colour=tuple(entry["value"]), label=_colour_label(i, entry))
        elif entry["kind"] == "material":
            self.items[index] = replace(i, value=entry["value"], label=f"{entry['label']} {_area_label(i.area)}")
        else:
            self.items[index] = replace(i, value=entry["value"], label=entry["label"], params=entry.get("params"))

    # the text is what the screens keep: a chip's x or swap rewrites it, then it is read again
    def text_without(self, index: int) -> str:
        """The text with item ``index``'s words taken out (its x)."""
        return _tidy(_edit(self.text, {span: "" for span in self.items[index].spans.values()}))

    def text_swapped(self, index: int, entry: dict) -> str:
        """The text with item ``index`` saying ``entry`` instead (a chip click): its colour, its
        material (added before the garment if it had none) or its noun."""
        i, new = self.items[index], entry["words"][0]
        part = {"colour": "colour", "material": "material"}.get(entry["kind"], "noun")
        if part in i.spans:
            start, end = i.spans[part]
            if self.text[start:start + 1].isupper():
                new = new[:1].upper() + new[1:]
            return _edit(self.text, {(start, end): new})
        if "noun" in i.spans:  # "greaves" -> "iron greaves"
            start, end = i.spans["noun"]
            return _edit(self.text, {(start, end): f"{new} {self.text[start:end]}"})
        return self.text


def _edit(text: str, changes: dict) -> str:
    for (start, end), new in sorted(changes.items(), reverse=True):
        text = text[:start] + new + text[end:]
    return text


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    for _ in range(2):  # a joining word left with nothing to join: "with," "and;"
        text = re.sub(r"(?i)\b(?:with|and|or|of|in|a|an|the|plus)[ \t]*(?=[,.;:!?]|$)", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;])[ \t]*(?=[,;.]|$)", "", text)
    text = re.sub(r"^[ \t]*[,;][ \t]*", "", text)
    return text.strip(" ,;")


def text_with_template(text: str, template: str, interpreter: "KeywordInterpreter") -> str:
    """``text`` starting from ``template`` (a template chip): its template word swapped, or the
    template's name put in front."""
    entry = next(e for e in interpreter.vocab.entries if e["kind"] == "template" and e["value"] == template)
    res = interpreter.interpret(text)
    at = next((n for n, i in enumerate(res.items) if i.kind == "template"), None)
    if at is not None:
        return res.text_swapped(at, entry)
    return f"{entry['label']}: {text.strip()}" if text.strip() else entry["label"]


def _area_label(area: str | None) -> str:
    return (area or "").replace("_", " ")


def _colour_label(i: Item, entry: dict) -> str:
    if i.kind == "crest_colour":
        return f"{entry['label']} crest"
    if i.kind == "finish":
        return f"{entry['label']} {_area_label(i.area)}"
    return f"{entry['label']} (main colour)"


# --------------------------------------------------------------------------- interpreters

class Interpreter(ABC):
    """Text (and an optional reference palette, F5) -> an Interpretation."""
    name = ""

    @abstractmethod
    def interpret(self, text: str, palette: list[Item] | None = None) -> Interpretation:
        ...


class KeywordInterpreter(Interpreter):
    name = "keywords"

    def __init__(self, vocab: Vocabulary | None = None):
        self.vocab = vocab or Vocabulary()

    def _matches(self, text: str) -> list[tuple[str, dict | None, str, tuple[int, int]]]:
        """(kind, entry, source words, (start, end) in the text): phrases longest first; "break" at
        punctuation; stopwords; unknown words."""
        found = list(_TOKEN.finditer(_flat(text)))
        toks = [m.group(0) for m in found]
        out, i = [], 0
        while i < len(toks):
            span = (found[i].start(), found[i].end())
            if not toks[i].isalpha():
                out.append(("break", None, toks[i], span))
                i += 1
                continue
            for n in range(min(self.vocab.longest, len(toks) - i), 0, -1):
                key = tuple(toks[i:i + n])
                if all(t.isalpha() for t in key) and key in self.vocab.phrases:
                    e = self.vocab.phrases[key]
                    out.append((e["kind"], e, " ".join(key), (found[i].start(), found[i + n - 1].end())))
                    i += n
                    break
            else:
                out.append(("stop" if toks[i] in self.vocab.stopwords else "unknown", None, toks[i], span))
                i += 1
        return out

    def interpret(self, text: str, palette: list[Item] | None = None) -> Interpretation:
        res = Interpretation(text)
        scope, material, colour, negate = "infantry", None, None, False

        def flush() -> None:  # a material or colour with nothing after it in its clause
            nonlocal material, colour
            if material:  # "in bronze": the torso
                e, words, span = material
                tint = tuple(colour[0]["value"]) if colour else None
                spans = {"material": span, **({"colour": colour[2]} if colour else {})}
                res.items.append(Item("finish", e["value"], (f"{colour[0]['label']} " if colour else "")
                                      + f"{e['label']} torso", area="torso", colour=tint, words=words, spans=spans))
                colour = None  # it tinted the material
            if colour:
                e, words, span = colour
                res.items.append(Item("unit_colour", None, f"{e['label']} (main colour)", colour=tuple(e["value"]),
                                      words=words, spans={"colour": span}))
            material = colour = None

        for kind, e, words, span in self._matches(text):
            noun = {"noun": span, **({"negation": negate} if negate else {})}
            if kind == "break":
                flush()
                scope, negate = "infantry", False
            elif kind == "unknown":
                if words not in res.leftovers:
                    res.leftovers.append(words)
            elif kind == "stop":
                continue
            elif kind == "template":
                res.items.append(Item("template", e["value"], e["label"], words=words, spans={"noun": span}))
            elif kind == "scope":
                scope = e["value"]
            elif kind == "negation":
                negate = span  # truthy: the next part is a "no ..."
            elif kind == "progression":
                res.items.append(Item("progression", e["value"], e["label"], words=words, spans={"noun": span}))
            elif kind == "material":
                material = (e, words, span)
            elif kind == "colour":
                colour = (e, words, span)
            elif kind == "garment":
                if negate:
                    res.items.append(Item("finish", None, f"no {e['label'].lower()}", area=e["value"], words=words,
                                          spans=noun))
                else:
                    m = material[0] if material else None
                    finish = m["value"] if m else ("colour" if colour else e["default"])
                    tint = tuple(colour[0]["value"]) if colour else None
                    name = (colour[0]["label"] + " " if colour else "") + (m["label"].lower() + " " if m else "")
                    label = (name + words).capitalize() if (m or colour) else f"{words.capitalize()}: {finish}"
                    spans = {**noun, **({"material": material[2]} if material else {}),
                             **({"colour": colour[2]} if colour else {})}
                    res.items.append(Item("finish", finish, label, area=e["value"], colour=tint,
                                          words=" ".join(x for x in ((material or (None, ""))[1], (colour or (None, ""))[1],
                                                                     words) if x), spans=spans))
                material = colour = None
                negate = False
            elif kind in ("weapon", "shield", "helmet", "crest"):
                utype = e.get("scope") if (kind == "weapon" and e.get("scope") and scope == "infantry") else scope
                value = None if negate else e["value"]
                if kind == "shield" and utype == "cavalry":
                    res.notes.append(f"\"{words}\": riders can't carry a shield yet, left out")
                elif kind == "helmet" and utype != "infantry":
                    res.notes.append(f"\"{words}\": only infantry can wear a Greek helmet; {utype} keep the base "
                                     "helmet (a crest works)")
                else:
                    label = f"no {e['label'].lower()}" if negate else e["label"]
                    owned = {"colour": colour[2]} if (colour and kind in ("crest", "shield") and not negate) else {}
                    res.items.append(Item(kind, value, label, utype=utype, params=e.get("params"), words=words,
                                          spans={**noun, **owned}))
                    if kind == "crest" and colour and not negate:
                        res.items.append(Item("crest_colour", None, f"{colour[0]['label']} crest",
                                              colour=tuple(colour[0]["value"]), words=colour[1],
                                              spans={"colour": colour[2]}))
                        colour = None
                    if kind == "shield" and colour and not negate:  # the hoplon's face is the main unit colour
                        res.items.append(Item("unit_colour", None, f"{colour[0]['label']} (main colour)",
                                              colour=tuple(colour[0]["value"]), words=colour[1],
                                              spans={"colour": colour[2]}))
                        colour = None
                    if kind == "helmet" and material:
                        res.notes.append(f"\"{material[1]} {words}\": helmets are bronze for now")
                        material = None
                negate = False
        flush()
        res.items += list(palette or [])
        return res
