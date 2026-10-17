"""Facts about vanilla tags at a start date, and which sprite packs could apply to them.

Used to pick an in-game test tag: one that exists at the start date and that no
vanilla or DLC sprite pack and no vanilla TAG_* entity touches, so whatever shows on
the map comes from the generated set.

Trigger evaluation is three-valued: True / False / None (unknown). Anything not
modelled here counts as unknown, and unknown counts as "might match".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from eu4unitgen.library.scanner import GameFiles, Index
from eu4unitgen.pdxscript.parser import Block, Pair, Scalar, parse, parse_bytes

START_DATE = (1444, 11, 11)
UNIT_SUFFIXES = ("INFANTRY", "CAVALRY", "ARTILLERY")


def parse_date(text: str) -> tuple[int, int, int] | None:
    parts = text.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def _history_entries(doc: Block, date: tuple[int, int, int]) -> list[Pair]:
    """Top-level pairs plus those in dated blocks up to ``date``, in time order."""
    undated = [p for p in doc.pairs() if parse_date(p.key.value) is None]
    dated = sorted(((parse_date(p.key.value), i, p) for i, p in enumerate(doc.pairs())
                    if parse_date(p.key.value) is not None and isinstance(p.value, Block)), key=lambda t: (t[0], t[1]))
    out = list(undated)
    for d, _, p in dated:
        if d <= date:
            out += list(p.value.pairs())
    return out


@dataclass
class TagFacts:
    tag: str
    country_file: str | None = None  # e.g. common/countries/Epirus.txt
    history_file: str | None = None  # e.g. history/countries/EPI - Epirus.txt
    graphical_culture: str | None = None
    primary_culture: str | None = None
    accepted_cultures: list[str] = field(default_factory=list)
    culture_group: str | None = None
    religion: str | None = None
    technology_group: str | None = None
    reforms: list[str] = field(default_factory=list)
    provinces: int = 0


def culture_groups(files: GameFiles) -> dict[str, str]:
    """culture -> group, from common/cultures/*.txt."""
    out: dict[str, str] = {}
    for _, _, data in files.iter_files("common/cultures", (".txt",)):
        for group in parse_bytes(data).pairs():
            if not isinstance(group.value, Block):
                continue
            for c in group.value.pairs():
                if isinstance(c.value, Block) and c.key.value not in ("male_names", "female_names", "dynasty_names"):
                    out[c.key.value] = group.key.value
    return out


def province_owners(files: GameFiles, date: tuple[int, int, int] = START_DATE) -> dict[str, int]:
    """tag -> number of provinces it owns at ``date``."""
    counts: dict[str, int] = {}
    for _, _, data in files.iter_files("history/provinces", (".txt",)):
        owner = None
        for p in _history_entries(parse_bytes(data), date):
            if p.key.value == "owner" and isinstance(p.value, Scalar):
                owner = p.value.value
        if owner:
            counts[owner] = counts.get(owner, 0) + 1
    return counts


def country_files(files: GameFiles) -> dict[str, str]:
    """tag -> common/countries/<file>.txt, from common/country_tags."""
    out: dict[str, str] = {}
    for _, _, data in files.iter_files("common/country_tags", (".txt",)):
        for p in parse_bytes(data).pairs():
            if isinstance(p.value, Scalar):
                out.setdefault(p.key.value, "common/" + p.value.value)
    return out


def unit_colours(files: GameFiles) -> dict[str, list[tuple[int, int, int]]]:
    """tag -> [color1, color2, color3] from common/country_colors: the colours PdxMeshColor
    multiplies into spec R/G/B (not the map ``color`` in the country file). In-game test 5
    (2026-10-14): HDA has no entry and its units showed red. UNVERIFIED: the engine's default
    for tags without an entry (the file's example template uses 205 32 26)."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for _, _, data in files.iter_files("common/country_colors", (".txt",)):
        for p in parse_bytes(data).pairs():
            if not isinstance(p.value, Block):
                continue
            cols = []
            for key in ("color1", "color2", "color3"):
                b = p.value.get_block(key)
                vals = b.bare() if b is not None else []
                if len(vals) != 3 or not all(v.isdigit() for v in vals):
                    break
                cols.append(tuple(min(255, int(v)) for v in vals))
            if len(cols) == 3:
                out[p.key.value] = cols
    return out


def tag_facts(files: GameFiles, date: tuple[int, int, int] = START_DATE,
              tags: list[str] | None = None) -> dict[str, TagFacts]:
    """Facts for every tag that owns at least one province at ``date``, plus ``tags`` even when they
    own none (a releasable tag such as a mod's MAC). ``files`` may be a LayeredFiles (a mod over
    the game): its country_tags, countries and history then win."""
    owners = province_owners(files, date)
    for t in tags or []:
        owners.setdefault(t, 0)
    groups = culture_groups(files)
    cfiles = country_files(files)
    history: dict[str, tuple[str, bytes]] = {}
    for path, _, data in files.iter_files("history/countries", (".txt",)):
        tag = path.rsplit("/", 1)[-1][:3].upper()  # the engine reads the tag from the name's first 3 letters
        history.setdefault(tag, (path, data))
    out = {}
    for tag, n in sorted(owners.items()):
        f = TagFacts(tag, cfiles.get(tag), provinces=n)
        if f.country_file:
            raw = files.read(f.country_file)
            if raw:
                f.graphical_culture = parse_bytes(raw).get_str("graphical_culture")
        if tag in history:
            f.history_file = history[tag][0]
            apply_history(f, history[tag][1], groups, date)
        else:
            f.culture_group = groups.get(f.primary_culture or "")
        out[tag] = f
    return out


def apply_history(f: TagFacts, data: bytes, groups: dict[str, str],
                  date: tuple[int, int, int] = START_DATE) -> TagFacts:
    """Fill ``f``'s history facts from a history/countries file at ``date`` (in place).
    The history fields are reset first, so this also re-reads a patched copy."""
    f.primary_culture, f.religion, f.technology_group = None, None, None
    f.accepted_cultures, f.reforms = [], []
    for p in _history_entries(parse_bytes(data), date):
        if not isinstance(p.value, Scalar):
            continue
        k, v = p.key.value, p.value.value
        if k == "primary_culture":
            f.primary_culture = v
        elif k == "add_accepted_culture":
            f.accepted_cultures.append(v)
        elif k == "remove_accepted_culture" and v in f.accepted_cultures:
            f.accepted_cultures.remove(v)
        elif k == "religion":
            f.religion = v
        elif k == "technology_group":
            f.technology_group = v
        elif k == "add_government_reform":
            f.reforms.append(v)
    f.culture_group = groups.get(f.primary_culture or "")
    return f


def province_cores(files: GameFiles, tag: str) -> list[str]:
    """history/provinces files that give ``tag`` a core (``add_core = TAG`` anywhere in the file)."""
    out = []
    for path, _, data in files.iter_files("history/provinces", (".txt",)):
        for p in _all_pairs(parse_bytes(data)):
            if p.key.value == "add_core" and isinstance(p.value, Scalar) and p.value.value == tag:
                out.append(path)
                break
    return sorted(out)


def province_file(files: GameFiles, province_id: int | str) -> str | None:
    """history/provinces/<id> - <name>.txt (the id is the name's leading number)."""
    for path, _, _ in files.iter_files("history/provinces", (".txt",)):
        head = path.rsplit("/", 1)[-1].split("-")[0].split(" ")[0].strip()
        if head == str(province_id):
            return path
    return None


def _all_pairs(doc) -> list[Pair]:
    out = []
    for p in doc.pairs():
        out.append(p)
        if isinstance(p.value, Block):
            out += list(p.value.pairs())
    return out


def start_level(files: GameFiles, tech_group: str | None) -> int | None:
    """The tech level a tech group starts at (common/technology.txt groups -> start_level)."""
    data = files.read("common/technology.txt")
    groups = parse_bytes(data).get_block("groups") if data else None
    group = groups.get_block(tech_group) if groups is not None and tech_group else None
    level = group.get_str("start_level") if group is not None else None
    return int(level) if level and level.isdigit() else None


# --------------------------------------------------------------------------- trigger evaluation

def _and(vals: list[bool | None]) -> bool | None:
    if any(v is False for v in vals):
        return False
    return True if all(v is True for v in vals) else None


def _or(vals: list[bool | None]) -> bool | None:
    if any(v is True for v in vals):
        return True
    return False if vals and all(v is False for v in vals) else None


def _eval(entries: list, f: TagFacts) -> list[bool | None]:
    out = []
    for e in entries:
        if not isinstance(e, Pair) or e.value is None:
            out.append(None)
            continue
        k, v = e.key.value.lower(), e.value
        if isinstance(v, Block):
            sub = _eval(v.entries, f)
            out.append({"or": _or(sub), "and": _and(sub), "not": None if _or(sub) is None else not _or(sub)}.get(k))
            continue
        if e.op != "=":
            out.append(None)
            continue
        val = v.value
        if k in ("tag", "was_tag"):
            out.append(val == f.tag if f.tag else None)  # no tag: a country of some culture, any tag
        elif k == "primary_culture":
            out.append(val == f.primary_culture if f.primary_culture else None)
        elif k == "culture":
            out.append(val == f.primary_culture if f.primary_culture else None)
        elif k == "culture_group":
            out.append(val == f.culture_group if f.culture_group else None)
        elif k in ("is_colonial_nation", "is_former_colonial_nation"):
            out.append(val == "no")  # no colonial nations exist at a vanilla start date
        elif k == "has_reform":
            out.append(val in f.reforms)  # at the start date; a reform adopted in play can still switch packs later
        elif k == "always":
            out.append(val == "yes")
        else:
            out.append(None)
    return out


def trigger_matches(trigger: str | None, facts: TagFacts) -> bool | None:
    """Does a sprite pack's country_trigger match this tag at the start date?"""
    if trigger is None:
        return None
    return _and(_eval(parse(trigger).entries, facts))


@dataclass
class Coverage:
    tag: str
    packs: list[tuple[str, str | None, bool | None]] = field(default_factory=list)  # (pack, dlc, match)
    graphical_culture_packs: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # vanilla/DLC TAG_* unit entities

    @property
    def clean(self) -> bool:
        """No tag/culture pack can match and no TAG_* entities exist."""
        return not self.packs and not self.entities


def coverage(idx: Index, facts: TagFacts) -> Coverage:
    cov = Coverage(facts.tag)
    for name, defs in sorted(idx.sprite_packs.items()):
        for d in defs:
            if d.get("graphical_cultures"):
                if facts.graphical_culture in d["graphical_cultures"]:
                    cov.graphical_culture_packs.append(name)
                continue  # gfx-culture packs are what the test replaces
            m = trigger_matches(d.get("country_trigger"), facts)
            if m is not False:
                cov.packs.append((name, d["dlc"], m))
    prefix = facts.tag + "_"
    cov.entities = sorted(n for n in idx.entities if n.startswith(prefix) and n[len(prefix):].split("_")[0] in UNIT_SUFFIXES)
    return cov


# --------------------------------------------------------------------------- names (Track F, F6)

_LOC_NAME = re.compile(r'^[ \t]*([A-Z][A-Z0-9]{2}):\d*[ \t]*"([^"\n]*)"', re.M)


def _loc_names(files) -> dict[str, str]:
    """TAG -> name from every ``*_l_english.yml`` under localisation/ (``replace/`` last, as the
    game lets it win)."""
    found = [(p, d) for p, _, d in files.iter_files("localisation", (".yml",), recursive=True)
             if p.lower().endswith("_l_english.yml")]
    found.sort(key=lambda pd: ("/replace/" in pd[0].lower(), pd[0].lower()))
    out: dict[str, str] = {}
    for _, data in found:
        for m in _LOC_NAME.finditer(data.decode("utf-8-sig", errors="replace")):
            out[m.group(1)] = m.group(2).strip()
    return out


def country_names(game: GameFiles, mod: GameFiles | None = None) -> dict[str, tuple[str, bool]]:
    """TAG -> (name, mod-only) for every tag in common/country_tags: the game's, then the mod's
    (its tags and localisation win). A tag without a name shows as itself."""
    tags = set(country_files(game))
    names = _loc_names(game)
    mod_only: set[str] = set()
    if mod is not None:
        mod_tags = set(country_files(mod))
        mod_only = mod_tags - tags
        tags |= mod_tags
        names.update(_loc_names(mod))
    return {t: (names.get(t) or t, t in mod_only) for t in sorted(tags)}
