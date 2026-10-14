"""Facts about vanilla tags at a start date, and which sprite packs could apply to them.

Used to pick an in-game test tag: one that exists at the start date and that no
vanilla or DLC sprite pack and no vanilla TAG_* entity touches, so whatever shows on
the map comes from the generated set.

Trigger evaluation is three-valued: True / False / None (unknown). Anything not
modelled here counts as unknown, and unknown counts as "might match".
"""
from __future__ import annotations

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


def tag_facts(files: GameFiles, date: tuple[int, int, int] = START_DATE) -> dict[str, TagFacts]:
    """Facts for every tag that owns at least one province at ``date``."""
    owners = province_owners(files, date)
    groups = culture_groups(files)
    cfiles = country_files(files)
    history = {}
    for path, _, data in files.iter_files("history/countries", (".txt",)):
        tag = path.rsplit("/", 1)[-1][:3].upper()
        history.setdefault(tag, data)
    out = {}
    for tag, n in sorted(owners.items()):
        f = TagFacts(tag, cfiles.get(tag), provinces=n)
        if f.country_file:
            raw = files.read(f.country_file)
            if raw:
                f.graphical_culture = parse_bytes(raw).get_str("graphical_culture")
        if tag in history:
            for p in _history_entries(parse_bytes(history[tag]), date):
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
        out[tag] = f
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
            out.append(val == f.tag)
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
