"""A nation to generate units for: tag, graphical culture, base set, units."""
from __future__ import annotations

from dataclasses import dataclass, field

from eu4unitgen.core.naming import TIERS, UNIT_TYPES, Naming, check_gfx, check_identifier, check_set_name, check_tag

STAT_KEYS = ("offensive_morale", "defensive_morale", "offensive_fire", "defensive_fire",
             "offensive_shock", "defensive_shock")

# Placeholder balance values, one row per tier, in STAT_KEYS order. Set real ones per mod.
DEFAULT_PIPS = {
    "infantry": {1: (1, 1, 0, 0, 1, 1), 2: (2, 2, 1, 1, 1, 1), 3: (2, 2, 2, 2, 2, 1), 4: (3, 3, 3, 2, 2, 2)},
    "cavalry": {1: (1, 1, 0, 0, 2, 1), 2: (2, 1, 0, 1, 2, 2), 3: (2, 2, 1, 1, 3, 2), 4: (3, 2, 1, 2, 4, 2)},
    "artillery": {1: (1, 1, 1, 1, 0, 0), 2: (1, 1, 2, 2, 1, 1), 3: (2, 2, 3, 3, 1, 2), 4: (2, 2, 4, 4, 2, 2)},
}
# Vanilla: maneuver 1 for infantry, 2 for cavalry and artillery (all 329 unit files, 1.37.5).
MANEUVER = {"infantry": 1, "cavalry": 2, "artillery": 2}
# Vanilla enables the first land units at MIL 1 (MIL 0 enables none) and the first artillery at MIL 7.
FIRST_UNIT_MIL = {"infantry": 1, "cavalry": 1, "artillery": 7}
TYPE_LABEL = {"infantry": "Infantry", "cavalry": "Cavalry", "artillery": "Artillery"}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


@dataclass
class UnitSpec:
    key: str  # script key, e.g. hr_mac_infantry_1
    type: str  # infantry | cavalry | artillery
    tier: int
    mil_tech: int  # MIL tech index (0-based, as in mil.txt order) whose block gets `enable = key`
    name: str
    description: str = ""
    stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_identifier(self.key, "unit key")
        if self.type not in UNIT_TYPES:
            raise ValueError(f"unit type {self.type!r}")
        if self.tier not in TIERS:
            raise ValueError(f"tier {self.tier!r}")
        if self.mil_tech < 0:
            raise ValueError("mil_tech must be >= 0")

    @property
    def maneuver(self) -> int:
        return MANEUVER[self.type]


@dataclass
class Nation:
    tag: str
    gfx_culture: str  # the nation's graphical culture; new ones get a default set + gfx pack
    base_gfx: str = "westerngfx"  # vanilla culture whose entities the Phase 1 set copies
    name: str = ""
    adjective: str = ""
    colour: tuple[int, int, int] | None = None  # None: leave the country file's color alone
    tech_group: str | None = "western"  # unit_type on infantry/cavalry; artillery never has one
    units: list[UnitSpec] = field(default_factory=list)
    set_name: str = ""  # names the pack, files and unit keys; defaults to the tag in lowercase
    gfx_set_base: str | None = None  # vanilla culture the gfx-culture default set copies; None: base_gfx

    def __post_init__(self) -> None:
        check_tag(self.tag)
        check_gfx(self.gfx_culture)
        check_gfx(self.base_gfx)
        if self.gfx_set_base is not None:
            check_gfx(self.gfx_set_base)
        if self.tech_group is not None:
            check_identifier(self.tech_group, "tech group")
        if self.colour is not None and (len(self.colour) != 3 or not all(0 <= c <= 255 for c in self.colour)):
            raise ValueError(f"colour {self.colour!r} must be three 0-255 values")
        self.set_name = check_set_name(self.set_name or self.tag.lower())
        self.name = self.name or self.tag
        self.adjective = self.adjective or self.name


def default_units(nation: Nation, naming: Naming, sprite_mil_tech: dict[int, int],
                  first_unit_mil: dict[str, int] | None = None) -> list[UnitSpec]:
    """One unit per type per tier (12). Tier 1 unlocks where vanilla unlocks its first units;
    tiers 2-4 unlock at the MIL tech that raises the sprite level, so the new unit and the
    new model arrive together."""
    units = []
    for tier in TIERS:
        for utype in UNIT_TYPES:
            mil = (first_unit_mil or FIRST_UNIT_MIL)[utype] if tier == 1 else sprite_mil_tech[tier]
            units.append(UnitSpec(
                key=naming.unit(nation.set_name, utype, tier),
                type=utype,
                tier=tier,
                mil_tech=mil,
                name=f"{nation.adjective} {TYPE_LABEL[utype]} {ROMAN[tier]}",
                description=f"{nation.adjective} {TYPE_LABEL[utype].lower()}, tier {tier}.",
                stats=dict(zip(STAT_KEYS, DEFAULT_PIPS[utype][tier])),
            ))
    return units
