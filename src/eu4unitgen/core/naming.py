"""Single source of truth for every name the tool generates (SPEC §3, CLAUDE.md rule 3).

Entity names, pdxmesh names, sprite pack names, unit keys, localisation keys and the
mod-relative paths of generated files are all built here and nowhere else. Names of
*vanilla* things the generator points at (base entities, weapons, cannons) come from
the base chain data, not from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNIT_TYPES = ("infantry", "cavalry", "artillery")
TIERS = (1, 2, 3, 4)
MESH_ROLES = ("infantry", "horse", "cannon")  # retexturable meshes per tier; the rider uses the infantry's
TEXTURE_KINDS = ("diffuse", "normal", "specular")

_TAG = re.compile(r"^[A-Z][A-Z0-9]{2}$")
_GFX = re.compile(r"^[a-z][a-z0-9_]*gfx$")
_PREFIX = re.compile(r"^[a-z][a-z0-9]*_$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LANG = re.compile(r"^[a-z_]+$")
_SET = re.compile(r"^[a-z][a-z0-9_]*$")

# Tags the engine reserves: rebels, pirates, natives, and the dynamic-tag ranges
# (letter + two digits, e.g. D00, C01, K12, T05, O00, E00, F00).
_RESERVED_TAGS = {"REB", "PIR", "NAT"}
_DYNAMIC_TAG = re.compile(r"^[CDEFKOT][0-9]{2}$")


class NamingError(ValueError):
    pass


def check_tag(tag: str) -> str:
    if not _TAG.match(tag or ""):
        raise NamingError(f"tag {tag!r} must be 3 characters: an uppercase letter then 2 uppercase letters/digits")
    if tag in _RESERVED_TAGS or _DYNAMIC_TAG.match(tag):
        raise NamingError(f"tag {tag!r} is reserved by the engine (rebels/pirates/natives/dynamic tags)")
    return tag


def check_gfx(gfx: str) -> str:
    # Every vanilla graphical culture is lowercase and ends in "gfx"; keep to that.
    if not _GFX.match(gfx or ""):
        raise NamingError(f"graphical culture {gfx!r} must be lowercase [a-z0-9_] and end in 'gfx'")
    return gfx


def check_prefix(prefix: str) -> str:
    if not _PREFIX.match(prefix or ""):
        raise NamingError(f"prefix {prefix!r} must be lowercase letters/digits followed by '_', e.g. 'hr_'")
    return prefix


def check_set_name(name: str) -> str:
    """A unit set: lowercase, used in pack, file and unit names (mac, macedonian_test).
    Names ending in 'gfx' are reserved for graphical cultures, whose files share the folders."""
    if not _SET.match(name or "") or name.endswith("gfx"):
        raise NamingError(f"set name {name!r} must be lowercase [a-z0-9_], start with a letter, not end in 'gfx'")
    return name


def check_identifier(name: str, what: str = "name") -> str:
    if not _IDENT.match(name or ""):
        raise NamingError(f"{what} {name!r} must be a script identifier [A-Za-z0-9_], not starting with a digit")
    return name


def _check_type(utype: str) -> str:
    if utype not in UNIT_TYPES:
        raise NamingError(f"unit type {utype!r} must be one of {UNIT_TYPES}")
    return utype


def _check_tier(tier: int) -> int:
    if tier not in TIERS:
        raise NamingError(f"tier {tier!r} must be one of {TIERS}")
    return tier


def _check_owner(owner: str) -> str:
    """Entity owners are a TAG (MAC_INFANTRY_1) or a graphical culture (hellenicgfx_INFANTRY_1)."""
    if _TAG.match(owner or ""):
        return check_tag(owner)
    return check_gfx(owner)


@dataclass(frozen=True)
class Naming:
    prefix: str = "hr_"

    def __post_init__(self) -> None:
        check_prefix(self.prefix)

    # ------------------------------------------------------------------ entities
    def unit_entity(self, owner: str, utype: str, tier: int) -> str:
        """Root map entity: MAC_INFANTRY_1, hellenicgfx_CAVALRY_3. Engine-mandated pattern."""
        return f"{_check_owner(owner)}_{_check_type(utype).upper()}_{_check_tier(tier)}"

    def rider_entity(self, owner: str, tier: int) -> str:
        return f"{self.unit_entity(owner, 'cavalry', tier)}_rider"

    def cannon_entity(self, owner: str, tier: int) -> str:
        return f"{self.unit_entity(owner, 'artillery', tier)}_cannon"

    def pdxmesh(self, set_name: str, role: str, tier: int) -> str:
        """Phase 2+: hr_mac_infantry_1_mesh. The _mesh suffix keeps pdxmesh and entity names apart."""
        return f"{self.prefix}{check_set_name(set_name)}_{check_identifier(role, 'role')}_{_check_tier(tier)}_mesh"

    # ------------------------------------------------------------------ textures (Phase 2)
    def texture(self, set_name: str, role: str, tier: int, kind: str) -> str:
        """hr_mac_infantry_1_diffuse.dds: a bare file name. It sits in the mod's copy of the vanilla
        mesh's folder (the engine prefixes that folder, SPEC §4); the prefix keeps it apart from
        vanilla names there."""
        if kind not in TEXTURE_KINDS:
            raise NamingError(f"texture kind {kind!r} must be one of {TEXTURE_KINDS}")
        return f"{self.prefix}{check_set_name(set_name)}_{check_identifier(role, 'role')}_{_check_tier(tier)}_{kind}.dds"

    @staticmethod
    def texture_guide_dir(set_name: str) -> str:
        """Mask-painting guides, under the output folder (not in the package): texture_guides/mac."""
        return f"texture_guides/{check_set_name(set_name)}"

    @staticmethod
    def texture_guide(tier: int, role: str, kind: str) -> str:
        if role not in MESH_ROLES or kind not in ("diffuse", "regions", "spec"):
            raise NamingError(f"texture guide {role!r}/{kind!r}")
        return f"{_check_tier(tier)}_{role}_{kind}.png"

    # ------------------------------------------------------------------ sprite packs
    def set_sprite_pack(self, set_name: str) -> str:
        """The TAG pack of a unit set: hr_mac_units, hr_macedonian_test_units."""
        return f"{self.prefix}{check_set_name(set_name)}_units"

    def gfx_sprite_pack(self, gfx: str) -> str:
        """Same pattern as vanilla westerngfx_sprite_pack."""
        return f"{check_gfx(gfx)}_sprite_pack"

    # ------------------------------------------------------------------ units + loc
    def unit(self, set_name: str, utype: str, tier: int) -> str:
        """Default unit key: hr_mac_infantry_1. Users may rename via unit_key()."""
        return f"{self.prefix}{check_set_name(set_name)}_{_check_type(utype)}_{_check_tier(tier)}"

    def unit_key(self, slug: str) -> str:
        """A user-chosen unit slug with the project prefix: 'hellenic_phalanx' -> hr_hellenic_phalanx."""
        slug = check_identifier(slug, "unit slug")
        return slug if slug.startswith(self.prefix) else f"{self.prefix}{slug}"

    @staticmethod
    def loc_name_key(unit: str) -> str:
        return check_identifier(unit, "unit")

    @staticmethod
    def loc_desc_key(unit: str) -> str:
        return f"{check_identifier(unit, 'unit')}DESCR"

    # ------------------------------------------------------------------ files (mod-relative, POSIX)
    def set_entities_file(self, set_name: str) -> str:
        return f"gfx/entities/{self.prefix}{check_set_name(set_name)}_units.asset"

    def gfx_entities_file(self, gfx: str) -> str:
        return f"gfx/entities/{self.prefix}{check_gfx(gfx)}_units.asset"

    def pdxmesh_file(self, set_name: str) -> str:
        return f"interface/assets/{self.prefix}{check_set_name(set_name)}.gfx"

    def set_pack_file(self, set_name: str) -> str:
        return f"gfx/sprite_packs/{self.prefix}{check_set_name(set_name)}_pack.txt"

    def gfx_pack_file(self, gfx: str) -> str:
        return f"gfx/sprite_packs/{self.prefix}{check_gfx(gfx)}_pack.txt"

    def pack_order_file(self, set_name: str) -> str:
        # One file per set so sets can be generated independently (any file in the folder loads).
        return f"gfx/sprite_packs_order/{self.prefix}{check_set_name(set_name)}_order.txt"

    def unit_file(self, unit: str) -> str:
        return f"common/units/{check_identifier(unit, 'unit')}.txt"

    def loc_file(self, set_name: str, language: str = "english") -> str:
        if not _LANG.match(language):
            raise NamingError(f"language {language!r}")
        return f"localisation/{self.prefix}{check_set_name(set_name)}_units_l_{language}.yml"

    @staticmethod
    def integration_file(set_name: str) -> str:
        """Hand-merge instructions for files the tool doesn't own, at the package root.
        Not loaded by the game."""
        check_set_name(set_name)
        return "INTEGRATION.md"

    @staticmethod
    def package_dir(set_name: str) -> str:
        """Drop-in package folder, relative to the output root: output/<set>/."""
        return check_set_name(set_name)

    @staticmethod
    def package_manifest(set_name: str) -> str:
        """Kept beside the package (relative to the output root), so the package folder holds
        only files meant for the mod."""
        return f"{TOOL_DIR}/{check_set_name(set_name)}.manifest.json"

    # ------------------------------------------------------------------ in-game test suite (test mod only)
    def priority_test_pack(self) -> str:
        """The rival TAG pack of the pack-priority test (SPEC §10)."""
        return f"{self.prefix}priority_test_units"

    def priority_test_pack_file(self) -> str:
        return f"gfx/sprite_packs/{self.prefix}priority_test_pack.txt"

    def priority_test_order_file(self) -> str:
        return f"gfx/sprite_packs_order/{self.prefix}priority_test_order.txt"

    def fallback_entities_file(self, tag: str) -> str:
        """TAG_* entities with no sprite pack (fallback case b)."""
        return f"gfx/entities/{self.prefix}{check_tag(tag).lower()}_fallback_units.asset"

    @staticmethod
    def test_variant_mod(variant: int) -> str:
        """eu4unitgen_test_v1, _v2: one mod per pack-priority variant; enable one at a time."""
        if variant not in (1, 2):
            raise NamingError(f"test variant {variant!r} must be 1 or 2")
        return f"{TEST_MOD_NAME}_v{variant}"

    @staticmethod
    def texture_test_mod() -> str:
        """eu4unitgen_test_textures: the meshsettings / texture-path test (SPEC §10 Phase 2)."""
        return f"{TEST_MOD_NAME}_textures"

    @staticmethod
    def mil_tech_file() -> str:
        return "common/technologies/mil.txt"

    @staticmethod
    def graphical_culture_file() -> str:
        return "common/graphicalculturetype.txt"

    @staticmethod
    def country_file(country_tags_value: str) -> str:
        """common/country_tags gives 'countries/Epirus.txt'; the file lives under common/."""
        return f"common/{country_tags_value}"


TOOL_DIR = ".eu4unitgen"  # manifest + integration notes; EU4 ignores folders it doesn't know
MANIFEST_FILE = f"{TOOL_DIR}/manifest.json"

# Throwaway in-game test mod (SPEC §10 Phase 1 in-game tests).
TEST_MOD_NAME = "eu4unitgen_test"
