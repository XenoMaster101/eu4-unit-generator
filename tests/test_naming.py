"""Tests for core/naming.py: every generated name and path, plus input checks."""
from __future__ import annotations

import pytest

from eu4unitgen.core.naming import (
    MANIFEST_FILE, TEST_MOD_NAME, Naming, NamingError, check_gfx, check_prefix, check_set_name, check_tag,
)

N = Naming("hr_")


def test_entity_names_match_engine_pattern():
    assert N.unit_entity("MAC", "infantry", 1) == "MAC_INFANTRY_1"
    assert N.unit_entity("MAC", "artillery", 4) == "MAC_ARTILLERY_4"
    assert N.rider_entity("MAC", 2) == "MAC_CAVALRY_2_rider"
    assert N.unit_entity("hellenicgfx", "cavalry", 3) == "hellenicgfx_CAVALRY_3"
    assert N.rider_entity("hellenicgfx", 3) == "hellenicgfx_CAVALRY_3_rider"
    assert N.cannon_entity("hellenicgfx", 1) == "hellenicgfx_ARTILLERY_1_cannon"


def test_pdxmesh_name_is_distinct_from_entity_namespace():
    assert N.pdxmesh("mac", "infantry", 1) == "hr_mac_infantry_1_mesh"
    assert not N.pdxmesh("mac", "infantry", 1).startswith("MAC_")


def test_sprite_packs_and_units():
    assert N.set_sprite_pack("mac") == "hr_mac_units"
    assert N.set_sprite_pack("macedonian_test") == "hr_macedonian_test_units"
    assert N.gfx_sprite_pack("hellenicgfx") == "hellenicgfx_sprite_pack"
    assert N.unit("mac", "cavalry", 2) == "hr_mac_cavalry_2"
    assert N.unit("macedonian_test", "infantry", 1) == "hr_macedonian_test_infantry_1"
    assert N.unit_key("hellenic_phalanx") == "hr_hellenic_phalanx"
    assert N.unit_key("hr_hellenic_phalanx") == "hr_hellenic_phalanx"
    assert (N.loc_name_key("hr_x"), N.loc_desc_key("hr_x")) == ("hr_x", "hr_xDESCR")


def test_file_paths():
    assert N.set_entities_file("mac") == "gfx/entities/hr_mac_units.asset"
    assert N.gfx_entities_file("hellenicgfx") == "gfx/entities/hr_hellenicgfx_units.asset"
    assert N.pdxmesh_file("mac") == "interface/assets/hr_mac.gfx"
    assert N.set_pack_file("mac") == "gfx/sprite_packs/hr_mac_pack.txt"
    assert N.gfx_pack_file("hellenicgfx") == "gfx/sprite_packs/hr_hellenicgfx_pack.txt"
    assert N.pack_order_file("mac") == "gfx/sprite_packs_order/hr_mac_order.txt"
    assert N.unit_file("hr_mac_infantry_1") == "common/units/hr_mac_infantry_1.txt"
    assert N.loc_file("mac") == "localisation/hr_mac_units_l_english.yml"
    assert N.loc_file("mac", "french").endswith("_l_french.yml")
    assert N.mil_tech_file() == "common/technologies/mil.txt"
    assert N.graphical_culture_file() == "common/graphicalculturetype.txt"
    assert N.integration_file("macedonian_test") == "INTEGRATION.md"
    assert N.package_dir("macedonian_test") == "macedonian_test"
    assert N.package_manifest("macedonian_test") == ".eu4unitgen/macedonian_test.manifest.json"
    assert N.country_file("countries/Haida.txt") == "common/countries/Haida.txt"
    assert MANIFEST_FILE == ".eu4unitgen/manifest.json"
    assert TEST_MOD_NAME == "eu4unitgen_test"


def test_prefix_is_configurable():
    n = Naming("tc_")
    assert n.set_sprite_pack("mac") == "tc_mac_units"
    assert n.set_entities_file("mac") == "gfx/entities/tc_mac_units.asset"
    assert n.unit_entity("MAC", "infantry", 1) == "MAC_INFANTRY_1"  # entity names are engine-fixed


@pytest.mark.parametrize("tag", ["MAC", "A01", "HR1"])
def test_good_tags(tag):
    assert check_tag(tag) == tag


@pytest.mark.parametrize("tag", ["mac", "MA", "MACE", "1AB", "M-C", "", "REB", "PIR", "NAT", "D00", "C12", "K05", "T99"])
def test_bad_tags(tag):
    with pytest.raises(NamingError):
        check_tag(tag)


@pytest.mark.parametrize("gfx, ok", [("hellenicgfx", True), ("hr_greekgfx", True), ("Hellenicgfx", False),
                                     ("hellenic", False), ("gfx", False), ("hellenic gfx", False)])
def test_gfx_names(gfx, ok):
    if ok:
        assert check_gfx(gfx) == gfx
    else:
        with pytest.raises(NamingError):
            check_gfx(gfx)


@pytest.mark.parametrize("prefix", ["hr", "HR_", "_", "h r_", "1x_"])
def test_bad_prefix(prefix):
    with pytest.raises(NamingError):
        check_prefix(prefix)


@pytest.mark.parametrize("call", [
    lambda: N.unit_entity("MAC", "navy", 1),
    lambda: N.unit_entity("MAC", "infantry", 5),
    lambda: N.unit_entity("MAC", "infantry", 0),
    lambda: N.unit_entity("mac", "infantry", 1),
    lambda: N.unit_key("2fast"),
    lambda: N.unit_key("bad-name"),
    lambda: N.loc_file("mac", "en-GB"),
    lambda: N.unit("MAC", "infantry", 1),
    lambda: N.set_sprite_pack("hellenicgfx"),
])
def test_bad_inputs_raise(call):
    with pytest.raises(NamingError):
        call()


@pytest.mark.parametrize("name, ok", [("mac", True), ("macedonian_test", True), ("m2", True), ("Mac", False),
                                      ("2mac", False), ("mac-test", False), ("", False), ("greekgfx", False)])
def test_set_names(name, ok):
    if ok:
        assert check_set_name(name) == name
    else:
        with pytest.raises(NamingError):
            check_set_name(name)


def test_naming_rejects_bad_prefix_at_construction():
    with pytest.raises(NamingError):
        Naming("HR")
