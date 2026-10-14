"""Tests for the Jinja2 templates: golden output, and every script template parses cleanly."""
from __future__ import annotations

import pytest

from eu4unitgen.core.nation import STAT_KEYS, UnitSpec
from eu4unitgen.pdxscript.parser import parse
from eu4unitgen.pdxscript.render import RenderError, render
from golden import check_golden

INFANTRY = {"name": "MAC_INFANTRY_2", "pdxmesh": "western_generic_02_type1", "version": 20,
            "states": [{"name": n, "animation": n} for n in ("idle", "walk", "attack", "retreat")],
            "attaches": [{"name": None, "nodes": [("Left_hand_node", "western_musket")]}]}
RIDER = {"name": "MAC_CAVALRY_1_rider", "pdxmesh": "western_generic_01_type1", "version": 20,
         "states": [{"name": n, "animation": f"horse_{n}"} for n in ("idle", "walk", "attack", "retreat")],
         "attaches": [{"name": None, "nodes": [("Right_hand_node", "cavalry_saber")]}]}
HORSE = {"name": "MAC_CAVALRY_1", "pdxmesh": "cavalry_horse", "version": 20,
         "states": [{"name": "idle", "animation": "horse_idle"}, {"name": "walk", "animation": "horse_walk"},
                    {"name": "retreat", "animation": "horse_walk"}, {"name": "attack", "animation": "horse_attack"}],
         "attaches": [{"name": "rider", "nodes": [("Saddle_Node", "MAC_CAVALRY_1_rider")]}]}
ARTILLERY = {"name": "MAC_ARTILLERY_1", "pdxmesh": "artillery_base_mesh", "version": 20,
             "states": [{"name": "attack", "animation": "artillery_base_attack"},
                        {"name": "idle", "animation": "artillery_base_idle"},
                        {"name": "walk", "animation": "artillery_base_moving"},
                        {"name": "retreat", "animation": "artillery_base_moving"}],
             "attaches": [{"name": "gunman", "nodes": [("infantry_node", "MAC_INFANTRY_1"),
                                                       ("cannon_node", "westerngfx_ARTILLERY_1_cannon")]}]}
SPRITES = [(u, [(i, f"MAC_{u.upper()}_{i + 1}") for i in range(4)]) for u in ("infantry", "cavalry", "artillery")]


def unit(utype: str) -> UnitSpec:
    return UnitSpec("hr_mac_" + utype + "_1", utype, 1, 1, "Macedonian X", stats=dict(zip(STAT_KEYS, range(6))))


def _parses_cleanly(text: str):
    doc = parse(text)
    assert doc.warnings == []
    return doc


def test_entities_template():
    out = render("entities.asset.j2", entities=[INFANTRY, RIDER, HORSE, ARTILLERY])
    check_golden("templates/entities.asset", out)
    doc = _parses_cleanly(out)
    ents = doc.get_all("entity")
    assert [e.get_str("name") for e in ents] == ["MAC_INFANTRY_2", "MAC_CAVALRY_1_rider", "MAC_CAVALRY_1", "MAC_ARTILLERY_1"]
    assert ents[3].get_block("attach").scalar_map() == {
        "name": "gunman", "infantry_node": "MAC_INFANTRY_1", "cannon_node": "westerngfx_ARTILLERY_1_cannon"}
    assert [s.get_str("animation") for s in ents[0].get_all("state")] == ["idle", "walk", "attack", "retreat"]


def test_tag_sprite_pack_template():
    out = render("sprite_pack.txt.j2", pack="hr_mac_units", trigger="tag = MAC", graphical_cultures=None, sprites=SPRITES)
    check_golden("templates/tag_pack.txt", out)
    pack = _parses_cleanly(out).get_block("hr_mac_units")
    assert pack.get_block("sprite_cavalry").scalar_map()["3"] == "MAC_CAVALRY_4"


def test_gfx_sprite_pack_template():
    sprites = [(u, [(i, f"hellenicgfx_{u.upper()}_{i + 1}") for i in range(4)]) for u in ("infantry", "cavalry", "artillery")]
    out = render("sprite_pack.txt.j2", pack="hellenicgfx_sprite_pack", trigger="always = no",
                 graphical_cultures=["hellenicgfx"], sprites=sprites)
    check_golden("templates/gfx_pack.txt", out)
    pack = _parses_cleanly(out).get_block("hellenicgfx_sprite_pack")
    assert pack.get_block("graphical_cultures").bare() == ["hellenicgfx"]


def test_pack_order_template():
    out = render("pack_order.txt.j2", orders=[("hr_mac_units", 110)])
    check_golden("templates/pack_order.txt", out)
    assert _parses_cleanly(out).scalar_map() == {"hr_mac_units": "110"}


@pytest.mark.parametrize("utype", ["infantry", "cavalry", "artillery"])
def test_unit_template(utype):
    out = render("unit.txt.j2", unit=unit(utype), tech_group="western", stat_keys=STAT_KEYS)
    check_golden(f"templates/unit_{utype}.txt", out)
    doc = _parses_cleanly(out)
    assert doc.get_str("type") == utype
    assert (doc.get_str("unit_type") is None) == (utype == "artillery")  # vanilla artillery has no unit_type
    assert doc.get_str("maneuver") == ("1" if utype == "infantry" else "2")


def test_loc_template():
    out = render("loc.yml.j2", language="english",
                 entries=[("hr_mac_infantry_1", "Macedonian Infantry I"), ("hr_mac_infantry_1DESCR", "Pikemen.")])
    check_golden("templates/loc_l_english.yml", out)
    assert out.splitlines()[0] == "l_english:"


def test_country_snippet_template():
    out = render("country_snippet.txt.j2", gfx="hellenicgfx", colour=(120, 30, 140),
                 units=["hr_mac_infantry_1", "hr_mac_cavalry_1"])
    check_golden("templates/country_snippet.txt", out)
    doc = _parses_cleanly(out)
    assert doc.get_block("historical_units").bare() == ["hr_mac_infantry_1", "hr_mac_cavalry_1"]
    assert doc.get_block("color").bare() == ["120", "30", "140"]
    no_colour = render("country_snippet.txt.j2", gfx="hellenicgfx", colour=None, units=["x"])
    assert "color" not in no_colour and _parses_cleanly(no_colour).get_str("graphical_culture") == "hellenicgfx"


def test_integration_template():
    from eu4unitgen.core.generator import Integration
    integ = Integration("HDA", "macedonian_test", "common/countries/Haida.txt", "hellenicgfx", None,
                        ["hr_macedonian_test_infantry_1"], {1: ["hr_macedonian_test_infantry_1"]}, ["hellenicgfx"])
    lines = render("country_snippet.txt.j2", gfx="hellenicgfx", colour=None, units=integ.historical_units)
    out = render("integration.md.j2", integ=integ, country_lines=lines, mil_file="common/technologies/mil.txt",
                 gct_file="common/graphicalculturetype.txt")
    check_golden("templates/INTEGRATION.md", out)
    assert "`common/countries/Haida.txt`" in out and "\tenable = hr_macedonian_test_infantry_1" in out
    assert "`color` is left as it is." in out


@pytest.mark.parametrize("bad", ['say "hi"', "two\nlines"])
def test_quoting_refuses_unsafe_text(bad):
    with pytest.raises(RenderError):
        render("entities.asset.j2", entities=[dict(INFANTRY, name=bad)])
    with pytest.raises(RenderError):
        render("loc.yml.j2", language="english", entries=[("k", bad)])


def test_missing_context_is_an_error():
    with pytest.raises(Exception):
        render("pack_order.txt.j2")
