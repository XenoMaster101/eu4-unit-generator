"""F3: looks and starter templates (core.look): compile into a valid character spec, per-tier steps,
and merging that keeps what the Advanced editor added."""
import pytest

from eu4unitgen.core.character import CharacterSpec
from eu4unitgen.core.finishes import split
from eu4unitgen.core.look import Look, LookError, compile_look, templates

SET = {"schema": 1, "set": {"tags": ["ATH"], "gfx": "hellenicgfx", "set_name": "ath"}}


def _parts(doc, key):
    return {p["kind"]: p for p in doc["loadout"]["slots"].get(key, {}).get("parts", [])}


def _finishes(doc, tier):
    return {r: c["finish"] for r, c in split(doc["textures"]["slots"][f"{tier}/infantry"]["ops"])[1].items()}


@pytest.mark.parametrize("name", ["hoplite", "phalangite", "companion_cavalry", "peltast"])
def test_every_template_compiles_to_a_valid_spec(name):
    doc = compile_look(Look.from_template(name), SET, {t: (140, 97, 90) for t in (1, 2, 3, 4)})
    spec = CharacterSpec.from_json(doc)  # the same checks as loading a file
    assert spec.loadout["bodies"] == {str(t): "western_generic_01_type1" for t in (1, 2, 3, 4)}
    for t in (1, 2, 3, 4):
        assert "4/crew" in doc["loadout"]["slots"]  # the gun crew never carries the infantry's shield
        assert _parts(doc, f"{t}/infantry")  # every tier's infantry carries something
    assert set(templates()) == {"hoplite", "phalangite", "companion_cavalry", "peltast"}


def test_hoplite_gets_heavier_with_tech_or_stays_the_same():
    look = Look.from_template("hoplite")
    doc = compile_look(look, SET)
    assert _parts(doc, "1/infantry")["greek_helmet"]["params"] == {"style": "pilos", "crest": "none"}
    assert _parts(doc, "4/infantry")["greek_helmet"]["params"] == {"style": "corinthian", "crest": "fore_aft"}
    assert _finishes(doc, 1)["area:shins"] == "skin" and _finishes(doc, 4)["area:shins"] == "bronze"
    assert "area:upper_torso" not in _finishes(doc, 3) and _finishes(doc, 4)["area:upper_torso"] == "bronze"
    assert doc["loadout"]["derived"]["1"] == {"add": [], "remove": ["helmet"]}  # Greek helmet replaces type1's
    assert set(_parts(doc, "2/rider")) == {"xyston", "crest"} and _parts(doc, "2/crew") == {}

    look.progression = "same"
    same = compile_look(look, SET)
    assert all(_parts(same, f"{t}/infantry")["greek_helmet"]["params"]["style"] == "corinthian" for t in (1, 2, 3, 4))
    assert all(_finishes(same, t) == _finishes(same, 1) for t in (2, 3, 4))


def test_edits_all_tiers_or_one_tier():
    look = Look.from_template("hoplite")
    look.edit(("roles", "infantry", "weapon"), "sarissa", tier=3)
    look.edit(("body", "shins"), "iron")  # all tiers, even tier 1 whose step says skin
    doc = compile_look(look, SET)
    assert [("sarissa" in _parts(doc, f"{t}/infantry")) for t in (1, 2, 3, 4)] == [False, False, True, False]
    assert all(_finishes(doc, t)["area:shins"] == "iron" for t in (1, 2, 3, 4))
    look.edit(("body", "arms"), None)  # back to the base texture everywhere
    assert all("area:arms" not in _finishes(compile_look(look, SET), t) for t in (1, 2, 3, 4))
    with pytest.raises(LookError):
        look.edit(("roles", "cavalry", "shield"), "hoplon")
        look.check()


def test_compiling_keeps_what_the_advanced_editor_added():
    doc = dict(SET, loadout={"schema": 1, "slots": {"1/infantry": {"parts": [
        {"kind": "horse_ears"}, {"kind": "xiphos"}]}}, "derived": {"1": ["tail"]}},
        textures={"schema": 1, "slots": {"1/infantry": {"base": None, "ops": [
            {"op": "recolour", "region": "cloth", "colour": [1, 2, 3]},
            {"op": "recolour", "region": "mask:paint_bronze.png", "colour": [125, 85, 38], "finish": "bronze"},
            {"op": "recolour", "region": "area:feet", "colour": [1, 1, 1], "finish": "colour"}]}}})
    out = compile_look(Look.from_template("hoplite"), doc)
    assert set(_parts(out, "1/infantry")) == {"horse_ears", "dory", "hoplon", "greek_helmet"}  # xiphos replaced
    assert out["loadout"]["derived"]["1"] == {"add": ["tail"], "remove": ["helmet"]}
    f = _finishes(out, 1)
    assert "mask:paint_bronze.png" in f and "area:feet" not in f  # painting stays; the look owns area: rows
    assert out["textures"]["slots"]["1/infantry"]["ops"][0]["region"] == "cloth"
    look = Look.from_template("hoplite")
    look.colours = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    assert compile_look(look, SET)["set"]["unit_colours"] == [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    assert Look.from_json(look.to_json()) == look
