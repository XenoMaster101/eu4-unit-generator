"""F10: the army the Home and Result screens edit (core.army) and the built-in emblems."""
import pytest

from eu4unitgen.core.army import Army, Edit
from eu4unitgen.core.character import CharacterSpec
from eu4unitgen.core.emblems import EMBLEMS, draw, write
from eu4unitgen.core.look import compile_look
from eu4unitgen.core.prompt import KeywordInterpreter, Vocabulary


@pytest.fixture
def interp(tmp_path):
    return KeywordInterpreter(Vocabulary(tmp_path / "vocab.json"))


def test_edits_survive_new_words_and_words_survive_edits(interp):
    army = Army("Hoplites with red crests and long spears", tag="ATH")
    army.add(Edit("part", ("infantry", "weapon"), "xiphos", tier=2))
    army.add(Edit("finish", ("shins",), "iron"))
    army.add(Edit("name", ("infantry",), "Thureophoroi"))
    look = army.look(interp)
    assert [look.tier(t)["roles"]["infantry"]["weapon"] for t in (1, 2, 3, 4)] == ["sarissa", "xiphos", "sarissa",
                                                                                  "sarissa"]
    assert all(look.tier(t)["body"]["shins"] == "iron" for t in (1, 2, 3, 4))
    assert look.tier(1)["roles"]["infantry"]["crest_colour"] == [170, 30, 30]
    army.text = "Hoplites with blue crests and long spears, bronze greaves"  # new words: edits replayed on top
    look = army.look(interp)
    assert look.tier(3)["roles"]["infantry"]["crest_colour"] == [40, 70, 160] and look.tier(4)["body"]["shins"] == "iron"
    assert look.names["infantry"] == "Thureophoroi" and look.names["cavalry"] == "Hippeis"
    army.add(Edit("part", ("infantry", "weapon"), "dory"))  # all tiers: the tier-2 edit goes
    assert [e.kind for e in army.edits].count("part") == 1
    assert all(army.look(interp).tier(t)["roles"]["infantry"]["weapon"] == "dory" for t in (1, 2, 3, 4))


def test_colours_and_materials(interp):
    army = Army("hoplites, linen cuirass", base_colours=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    army.add(Edit("unit_colour", (0,), [200, 0, 0]))
    army.add(Edit("material_colour", ("linen",), [90, 30, 30]))
    army.add(Edit("crest_colour", (), [255, 255, 255]))
    look = army.look(interp)
    assert look.colours == [[200, 0, 0], [4, 5, 6], [7, 8, 9]] and look.crest_colour == [255, 255, 255]
    assert look.tier(4)["body"]["torso"] == ["linen", [90, 30, 30]]
    army.progression = "same"
    assert army.look(interp).progression == "same"
    doc = compile_look(army.look(interp), {"schema": 1, "set": {"tags": ["ATH"], "gfx": "hellenicgfx"}})
    CharacterSpec.from_json(doc)
    with pytest.raises(ValueError):
        army.add(Edit("wings", (), 1))


def test_emblems(tmp_path):
    for name in EMBLEMS:
        img = draw(name, (10, 20, 30))
        assert img.mode == "RGBA" and img.getpixel((0, 0))[3] == 0  # transparent around the shape
        assert (10, 20, 30, 255) in [c for _, c in img.getcolors(1 << 16)]
    path = write("star", tmp_path / "e", (1, 2, 3))
    assert path.name == "emblem_star_010203.png" and path.is_file()
    army = Army("hoplites")
    army.add(Edit("emblem", (), str(path)))
    assert army.look(KeywordInterpreter(Vocabulary(tmp_path / "v.json"))).emblem == str(path)
