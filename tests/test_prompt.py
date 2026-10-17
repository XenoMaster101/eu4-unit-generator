"""F4: the offline keyword interpreter (core.prompt): synonyms, scopes, negation, leftovers, the
user's own synonyms, suggestions while typing, and editable items."""
import pytest

from eu4unitgen.core.character import CharacterSpec
from eu4unitgen.core.look import compile_look
from eu4unitgen.core.prompt import KeywordInterpreter, Vocabulary, VocabularyError


@pytest.fixture
def vocab(tmp_path):
    return Vocabulary(tmp_path / "user" / "vocabulary.json")


@pytest.fixture
def read(vocab):
    return KeywordInterpreter(vocab).interpret


@pytest.mark.parametrize("words", ["pike", "sarissa", "long spear", "long pikes"])
def test_weapon_synonyms(read, words):
    look = read(f"soldiers with {words}").look()
    assert all(look.tier(t)["roles"]["infantry"]["weapon"] == "sarissa" for t in (1, 2, 3, 4))


@pytest.mark.parametrize("words", ["red", "crimson", "scarlet", "blood-red"])
def test_colour_synonyms(read, words):
    r = read(f"{words} crests")
    assert r.look().crest_colour == [170, 30, 30] and [i.kind for i in r.items] == ["crest", "crest_colour"]


def test_a_full_description(read):
    r = read("Bronze-armoured hoplites with red horsehair crests, round shields and long spears; linen cuirass, "
             "bare arms. Cavalry with lances and white plumes.")
    assert not r.leftovers and not r.notes
    look = r.look()
    t4 = look.tier(4)
    assert t4["roles"]["infantry"]["weapon"] == "sarissa" and t4["roles"]["infantry"]["shield"] == "hoplon"
    assert t4["body"]["upper_torso"] == "bronze" and t4["body"]["torso"] == "linen" and t4["body"]["arms"] == "skin"
    assert t4["roles"]["cavalry"] == {"weapon": "xyston", "crest": {"orientation": "fore_aft"}}
    assert look.crest_colour == [235, 230, 215]  # the last crest colour named
    CharacterSpec.from_json(compile_look(look, {"schema": 1, "set": {"tags": ["ATH"], "gfx": "hellenicgfx"}}))


def test_negation_scope_notes_and_leftovers(read):
    r = read("peltasts with javelins, no shield, a red tunic; cavalry with shields and helmets, same look on all tiers")
    assert r.leftovers == ["javelins"]
    assert len(r.notes) == 2 and "riders can't carry a shield" in r.notes[0]
    look = r.look()
    assert look.template == "peltast" and look.progression == "same"
    assert "shield" not in look.tier(2)["roles"]["infantry"] and look.tier(2)["body"]["torso"] == ["colour", [170, 30, 30]]
    assert "Red tunic" in [i.text() for i in r.items]
    assert read("hoplites without greaves").look().tier(4)["body"].get("shins") is None


def test_colour_alone_is_the_main_unit_colour(read):
    look = read("blue hoplites").look()
    assert look.colours[0] == [40, 70, 160]
    assert read("hoplites with green shields").look().colours[0] == [40, 110, 50]


def test_user_synonyms_are_saved_and_win(vocab, tmp_path):
    k = KeywordInterpreter(vocab)
    assert k.interpret("doru").leftovers == ["doru"]
    vocab.add_synonym("Doru", "weapon:dory")
    assert vocab.user_file.is_file()
    again = Vocabulary(vocab.user_file)  # a new session reads it back
    look = KeywordInterpreter(again).interpret("phalangites with doru").look()
    assert look.tier(4)["roles"]["infantry"]["weapon"] == "dory"
    assert ("weapon:sarissa", "Sarissa (long pike) (weapon)") in again.meanings()
    for bad in (("", "weapon:dory"), ("x1", "weapon:dory"), ("doru", "weapon:musket")):
        with pytest.raises(VocabularyError):
            vocab.add_synonym(*bad)


def test_suggestions_while_typing(vocab):
    assert vocab.suggest("crim") == ["crimson"]
    assert vocab.suggest("bronze gr")[:2] == ["greave", "greaves"]  # a material wants a garment next
    assert "greaves" in vocab.suggest("bronze ") and vocab.suggest("") == []


def test_chips_remove_and_swap(read, vocab):
    r = read("hoplites with long spears and red crests")
    i = next(n for n, it in enumerate(r.items) if it.kind == "weapon")
    r.swap(i, next(e for e in vocab.alternatives("weapon") if e["value"] == "xiphos"))
    assert r.look().tier(3)["roles"]["infantry"]["weapon"] == "xiphos"
    c = next(n for n, it in enumerate(r.items) if it.kind == "crest_colour")
    r.swap(c, next(e for e in vocab.alternatives("colour") if e.get("id") == "blue"))
    assert r.look().crest_colour == [40, 70, 160] and r.items[c].text() == "Blue crest"
    r.remove(c)
    assert r.look().crest_colour is None
