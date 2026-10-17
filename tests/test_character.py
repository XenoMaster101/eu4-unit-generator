"""Character specs (SPEC Track R, data-driven characters): one JSON per character."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eu4unitgen.core.character import CharacterError, CharacterSpec, from_parts, rebase_paths

LOADOUT = {"schema": 1, "derived": {"1": {"add": ["tail"], "remove": ["helmet"]}},
           "slots": {"1/infantry": {"parts": [{"kind": "hoplon", "blazon": "art/lion.png"}]}}}
RECIPE = {"schema": 1, "slots": {"1/infantry": {"ops": [{"op": "pattern", "region": "mask:masks/stripes.png",
                                                         "image": "art/stripes.png", "mode": "multiply", "opacity": 0.5}]}}}


def test_spec_round_trip_and_set_spec(tmp_path):
    doc = {"schema": 1, "name": "hoplite", "set": {"tags": ["hda"], "gfx": "hellenicgfx", "set_name": "hoplite_test",
                                                   "unit_colours": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
           "loadout": LOADOUT, "hair": {"colour": [100, 70, 40]}, "export": {"prefix": "hr_"},
           "test": {"sprite_techs": [2, 3, 4]}}
    (tmp_path / "c.json").write_text(json.dumps(doc), encoding="utf-8")
    spec = CharacterSpec.load(tmp_path / "c.json")
    assert spec.to_json() == doc
    s = spec.set_spec()
    assert s.tags == ["HDA"] and s.set_name == "hoplite_test" and s.unit_colours == {"HDA": ((1, 2, 3), (4, 5, 6), (7, 8, 9))}
    assert s.loadout.derived == {1: {"add": ["tail"], "remove": ["helmet"]}} and s.loadout.hair == {"colour": [100, 70, 40]}
    assert s.loadout.path("art/lion.png") == tmp_path / "art/lion.png"  # relative to the spec
    assert spec.test_options()["sprite_techs"] == {2: 2, 3: 3, 4: 4}


@pytest.mark.parametrize("doc, msg", [
    ({"schema": 2, "set": {"gfx": "x"}}, "schema"),
    ({"schema": 1, "set": {"tags": ["HDA"]}}, "gfx"),
    ({"schema": 1, "set": {"gfx": "hellenicgfx"}, "colours": {}}, "unknown keys \['colours'\]"),
    ({"schema": 1, "set": {"gfx": "hellenicgfx", "tgas": ["HDA"]}}, "set: unknown keys"),
    ({"schema": 1, "set": {"gfx": "hellenicgfx"}, "test": {"tag": "HDA"}}, "test: unknown keys"),
    ({"schema": 1, "set": {"gfx": "hellenicgfx", "tags": ["HDA"], "unit_colours": [[1, 2, 3]]}}, "three"),
    ({"schema": 1, "set": {"gfx": "hellenicgfx", "tags": ["HDA"]}, "loadout": {"schema": 1, "derived": {"9": ["tail"]}}},
     "derived"),
])
def test_spec_errors(doc, msg):
    with pytest.raises(CharacterError, match=msg):
        CharacterSpec.from_json(doc).set_spec()


def test_converter_embeds_and_rebases(tmp_path):
    old = tmp_path / "old" / "sets"
    old.mkdir(parents=True)
    (old / "loadout.json").write_text(json.dumps(LOADOUT), encoding="utf-8")
    (old / "recipe.json").write_text(json.dumps(RECIPE), encoding="utf-8")
    out = tmp_path / "specs"
    out.mkdir()
    spec = from_parts(out, tags=["HDA"], gfx="hellenicgfx", set_name="t", unit_colours={"HDA": ((1, 2, 3),) * 3},
                      loadout=old / "loadout.json", recipe=old / "recipe.json", sprite_techs=[2, 3, 4], prefix="hr_")
    doc = spec.to_json()
    assert doc["set"]["unit_colours"] == [[1, 2, 3]] * 3  # one triple for every tag
    assert doc["loadout"]["slots"]["1/infantry"]["parts"][0]["blazon"] == "../old/sets/art/lion.png"
    op = doc["textures"]["slots"]["1/infantry"]["ops"][0]
    assert op["image"] == "../old/sets/art/stripes.png" and op["region"] == "mask:../old/sets/masks/stripes.png"
    assert doc["test"] == {"sprite_techs": [2, 3, 4]} and doc["export"] == {"prefix": "hr_"}
    absolute = str((tmp_path / "abs.png").resolve())
    assert rebase_paths({"blazon": absolute}, old, out) == {"blazon": absolute}  # absolute paths stay


def test_u6_unsaved_images_resolve_first_and_are_written_on_save(tmp_path):
    """U6: painted masks live in the recipe until it (or the character) is saved."""
    from PIL import Image
    from eu4unitgen.core.character import rebase_paths
    from eu4unitgen.core.recipe import SlotRecipe, TextureRecipe, apply_slot

    recipe = TextureRecipe(root=tmp_path / "old")
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (0, 0, 4, 8))
    recipe.images["paint_bronze.png"] = mask
    slot = SlotRecipe(ops=[{"op": "recolour", "region": "mask:paint_bronze.png", "colour": [200, 100, 0],
                            "detail": 0.0, "protect": False}])
    res = apply_slot(recipe, slot, Image.new("RGBA", (8, 8), (50, 50, 50, 255)), None, "PdxMeshStandard")
    assert res.diffuse.getpixel((1, 1))[:3] == (200, 100, 0) and res.diffuse.getpixel((6, 1))[:3] == (50, 50, 50)

    recipe.slots[(1, "infantry")] = slot
    doc = rebase_paths(recipe.to_json(), tmp_path / "old", tmp_path / "new", {"paint_bronze.png"})
    assert doc["slots"]["1/infantry"]["ops"][0]["region"] == "mask:paint_bronze.png"
    assert rebase_paths(recipe.to_json(), tmp_path / "old", tmp_path / "new")["slots"]["1/infantry"]["ops"][0][
        "region"] == "mask:../old/paint_bronze.png"

    (tmp_path / "new").mkdir()
    recipe.save(tmp_path / "new" / "recipe.json")
    assert not recipe.images and recipe.root == tmp_path / "new"
    assert Image.open(tmp_path / "new" / "paint_bronze.png").getpixel((1, 1)) == 255
    assert TextureRecipe.load(tmp_path / "new" / "recipe.json").image("paint_bronze.png").getpixel((6, 1)) == 0
