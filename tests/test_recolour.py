"""Phase 2 steps 4-6: regions, protect masks, recolour operations, recipes, generate(recipe=)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from eu4unitgen.core.generator import GenerationError, generate
from eu4unitgen.core.recipe import RecipeError, SlotRecipe, TextureRecipe, apply_slot, parse_slot
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.pdxscript.parser import parse_bytes
from eu4unitgen.textures import recolour as rc
from eu4unitgen.textures.dds import read_dds
from eu4unitgen.textures.regions import auto_regions, guide, protect_mask
from eu4unitgen.validate import validate
from fakes import make_textures_decodable, synthetic_diffuse, synthetic_spec

SKIN, GREY, PRIMARY, SECONDARY, METAL = (5, 30), (40, 20), (40, 20), (40, 50), (40, 3)


def _v(mask: Image.Image, xy) -> int:
    return mask.getpixel(xy)


# --------------------------------------------------------------------------- regions

def test_auto_regions_on_a_colour_shader_base():
    r = auto_regions(synthetic_diffuse(), synthetic_spec(), colour_shader=True)
    assert _v(r["skin"], SKIN) == 255 and _v(r["cloth"], SKIN) == 0
    assert _v(r["country1"], PRIMARY) == 255 and _v(r["country2"], SECONDARY) == 255
    assert _v(r["metal"], METAL) == 255 and _v(r["skin"], METAL) == 0
    assert _v(r["cloth"], PRIMARY) == 0 and _v(r["cloth"], (40, 1)) == 0
    assert r["skin"].size == (64, 64)


def test_auto_regions_without_country_colours():
    r = auto_regions(synthetic_diffuse(), synthetic_spec(), colour_shader=False)
    assert r["country1"].getbbox() is None
    assert _v(r["cloth"], PRIMARY) == 255  # no country regions on PdxMeshStandard


def test_protect_mask_defaults_and_override():
    r = auto_regions(synthetic_diffuse(), synthetic_spec(), True)
    p = protect_mask(r)
    assert _v(p, SKIN) == 255 and _v(p, METAL) == 0
    assert _v(protect_mask(r, metal=True), METAL) == 255
    override = Image.new("L", (16, 16), 0)
    assert protect_mask(r, override=override).getbbox() is None  # base-provided mask replaces auto
    assert guide(synthetic_diffuse(), r).size == (64, 64)


# --------------------------------------------------------------------------- operations

def test_recolour_keeps_protected_skin():
    d = synthetic_diffuse()
    r = auto_regions(d, synthetic_spec(), True)
    everything = Image.new("L", d.size, 255)
    out = rc.recolour(d, rc.effective(everything, protect_mask(r)), (0, 0, 255))
    assert out.getpixel(SKIN)[:3] == (210, 150, 120)
    b = out.getpixel(GREY)
    assert b[2] > 200 and b[0] < 20


def test_hue_shift_inside_mask_only():
    d = synthetic_diffuse()
    mask = Image.new("L", d.size, 0)
    mask.paste(255, (0, 0, 16, 64))
    out = rc.hue_shift(d, 180, mask)
    assert out.getpixel((5, 5))[:3] != (210, 150, 120) and out.getpixel((20, 5))[:3] == (210, 150, 120)


def test_pattern_spec_paint_gloss():
    d = synthetic_diffuse()
    mask = Image.new("L", d.size, 255)
    pat = Image.new("RGB", (4, 4), (255, 0, 0))
    assert rc.pattern_overlay(d, mask, pat, "normal").getpixel(GREY)[:3] == (255, 0, 0)
    assert rc.pattern_overlay(d, mask, pat, "multiply").getpixel(GREY)[:3] == (128, 0, 0)
    s = rc.spec_paint(synthetic_spec(), mask, "country3", "set")
    assert s.getpixel(SKIN)[2] == 255
    s = rc.spec_paint(s, mask, "country1", "clear")
    assert s.getpixel(PRIMARY)[0] == 0
    assert rc.gloss(synthetic_spec(), mask, rc.GLOSS_PRESETS["metal"]).getpixel(SKIN)[3] == 33
    with pytest.raises(rc.RecolourError):
        rc.pattern_overlay(d, mask, pat, "screen")


def test_dominant_colours_are_distinct_and_ranked():
    ref = Image.new("RGB", (300, 100), (150, 20, 30))
    ref.paste((210, 170, 40), (160, 0, 250, 100))
    ref.paste((25, 35, 90), (250, 0, 300, 100))
    assert rc.dominant_colours(ref, 3) == [(150, 20, 30), (210, 170, 40), (25, 35, 90)]
    assert len(rc.dominant_colours(Image.new("RGB", (8, 8), (1, 2, 3)), 2)) == 2


def test_shade_follows_the_pdxmeshcolor_formula():
    d = Image.new("RGB", (4, 4), (200, 200, 200))
    s = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert rc.shade(d, s, [(255, 0, 0), (0, 0, 0), (0, 0, 0)], "PdxMeshColor").getpixel((0, 0)) == (200, 0, 0)
    assert rc.shade(d, s, [(255, 0, 0), (0, 0, 0), (0, 0, 0)], "PdxMeshStandard").getpixel((0, 0)) == (200, 200, 200)
    half = Image.new("RGBA", (4, 4), (128, 0, 0, 0))  # s = 0.5: col * (1 - .5 + .25 * C)
    assert rc.shade(d, half, [(0, 0, 0)] * 3, "PdxMeshColor").getpixel((0, 0))[0] in (99, 100, 101)


# --------------------------------------------------------------------------- recipes

def _ref(tmp_path: Path) -> Path:
    ref = Image.new("RGB", (100, 100), (150, 20, 30))
    ref.paste((210, 170, 40), (60, 0, 100, 100))
    ref.save(tmp_path / "ref.png")
    return tmp_path / "ref.png"


def test_recipe_round_trip_and_errors(tmp_path):
    doc = {"schema": 1, "slots": {"1/infantry": {"ops": [{"op": "recolour", "region": "cloth", "colour": [1, 2, 3]}]},
                                  "2/horse": {"protect": {"skin": False, "metal": True}, "ops": []}}}
    (tmp_path / "r.json").write_text(json.dumps(doc))
    r = TextureRecipe.load(tmp_path / "r.json")
    assert r.root == tmp_path and list(r.active()) == [(1, "infantry")]
    assert r.slots[(2, "horse")].protect_metal and not r.slots[(2, "horse")].protect_skin
    assert TextureRecipe.from_json(r.to_json()).to_json() == r.to_json()
    for bad in ({"schema": 2}, {"schema": 1, "slots": {"5/infantry": {}}}, {"schema": 1, "slots": {"1/rider": {}}},
                {"schema": 1, "slots": {"1/horse": {"ops": [{"op": "melt"}]}}},
                {"schema": 1, "slots": {"1/horse": {"ops": [{"op": "recolour", "region": "hat", "colour": [1, 2, 3]}]}}},
                {"schema": 1, "slots": {"1/horse": {"ops": [{"op": "recolour", "region": "cloth", "colour": [1, 2]}]}}},
                {"schema": 1, "slots": {"1/horse": {"ops": [{"op": "gloss", "region": "metal", "preset": "wet"}]}}}):
        with pytest.raises(RecipeError):
            TextureRecipe.from_json(bad)
    assert parse_slot("4/cannon") == (4, "cannon")


def test_palette_recolours_country_regions_fixed_and_protects_skin(tmp_path):
    _ref(tmp_path)
    recipe = TextureRecipe(root=tmp_path)
    slot = SlotRecipe(ops=[{"op": "palette", "image": "ref.png", "targets": ["country1", "country2"]}])
    res = apply_slot(recipe, slot, synthetic_diffuse(), synthetic_spec(), "PdxMeshColor")
    assert res.diffuse.getpixel(SKIN)[:3] == (210, 150, 120)
    p1, p2 = res.diffuse.getpixel(PRIMARY), res.diffuse.getpixel(SECONDARY)
    assert p2[0] > 120 and p2[1] < 60  # country2 is the larger region: the most common colour (red)
    assert p1[0] > 150 and p1[1] > 120  # gold
    assert res.specular.getpixel(PRIMARY)[0] == 0 and res.specular.getpixel(SECONDARY)[1] == 0  # fixed: cleared
    assert any(n.startswith("palette:") for n in res.notes)
    keep = SlotRecipe(ops=[{"op": "palette", "image": "ref.png", "targets": ["country1"], "fixed": False}])
    assert apply_slot(recipe, keep, synthetic_diffuse(), synthetic_spec(), "PdxMeshColor").specular is None


def test_spec_paint_refused_on_standard_shader(tmp_path):
    Image.new("L", (8, 8), 255).save(tmp_path / "m.png")
    slot = SlotRecipe(ops=[{"op": "spec_paint", "channel": "country1", "mask": "m.png"}])
    with pytest.raises(RecipeError, match="ignores spec"):
        apply_slot(TextureRecipe(root=tmp_path), slot, synthetic_diffuse(), synthetic_spec(), "PdxMeshStandard")


# --------------------------------------------------------------------------- generate(recipe=)

@pytest.fixture
def tex_game(fake_install):
    make_textures_decodable(fake_install)
    files = GameFiles(fake_install)
    yield files, build_index(files)
    files.close()


def _recipe(tmp_path: Path) -> TextureRecipe:
    _ref(tmp_path)
    return TextureRecipe.from_json({"schema": 1, "slots": {
        "1/infantry": {"ops": [{"op": "palette", "image": "ref.png"}]},
        "1/horse": {"ops": [{"op": "recolour", "region": "cloth", "colour": [20, 20, 120]}]},
        "1/cannon": {"ops": [{"op": "hue_shift", "region": "cloth", "degrees": 90}]},
        "3/infantry": {"ops": [{"op": "gloss", "region": "metal", "preset": "metal"}]},
    }}, tmp_path)


def test_generate_with_a_recipe(project, nation, tex_game, tmp_path):
    files, idx = tex_game
    out = generate(project, nation, files, idx, recipe=_recipe(tmp_path))
    paths = set(out.files)
    for p in ("interface/assets/hr_mac.gfx", "gfx/models/Units/hr_mac_infantry_1_diffuse.dds",
              "gfx/models/Units/hr_mac_infantry_1_specular.dds", "gfx/models/Units/hr_mac_horse_1_diffuse.dds",
              "gfx/models/Units/hr_mac_cannon_1_diffuse.dds", "gfx/models/Units/hr_mac_infantry_3_specular.dds"):
        assert p in paths, p
    assert "gfx/models/Units/hr_mac_infantry_3_diffuse.dds" not in paths  # gloss only changes spec
    gfx = parse_bytes(out.files["interface/assets/hr_mac.gfx"].data)
    meshes = {m.get_str("name"): m.get_block("meshsettings") for m in gfx.get_block("objectTypes").get_all("pdxmesh")}
    assert meshes["hr_mac_infantry_3_mesh"].get_str("texture_diffuse") == "g3_d.dds"  # unchanged: vanilla name
    assert meshes["hr_mac_infantry_3_mesh"].get_str("texture_specular") == "hr_mac_infantry_3_specular.dds"
    ents = {e.get_str("name"): e for e in parse_bytes(out.files["gfx/entities/hr_mac_units.asset"].data).get_all("entity")}
    assert ents["MAC_INFANTRY_3"].get_str("pdxmesh") == "hr_mac_infantry_3_mesh"
    assert ents["MAC_CAVALRY_3_rider"].get_str("pdxmesh") == "hr_mac_infantry_3_mesh"
    assert ents["MAC_CAVALRY_3"].get_str("pdxmesh") == "horse_3"  # no horse slot at tier 3
    assert "hr_mac_" not in out.files["gfx/entities/hr_hellenicgfx_units.asset"].text  # defaults stay vanilla
    report = validate(project, nation, out, files, idx)
    assert report.errors == [] and report.warnings == [], "\n".join(map(str, report.issues))
    assert any(i.rule == "generator_notes" and "palette" in i.message for i in report.issues)
    infantry = read_dds(out.files["gfx/models/Units/hr_mac_infantry_1_diffuse.dds"].data)
    assert abs(infantry.getpixel(SKIN)[0] - 210) < 12  # skin protected (DXT1 tolerance)


def test_bad_recipe_slot_is_a_generation_error(project, nation, tex_game, tmp_path):
    files, idx = tex_game
    Image.new("L", (8, 8), 255).save(tmp_path / "m.png")
    recipe = TextureRecipe.from_json({"schema": 1, "slots": {
        "1/cannon": {"ops": [{"op": "spec_paint", "channel": "country1", "mask": "m.png"}]}}}, tmp_path)
    with pytest.raises(GenerationError, match="1/cannon"):
        generate(project, nation, files, idx, recipe=recipe)


def test_pipeline_takes_a_recipe_file(fake_install, tmp_path):
    make_textures_decodable(fake_install)
    path = tmp_path / "recipe.json"
    _recipe(tmp_path).save(path)
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out", mod_dir=tmp_path / "mods"))
    try:
        res = s.validate(SetSpec(["MAC"], "hellenicgfx", texture_recipe=path))
        assert res[0].report.errors == []
        written = s.generate(SetSpec(["MAC"], "hellenicgfx", texture_recipe=path))[0].written
        assert any(p.endswith("hr_mac_infantry_1_diffuse.dds") for p in written)
        (tmp_path / "bad.json").write_text("{")
        with pytest.raises(SpecError, match="can't read texture recipe"):
            s.validate(SetSpec(["MAC"], "hellenicgfx", texture_recipe=tmp_path / "bad.json"))
    finally:
        s.close()


def test_simple_slot_round_trip_keeps_unknown_ops():
    from eu4unitgen.core.recipe import SimpleSlot
    f = SimpleSlot(palette_image="r.png", palette_targets=["country1", "cloth"], cloth_colour=(1, 2, 3),
                   pattern_image="p.png", spec_masks={"country2": "m.png"}, gloss_preset="cloth",
                   extra=[{"op": "hue_shift", "region": "cloth", "degrees": 10}])
    slot = f.to_slot()
    assert [op["op"] for op in slot.ops] == ["palette", "recolour", "pattern", "spec_paint", "gloss", "hue_shift"]
    assert SimpleSlot.from_slot(slot) == f
    with pytest.raises(RecipeError):
        SimpleSlot(pattern_image="p.png", pattern_region="hat").to_slot()


def test_unit_colours_and_preview_fallback(fake_install, tmp_path):
    from eu4unitgen.library.tags import unit_colours
    from eu4unitgen.export.pipeline import DEFAULT_UNIT_COLOURS
    make_textures_decodable(fake_install)
    (fake_install / "common/country_colors").mkdir(parents=True)
    (fake_install / "common/country_colors/00_country_colors.txt").write_text(
        "# made up\nMAC = {\n\tcolor1= { 0 40 104 }\n\tcolor2= { 250 184 7 }\n\tcolor3= { 0 40 104 }\n}\n"
        "BAD = { color1 = { 1 2 } }\n")
    files = GameFiles(fake_install)
    try:
        assert unit_colours(files) == {"MAC": [(0, 40, 104), (250, 184, 7), (0, 40, 104)]}
    finally:
        files.close()
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out", mod_dir=tmp_path / "mods"))
    try:
        slot = SlotRecipe(ops=[{"op": "hue_shift", "region": "cloth", "degrees": 90}])
        mac = s.texture_preview(SetSpec(["MAC"], "hellenicgfx"), 1, "infantry", TextureRecipe(), slot)
        assert not any("no common/country_colors" in n for n in mac.notes)
        assert mac.before.getpixel((40, 20))[2] > mac.before.getpixel((40, 20))[0]  # primary region tinted navy
        aaa = s.texture_preview(SetSpec(["AAA"], "hellenicgfx"), 1, "infantry", TextureRecipe(), slot)
        assert any("no common/country_colors" in n for n in aaa.notes)
        assert DEFAULT_UNIT_COLOURS[0] == (205, 32, 26)
    finally:
        s.close()


def test_unit_colours_rule_integration_and_test_mod(project, nation, tex_game, tmp_path):
    from eu4unitgen.core.nation import Nation
    from eu4unitgen.export.testmod import build_test_mod
    files, idx = tex_game
    bare = Nation("MAC", "hellenicgfx")
    report = validate(project, bare, generate(project, bare, files, idx), files, idx)
    assert any(i.rule == "unit_colours" and i.severity == "info" and "no entry" in i.message for i in report.issues)
    out = generate(project, nation, files, idx)  # conftest MAC sets unit colours
    report = validate(project, nation, out, files, idx)
    assert not any(i.rule == "unit_colours" for i in report.issues)
    assert "\tcolor2 = { 210 170 40 }" in out.files["INTEGRATION.md"].text
    with pytest.raises(ValueError):
        Nation("MAC", "hellenicgfx", unit_colours=((1, 2, 3), (4, 5, 6)))

    (tmp_path / "mods").mkdir()
    cc = files.root / "common/country_colors"
    cc.mkdir(parents=True)
    (cc / "00_country_colors.txt").write_text("# made up\nAAA = {\n\tcolor1= { 1 2 3 }\n\tcolor2= { 4 5 6 }\n"
                                              "\tcolor3= { 7 8 9 }\n}\n")
    tagged = Nation("AAA", "hellenicgfx", set_name="macedonian_test", unit_colours=nation.unit_colours["MAC"])
    fresh = GameFiles(files.root)  # GameFiles caches folder listings; read the new file with a new one
    try:
        tm = build_test_mod(project, tagged, fresh, build_index(fresh), tmp_path / "mods")
    finally:
        fresh.close()
    patched = tm.package.files["common/country_colors/00_country_colors.txt"].text
    assert patched.replace("\r\n", "\n").startswith("# made up\nAAA = {\n\tcolor1 = { 150 20 30 }")
    assert patched.count("AAA") == 1
    from eu4unitgen.core.generator import patch_country_colours
    appended = patch_country_colours(b"X = {\n\tcolor1= { 1 2 3 }\n}", "BBB = {\n\tcolor1 = { 4 5 6 }\n}\n", "BBB")
    assert b"}\nBBB = {" in appended  # no final newline in the original: the entry still starts a line
    if "\r\n" in patched:  # a CRLF file stays CRLF throughout
        assert "\n" not in patched.replace("\r\n", "")
    assert not [i for i in tm.report.issues if i.rule == "unit_colours"]
