"""Fixes A-C: facts and country files through a mod, a test mod built on a copy of a base mod,
and the test-only 'give the tag its cores'. The base mod here is made up."""
from __future__ import annotations

from pathlib import Path

import pytest

from eu4unitgen.core.generator import generate, patch_province_owner
from eu4unitgen.core.naming import Naming
from eu4unitgen.export.pipeline import Session, SetSpec, Settings, SpecError
from eu4unitgen.library.scanner import GameFiles, LayeredFiles
from eu4unitgen.library.tags import province_cores, tag_facts
from eu4unitgen.validate import validate
from fakes import MIL

N = Naming("hr_")


def make_base_mod(root: Path) -> Path:
    """A mod adding the landless tag MAC (eastern tech group, capital 3, a core on province 2)."""
    files = {
        "descriptor.mod": 'name="Fake\'s Mod"\nsupported_version="v9.99.*"\nremote_file_id="123"\n',
        "common/country_tags/00_countries.txt": 'MAC = "countries/Macedonia.txt"\n',
        "common/countries/Macedonia.txt": "graphical_culture = easterngfx\ncolor = { 235 200 0 }\n"
                                          "historical_units = {\n\tw_medieval_infantry\n}\n",
        "history/countries/MAC = Macedonia.txt": "government = monarchy\ntechnology_group = eastern\n"
                                                 "primary_culture = alphan\ncapital = 3\n",
        "history/provinces/2 - Two.txt": "owner = AAA\ncontroller = AAA\nadd_core = AAA\nadd_core = MAC\n",
        "common/technologies/mil.txt": MIL + "\n# the mod's own copy\n",
        "common/country_colors/00_country_colors.txt": "GRE = {\n\tcolor1= { 1 2 3 }\n\tcolor2= { 4 5 6 }\n"
                                                       "\tcolor3= { 7 8 9 }\n}\n",
        "common/technology.txt": "groups = {\n\twestern = { start_level = 3 }\n\teastern = { start_level = 2 }\n}\n",
        "map/x.bmp": b"BM\x00\x01binary",
        ".git/HEAD": "ref: refs/heads/main\n",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return root


@pytest.fixture
def base_mod(tmp_path) -> Path:
    return make_base_mod(tmp_path / "Fake_Mod")


def test_layered_facts_find_a_landless_mod_tag(game, base_mod):
    files, _ = game
    mod = GameFiles(base_mod, include_dlc=False)
    try:
        layered = LayeredFiles(mod, files)
        facts = tag_facts(layered, tags=["MAC"])
        mac = facts["MAC"]
        assert mac.provinces == 0 and mac.technology_group == "eastern" and mac.primary_culture == "alphan"
        assert mac.country_file == "common/countries/Macedonia.txt" and mac.graphical_culture == "easterngfx"
        assert "AAA" in facts  # vanilla tags still there
        assert province_cores(layered, "MAC") == ["history/provinces/2 - Two.txt"]
    finally:
        mod.close()


def test_patch_province_owner():
    data = b"owner = AAA\ncontroller = AAA\nadd_core = AAA\n1500.1.1 = { owner = BBB }\n"
    out = patch_province_owner(data, "MAC").decode()
    assert out.startswith("owner = MAC\ncontroller = MAC\nadd_core = AAA\n1500.1.1 = { owner = BBB }")
    assert out.rstrip().endswith("add_core = MAC")
    bare = patch_province_owner(b"1400.1.1 = { owner = CCC }\n", "MAC").decode()
    assert "owner = MAC" in bare  # dated owner before the start date replaced


def test_generate_through_the_mod_names_its_country_file(project, game, base_mod):
    from eu4unitgen.core.nation import Nation
    files, idx = game
    mod = GameFiles(base_mod, include_dlc=False)
    try:
        gen = generate(project, Nation("MAC", "hellenicgfx"), files, idx, mod=mod)
        assert "In `common/countries/Macedonia.txt`" in gen.files["INTEGRATION.md"].text
        assert gen.files["common/technologies/mil.txt"].based_on == "mod"
    finally:
        mod.close()


@pytest.fixture
def session(fake_install, tmp_path):
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out", stage_dir=tmp_path / "stage",
                         mod_dir=tmp_path / "eu4mods"))
    yield s
    s.close()


def test_base_mod_test_mod(session, base_mod, tmp_path):
    spec = SetSpec(tags=["MAC"], gfx="hellenicgfx", set_name="macedonian",
                   unit_colours=((120, 30, 140), (210, 170, 40), (230, 225, 210)))
    assert session.nations(spec, base_mod)[0].tech_group == "eastern"  # fix A: from the mod's history
    assert session.nations(spec)[0].tech_group == "western"  # vanilla only: MAC unknown
    res = session.build_test_mod(spec, base_mod=base_mod, give_cores=True, sprite_techs={2: 4, 3: 5, 4: 6})
    (name, report), = res.reports
    assert name == "eu4unitgen_test_fake_mod"
    assert not report.errors, [str(i) for i in report.errors]
    assert sum(i.rule == "test_owners" for i in report.issues) == 2  # province 2 (core) and 3 (capital)
    stage = tmp_path / "stage"
    outer = (stage / "eu4unitgen_test_fake_mod.mod").read_text(encoding="utf-8")
    assert 'name="Fake\'s Mod - unitgen test"' in outer and "remote_file_id" not in outer
    mod = stage / "eu4unitgen_test_fake_mod"
    assert (mod / "map/x.bmp").read_bytes() == b"BM\x00\x01binary"  # copied unchanged
    assert not (mod / ".git").exists()  # never copied
    assert "Fake's Mod - unitgen test" in (mod / "descriptor.mod").read_text(encoding="utf-8")
    country = (mod / "common/countries/Macedonia.txt").read_text(encoding="utf-8")
    assert "graphical_culture = hellenicgfx" in country and "hr_macedonian_infantry_1" in country
    assert "235 200 0" in country  # the rest of the mod's file kept
    mil = (mod / "common/technologies/mil.txt").read_text(encoding="utf-8")
    assert "# the mod's own copy" in mil and "enable = hr_macedonian_infantry_1" in mil
    assert "owner = MAC" in (mod / "history/provinces/2 - Two.txt").read_text(encoding="utf-8")
    assert "owner = MAC" in (mod / "history/provinces/3 - Three.txt").read_text(encoding="utf-8")
    colours = (mod / "common/country_colors/00_country_colors.txt").read_text(encoding="utf-8")
    assert "MAC" in colours
    assert (mod / "gfx/sprite_packs/hr_macedonian_pack.txt").is_file()
    assert (base_mod / "common/countries/Macedonia.txt").read_text(encoding="utf-8").startswith(
        "graphical_culture = easterngfx")  # the base mod is only read
    with pytest.raises(SpecError, match="base mod"):
        session.build_test_mod(spec, give_cores=True)
