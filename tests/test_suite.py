"""Tests for the in-game test suite: test-mod --suite (export/testmod.build_suite)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from eu4unitgen.cli import main as cli_main
from eu4unitgen.core.generator import GeneratedFile, GenerationError, patch_history_culture
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.export.testmod import build_suite
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.library.tags import culture_groups, tag_facts
from eu4unitgen.pdxscript.parser import parse_bytes
from eu4unitgen.validate import validate

# Three more 1444 tags with culture betan, which the DLC culture-group pack matches; they
# become clean once the suite gives them AAA's culture (alphan).
EXTRA = {"EEE": ("Eel", 3), "FFF": ("Fen", 2), "GGG": ("Gull", 1)}


def _w(root: Path, rel: str, text: str, append: bool = False) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="latin-1", newline="\n") as fh:
        fh.write(text)


@pytest.fixture
def suite_install(fake_install: Path) -> Path:
    prov = 10
    for tag, (name, n) in EXTRA.items():
        _w(fake_install, "common/country_tags/00_countries.txt", f'{tag} = "countries/{name}.txt"\n', append=True)
        _w(fake_install, f"common/countries/{name}.txt", "graphical_culture = westerngfx\ncolor = { 1 2 3 }\n")
        _w(fake_install, f"history/countries/{tag} - {name}.txt",
           "primary_culture = betan\ntechnology_group = western\n1300.1.1 = { primary_culture = betan }\n"
           "1600.1.1 = { primary_culture = betan }\n")
        for _ in range(n):
            _w(fake_install, f"history/provinces/{prov} - P{prov}.txt", f"owner = {tag}\n")
            prov += 1
    return fake_install


@pytest.fixture
def world(suite_install):
    files = GameFiles(suite_install)
    yield files, build_index(files), tag_facts(files), culture_groups(files)
    files.close()


@pytest.fixture
def suite(world, tmp_path):
    files, idx, facts, groups = world
    project = Project(mod_root=tmp_path / "m", eu4_path=files.root, owned_dlc=["dlc999_test"])
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test")
    return build_suite(project, nation, files, idx, tmp_path / "mods", facts, groups, rival_gfx="westerngfx",
                       test_sprite_techs={2: 2, 3: 4, 4: 5}, fallback_tags={"a": "EEE", "b": "FFF", "c": "GGG"})


def _file(tm, rel):
    return next(f for f in tm.plan.files if f.path.endswith(rel))


def test_two_variants_validate_clean(suite):
    assert [tm.plan.manifest_rel.split("/")[0] for tm in suite.variants] == ["eu4unitgen_test_v1", "eu4unitgen_test_v2"]
    for tm in suite.variants:
        assert tm.report.errors == [] and tm.report.warnings == [], "\n".join(map(str, tm.report.issues))
        assert tm.plan.can_apply


def test_fallback_tags_picked_by_size_one_per_case(world, tmp_path):
    files, idx, facts, groups = world
    project = Project(mod_root=tmp_path / "m", eu4_path=files.root)
    auto = build_suite(project, Nation("AAA", "hellenicgfx", set_name="macedonian_test"), files, idx,
                       tmp_path / "mods", facts, groups, rival_gfx="westerngfx")
    # BBB (culture-group pack) qualifies too; ties on size go by tag. TST never: tag pack + TST_*.
    assert [(c.case, c.tag) for c in auto.fixtures.cases] == [("a", "EEE"), ("b", "FFF"), ("c", "BBB")]
    assert auto.fixtures.culture == "alphan"


def test_pack_orders_swap_between_variants(suite):
    v1, v2 = suite.variants
    order = lambda tm, f: parse_bytes(_file(tm, f).data).pairs().__next__().value.value
    assert order(v1, "hr_macedonian_test_order.txt") == "110" and order(v1, "hr_priority_test_order.txt") == "111"
    assert order(v2, "hr_macedonian_test_order.txt") == "111" and order(v2, "hr_priority_test_order.txt") == "110"
    same = {f.path.split("/", 1)[1]: f.data for f in v1.plan.files if "/" in f.path and "order.txt" not in f.path
            and not f.path.endswith(("descriptor.mod", "manifest.json"))}
    other = {f.path.split("/", 1)[1]: f.data for f in v2.plan.files if f.path.split("/", 1)[-1] in same}
    assert same == other  # everything else is identical


def test_rival_pack_targets_the_main_tag(suite):
    pack = parse_bytes(_file(suite.variants[0], "hr_priority_test_pack.txt").data).get_block("hr_priority_test_units")
    assert pack.get_block("country_trigger").scalar_map() == {"tag": "AAA"}
    assert pack.get_block("sprite_cavalry").scalar_map()["3"] == "westerngfx_CAVALRY_4"


def test_case_files(suite):
    tm = suite.variants[0]
    paths = {f.path for f in tm.plan.files}
    v = "eu4unitgen_test_v1"
    assert f"{v}/gfx/entities/hr_fff_fallback_units.asset" in paths  # case b only
    assert not any("hr_eee" in p or "hr_ggg" in p for p in paths)
    assert not any("sprite_packs/hr_fff" in p for p in paths)
    gull = parse_bytes(_file(tm, "common/countries/Gull.txt").data)
    assert gull.get_str("graphical_culture") == "hellenicgfx"
    assert f"{v}/common/countries/Eel.txt" not in paths  # case a keeps its own look
    hist = _file(tm, "history/countries/EEE - Eel.txt").data
    assert hist == (b"primary_culture = alphan\ntechnology_group = western\n1300.1.1 = { primary_culture = alphan }\n"
                    b"1600.1.1 = { primary_culture = betan }\n")  # later changes left alone


def test_history_patch_needs_a_culture():
    with pytest.raises(GenerationError):
        patch_history_culture(b"technology_group = western\n", "alphan")


def test_explicit_fallback_tags_must_be_cleanable(world, tmp_path):
    files, idx, facts, groups = world
    project = Project(mod_root=tmp_path / "m", eu4_path=files.root)
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test")
    with pytest.raises(GenerationError, match="TST"):  # tag pack + TST_* entities
        build_suite(project, nation, files, idx, tmp_path / "mods", facts, groups, rival_gfx="westerngfx",
                    fallback_tags={"a": "TST", "b": "FFF", "c": "GGG"})


def test_main_tag_must_be_clean(world, tmp_path):
    files, idx, facts, groups = world
    project = Project(mod_root=tmp_path / "m", eu4_path=files.root)
    with pytest.raises(GenerationError, match="clean"):
        build_suite(project, Nation("EEE", "hellenicgfx"), files, idx, tmp_path / "mods", facts, groups,
                    rival_gfx="westerngfx")


def _revalidate(suite, world, tmp_path, rel, old, new):
    files, idx, *_ = world
    tm = suite.variants[0]
    pkg = tm.package
    f = pkg.files[rel]
    assert old in f.data, old
    pkg.files[rel] = GeneratedFile(f.path, f.data.replace(old, new), f.role, f.based_on, f.game_path)
    project = Project(mod_root=tmp_path / "m", eu4_path=files.root, owned_dlc=["dlc999_test"],
                      tag_pack_order=suite.fixtures.main_order)
    report = validate(project, suite.nation, pkg, files, idx)
    return "\n".join(i.message for i in report.issues if i.rule == "test_fixtures" and i.severity == "error")


def test_rule_catches_culture_left_matchable(suite, world, tmp_path):
    msg = _revalidate(suite, world, tmp_path, "history/countries/EEE - Eel.txt",
                      b"1300.1.1 = { primary_culture = alphan }", b"1300.1.1 = { primary_culture = betan }")
    assert "EEE" in msg and "dlc999_grp_sprite_pack" in msg


def test_rule_catches_case_c_gfx_and_rival_trigger(suite, world, tmp_path):
    msg = _revalidate(suite, world, tmp_path, "common/countries/Gull.txt", b"hellenicgfx", b"westerngfx")
    assert "Gull.txt" in msg
    msg = _revalidate(suite, world, tmp_path, "gfx/sprite_packs/hr_priority_test_pack.txt", b"tag = AAA", b"tag = BBB")
    assert "country_trigger" in msg


def test_rule_catches_orders_not_one_apart(suite, world, tmp_path):
    msg = _revalidate(suite, world, tmp_path, "gfx/sprite_packs_order/hr_priority_test_order.txt", b"111", b"115")
    assert "one step apart" in msg


def test_cli_stages_both_variants_and_prints_checklist(suite_install, tmp_path, capsys):
    stage = tmp_path / "stage"
    code = cli_main(["test-mod", "--suite", "--tag", "AAA", "--gfx", "hellenicgfx", "--set", "macedonian_test",
                     "--rival-gfx", "westerngfx", "--eu4", str(suite_install), "--mod-dir", str(tmp_path / "mods"),
                     "--stage-dir", str(stage), "--test-sprite-techs", "2", "4", "5"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (stage / "eu4unitgen_test_v1.mod").is_file() and (stage / "eu4unitgen_test_v2.mod").is_file()
    assert (stage / "eu4unitgen_test_v2" / ".eu4unitgen" / "manifest.json").is_file()
    assert "IN-GAME TEST CHECKLIST" in out and "[ ] Fb  FFF" in out and "[ ] Fc  BBB" in out and "[ ] T4  AAA" in out
    assert "can't be told apart" in out  # everything is westerngfx in the fake install
    assert not (tmp_path / "mods").exists()  # staging never touches the mod directory


@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_real_install_suite_validates_clean_with_distinct_looks(tmp_path):
    project = Project(mod_root=tmp_path / "m", owned_dlc=["dlc116_scandinavian_unit_pack"])
    files = GameFiles(project.eu4_path)
    try:
        idx = build_index(files)
        facts = tag_facts(files)
        nation = Nation("HDA", "hellenicgfx", set_name="macedonian_test", gfx_set_base="muslimgfx",
                        tech_group=facts["HDA"].technology_group)
        suite = build_suite(project, nation, files, idx, tmp_path / "mods", facts, culture_groups(files),
                            test_sprite_techs={2: 2, 3: 3, 4: 4})
    finally:
        files.close()
    for tm in suite.variants:
        assert tm.report.errors == [] and tm.report.warnings == [], "\n".join(map(str, tm.report.issues))
    assert {c.vanilla_gfx for c in suite.fixtures.cases} == {"africangfx", "indiangfx", "southamericagfx"}
    assert suite.notes == []  # every outcome has its own look
