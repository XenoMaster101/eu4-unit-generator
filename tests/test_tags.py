"""Tests for library/tags.py: 1444 facts and sprite-pack trigger coverage."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.library.tags import (
    TagFacts, coverage, country_files, culture_groups, parse_date, province_owners, tag_facts, trigger_matches,
)


@pytest.fixture
def facts(game):
    files, _ = game
    return tag_facts(files)


def test_parse_date():
    assert parse_date("1444.11.11") == (1444, 11, 11)
    assert parse_date("owner") is None and parse_date("1.2") is None


def test_owners_at_start_date(game):
    files, _ = game
    assert province_owners(files) == {"TST": 1, "AAA": 1, "BBB": 1}  # CCC only before 1440 and after 1500
    assert province_owners(files, (1460, 1, 1)) == {"TST": 2, "BBB": 1}


def test_tag_facts_apply_dated_history(facts):
    assert set(facts) == {"TST", "AAA", "BBB"}
    aaa = facts["AAA"]
    assert (aaa.primary_culture, aaa.culture_group, aaa.technology_group) == ("alphan", "test_group", "western")
    assert aaa.reforms == ["feudalism_reform"] and aaa.country_file == "common/countries/Alpha.txt"
    assert aaa.graphical_culture == "westerngfx"


def test_culture_groups_skip_name_lists(game):
    groups = culture_groups(game[0])
    assert groups == {"testish": "test_group", "alphan": "test_group", "betan": "other_group"}


def test_country_files(game):
    assert country_files(game[0])["BBB"] == "common/countries/Beta.txt"


F = TagFacts("AAA", primary_culture="alphan", culture_group="test_group", reforms=["feudalism_reform"])


@pytest.mark.parametrize("trigger, expected", [
    ("tag = AAA", True),
    ("tag = BBB", False),
    ("was_tag = AAA", True),
    ("primary_culture = alphan", True),
    ("primary_culture = betan", False),
    ("culture = alphan", True),
    ("culture_group = test_group", True),
    ("OR = { tag = BBB culture_group = other_group }", False),
    ("OR = { tag = BBB culture_group = test_group }", True),
    ("culture_group = test_group NOT = { tag = AAA }", False),
    ("culture_group = test_group NOT = { tag = BBB }", True),
    ("is_colonial_nation = yes", False),
    ("OR = { is_colonial_nation = yes is_former_colonial_nation = yes }", False),
    ("has_reform = pirate_republic_reform culture_group = test_group", False),
    ("has_reform = feudalism_reform", True),
    ("always = no", False),
    ("religion = catholic", None),  # not modelled: might match
    ("OR = { religion = catholic tag = AAA }", True),
    ("AND = { religion = catholic tag = BBB }", False),
    ("num_of_cities >= 3", None),
])
def test_trigger_evaluation(trigger, expected):
    assert trigger_matches(trigger, F) is expected


def test_coverage_finds_the_clean_tag(game, facts, project):
    _, idx = game
    cov = {t: coverage(idx, f) for t, f in facts.items()}
    assert [t for t, c in cov.items() if c.clean] == ["AAA"]
    assert [p for p, _, _ in cov["TST"].packs] == ["dlc999_tst_sprite_pack", "tst_base_sprite_pack"]
    assert cov["TST"].entities == ["TST_INFANTRY_1"]
    assert [p for p, _, _ in cov["BBB"].packs] == ["dlc999_grp_sprite_pack"]
    assert cov["AAA"].graphical_culture_packs == ["westerngfx_sprite_pack"]


@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_real_install_test_tag():
    files = GameFiles(Path(os.environ["EU4_PATH"]))
    try:
        idx = build_index(files)
        facts = tag_facts(files)
    finally:
        files.close()
    assert len(facts) > 500
    clean = sorted(t for t, f in facts.items() if coverage(idx, f).clean)
    assert clean == ["HDA"]  # 1.37.5: every other 1444 tag has a matching vanilla/DLC pack
    hda = facts["HDA"]
    assert (hda.primary_culture, hda.country_file) == ("haida", "common/countries/Haida.txt")
    assert coverage(idx, facts["BYZ"]).packs and not coverage(idx, facts["BYZ"]).clean
