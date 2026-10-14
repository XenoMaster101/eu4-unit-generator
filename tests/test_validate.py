"""Tests for validate/: a clean set passes, and each rule catches the fault it exists for."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from eu4unitgen.core.generator import GeneratedFile, generate
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.validate import validate
from eu4unitgen.validate.graph import Node
from eu4unitgen.validate.rules import COVERAGE, RULES, STANDALONE_RULES

ENTS = "gfx/entities/hr_mac_units.asset"


@pytest.fixture
def setup(project, nation, game):
    files, idx = game
    return project, nation, generate(project, nation, files, idx), files, idx


def run(setup, mod: Path | None = None, **kw):
    project, nation, gs, files, idx = setup
    return validate(project, nation, gs, files, idx, GameFiles(mod, include_dlc=False) if mod else None, **kw)


def edit(setup, path: str, old: str, new: str, count: int = 1) -> None:
    gs = setup[2]
    f = gs.files[path]
    text = f.data.decode("utf-8-sig" if f.data.startswith(b"\xef\xbb\xbf") else "utf-8")
    assert old in text, old
    text = text.replace(old, new, count)
    data = (b"\xef\xbb\xbf" if f.data.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
    gs.files[path] = GeneratedFile(f.path, data, f.role, f.based_on)


def rules_hit(report, severity="error") -> set[str]:
    return {i.rule for i in report.issues if i.severity == severity}


def messages(report, rule) -> str:
    return "\n".join(i.message for i in report.issues if i.rule == rule)


# --------------------------------------------------------------------------- clean + coverage

def test_clean_set_has_no_errors_or_warnings(setup):
    report = run(setup)
    assert report.errors == [] and report.warnings == [], "\n".join(map(str, report.issues))
    assert {i.rule for i in report.issues} == {"tech_enables", "generator_notes"}  # two notes


def test_graph_reaches_the_whole_chain(setup):
    g = run(setup).graph
    for node in (Node("entity", "MAC_CAVALRY_3_rider"), Node("entity", "westerngfx_CAVALRY_1_rider"),
                 Node("entity", "w_saber"), Node("pdxmesh", "horse_3"), Node("file", "gfx/models/Units/horse_3.mesh"),
                 Node("file", "gfx/models/Units/h_s.dds"), Node("anim_type", "rider_idle_animation"),
                 Node("file", "gfx/models/Units/animation/rider_idle.anim")):
        assert node in g.reached, node
    assert g.missing() == []


def test_every_generated_file_is_covered_by_rules(setup):
    for path in setup[2].files:
        matched = [rules for pat, rules in COVERAGE.items() if re.search(pat, path)]
        assert matched, f"no validator rule covers {path}"
        assert all(r in RULES or r in STANDALONE_RULES for rules in matched for r in rules)


# --------------------------------------------------------------------------- fault injection

def test_missing_attach_target(setup):
    edit(setup, ENTS, 'cannon_node = "westerngfx_ARTILLERY_1_cannon"', 'cannon_node = "nope_cannon"')
    report = run(setup)
    assert "missing_nodes" in rules_hit(report)
    assert "nope_cannon" in messages(report, "missing_nodes")


def test_bad_crew_breaks_artillery(setup):
    edit(setup, ENTS, 'infantry_node = "MAC_INFANTRY_1"', 'infantry_node = "MAC_INFANTRY_9"')
    assert "gunman" in messages(run(setup), "missing_nodes")


def test_missing_pdxmesh_and_mesh_file(setup, fake_install):
    edit(setup, ENTS, 'pdxmesh = "generic_01"', 'pdxmesh = "no_such_mesh"')
    (fake_install / "gfx/models/Units/horse_3.mesh").unlink()
    project, nation, gs, _, _ = setup
    files = GameFiles(fake_install)
    report = validate(project, nation, gs, files, build_index(files))
    text = messages(report, "missing_nodes")
    assert "no_such_mesh" in text and "horse_3.mesh" in text


def test_missing_texture_and_anim_file(setup, fake_install):
    (fake_install / "gfx/models/Units/h_s.dds").unlink()
    (fake_install / "gfx/models/Units/animation/rider_idle.anim").unlink()
    project, nation, gs, _, _ = setup
    files = GameFiles(fake_install)
    text = messages(validate(project, nation, gs, files, build_index(files)), "missing_nodes")
    assert "h_s.dds" in text and "rider_idle.anim" in text


def test_state_id_missing_from_pdxmesh(setup):
    edit(setup, ENTS, 'state = { name = "idle"    animation = "idle" }', 'state = { name = "idle"    animation = "horse_gallop" }')
    assert "horse_gallop" in messages(run(setup), "state_animations")


def test_rider_on_mesh_without_horse_ids(setup):
    edit(setup, ENTS, 'name = "MAC_CAVALRY_1_rider"\n\tpdxmesh = "generic_01"', 'name = "MAC_CAVALRY_1_rider"\n\tpdxmesh = "musket"')
    assert "rider pdxmesh 'musket' lacks" in messages(run(setup), "state_animations")


def test_crew_without_artillery_attack(setup):
    edit(setup, ENTS, 'infantry_node = "MAC_INFANTRY_1"', 'infantry_node = "westerngfx_CAVALRY_1"')
    assert "artillery_attack" in messages(run(setup), "state_animations")


def test_version_collision_with_vanilla_and_dlc(setup, project):
    project.owned_dlc = ["dlc999_test"]
    edit(setup, ENTS, 'name = "MAC_INFANTRY_1"', 'name = "TST_INFANTRY_1"')
    report = run(setup)
    text = messages(report, "version_collisions")
    assert "vanilla" in text and "dlc999_test" in text and "ours wins" in text
    edit(setup, ENTS, "version = 20", "version = 4", count=1)  # the edited entity is first in the file
    assert any(i.rule == "version_collisions" and "ties" in i.message for i in run(setup).errors)


def test_generated_twice(setup):
    project, nation, gs, files, idx = setup
    dup = gs.files["gfx/entities/hr_mac_units.asset"]
    gs.files["gfx/entities/hr_zzz_units.asset"] = GeneratedFile("gfx/entities/hr_zzz_units.asset", dup.data)
    assert "generated twice" in messages(run(setup), "version_collisions")


def test_collision_with_existing_mod_file(setup, mod_root):
    (mod_root / "gfx/entities").mkdir(parents=True)
    (mod_root / "gfx/entities/old.asset").write_text('entity = { name = "MAC_INFANTRY_1" version = 30 }\n')
    report = run(setup, mod_root)
    assert any(i.rule == "version_collisions" and "mod" in i.message for i in report.errors)


def test_mod_file_on_same_path_is_replaced_not_collided(setup, mod_root):
    (mod_root / "gfx/entities").mkdir(parents=True)
    (mod_root / ENTS).write_text('entity = { name = "MAC_INFANTRY_1" version = 99 }\n')
    assert "version_collisions" not in rules_hit(run(setup, mod_root), "error")


def test_orphan_entity(setup):
    edit(setup, ENTS, "\nentity = {", '\nentity = { name = "MAC_INFANTRY_9" pdxmesh = "generic_01" version = 20 }\nentity = {')
    report = run(setup)
    assert "MAC_INFANTRY_9 is not used" in messages(report, "orphan_entities")
    assert "unexpected generated entity name 'MAC_INFANTRY_9'" in messages(report, "naming")


def test_naming_catches_missing_entity(setup):
    edit(setup, ENTS, 'name = "MAC_ARTILLERY_4"', 'name = "MAC_ARTILLERY_X"')
    assert "MAC_ARTILLERY_4" in messages(run(setup), "naming")


def test_sprite_pack_slot_wrong(setup):
    edit(setup, "gfx/sprite_packs/hr_mac_pack.txt", "2 = MAC_CAVALRY_3", "2 = MAC_CAVALRY_2")
    assert "sprite_cavalry 2 should be MAC_CAVALRY_3" in messages(run(setup), "sprite_packs")


def test_gfx_pack_trigger_and_cultures(setup):
    edit(setup, "gfx/sprite_packs/hr_hellenicgfx_pack.txt", "always = no", "always = yes")
    edit(setup, "gfx/sprite_packs/hr_hellenicgfx_pack.txt", "{ hellenicgfx }", "{ westerngfx }")
    text = messages(run(setup), "sprite_packs")
    assert "always = no" in text and "graphical_cultures" in text


def test_pack_order_missing(setup):
    edit(setup, "gfx/sprite_packs_order/hr_mac_order.txt", "hr_mac_units = 110", "")
    assert "no sprite_packs_order entry" in messages(run(setup), "pack_order")


def test_pack_order_shadowing(setup, project):
    project.owned_dlc = ["dlc999_test"]
    edit(setup, "gfx/sprite_packs_order/hr_mac_order.txt", "hr_mac_units = 110", "hr_mac_units = 100")
    report = run(setup)
    assert "dlc999_tst_sprite_pack (order 100" in messages(report, "pack_order")
    assert "tst_base_sprite_pack" not in messages(report, "pack_order")  # tag = TST provably excludes MAC


def test_unowned_dlc_pack_is_not_loaded(setup):
    edit(setup, "gfx/sprite_packs_order/hr_mac_order.txt", "hr_mac_units = 110", "hr_mac_units = 100")
    assert "dlc999" not in messages(run(setup), "pack_order")


def test_dlc_only_reference(setup, project):
    project.owned_dlc = ["dlc999_test"]
    edit(setup, ENTS, 'infantry_node = "MAC_INFANTRY_1"', 'infantry_node = "TST_INFANTRY_1"')
    assert "dlc999_test" in messages(run(setup), "dlc_dependency")


def test_graphical_culture_not_listed(setup):
    edit(setup, "common/graphicalculturetype.txt", "hellenicgfx\n", "")
    assert "not in common/graphicalculturetype.txt" in messages(run(setup), "graphical_culture")


def test_graphical_culture_list_dropping_vanilla(setup):
    edit(setup, "common/graphicalculturetype.txt", "easterngfx\n", "")
    assert "drops vanilla cultures: easterngfx" in messages(run(setup), "graphical_culture")


def test_unit_file_faults(setup):
    edit(setup, "common/units/hr_mac_infantry_1.txt", "unit_type = western", "unit_type = martian")
    edit(setup, "common/units/hr_mac_cavalry_1.txt", "maneuver = 2", "maneuver = 1")
    edit(setup, "common/units/hr_mac_artillery_1.txt", "offensive_fire = 1", "offensive_fire = lots")
    report = run(setup)
    text = messages(report, "units")
    assert "martian" in text and "offensive_fire" in text
    assert any(i.rule == "units" and "maneuver" in i.message for i in report.warnings)


def test_missing_loc_key(setup):
    edit(setup, "localisation/hr_mac_units_l_english.yml", " hr_mac_cavalry_2DESCR:0", " hr_mac_cavalry_2DESKR:0")
    assert "hr_mac_cavalry_2DESCR" in messages(run(setup), "localisation")


def test_loc_without_bom(setup):
    gs = setup[2]
    f = gs.files["localisation/hr_mac_units_l_english.yml"]
    gs.files[f.path] = GeneratedFile(f.path, f.data[3:])
    assert "BOM" in messages(run(setup), "file_encoding")


def test_unit_never_enabled(setup):
    edit(setup, "common/technologies/mil.txt", "\tenable = hr_mac_artillery_1\n", "")
    assert "hr_mac_artillery_1 is never enabled" in messages(run(setup), "tech_enables")


def test_enable_without_unit_file(setup):
    edit(setup, "common/technologies/mil.txt", "\tenable = hr_mac_artillery_1\n", "\tenable = hr_mac_artillery_1\n\tenable = hr_ghost\n")
    assert "hr_ghost, which has no unit file" in messages(run(setup), "tech_enables")


def test_unbalanced_braces(setup):
    edit(setup, "gfx/sprite_packs/hr_mac_pack.txt", "\n}\n", "\n")
    assert "end of file inside a block" in messages(run(setup), "parse_clean")


def test_rules_subset(setup):
    report = run(setup, rules=["naming"])
    assert report.ok and report.issues == []


# --------------------------------------------------------------------------- real install

@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_mac_validates_clean_against_real_install(tmp_path):
    project = Project(mod_root=tmp_path / "mod", owned_dlc=["dlc116_scandinavian_unit_pack"])
    files = GameFiles(project.eu4_path)
    try:
        idx = build_index(files)
        nation = Nation("MAC", "hellenicgfx", name="Macedon", adjective="Macedonian")
        report = validate(project, nation, generate(project, nation, files, idx), files, idx)
    finally:
        files.close()
    assert report.errors == [] and report.warnings == [], "\n".join(map(str, report.issues))
    assert len(report.graph.edges) > 400


# --------------------------------------------------------------------------- integration, country files, mod checks

def test_integration_missing_enable(setup):
    edit(setup, "INTEGRATION.md", "\tenable = hr_mac_cavalry_3\n", "")
    assert "hr_mac_cavalry_3 missing" in messages(run(setup), "integration")


def test_integration_data_mismatch(setup):
    setup[2].integration.historical_units = setup[2].integration.historical_units[:-1]
    assert "historical_units" in messages(run(setup), "integration")


def test_country_file_rule(setup):
    gs = setup[2]
    bad = b"graphical_culture = nopegfx\nhistorical_units = {\n\thr_mac_infantry_1\n\thr_ghost\n}\n"
    gs.files["common/countries/Alpha.txt"] = GeneratedFile("common/countries/Alpha.txt", bad)
    text = messages(run(setup), "country_files")
    assert "nopegfx" in text and "hr_ghost" in text and "hr_mac_infantry_1" not in text


def test_mod_checks_are_off_by_default(project, mod_root):
    (mod_root / "gfx/entities").mkdir(parents=True)
    assert project.check_mod_collisions is False and project.mod_files() is None
    project.check_mod_collisions = True
    assert project.mod_files() is not None


def test_mod_checks_off_means_no_collision_with_mod(setup, mod_root):
    (mod_root / "gfx/entities").mkdir(parents=True)
    (mod_root / "gfx/entities/old.asset").write_text('entity = { name = "MAC_INFANTRY_1" version = 30 }\n')
    p, nation, gs, files, idx = setup
    assert validate(p, nation, gs, files, idx, p.mod_files()).ok  # off: the mod folder isn't read
    p.check_mod_collisions = True
    assert not validate(p, nation, gs, files, idx, p.mod_files()).ok


def test_sprite_levels_must_rise_with_tech(setup):
    edit(setup, "common/technologies/mil.txt", "\tsprite_level = 2\n", "")
    edit(setup, "common/technologies/mil.txt", "\tsprite_level = 4\n", "\tsprite_level = 4\n\tsprite_level = 2\n")
    assert "must rise with the tech" in messages(run(setup), "tech_enables")


def test_sprite_level_missing_or_doubled(setup):
    edit(setup, "common/technologies/mil.txt", "\tsprite_level = 3\n", "\tsprite_level = 2\n")
    assert "exactly once" in messages(run(setup), "tech_enables")
