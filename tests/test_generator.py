"""Tests for core/generator.py on the synthetic install, plus a real-install run."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from eu4unitgen.core.generator import (
    OWNED, PATCHED, SNIPPET, BaseChain, GenerationError, generate, patch_graphical_cultures, patch_mil_enables,
)
from eu4unitgen.core.nation import Nation, UnitSpec
from eu4unitgen.core.project import Project
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.pdxscript.parser import Block, parse_bytes
from golden import check_golden


@pytest.fixture
def generated(project, nation, game):
    files, idx = game
    return generate(project, nation, files, idx)


def _entities(gs, rel) -> dict[str, Block]:
    doc = parse_bytes(gs.files[rel].data)
    return {e.get_str("name"): e for e in doc.get_all("entity")}


def test_file_list_and_roles(generated):
    roles = {p: f.role for p, f in generated.files.items()}
    assert roles == {
        "gfx/entities/hr_mac_units.asset": OWNED,
        "gfx/entities/hr_hellenicgfx_units.asset": OWNED,
        "gfx/sprite_packs/hr_mac_pack.txt": OWNED,
        "gfx/sprite_packs/hr_hellenicgfx_pack.txt": OWNED,
        "gfx/sprite_packs_order/hr_mac_order.txt": OWNED,
        **{f"common/units/hr_mac_{u}_{t}.txt": OWNED for t in range(1, 5) for u in ("infantry", "cavalry", "artillery")},
        "localisation/hr_mac_units_l_english.yml": OWNED,
        "common/technologies/mil.txt": PATCHED,
        "common/graphicalculturetype.txt": PATCHED,
        "INTEGRATION.md": SNIPPET,
    }


def test_golden_output(generated):
    for rel, f in generated.files.items():
        check_golden(f"generator/MAC/{rel}", f.data)


def test_every_script_file_parses_cleanly(generated):
    for f in generated.script_files():
        if f.path.endswith((".yml", ".md")):
            continue
        assert parse_bytes(f.data).warnings == [], f.path


def test_entities_are_full_definitions_without_clones(generated):
    """Clones of vanilla entities fail in game: hr_*.asset loads before western_units.asset."""
    for rel in ("gfx/entities/hr_mac_units.asset", "gfx/entities/hr_hellenicgfx_units.asset"):
        ents = _entities(generated, rel)
        assert len(ents) == 16  # 4 tiers x (infantry, rider, horse, artillery)
        assert all(e.get_str("clone") is None and e.get_str("pdxmesh") for e in ents.values()), rel
    ents = _entities(generated, "gfx/entities/hr_mac_units.asset")
    assert ents["MAC_CAVALRY_2"].get_str("pdxmesh") == "horse"  # inherited by westerngfx_CAVALRY_2
    assert ents["MAC_CAVALRY_3"].get_str("pdxmesh") == "horse_3"  # westerngfx_CAVALRY_3's own horse
    assert ents["MAC_CAVALRY_4_rider"].get_str("pdxmesh") == "generic_04"
    for t in range(1, 5):
        assert ents[f"MAC_ARTILLERY_{t}"].get_str("pdxmesh") == "artillery_base"


def test_full_entities_restate_inherited_states_and_attaches(generated):
    ents = _entities(generated, "gfx/entities/hr_mac_units.asset")
    states = lambda e: [(s.get_str("name"), s.get_str("animation")) for s in e.get_all("state")]
    assert states(ents["MAC_CAVALRY_2_rider"]) == [("idle", "horse_idle"), ("walk", "horse_walk"),
                                                   ("attack", "horse_attack"), ("retreat", "horse_retreat")]
    assert ents["MAC_CAVALRY_2_rider"].get_block("attach").scalar_map() == {"Right_hand_node": "w_saber"}
    assert ("retreat", "horse_walk") in states(ents["MAC_CAVALRY_4"])
    assert ("attack", "artillery_base_attack") in states(ents["MAC_ARTILLERY_3"])
    assert len(ents["MAC_ARTILLERY_3"].get_all("attach")) == 1  # the gunman attach is replaced, not added


def test_attachments_wire_the_set_together(generated):
    ents = _entities(generated, "gfx/entities/hr_mac_units.asset")
    assert ents["MAC_INFANTRY_1"].get_block("attach").scalar_map() == {"Right_hand_node": "w_halbard"}
    assert ents["MAC_INFANTRY_2"].get_block("attach").scalar_map() == {"Left_hand_node": "w_musket"}
    assert ents["MAC_CAVALRY_3"].get_block("attach").scalar_map() == {"name": "rider", "Saddle_Node": "MAC_CAVALRY_3_rider"}
    assert ents["MAC_ARTILLERY_2"].get_block("attach").scalar_map() == {
        "name": "gunman", "infantry_node": "MAC_INFANTRY_2", "cannon_node": "westerngfx_ARTILLERY_2_cannon"}
    assert all(e.get_str("version") == "20" for e in ents.values())


def test_gfx_default_set_uses_its_own_infantry_as_crew(generated):
    ents = _entities(generated, "gfx/entities/hr_hellenicgfx_units.asset")
    assert ents["hellenicgfx_ARTILLERY_1"].get_block("attach").scalar_map()["infantry_node"] == "hellenicgfx_INFANTRY_1"
    assert ents["hellenicgfx_CAVALRY_1"].get_block("attach").scalar_map()["Saddle_Node"] == "hellenicgfx_CAVALRY_1_rider"


def test_infantry_states_copied_without_commented_death(generated):
    ents = _entities(generated, "gfx/entities/hr_mac_units.asset")
    assert [s.get_str("name") for s in ents["MAC_INFANTRY_1"].get_all("state")] == ["idle", "walk", "attack", "retreat"]


def test_mil_enables_land_in_the_right_techs(generated):
    doc = parse_bytes(generated.files["common/technologies/mil.txt"].data)
    techs = doc.get_all("technology")
    enabled = {i: [v.value for v in t.get_all("enable")] for i, t in enumerate(techs)}
    assert enabled[1] == ["w_medieval_infantry", "hr_mac_infantry_1", "hr_mac_cavalry_1"]
    assert enabled[2] == ["w_cannon"]
    assert enabled[7] == ["hr_mac_artillery_1"]
    assert enabled[3] == ["hr_mac_infantry_2", "hr_mac_cavalry_2", "hr_mac_artillery_2"]  # sprite level 2 tech
    assert generated.files["common/technologies/mil.txt"].based_on == "vanilla"


def test_mil_patch_keeps_everything_else_byte_identical(fake_install, generated):
    original = (fake_install / "common/technologies/mil.txt").read_bytes()
    patched = generated.files["common/technologies/mil.txt"].data
    added = [ln for ln in patched.splitlines() if ln not in original.splitlines()]
    assert all(ln.strip().startswith(b"enable = hr_mac_") for ln in added) and len(added) == 12
    assert b"\n".join(ln for ln in patched.splitlines() if ln not in added) == original.rstrip(b"\n")


def test_mil_patch_is_idempotent(generated, nation):
    once = generated.files["common/technologies/mil.txt"].data
    assert patch_mil_enables(once, nation.units) == once


def test_mil_patch_rejects_missing_tech(fake_install):
    data = (fake_install / "common/technologies/mil.txt").read_bytes()
    with pytest.raises(GenerationError):
        patch_mil_enables(data, [UnitSpec("hr_x", "infantry", 1, 99, "X")])


def test_graphical_culture_list(generated):
    assert generated.files["common/graphicalculturetype.txt"].data == b"westerngfx\neasterngfx\nhellenicgfx\n"
    assert patch_graphical_cultures(b"westerngfx\r\n", ["westerngfx", "a_gfx"]) == b"westerngfx\r\na_gfx\r\n"


def test_mod_copy_of_shared_files_wins(project, nation, game, mod_root):
    (mod_root / "common").mkdir(parents=True)
    (mod_root / "common/graphicalculturetype.txt").write_bytes(b"hellenicgfx\nromangfx\n")
    files, idx = game
    gs = generate(project, nation, files, idx, mod=GameFiles(mod_root, include_dlc=False))
    f = gs.files["common/graphicalculturetype.txt"]
    assert (f.based_on, f.data) == ("mod", b"hellenicgfx\nromangfx\n")


def test_localisation_has_bom_and_all_keys(generated):
    data = generated.files["localisation/hr_mac_units_l_english.yml"].data
    assert data.startswith(b"\xef\xbb\xbfl_english:\n")
    text = data.decode("utf-8-sig")
    for t in range(1, 5):
        for u in ("infantry", "cavalry", "artillery"):
            assert f" hr_mac_{u}_{t}:0 " in text and f" hr_mac_{u}_{t}DESCR:0 " in text


def test_new_culture_note(generated):
    assert any("hellenicgfx" in n and "UNVERIFIED" in n for n in generated.notes)


def test_existing_culture_gets_no_note(project, game):
    files, idx = game
    gs = generate(project, Nation("MAC", "easterngfx"), files, idx)
    assert gs.notes == []
    assert gs.files["common/graphicalculturetype.txt"].data == b"westerngfx\neasterngfx\n"


def test_base_culture_must_exist(game):
    _, idx = game
    with pytest.raises(GenerationError):
        BaseChain(idx, "nopegfx")


def test_project_prefix_flows_through(fake_install, mod_root, nation, game):
    files, idx = game
    gs = generate(Project(mod_root=mod_root, eu4_path=fake_install, prefix="tc_", entity_version=7, tag_pack_order=150),
                  nation, files, idx)
    assert "gfx/entities/tc_mac_units.asset" in gs.files
    assert b"tc_mac_units = 150" in gs.files["gfx/sprite_packs_order/tc_mac_order.txt"].data
    assert b"version = 7" in gs.files["gfx/entities/tc_mac_units.asset"].data


# --------------------------------------------------------------------------- real install

@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_generate_mac_against_real_install(tmp_path):
    project = Project(mod_root=tmp_path / "mod")
    files = GameFiles(project.eu4_path)
    try:
        idx = build_index(files)
        gs = generate(project, Nation("MAC", "hellenicgfx", name="Macedon", adjective="Macedonian"), files, idx)
    finally:
        files.close()
    ents = _entities(gs, "gfx/entities/hr_mac_units.asset")
    assert ents["MAC_INFANTRY_1"].get_str("pdxmesh") == "western_generic_01_type1"
    assert ents["MAC_CAVALRY_3"].get_str("pdxmesh") == "cavalry_horse_3_mesh"
    assert ents["MAC_CAVALRY_3_rider"].get_block("attach").scalar_map() == {"Right_hand_node": "cavalry_saber"}
    assert all(e.get_str("clone") is None for e in ents.values())
    assert ents["MAC_ARTILLERY_4"].get_block("attach").scalar_map()["cannon_node"] == "westerngfx_ARTILLERY_4_cannon"
    mil = parse_bytes(gs.files["common/technologies/mil.txt"].data)
    techs = mil.get_all("technology")
    assert "hr_mac_infantry_2" in [v.value for v in techs[12].get_all("enable")]
    vanilla = Path(os.environ["EU4_PATH"], "common/technologies/mil.txt").read_bytes()
    assert len(gs.files["common/technologies/mil.txt"].data.splitlines()) == len(vanilla.splitlines()) + 12


def test_package_files_leave_out_patched_game_files(generated):
    paths = {f.path for f in generated.package_files()}
    assert "INTEGRATION.md" in paths and "gfx/entities/hr_mac_units.asset" in paths
    assert "common/technologies/mil.txt" not in paths and "common/graphicalculturetype.txt" not in paths
    assert len(paths) == len(generated.files) - 2


def test_patch_sprite_levels_moves_only_levels_2_to_4(fake_install):
    from eu4unitgen.core.generator import patch_sprite_levels
    data = (fake_install / "common/technologies/mil.txt").read_bytes()
    out = patch_sprite_levels(data, {2: 1, 3: 2, 4: 4})
    techs = parse_bytes(out).get_all("technology")
    where = {v.value: i for i, t in enumerate(techs) for v in t.get_all("sprite_level")}
    assert where == {"1": 0, "2": 1, "3": 2, "4": 4}
    assert b"\tyear = 1360\n\tsprite_level = 2\n" in out  # placed after the year line
    def others(b: bytes) -> list[bytes]:
        return [ln for ln in b.splitlines() if not ln.strip().startswith(b"sprite_level = ")]
    assert others(out) == others(data)  # only sprite_level lines moved; everything else byte-identical
    with pytest.raises(GenerationError):
        patch_sprite_levels(data, {2: 99})


def test_test_sprite_techs_reach_units_and_mil(project, game):
    files, idx = game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test")
    gs = generate(project, nation, files, idx, test_sprite_techs={2: 1, 3: 2, 4: 4})
    mil = {u.key: u.mil_tech for u in nation.units}
    assert mil["hr_macedonian_test_infantry_2"] == 1 and mil["hr_macedonian_test_cavalry_4"] == 4
    assert mil["hr_macedonian_test_artillery_1"] == 1  # tier 1 all at MIL 1 in test mode
    assert any("TEST ONLY" in n for n in gs.notes)


def test_real_packages_never_move_sprite_levels(generated, fake_install):
    vanilla = (fake_install / "common/technologies/mil.txt").read_bytes()
    removed = [ln for ln in vanilla.splitlines() if ln not in generated.files["common/technologies/mil.txt"].data.splitlines()]
    assert removed == []
