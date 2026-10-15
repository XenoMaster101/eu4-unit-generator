"""Tests for library/scanner.py.

All fixtures are synthetic: a fake install is written to tmp_path with hand-built
meshes and DDS headers. No Paradox files are used except by the `vanilla` test,
which reads the user's install in place when EU4_PATH is set.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from eu4unitgen.library import scanner
from fakes import make_dds, make_mesh
from eu4unitgen.library.scanner import (
    GameFiles, build_base_models, build_index, parse_pdx_mesh, read_dds_header, summarize_mesh,
)

COMMITTED = Path(scanner.__file__).with_name("base_models.json")


# --------------------------------------------------------------------------- fake install

INFANTRY_BONES = ["root", "L_hand", "Left_hand_node", "R_hand", "Right_hand_node"]
HORSE_BONES = ["Root", "HorseSpineTop", "Saddle_Node"]

ENTITIES = """
entity = {
	name = "westerngfx_INFANTRY_1"
	pdxmesh = "generic_01"
	state = { name = "idle"		animation = "idle" }
	state = { name = "walk"		animation = "walk" }
	#state = { name = "death"	animation = "death" }
	attach = { Right_hand_node = "halbard" }
}
entity = {
	name = "westerngfx_INFANTRY_2"
	pdxmesh = "generic_02"
	state = { name = "idle"		animation = "idle" }
	attach = { Left_hand_node = "musket" }
}
entity = {
	name = "westerngfx_CAVALRY_1_rider"
	pdxmesh = "generic_01"
	state = { name = "idle"		animation = "horse_idle" }
	attach = { Right_hand_node = "saber" }
}
entity = {
	clone = "westerngfx_CAVALRY_1_rider"
	name = "westerngfx_CAVALRY_2_rider"
	pdxmesh = "generic_02"
}
entity = {
	name = "westerngfx_CAVALRY_1"
	pdxmesh = "horse"
	state = { name = "idle"		animation = "horse_idle" }
	attach = { name = "rider" Saddle_Node = "westerngfx_CAVALRY_1_rider" }
}
entity = {
	clone = "westerngfx_CAVALRY_1"
	name = "westerngfx_CAVALRY_2"
	attach = { name = "rider" Saddle_Node = "westerngfx_CAVALRY_2_rider" }
}
entity = {
	name = "westerngfx_ARTILLERY_1"
	pdxmesh = "artillery_base"
	attach = { name = "gunman" infantry_node = "westerngfx_INFANTRY_1" cannon_node = "westerngfx_ARTILLERY_1_cannon" }
}
entity = {
	name = "westerngfx_ARTILLERY_1_cannon"
	pdxmesh = "cannon"
	state = { name = "attack" animation = "artillery_attack" event = { time = 0.0 node = "cannon_muzzle" particle = "artillery_cannon" } }
}
entity = {
	name = "TST_INFANTRY_1"
	pdxmesh = "TST_01"
	state = { name = "idle"		animation = "idle" }
	attach = { Right_hand_node = "halbard" }
}
entity = {
	name = "TST_INFANTRY_2"
	pdxmesh = "generic_02"
	attach = { Left_hand_node = "musket" }
}
entity = {
	clone = "westerngfx_CAVALRY_1_rider"
	name = "TST_CAVALRY_1_rider"
	pdxmesh = "TST_01"
}
entity = {
	clone = "westerngfx_CAVALRY_1"
	name = "TST_CAVALRY_1"
	attach = { name = "rider" Saddle_Node = "TST_CAVALRY_1_rider" }
}
entity = {
	clone = "westerngfx_ARTILLERY_1"
	name = "TST_ARTILLERY_1"
	attach = { name = "gunman" infantry_node = "TST_INFANTRY_1" cannon_node = "westerngfx_ARTILLERY_1_cannon" }
}
"""

WEAPONS = """
entity = { name = "halbard" pdxmesh = "halbard" }
entity = { name = "saber" pdxmesh = "halbard" }
entity = {
	name = "musket"
	pdxmesh = "halbard"
	state = { name = "attack" event = { time = 0.90 node = "muzzle" particle = "musket_muzzle" } }
}
"""

INFANTRY_ANIMS = """
		animation = { id = "idle"			type = "inf_idle_animation" }
		animation = { id = "walk"			type = "inf_walk_animation" }
		animation = { id = "horse_idle"		type = "rider_idle_animation" }
"""

GFX = f"""
objectTypes = {{
	pdxmesh = {{ name = "generic_01" file = "gfx/models/units/generic_01.mesh" {INFANTRY_ANIMS} scale = 0.5 }}
	pdxmesh = {{ name = "generic_02" file = "gfx/models/Units/generic_02.mesh" {INFANTRY_ANIMS} scale = 0.5 }}
	pdxmesh = {{ name = "TST_01" file = "gfx/models/Units/TST_01.mesh" {INFANTRY_ANIMS} scale = 0.5 }}
	pdxmesh = {{ name = "horse" file = "gfx/models/Units/horse.mesh"
		animation = {{ id = "horse_idle" type = "horse_idle_animation" }} scale = 0.48 }}
	pdxmesh = {{ name = "artillery_base" file = "gfx/models/Units/artillery_base.mesh" scale = 1.0 }}
	pdxmesh = {{ name = "cannon" file = "gfx/models/Units/cannon.mesh"
		animation = {{ id = "artillery_attack" type = "cannon_attack_animation" }} scale = 1.1 }}
	pdxmesh = {{ name = "halbard" file = "gfx/models/Units/halbard.mesh" scale = 0.5 }}
}}
"""

ANIMS = """
animation = { name = "inf_idle_animation" file = "inf_idle.anim" }
animation = { name = "inf_walk_animation" file = "inf_walk.anim" }
animation = { name = "rider_idle_animation" file = "rider_idle.anim" }
animation = { name = "horse_idle_animation" file = "horse_idle.anim" }
animation = { name = "cannon_attack_animation" file = "cannon_attack.anim" }
"""

PACKS_BASE = """
westerngfx_sprite_pack = {
	country_trigger = { always = no }
	graphical_cultures = { westerngfx }
	sprite_infantry = { 0 = westerngfx_INFANTRY_1 1 = westerngfx_INFANTRY_2 }
	sprite_cavalry = { 0 = westerngfx_CAVALRY_1 1 = westerngfx_CAVALRY_2 }
	sprite_artillery = { 0 = westerngfx_ARTILLERY_1 1 = westerngfx_ARTILLERY_1 }
}
"""

PACKS_TAG = """
tst_base_sprite_pack = {
	country_trigger = { tag = TST }
	sprite_infantry = { 0 = TST_INFANTRY_1 1 = TST_INFANTRY_2 }
	sprite_cavalry = { 0 = TST_CAVALRY_1 1 = westerngfx_CAVALRY_2 }
	sprite_artillery = { 0 = TST_ARTILLERY_1 1 = westerngfx_ARTILLERY_1 }
}
"""

MIL = """
technology = { year = 1350 sprite_level = 1 }
technology = { year = 1375 }
technology = { year = 1400 sprite_level = 2 }
technology = { year = 1805 #sprite_level = 5
}
"""

DLC_ENTITIES = """
entity = {
	name = "TST_INFANTRY_1"
	pdxmesh = "generic_02"
	attach = { Right_hand_node = "halbard" }
	version = 4
}
"""

DLC_PACK = """
dlc999_tst_sprite_pack = {
	country_trigger = { primary_culture = testish }
	sprite_infantry = { 0 = TST_INFANTRY_1 }
}
"""


def _write(root: Path, rel: str, data: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="latin-1")
    else:
        path.write_bytes(data)


@pytest.fixture
def fake_eu4(tmp_path: Path) -> Path:
    root = tmp_path / "eu4"
    _write(root, "eu4_branch.txt", "release_9.99\n")
    _write(root, "eu4_rev.txt", "abc123")
    _write(root, "common/graphicalculturetype.txt", "westerngfx\neasterngfx\n")
    _write(root, "common/technologies/mil.txt", MIL)
    _write(root, "gfx/entities/test_units.asset", ENTITIES)
    _write(root, "gfx/entities/weapons.asset", WEAPONS)
    _write(root, "interface/assets/test.gfx", GFX)
    _write(root, "gfx/models/Units/animation/animation.asset", ANIMS)
    for anim in ("inf_idle", "inf_walk", "rider_idle", "horse_idle"):  # cannon_attack.anim left missing
        _write(root, f"gfx/models/Units/animation/{anim}.anim", b"")
    _write(root, "gfx/sprite_packs/+00_base_sprites.txt", PACKS_BASE)
    _write(root, "gfx/sprite_packs/+01_tst_sprites.txt", PACKS_TAG)
    _write(root, "gfx/sprite_packs_order/00_sprite_pack_orders.txt", "# higher wins\ndlc999_tst_sprite_pack = 100\n")

    units = "gfx/models/Units"
    loc = {"L_attach_hand_ctrl": "L_hand", "attack_spine_ctrl": None}
    _write(root, f"{units}/generic_01.mesh", make_mesh("PdxMeshColor", "g1_diffuse.dds", "g1_normal.dds", "g1_spec.dds", INFANTRY_BONES, loc))
    _write(root, f"{units}/generic_02.mesh", make_mesh("PdxMeshColor", "g2_diffuse.dds", "g1_normal.dds", "g1_spec.dds", INFANTRY_BONES))
    _write(root, f"{units}/TST_01.mesh", make_mesh("PdxMeshStandard", "tst_diffuse.dds", "g1_normal.dds", "g1_spec.dds", INFANTRY_BONES))
    _write(root, f"{units}/horse.mesh", make_mesh("PdxMeshColor", "horse_diffuse.dds", "horse_normal.dds", "horse_spec.dds", HORSE_BONES))
    _write(root, f"{units}/artillery_base.mesh", make_mesh("PdxMeshStandard", "nospec.dds", "nonormal.dds", "nospec.dds", ["root", "infantry_node", "cannon_node"]))
    _write(root, f"{units}/cannon.mesh", make_mesh("PdxMeshStandard", "cannon_diffuse.dds", "cannon_normal.dds", "cannon_spec.dds", ["Root", "Cannon"], {"cannon_muzzle": "Cannon"}))
    _write(root, f"{units}/halbard.mesh", make_mesh("PdxMeshStandard", "weapons_diffuse.dds", "weapons_normal.dds", "weapons_spec.dds"))
    for tex in ("g1_diffuse", "g2_diffuse", "tst_diffuse", "horse_diffuse", "cannon_diffuse", "weapons_diffuse",
                "g1_normal", "horse_normal", "cannon_normal", "weapons_normal"):
        _write(root, f"{units}/{tex}.dds", make_dds(512, 512, 10, b"DXT1"))
    for tex in ("g1_spec", "horse_spec", "cannon_spec", "weapons_spec", "nospec", "nonormal"):
        _write(root, f"{units}/{tex}.dds", make_dds(512, 512, 10, b"DXT5"))

    zpath = root / "dlc" / "dlc999_test" / "dlc999.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("gfx/entities/dlc_tst_units.asset", DLC_ENTITIES)
        zf.writestr("gfx/sprite_packs/999_sprites.txt", DLC_PACK)
    return root


@pytest.fixture
def built(fake_eu4: Path) -> dict:
    files = GameFiles(fake_eu4)
    try:
        return build_base_models(files, build_index(files), "TST", "westerngfx")
    finally:
        files.close()


# --------------------------------------------------------------------------- mesh + DDS

def test_mesh_summary_reads_shader_textures_bones_locators():
    raw = make_mesh("PdxMeshColor", "d.dds", "n.dds", "s.dds", INFANTRY_BONES, {"muzzle": "joint1", "trigger": None})
    summary = summarize_mesh(parse_pdx_mesh(raw))
    assert summary["shapes"] == [{"shape": "bodyShape", "shader": "PdxMeshColor",
                                  "diffuse": "d.dds", "normal": "n.dds", "specular": "s.dds"}]
    assert summary["bones"] == INFANTRY_BONES
    assert summary["locators"] == {"muzzle": "joint1", "trigger": None}


def test_mesh_rejects_non_pdx_data():
    with pytest.raises(ValueError):
        parse_pdx_mesh(b"OBJ\0garbage")


def test_dds_header_fields():
    assert read_dds_header(make_dds(512, 256, 10, b"DXT5")) == {
        "width": 512, "height": 256, "mips": 10, "format": "DXT5", "dx10_header": False}
    dx10 = read_dds_header(make_dds(64, 64, 7, b"DX10"))
    assert dx10["dx10_header"] is True and dx10["dxgi_format"] == 71


# --------------------------------------------------------------------------- file access + index

def test_lookup_ignores_path_case(fake_eu4: Path):
    files = GameFiles(fake_eu4, include_dlc=False)
    assert files.read("GFX/Models/UNITS/generic_01.mesh") is not None
    assert files.read("gfx/models/units/nope.mesh") is None


def test_index_collects_every_layer(fake_eu4: Path):
    files = GameFiles(fake_eu4)
    try:
        idx = build_index(files)
    finally:
        files.close()
    assert idx.graphical_cultures == ["westerngfx", "easterngfx"]
    assert idx.sprite_levels == {"1": {"mil_tech": 0, "year": 1350}, "2": {"mil_tech": 2, "year": 1400}}
    assert idx.game_version == {"eu4_branch": "release_9.99", "eu4_rev": "abc123"}
    assert idx.pdxmesh("horse")["defined_in"] == "interface/assets/test.gfx"
    assert idx.animation_type("inf_idle_animation")["file"] == "gfx/models/Units/animation/inf_idle.anim"
    assert idx.pack_order["dlc999_tst_sprite_pack"][0]["order"] == 100
    musket = idx.entity("musket")
    assert musket["states"] == [{"name": "attack", "events": [{"time": 0.9, "node": "muzzle", "particle": "musket_muzzle"}]}]
    # The commented-out death state must not appear.
    assert [s["name"] for s in idx.entity("westerngfx_INFANTRY_1")["states"]] == ["idle", "walk"]


def test_dlc_definition_wins_by_version_only_when_dlc_included(fake_eu4: Path):
    files = GameFiles(fake_eu4)
    try:
        idx = build_index(files)
    finally:
        files.close()
    assert idx.entity("TST_INFANTRY_1")["pdxmesh"] == "TST_01"
    effective = idx.entity("TST_INFANTRY_1", base_only=False)
    assert (effective["dlc"], effective["version"], effective["pdxmesh"]) == ("dlc999_test", 4, "generic_02")


# --------------------------------------------------------------------------- base_models.json

def test_reference_tiers_follow_clone_chain(built: dict):
    t1 = built["reference_nation"]["tiers"]["1"]
    assert t1["infantry"] == {"entity": "TST_INFANTRY_1", "clones": None, "pdxmesh": "TST_01",
                              "weapon_node": "Right_hand_node", "weapon_entity": "halbard"}
    cav = t1["cavalry"]
    assert (cav["horse_pdxmesh"], cav["rider_pdxmesh"], cav["rider_weapon_entity"]) == ("horse", "TST_01", "saber")
    art = t1["artillery"]
    assert (art["base_pdxmesh"], art["crew_entity"], art["cannon_pdxmesh"]) == ("artillery_base", "TST_INFANTRY_1", "cannon")
    assert built["reference_nation"]["tiers"]["2"]["infantry"]["weapon_node"] == "Left_hand_node"
    assert built["reference_nation"]["sprite_pack"]["order"] is None


def test_country_colour_flag_and_recolour_bases(built: dict):
    pdx = built["pdxmeshes"]
    assert pdx["generic_01"]["uses_country_colour"] is True
    assert pdx["TST_01"]["uses_country_colour"] is False
    bases = built["recolour_bases"]["tiers"]
    assert bases["1"]["infantry"] == {"pdxmesh": "generic_01", "uses_country_colour": True}
    assert bases["2"]["cavalry_rider"]["pdxmesh"] == "generic_02"
    assert bases["1"]["artillery_cannon"] == {"pdxmesh": "cannon", "uses_country_colour": False}


def test_skeletons_grouped_and_named_after_generic_mesh(built: dict):
    skel = built["skeletons"]["generic_01"]
    assert skel["bones"] == INFANTRY_BONES
    assert skel["attach_bones"] == ["Left_hand_node", "Right_hand_node"]
    assert skel["used_by"] == ["TST_01", "generic_01", "generic_02"]
    assert built["pdxmeshes"]["halbard"]["skeleton"] is None


def test_textures_and_animations_resolved(built: dict):
    assert built["textures"]["g1_spec.dds"]["format"] == "DXT5"
    assert built["pdxmeshes"]["generic_01"]["texture_dir"] == "gfx/models/units"
    assert built["animation_types"]["inf_idle_animation"]["file_exists"] is True
    assert built["animation_types"]["cannon_attack_animation"]["file_exists"] is False


def test_dlc_override_reported(built: dict):
    (ov,) = built["reference_nation"]["dlc_overrides"]
    assert ov["pack"] == "dlc999_tst_sprite_pack" and ov["pack_order"] == 100
    assert ov["partial"] == {"infantry": [0], "cavalry": [], "artillery": []}
    assert ov["redefined_entities"] == [{"name": "TST_INFANTRY_1", "version": 4}]


def test_no_dlc_mode_skips_zips(fake_eu4: Path):
    files = GameFiles(fake_eu4, include_dlc=False)
    doc = build_base_models(files, build_index(files), "TST", "westerngfx")
    assert doc["reference_nation"]["dlc_overrides"] == []


def test_unknown_tag_raises(fake_eu4: Path):
    files = GameFiles(fake_eu4, include_dlc=False)
    with pytest.raises(LookupError):
        build_base_models(files, build_index(files), "XXX", "westerngfx")


def test_output_is_json_serialisable(built: dict):
    assert json.loads(json.dumps(built)) == built


# --------------------------------------------------------------------------- committed library file

def test_committed_base_models_shape():
    doc = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert doc["schema_version"] == scanner.SCHEMA_VERSION
    for name, pdx in doc["pdxmeshes"].items():
        assert isinstance(pdx["uses_country_colour"], bool), name
    for tier in "1234":
        base = doc["recolour_bases"]["tiers"][tier]["infantry"]
        assert base["pdxmesh"].startswith("western_generic_") and base["uses_country_colour"] is True


@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_vanilla_install_regenerates_committed_file():
    files = GameFiles(Path(os.environ["EU4_PATH"]))
    try:
        doc = build_base_models(files, build_index(files), "DAN", "westerngfx")
    finally:
        files.close()
    assert doc == json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_bind_pose_inverts_tx():
    from eu4unitgen.library.scanner import bind_pose
    # tx = inverse bind: identity rotation, translation (0, -14.5, 0) -> bone at y = 14.5
    assert bind_pose([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -14.5, 0]) == [1, 0, 0, 0, 0, 1, 0, 14.5, 0, 0, 1, 0]
    # 180 degrees about X (Right_hand_node-like): inverse of itself; position -R^T t
    pose = bind_pose([1, 0, 0, 0, -1, 0, 0, 0, -1, -6, 10, 0])
    assert [round(v, 6) + 0.0 for v in pose] == [1, 0, 0, 6, 0, -1, 0, 10, 0, 0, -1, 0]
    with pytest.raises(ValueError):
        bind_pose([0.0] * 12)
