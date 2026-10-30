"""Track T: a tier's cavalry played by a soldier on foot (core.loadout foot_cavalry), and the musketeer stance."""
import pytest

from eu4unitgen.core.loadout import STANCE_TYPES, Loadout, LoadoutError
from eu4unitgen.core.naming import Naming


def _entry(**kw):
    return {"geometry": "g.json", "textures": {"diffuse": "d.png", "specular": "s.png", "normal": "n.png"},
            "rights": "private", **kw}


def test_foot_cavalry_entries_name_their_stance_and_weapons():
    lo = Loadout.from_json({"schema": 1, "foot_cavalry": {
        "1": _entry(stance="polearm_twohanded", weapons={"Right_hand_node": "muslim_spear"}),
        "3": _entry(stance="musketeer", weapons={"Left_hand_node": "western_musket"}, scale=0.49)}})
    assert lo.foot_cavalry[1]["base"] == "western_generic_01_type1" and lo.foot_cavalry[3]["scale"] == 0.49
    assert lo.foot_cavalry[3]["weapons"] == {"Left_hand_node": "western_musket"} and lo.changes_anything()
    assert Loadout.from_json(lo.to_json()).foot_cavalry == lo.foot_cavalry
    with pytest.raises(LoadoutError, match="stance"):
        Loadout.from_json({"schema": 1, "foot_cavalry": {"1": _entry(stance="dance")}})
    with pytest.raises(LoadoutError, match="weapons"):
        Loadout.from_json({"schema": 1, "foot_cavalry": {"1": _entry(weapons={"head": "x"})}})


def test_the_musketeer_set_is_the_base_games_and_names_come_from_naming():
    assert STANCE_TYPES["musketeer"] == {"idle": "musketeer_02_idle_animation", "walk": "musketeer_02_moving_animation",
                                         "attack": "rifleman_attack_animation"}
    assert STANCE_TYPES["rifleman"] == {"idle": "rifleman_idle_animation", "walk": "rifleman_moving_animation",
                                        "attack": "rifleman_attack_animation"}
    n = Naming()
    assert n.fresh_foot_cavalry_pdxmesh("tracen_academy", 2) == "hr_tracen_academy_fresh_cavalry_2_mesh"
    assert n.fresh_foot_cavalry_file("tracen_academy", 2).endswith("/hr_tracen_academy_fresh_cavalry_2.mesh")


def test_an_export_breakdown_names_every_part_and_how_to_use_a_culture_pack(tmp_path):
    from eu4unitgen.export import modexport
    files = ["ta.mod", "ta/descriptor.mod", "ta/common/graphicalculturetype.txt", "ta/gfx/entities/x.asset",
             "ta/gfx/sprite_packs/p.txt", "ta/gfx/models/units/a.mesh", "ta/gfx/models/units/a_diffuse.dds",
             "ta/interface/assets/x.gfx"]
    text = modexport.breakdown(files, "ta", "culture_pack", tmp_path, gfx="tagfx", display="TA")
    assert "ta/gfx/models/  (2 files)" in text and "the unit definitions" in text
    assert "graphical_culture = tagfx" in text and "delete it first" in text
    assert "checksum" in modexport.breakdown(files, "ta", "normal", tmp_path, tag="BYZ")


def test_starter_units_and_extras_are_read_from_their_folders(tmp_path, monkeypatch):
    from eu4unitgen.core import library
    root = tmp_path / "starter"
    library.write_unit(root / "units" / "byz", {"type": "cavalry", "tiers": {"1": {"horse": "starter:h"}}})
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    us = library.starter_units(root)
    assert [u["name"] for u in us] == ["byz"] and us[0]["starter"]
    assert "mambo_import_test" not in library.STARTER and "mambo_import_test" in library.EXTRAS
    assert "byzantine_cataphract_cavalry" in library.STARTER_UNITS and "tracen_academy" in library.EXTRA_UNITS


def test_a_starter_parts_own_candidate_is_used_in_place(tmp_path):
    # the fresh-install test (Release 1.0 R5): a starter part's one candidate is its raw.glb; picking it copied the
    # file onto itself (shutil.SameFileError) and the default library never fitted on a new PC
    import json
    from eu4unitgen.ref3d import flow
    tr = tmp_path / "trellis"
    tr.mkdir()
    (tr / "raw.glb").write_bytes(b"glb")
    (tr / "candidates.json").write_text(json.dumps({"candidates": [{"glb": str(tr / "raw.glb")}]}), encoding="utf-8")
    flow.use_candidate(tmp_path, 0)
    assert (tr / "raw.glb").read_bytes() == b"glb"
