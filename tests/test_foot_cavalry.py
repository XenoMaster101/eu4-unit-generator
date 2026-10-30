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
    n = Naming()
    assert n.fresh_foot_cavalry_pdxmesh("tracen_academy", 2) == "hr_tracen_academy_fresh_cavalry_2_mesh"
    assert n.fresh_foot_cavalry_file("tracen_academy", 2).endswith("/hr_tracen_academy_fresh_cavalry_2.mesh")
