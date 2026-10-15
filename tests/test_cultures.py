"""Culture-triggered sprite packs: a set targets tags, primary cultures, or both, in one trigger."""
from __future__ import annotations

import dataclasses

import pytest

from eu4unitgen.core.generator import generate, set_trigger
from eu4unitgen.core.naming import Naming, NamingError
from eu4unitgen.core.nation import Nation
from eu4unitgen.validate import validate

N = Naming("hr_")


def test_set_trigger_forms():
    assert set_trigger(["MAC"]) == "tag = MAC"
    assert set_trigger([], ["macedonian"]) == "primary_culture = macedonian"
    assert set_trigger(["MAC"], ["macedonian"]) == "OR = { tag = MAC primary_culture = macedonian }"
    assert set_trigger(["MAC", "EPI"], ["macedonian", "epirote"]) == (
        "OR = { tag = MAC tag = EPI primary_culture = macedonian primary_culture = epirote }")


def test_nation_cultures_and_owner():
    n = Nation([], "hellenicgfx", set_name="macedonian", cultures=["macedonian"])
    assert n.tag is None and n.tags == [] and n.name == "Macedonian"
    assert N.entity_owner(n.tags, n.set_name) == "hr_macedonian"
    assert N.unit_entity("hr_macedonian", "infantry", 1) == "hr_macedonian_INFANTRY_1"
    assert N.entity_owner(["MAC"], "mac") == "MAC"
    for kwargs, msg in (({"tag": [], "cultures": ["macedonian"]}, "needs a set name"),
                        ({"tag": [], "cultures": []}, "at least one tag or culture"),
                        ({"tag": "MAC", "cultures": ["Macedonian"]}, "culture"),
                        ({"tag": "MAC", "cultures": ["greek", "greek"]}, "listed twice"),
                        ({"tag": [], "cultures": ["greek"], "set_name": "greek_set",
                          "unit_colours": ((1, 2, 3), (4, 5, 6), (7, 8, 9))}, "per tag")):
        with pytest.raises((ValueError, NamingError), match=msg):
            Nation(gfx_culture="hellenicgfx", **kwargs)
    with pytest.raises(NamingError):
        N.unit_entity("Bad Owner", "infantry", 1)


def test_tag_and_culture_set_is_clean(project, game):
    files, idx = game
    nation = Nation("MAC", "hellenicgfx", cultures=["alphan"])
    gen = generate(project, nation, files, idx)
    pack = gen.files[N.set_pack_file("mac")].text
    assert "country_trigger = { OR = { tag = MAC primary_culture = alphan } }" in pack
    assert "primary_culture = alphan" in gen.files["INTEGRATION.md"].text
    report = validate(project, nation, gen, files, idx)
    assert not report.errors, [str(i) for i in report.errors]
    assert not [i for i in report.issues if i.rule in ("cultures", "pack_order") and i.severity != "info"], \
        [str(i) for i in report.issues]


def test_culture_only_set(project, game):
    files, idx = game
    nation = Nation([], "hellenicgfx", set_name="alphan_set", cultures=["alphan"])
    gen = generate(project, nation, files, idx)
    pack = gen.files[N.set_pack_file("alphan_set")].text
    assert "country_trigger = { primary_culture = alphan }" in pack and "0 = hr_alphan_set_INFANTRY_1" in pack
    assert 'name = "hr_alphan_set_CAVALRY_1_rider"' in gen.files[N.set_entities_file("alphan_set")].text
    integ = gen.files["INTEGRATION.md"].text
    assert "names no tag" in integ and "primary culture\nalphan" in integ
    report = validate(project, nation, gen, files, idx)
    assert not report.errors, [str(i) for i in report.errors]


def test_culture_overlap_and_unknown_culture_warnings(project, game):
    files, idx = game
    # fake DLC pack: primary_culture = testish at order 100; ours at 50 is below it
    project = dataclasses.replace(project, owned_dlc=["dlc999_test"])  # unowned DLC packs never load
    low = dataclasses.replace(project, tag_pack_order=50)
    nation = Nation("MAC", "hellenicgfx", cultures=["testish", "betan", "nosuch"])
    report = validate(low, nation, generate(low, nation, files, idx), files, idx)
    msgs = [i.message for i in report.warnings if i.rule == "pack_order"]
    assert any("dlc999_tst_sprite_pack" in m and "every testish country" in m for m in msgs), msgs
    # the culture-group pack (culture_group = other_group, betan's group) has no order entry: 0 < 50
    assert not any("dlc999_grp_sprite_pack" in m for m in msgs), msgs
    cult = [i for i in report.warnings if i.rule == "cultures"]
    assert len(cult) == 1 and "'nosuch'" in cult[0].message
    nation = Nation("MAC", "hellenicgfx", cultures=["betan"])
    report = validate(dataclasses.replace(project, tag_pack_order=0), nation,
                      generate(dataclasses.replace(project, tag_pack_order=0), nation, files, idx), files, idx)
    msgs = [i.message for i in report.warnings if i.rule == "pack_order"]
    assert any("dlc999_grp_sprite_pack" in m and "every betan country" in m for m in msgs), msgs
