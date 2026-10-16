"""Derived bodies through a loadout (SPEC Track R): the tier's infantry wears its base-game body
with skinned additions appended; rule derived_body checks the result."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from eu4unitgen.blender.runner import JobResult
from eu4unitgen.core.derived_body import check_derived
from eu4unitgen.core.generator import GenerationError, generate
from eu4unitgen.core.loadout import Loadout, LoadoutError
from eu4unitgen.core.naming import Naming
from eu4unitgen.library.scanner import parse_pdx_mesh, write_pdx_mesh
from eu4unitgen.validate import validate
from test_parts import FakeRunner

N = Naming("hr_")
BONES = ["root", "lower_spine", "mid_spine", "top_spine", "head", "Left_hand_node", "Right_hand_node", "attach_back",
         "hip", "attach_free"]
LOADOUT = {"schema": 1, "derived": {"1": ["tail", "tuft"]},
           "slots": {"1/infantry": {"parts": [{"kind": "dory"}, {"kind": "hoplon"}]},
                     "1/crew": {"parts": []}}}


def skinned_body(diff="g1_d.dds") -> bytes:
    """A made-up one-shape skinned body with the generic infantry's bone names (attach_free's bind
    matrix all zeros, as in vanilla), one triangle on ``hip``."""
    ident = [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    bones = [{"name": b, "props": {"ix": [i], **({"pa": [0]} if i else {}),
                                   "tx": [0.0] * 12 if b == "attach_free" else ident}, "children": []}
             for i, b in enumerate(BONES)]
    mesh = {"name": "mesh", "props": {"p": [0.0, 8, 0, 1, 8, 0, 0, 9, 0], "n": [0.0, 0, -1] * 3,
                                      "ta": [1.0, 0, 0, 1] * 3, "u0": [0.0, 0, 1, 0, 0, 1], "tri": [0, 2, 1]},
            "children": [{"name": "aabb", "props": {"min": [0.0, 8, 0], "max": [1.0, 9, 0]}, "children": []},
                         {"name": "material", "props": {"shader": ["PdxMeshColor"], "diff": [diff], "n": ["g_n.dds"],
                                                        "spec": ["g_s.dds"]}, "children": []},
                         {"name": "skin", "props": {"bones": [4], "ix": [8, -1, -1, -1] * 3, "w": [1.0, 0, 0, 0] * 3},
                          "children": []}]}
    return write_pdx_mesh({"name": "root", "props": {"pdxasset": [1, 0]}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": "bodyShape", "props": {}, "children": [mesh, {"name": "skeleton", "props": {}, "children": bones}]}]},
        {"name": "locator", "props": {}, "children": []}]})


class AdditionsRunner(FakeRunner):
    """FakeRunner that also 'builds' skinned additions: a 2-ring strip per addition."""

    def run_job(self, job, args, timeout=None):
        if job != "build_additions":
            return super().run_job(job, args, timeout)
        self.calls.append((job, args))
        out = []
        for k, add in enumerate(args["additions"]):
            z = 1.0 + k
            out.append({"name": add["name"], "p": [0.0, 8, z, 0.3, 8, z, 0, 6, z + 1, 0.3, 6, z + 1],
                        "n": [0.0, 0, 1] * 4, "u0": [0.21, 0.06, 0.22, 0.06, 0.21, 0.07, 0.22, 0.07],
                        "tri": [0, 1, 2, 1, 3, 2], "rings": [0, 0, 1, 1]})
        return JobResult(job, 0, {"additions": out}, "fake log")


@pytest.fixture
def body_game(game, fake_install: Path):
    """The fake install with a skinned tier-1 body."""
    (fake_install / "gfx/models/Units/generic_01.mesh").write_bytes(skinned_body())
    return game


def test_loadout_derived_round_trip_and_errors():
    lo = Loadout.from_json(LOADOUT)
    assert lo.derived == {1: ["tail", "tuft"]} and lo.needs_blender() and lo.changes_anything()
    assert Loadout.from_json(lo.to_json()).to_json() == lo.to_json()
    for bad in ({"1": ["wings"]}, {"5": ["tail"]}, {"1": []}, {"1": ["tail", "tail"]}):
        with pytest.raises(LoadoutError, match="derived"):
            Loadout.from_json({"schema": 1, "derived": bad})


def test_generate_with_a_derived_body(project, nation, body_game):
    files, idx = body_game
    runner = AdditionsRunner()
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=runner)
    file = "gfx/models/Units/hr_mac_body_1.mesh"
    f = gen.files[file]
    assert f.derived_from == "gfx/models/Units/generic_01.mesh" and not f.procedural
    problems, counts = check_derived(files.read("gfx/models/Units/generic_01.mesh"), f.data)
    assert problems == [] and counts["added_vertices"] == 8 and counts["added_triangles"] == 4
    ents = {e["name"]: e for e in _entities(gen)}
    assert ents["MAC_INFANTRY_1"]["pdxmesh"] == "hr_mac_body_1_mesh"
    assert ents["MAC_ARTILLERY_1_crew"]["pdxmesh"] == "generic_01"  # the crew keeps the base body
    assert ents["MAC_CAVALRY_1_rider"]["pdxmesh"] == "generic_01"  # so does the rider
    gfx = gen.files[N.pdxmesh_file("mac")].text
    block = gfx[gfx.index('name = "hr_mac_body_1_mesh"'):]
    block = block[:block.index("}\n\tpdxmesh") if "\tpdxmesh" in block else len(block)]
    assert f'file = "{file}"' in block and "spear_shield_attack_animation" in block and "meshsettings" not in block
    assert gen.loadout.stances["MAC_INFANTRY_1"] == ("spear_shield", "hr_mac_body_1_mesh")
    assert [j for j, _ in runner.calls].count("build_additions") == 1
    report = validate(project, nation, gen, files, idx)
    assert not report.errors, [str(i) for i in report.errors]
    assert any(i.rule == "derived_body" and i.severity == "info" for i in report.issues)


def _entities(gen):
    from eu4unitgen.pdxscript.parser import parse_bytes

    out = {}
    for f in gen.files.values():
        if f.path.endswith(".asset") and "entities" in f.path:
            for p in parse_bytes(f.data).pairs("entity"):
                out[p.value.get_str("name")] = {"name": p.value.get_str("name"), "pdxmesh": p.value.get_str("pdxmesh")}
    return out.values()


def test_derived_body_rule_catches_changes(project, nation, body_game):
    files, idx = body_game
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=AdditionsRunner())
    file = "gfx/models/Units/hr_mac_body_1.mesh"
    tree = parse_pdx_mesh(gen.files[file].data)
    shape = tree["children"][0]["children"][0]
    mesh, skeleton = shape["children"]
    skeleton["children"][-1]["props"]["tx"] = [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # attach_free -> identity
    gen.files[file] = dataclasses.replace(gen.files[file], data=write_pdx_mesh(tree))
    report = validate(project, nation, gen, files, idx, rules=["derived_body"])
    assert any("skeleton differs" in i.message for i in report.errors), [str(i) for i in report.issues]
    # weights outside a chain: the tuft (rigid on head) put on hip
    tree = parse_pdx_mesh(generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT),
                                   runner=AdditionsRunner()).files[file].data)
    skin = tree["children"][0]["children"][0]["children"][0]["children"][2]["props"]
    skin["ix"][-4] = BONES.index("hip")
    gen.files[file] = dataclasses.replace(gen.files[file], data=write_pdx_mesh(tree))
    report = validate(project, nation, gen, files, idx, rules=["derived_body"])
    assert any("tuft: weights on ['hip'], outside its chain ['head']" in i.message for i in report.errors), \
        [str(i) for i in report.issues]


def test_derived_body_needs_a_skinned_base_and_blender(project, nation, game):
    files, idx = game  # the plain fake body: no skin
    with pytest.raises(GenerationError, match="derived body"):
        generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=AdditionsRunner())
