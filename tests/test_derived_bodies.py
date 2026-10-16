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

    def __init__(self, deform=None):
        super().__init__()
        self.deform = deform  # name -> metric overrides for check_deform

    def run_job(self, job, args, timeout=None):
        if job == "check_deform":
            self.calls.append((job, args))
            ok = {"str_min": 1.0, "str_max": 1.0, "str_p05": 1.0, "str_p95": 1.0, "pen_depth": 0.0, "pen_pct": 0.0,
                  "flips": 0, "drift": 0.0, "behind": 1.0, "worst_frame": 1}
            anims = {a["id"]: {"frames": [1, 10], "additions": {s["name"]: {**ok, **(self.deform or {}).get(s["name"], {})}
                                                               for s in args["additions"]}}
                     for a in args["anims"]}
            return JobResult(job, 0, {"anims": anims}, "fake log")
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
def body_game(game, fake_install: Path, monkeypatch):
    """The fake install with a skinned tier-1 body (and a hair region for its additions' UVs)."""
    from eu4unitgen.core import derived_body

    (fake_install / "gfx/models/Units/generic_01.mesh").write_bytes(skinned_body())
    monkeypatch.setitem(derived_body.HAIR_REGIONS, "gfx/models/units/generic_01.mesh", (100, 20, 120, 50))
    return game


def test_loadout_derived_round_trip_and_errors():
    lo = Loadout.from_json(LOADOUT)
    assert lo.derived == {1: {"add": ["tail", "tuft"], "remove": []}} and lo.needs_blender() and lo.changes_anything()
    both = Loadout.from_json({"schema": 1, "derived": {"2": {"add": ["tail"], "remove": ["helmet"]}}})
    assert both.derived == {2: {"add": ["tail"], "remove": ["helmet"]}}
    assert both.to_json()["derived"] == {"2": {"add": ["tail"], "remove": ["helmet"]}}
    only = Loadout.from_json({"schema": 1, "derived": {"2": {"remove": ["helmet"]}}})
    assert not only.needs_blender() and only.changes_anything()
    assert Loadout.from_json(lo.to_json()).to_json() == lo.to_json()
    for bad in ({"1": ["wings"]}, {"5": ["tail"]}, {"1": []}, {"1": ["tail", "tail"]}, {"1": {"remove": []}},
                {"1": {"add": ["tail"], "hat": []}}, {"1": {"remove": ["helmet", "helmet"]}}):
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


def test_horse_ears_fit_the_head():
    from eu4unitgen.core.fitting import BodyFit, fitted_params
    from eu4unitgen.core.parts import KINDS, PartSpec, resolve

    fit = BodyFit("x.mesh", stature=14.499, head={"top": 2.152, "xmin": -0.988, "xmax": 0.988, "zmin": -1.269,
                                                  "zmax": 1.107, "count": 10})
    assert fitted_params("horse_ears", fit) == {"base": 1.702, "spread": 0.622, "length": 1.3, "width": 0.5,
                                                "depth": 0.2}
    part = resolve(PartSpec("horse_ears"), "infantry", fit)
    assert part.bone == "head" and part.shader == "PdxMeshStandard" and KINDS["horse_ears"].builder == "ears"


def test_checklist_lists_the_derived_body(project, nation, body_game):
    from eu4unitgen.export.testmod import loadout_checklist

    files, idx = body_game
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=AdditionsRunner())
    text = loadout_checklist(gen, "eu4unitgen_test", None)
    assert "derived bodies" in text.splitlines()[0]
    assert "[ ] 1D1 infantry wears a derived body (gfx/models/Units/hr_mac_body_1.mesh, from generic_01.mesh)" in text
    assert "[ ] 1D5 the gun crew and the cavalry rider wear the plain base body (no tail or tuft)" in text
    assert "SIDE-VIEW CHECK" in text


@pytest.fixture
def removable_helmet(monkeypatch):
    """The fake body's one triangle counts as its 'helmet' (on hip, any UV)."""
    from eu4unitgen.core import derived_body

    monkeypatch.setitem(derived_body.REMOVALS, "gfx/models/units/generic_01.mesh",
                        {"helmet": derived_body.Removal("hip", (0, 0, 512, 512))})


def test_generate_with_a_removed_part(project, nation, body_game, removable_helmet):
    files, idx = body_game
    lo = Loadout.from_json({"schema": 1, "derived": {"1": {"add": ["tail"], "remove": ["helmet"]}},
                            "slots": {"1/infantry": {"parts": [{"kind": "dory"}]}}})
    gen = generate(project, nation, files, idx, loadout=lo, runner=AdditionsRunner())
    data = gen.files["gfx/models/Units/hr_mac_body_1.mesh"].data
    base = files.read("gfx/models/Units/generic_01.mesh")
    problems, counts = check_derived(base, data, [0])
    assert problems == [] and counts["removed_triangles"] == 1 and counts["added_triangles"] == 2
    assert check_derived(base, data)[0] == ["the base's triangles changed"]  # without the removal it's a change
    assert gen.loadout.derived[1].removed == ["helmet"]
    report = validate(project, nation, gen, files, idx)
    assert not report.errors, [str(i) for i in report.errors]
    assert any("- helmet (1 triangles removed)" in i.message for i in report.issues if i.rule == "derived_body")


def test_unknown_removal_is_refused(project, nation, body_game):
    files, idx = body_game
    lo = Loadout.from_json({"schema": 1, "derived": {"1": {"remove": ["helmet"]}}})
    with pytest.raises(GenerationError, match="no removable 'helmet'"):
        generate(project, nation, files, idx, loadout=lo, runner=AdditionsRunner())


def test_deformation_check_runs_and_reaches_the_validator(project, nation, body_game):
    files, idx = body_game
    runner = AdditionsRunner(deform={"tuft": {"pen_depth": 0.6, "pen_pct": 20.0}})
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=runner)
    calls = [a for j, a in runner.calls if j == "check_deform"]
    assert len(calls) == 1 and [a["id"] for a in calls[0]["anims"]] == ["idle", "walk", "attack"]
    assert [s["name"] for s in calls[0]["additions"]] == ["tail", "tuft"]
    assert calls[0]["additions"][0]["start"] == 3 and calls[0]["additions"][0]["exempt"] == [0, 1]
    report = validate(project, nation, gen, files, idx, rules=["derived_body"])
    errs = [i.message for i in report.errors]
    assert any("C2 tuft in attack" in m for m in errs) and any("C2 tuft in idle" in m for m in errs), errs
    lo = Loadout.from_json({**LOADOUT, "accept_clipping": ["tuft"]})
    gen = generate(project, nation, files, idx, loadout=lo, runner=AdditionsRunner(deform={"tuft": {"pen_depth": 0.6}}))
    report = validate(project, nation, gen, files, idx, rules=["derived_body"])
    assert not report.errors and any("accepted as a known limit" in i.message for i in report.warnings)


def test_judge_criteria():
    from eu4unitgen.core.deform_check import judge

    ok = {"str_min": 1.0, "str_max": 1.0, "str_p05": 1.0, "str_p95": 1.0, "pen_depth": 0.0, "pen_pct": 0.0,
          "flips": 0, "drift": 0.0, "behind": 1.0, "worst_frame": 3}
    res = {"anims": {"walk": {"additions": {"tail": {**ok, "str_min": 0.8}, "ears": {**ok, "behind": None}}},
                     "attack": {"additions": {"tail": {**ok, "pen_pct": 2.0, "pen_depth": 0.25, "flips": 1, "drift": 0.1,
                                                       "behind": -0.1}}}}}
    found = judge(res)
    assert ("error", "C1 tail in walk (worst frame 3): stretch 0.80-1.00 (p05-p95 1.00-1.00)") in found
    assert not any("C2" in m for _, m in found)  # attack allows 3% / 0.30
    assert ("error", "C3 tail in attack (worst frame 3): 1 triangles flip") in found
    assert any(s == "warning" and m.startswith("C4 tail in attack") for s, m in found)
    assert any(s == "warning" and m.startswith("C5 tail in attack") for s, m in found)
    assert not any("ears" in m for _, m in found)


def test_greek_helmet_fits_the_bare_head():
    from eu4unitgen.core.fitting import BodyFit, fitted_params
    from eu4unitgen.core.parts import KINDS, PartSpec, resolve

    bare = BodyFit("x.mesh", stature=14.499, head={"top": 1.795, "xmin": -0.72, "xmax": 0.72, "zmin": -1.269,
                                                   "zmax": 0.846, "count": 10})
    assert fitted_params("greek_helmet", bare) == {"top": 1.915, "half_width": 0.85, "front": -1.115, "back": 1.046}
    part = resolve(PartSpec("greek_helmet", {"style": "corinthian", "crest": "transverse"}), "infantry", bare)
    assert part.bone == "head" and KINDS["greek_helmet"].builder == "helmet" and len(part.zones) == 3


def test_body_without_a_hair_region_is_refused(project, nation, body_game, monkeypatch):
    from eu4unitgen.core import derived_body

    monkeypatch.delitem(derived_body.HAIR_REGIONS, "gfx/models/units/generic_01.mesh")
    files, idx = body_game
    with pytest.raises(GenerationError, match="no known hair region"):
        generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=AdditionsRunner())


def test_additions_span_the_hair_region(project, nation, body_game):
    files, idx = body_game
    runner = AdditionsRunner()
    generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=runner)
    job = next(a for j, a in runner.calls if j == "build_additions")
    assert job["patch_px"] == [100, 20, 120, 50]


def test_atlas_images_paint_each_triangle_in_its_zone():
    from eu4unitgen.textures.procedural import Zone, atlas_images

    zones = [Zone((110, 75, 40), 3), Zone((180, 130, 60), 13), Zone((0, 0, 0), 5, "country1")]
    tris = [[1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.4], [2, 0.6, 0.6, 0.9, 0.6, 0.6, 0.9]]
    imgs = atlas_images(zones, 64, tris)
    assert imgs["diffuse"].size == (64, 64)
    assert imgs["diffuse"].getpixel((10, 10))[:3] == (180, 130, 60)  # zone 1
    assert imgs["diffuse"].getpixel((42, 42))[:3] == (170, 170, 170)  # country zone: neutral grey
    assert imgs["specular"].getpixel((42, 42))[:3] == (255, 0, 0)  # its mask in spec R
    assert imgs["diffuse"].getpixel((60, 2))[:3] == (110, 75, 40)  # background: zone 0
