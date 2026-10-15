"""Phase 3: mesh writer, skeleton facts, procedural parts, attach rules, the parts spike test mod."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from eu4unitgen.blender.runner import BlenderRunner, JobResult
from eu4unitgen.core.naming import Naming, NamingError
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.parts import PartSpec, attach_parts, build_parts
from eu4unitgen.core.project import default_blender_path
from eu4unitgen.export.testmod import build_parts_test, parts_checklist
from eu4unitgen.library.scanner import (
    GameFiles, build_index, parse_pdx_mesh, read_dds_header, strip_mesh_props, summarize_mesh, write_pdx_mesh,
)
from eu4unitgen.validate import validate
from fakes import make_mesh

N = Naming("hr_")


def _static_mesh(shape: str, textures: dict, zmin=-12.0, zmax=15.0, sphere=True, tris=12) -> bytes:
    """A tiny static mesh in io_pdx_mesh's export layout (made up)."""
    mesh_props = {"p": [0.0, 0.0, zmin, 0.3, 0.0, zmax, -0.3, 0.2, 0.0], "n": [0.0, 1.0, 0.0] * 3,
                  "ta": [1.0, 0.0, 0.0, 1.0] * 3, "u0": [0.0, 0.0] * 3, "tri": [0, 1, 2] * tris}
    if sphere:
        mesh_props["boundingsphere"] = [0.0, 0.0, 1.0, 14.0]
    mat = {"shader": ["PdxMeshStandard"], "diff": [textures["diff"]], "n": [textures["n"]], "spec": [textures["spec"]]}
    tree = {"name": "root", "props": {"pdxasset": [1, 0]}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": shape, "props": {}, "children": [
                {"name": "mesh", "props": mesh_props, "children": [
                    {"name": "aabb", "props": {"min": [-0.3, 0.0, zmin], "max": [0.3, 0.2, zmax]}, "children": []},
                    {"name": "material", "props": mat, "children": []}]}]}]},
        {"name": "locator", "props": {}, "children": []}]}
    return write_pdx_mesh(tree)


class FakeRunner:
    """Stands in for BlenderRunner: 'builds' each requested part as a tiny static mesh."""

    def __init__(self):
        self.calls = []

    def run_job(self, job, args, timeout=None):
        self.calls.append((job, args))
        out = []
        for p in args["parts"]:
            Path(p["out_mesh"]).write_bytes(_static_mesh(p["shape"], p["textures"]))
            out.append({"kind": p["kind"], "out_mesh": p["out_mesh"], "verts": 3, "faces": 1})
        return JobResult(job, 0, {"parts": out}, "fake log")


# --------------------------------------------------------------------------- mesh writer / skeleton facts

def test_mesh_writer_round_trips_and_strips():
    data = make_mesh("PdxMeshColor", "d.dds", "n.dds", "s.dds", ["root", "head"], {"loc": "head"})
    assert write_pdx_mesh(parse_pdx_mesh(data)) == data
    raw = _static_mesh("xShape", {"diff": "d.dds", "n": "n.dds", "spec": "s.dds"})
    assert b"boundingsphere" in raw and b"boundingsphere" not in strip_mesh_props(raw)
    s = summarize_mesh(parse_pdx_mesh(raw))
    assert s["aabb"] == {"min": [pytest.approx(-0.3), 0.0, -12.0], "max": [pytest.approx(0.3), pytest.approx(0.2), 15.0]}


@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="needs a real EU4 install (EU4_PATH)")
def test_vanilla_skeleton_facts():
    """SPEC §10 Phase 3: generic infantry share one skeleton; head joint ~14.5 up; weapons run along +Z."""
    files = GameFiles(Path(os.environ["EU4_PATH"]), include_dlc=False)
    try:
        west = summarize_mesh(parse_pdx_mesh(files.read("gfx/models/Units/western_generic_01_type1.mesh")))
        assert west["skeleton"]["head"]["parent"] == "top_spine"
        assert west["skeleton"]["head"]["bind"][7] == pytest.approx(14.4988, abs=1e-3)
        assert west["skeleton"]["attach_free"]["bind"] is None  # all-zero tx in vanilla
        for other in ("eastern_generic_01_type1", "indian_generic_01_type1"):
            assert summarize_mesh(parse_pdx_mesh(files.read(f"gfx/models/Units/{other}.mesh")))["bones"] == west["bones"]
        hal = summarize_mesh(parse_pdx_mesh(files.read("gfx/models/Units/western_halbard.mesh")))["aabb"]
        assert hal["max"][2] - hal["min"][2] > 20 and hal["max"][0] - hal["min"][0] < 1
    finally:
        files.close()


# --------------------------------------------------------------------------- parts

def test_part_names():
    assert N.part("mac", "dory", 1) == "hr_mac_dory_1"
    assert N.part_mesh_file("mac", "crest", 2) == "gfx/models/units/hr_mac/hr_mac_crest_2.mesh"
    assert N.part_texture("mac", "dory", 1, "specular") == "hr_mac_dory_1_specular.dds"
    with pytest.raises(NamingError):
        N.part("mac", "pilum", 1)


def test_build_parts_with_a_fake_runner():
    runner = FakeRunner()
    built = build_parts(runner, N, "mac", [PartSpec("dory", 1), PartSpec("crest", 1)], 20)
    job, args = runner.calls[0]
    assert job == "build_parts" and [p["kind"] for p in args["parts"]] == ["dory", "crest"]
    assert args["parts"][0]["textures"]["diff"] == "hr_mac_dory_1_diffuse.dds"
    paths = {f.path: f for f in built.files}
    mesh = paths["gfx/models/units/hr_mac/hr_mac_dory_1.mesh"]
    assert mesh.procedural and b"boundingsphere" not in mesh.data
    for kind, fmt in (("diffuse", "DXT1"), ("normal", "DXT1"), ("specular", "DXT5")):
        dds = paths[f"gfx/models/units/hr_mac/hr_mac_dory_1_{kind}.dds"]
        assert dds.procedural and read_dds_header(dds.data)["format"] == fmt
    assert built.meshes[0] == {"name": "hr_mac_dory_1_mesh", "file": "gfx/models/units/hr_mac/hr_mac_dory_1.mesh",
                               "animations": {}, "scale": 0.5, "extra": {}, "settings": []}
    assert built.attach == {(1, "Right_hand_node"): "hr_mac_dory_1", (1, "head"): "hr_mac_crest_1"}
    with pytest.raises(Exception, match="can't be built yet"):
        build_parts(runner, N, "mac", [PartSpec("hoplon", 1)], 20)


def test_attach_parts_replaces_the_hand_weapon_and_adds_head():
    ents = [{"name": "MAC_INFANTRY_1", "attaches": [{"name": None, "nodes": [("Right_hand_node", "western_halbard")]}]}]
    attach_parts(ents, "MAC", {(1, "Right_hand_node"): "hr_mac_dory_1", (1, "head"): "hr_mac_crest_1"}, N)
    assert ents[0]["attaches"] == [{"name": None, "nodes": [("Right_hand_node", "hr_mac_dory_1"),
                                                            ("head", "hr_mac_crest_1")]}]


@pytest.fixture
def parts_test(project, game, tmp_path):
    files, idx = game
    (tmp_path / "mods").mkdir()
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test")
    return build_parts_test(project, nation, files, idx, tmp_path / "mods", FakeRunner(), {2: 2, 3: 3, 4: 4})


def test_parts_test_mod_and_rules(parts_test):
    """The fake skeleton has no 'head' bone: attach_bones must refuse it; everything else is clean."""
    tm = parts_test.mod
    paths = {f.path for f in tm.plan.files}
    for p in ("eu4unitgen_test_parts/gfx/models/units/hr_macedonian_test/hr_macedonian_test_dory_1.mesh",
              "eu4unitgen_test_parts/gfx/models/units/hr_macedonian_test/hr_macedonian_test_crest_1_diffuse.dds",
              "eu4unitgen_test_parts/interface/assets/hr_macedonian_test.gfx"):
        assert p in paths, p
    errors = [str(i) for i in tm.report.errors]
    assert len(errors) >= 1 and all("attach_bones" in e and "'head'" in e for e in errors), errors
    assert not tm.report.warnings, tm.report.warnings
    assert "H1 crest" in parts_checklist(parts_test, {2: 2})


def test_part_mesh_rule_catches_bad_meshes(project, game, parts_test):
    from eu4unitgen.core.generator import GeneratedFile
    files, idx = game
    pkg = parts_test.mod.package
    rel = "gfx/models/units/hr_macedonian_test/hr_macedonian_test_dory_1.mesh"
    tex = {"diff": "hr_macedonian_test_dory_1_diffuse.dds", "n": "hr_macedonian_test_dory_1_normal.dds",
           "spec": "hr_macedonian_test_dory_1_specular.dds"}
    pkg.files[rel] = GeneratedFile(rel, _static_mesh("dShape", tex, tris=5000), procedural=True)
    msgs = [i.message for i in validate(project, parts_test.nation, pkg, files, idx).errors if i.rule == "part_mesh"]
    assert any("boundingsphere" in m for m in msgs) and any("5000 triangles" in m for m in msgs)


@pytest.mark.skipif(default_blender_path() is None or not default_blender_path().is_file(),
                    reason="pinned Blender not installed")
def test_real_blender_builds_parts_in_vanilla_convention(tmp_path):
    built = build_parts(BlenderRunner(default_blender_path(), log_dir=tmp_path), N, "mac",
                        [PartSpec("dory", 1), PartSpec("crest", 1)], 20)
    meshes = {f.path: summarize_mesh(parse_pdx_mesh(f.data)) for f in built.files if f.path.endswith(".mesh")}
    dory = meshes["gfx/models/units/hr_mac/hr_mac_dory_1.mesh"]["aabb"]
    assert dory["min"][2] < -11 and dory["max"][2] > 13  # along +Z like the halberd, grip near 0
    assert max(abs(v) for v in dory["min"][:2] + dory["max"][:2]) < 1
    crest = meshes["gfx/models/units/hr_mac/hr_mac_crest_1.mesh"]["aabb"]
    assert 0.5 < crest["min"][1] < crest["max"][1] < 4  # above the head joint, Y-up
