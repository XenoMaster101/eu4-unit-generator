"""Phase 3: mesh writer, skeleton facts, the procedural part library, fitting, loadouts in the
generator, the loadout rule, the parts test mod (in-game test 8)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
from PIL import Image

from eu4unitgen.blender.runner import BlenderRunner, JobResult
from eu4unitgen.core.fitting import REFERENCE_STATURE, fitted_params, measure_body
from eu4unitgen.core.generator import GeneratedFile, GenerationError, generate
from eu4unitgen.core.loadout import Loadout, LoadoutError, RoleLoadout
from eu4unitgen.core.naming import Naming, NamingError
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.parts import KINDS, PartBuild, PartError, PartSpec, build_parts, resolve
from eu4unitgen.core.project import default_blender_path
from eu4unitgen.export.testmod import build_test_mod, loadout_checklist
from eu4unitgen.library.scanner import (
    GameFiles, parse_pdx_mesh, read_dds_header, strip_mesh_props, summarize_mesh, write_pdx_mesh,
)
from eu4unitgen.textures.dds import read_dds
from eu4unitgen.textures.procedural import DISC_CORNERS, Zone, part_images, shader_for, zone_uvs
from eu4unitgen.validate import validate
from fakes import make_mesh, make_textures_decodable

N = Naming("hr_")


def _static_mesh(shape: str, textures: dict, zmin=-12.0, zmax=15.0, sphere=True, tris=12,
                 shader="PdxMeshStandard") -> bytes:
    """A tiny static mesh in io_pdx_mesh's export layout (made up)."""
    mesh_props = {"p": [0.0, 0.0, zmin, 0.3, 0.0, zmax, -0.3, 0.2, 0.0], "n": [0.0, 1.0, 0.0] * 3,
                  "ta": [1.0, 0.0, 0.0, 1.0] * 3, "u0": [0.0, 0.0] * 3, "tri": [0, 1, 2] * tris}
    if sphere:
        mesh_props["boundingsphere"] = [0.0, 0.0, 1.0, 14.0]
    mat = {"shader": [shader], "diff": [textures["diff"]], "n": [textures["n"]], "spec": [textures["spec"]]}
    tree = {"name": "root", "props": {"pdxasset": [1, 0]}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": shape, "props": {}, "children": [
                {"name": "mesh", "props": mesh_props, "children": [
                    {"name": "aabb", "props": {"min": [-0.3, 0.0, zmin], "max": [0.3, 0.2, zmax]}, "children": []},
                    {"name": "material", "props": mat, "children": []}]}]}]},
        {"name": "locator", "props": {}, "children": []}]}
    return write_pdx_mesh(tree)


class FakeRunner:
    """Stands in for BlenderRunner: 'builds' each requested part as a tiny static mesh with the
    entry's shader, grip inside it (z -12..15)."""

    def __init__(self):
        self.calls = []

    def run_job(self, job, args, timeout=None):
        self.calls.append((job, args))
        out = []
        for p in args["parts"]:
            Path(p["out_mesh"]).write_bytes(_static_mesh(p["shape"], p["textures"], shader=p["shader"]))
            out.append({"kind": p["kind"], "out_mesh": p["out_mesh"], "verts": 3, "faces": 1})
        return JobResult(job, 0, {"parts": out}, "fake log")


def _skinned_body(top=2.15, half_z=1.27, height=16.65, joint=REFERENCE_STATURE) -> bytes:
    """A made-up skinned body: a head joint at y ``joint`` with a few vertices skinned to it, one
    vertex on the root, and a bounding box ``height`` tall."""
    head = [(0.8, top, 0.0), (-0.8, 1.0, 0.0), (0.0, 1.5, -half_z), (0.0, 1.2, half_z * 0.9)]
    p = [c for x, y, z in head for c in (x, joint + y, z)] + [0.0, 1.0, 0.0]
    ix = [1, -1, -1, -1] * len(head) + [0, -1, -1, -1]
    w = [1.0, 0.0, 0.0, 0.0] * (len(head) + 1)
    ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tree = {"name": "root", "props": {"pdxasset": [1, 0]}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": "bodyShape", "props": {}, "children": [
                {"name": "mesh", "props": {"p": p, "tri": [0, 1, 2]}, "children": [
                    {"name": "aabb", "props": {"min": [-6.0, 0.0, -1.5], "max": [6.0, height, 1.5]}, "children": []},
                    {"name": "material", "props": {"shader": ["PdxMeshColor"]}, "children": []},
                    {"name": "skin", "props": {"bones": [4], "ix": ix, "w": w}, "children": []}]},
                {"name": "skeleton", "props": {}, "children": [
                    {"name": "root", "props": {"ix": [0], "tx": ident + [0.0, 0.0, 0.0]}, "children": []},
                    {"name": "head", "props": {"ix": [1], "pa": [0], "tx": ident + [0.0, -joint, 0.0]},
                     "children": []}]}]}]}]}
    return write_pdx_mesh(tree)


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


# --------------------------------------------------------------------------- names, textures, specs

def test_part_names():
    assert N.part("mac", "dory", 1) == "hr_mac_dory_1"
    assert N.part("mac", "crest", 2, "rider") == "hr_mac_crest_rider_2"
    assert N.part_pdxmesh("mac", "crest", 2, "crew") == "hr_mac_crest_crew_2_mesh"
    assert N.part_mesh_file("mac", "crest", 2) == "gfx/models/units/hr_mac/hr_mac_crest_2.mesh"
    assert N.part_texture("mac", "dory", 1, "specular", "rider") == "hr_mac_dory_rider_1_specular.dds"
    assert N.crew_entity("MAC", 3) == "MAC_ARTILLERY_3_crew"
    assert N.part_preview_dir("mac") == "part_previews/mac" and N.part_preview(2, "crew") == "2_crew.png"
    for bad in (lambda: N.part("mac", "pilum", 1), lambda: N.part("mac", "dory", 1, "horse")):
        with pytest.raises(NamingError):
            bad()


def test_procedural_textures_strip_disc_and_blazon():
    zones = [Zone((110, 75, 40), 3), Zone((180, 130, 60), 13)]
    assert zone_uvs(2, "strip") == [(0.25, 0.5), (0.75, 0.5)]
    imgs = part_images(zones)
    assert imgs["diffuse"].getpixel((5, 30))[:3] == (110, 75, 40) and imgs["specular"].getpixel((60, 30)) == (0, 0, 0, 33)
    assert shader_for(zones) == "PdxMeshStandard"
    disc = [Zone((1, 2, 3), 10, "country2"), Zone((180, 130, 60), 13), Zone((120, 85, 50), 3)]
    assert shader_for(disc) == "PdxMeshColor"
    for u, v in DISC_CORNERS:  # corner swatches lie outside the face circle
        assert ((u - 0.5) ** 2 + (v - 0.5) ** 2) ** 0.5 > 0.55
    imgs = part_images(disc, "disc")
    assert imgs["specular"].getpixel((128, 128)) == (0, 255, 0, 26)  # face: secondary country colour
    assert imgs["specular"].getpixel((4, 4))[:3] == (0, 0, 0)  # rim corner: fixed colour
    blazon = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    blazon.paste((250, 250, 250, 255), (0, 0, 32, 16))  # top half painted
    imgs = part_images(disc, "disc", blazon)
    assert imgs["diffuse"].getpixel((128, 60))[:3] == (250, 250, 250)
    assert imgs["specular"].getpixel((128, 60)) == (0, 0, 0, 26)  # under the blazon: no country colour
    assert imgs["specular"].getpixel((128, 200)) == (0, 255, 0, 26)
    flipped = part_images(disc, "disc", blazon, blazon_rotate=180)
    assert flipped["diffuse"].getpixel((128, 200))[:3] == (250, 250, 250)
    with pytest.raises(ValueError):
        part_images(zones, "strip", blazon)


def test_part_spec_checks():
    PartSpec("hoplon", {"radius": 4.0, "blazon_rotate": "90"}, {"face": [1, 2, 3], "rim": "country3"}, "x.png").check()
    for bad, msg in ((PartSpec("pilum"), "unknown part kind"),
                     (PartSpec("dory", {"length": 3}), "unknown param"),
                     (PartSpec("dory", {"tip": 500}), "from 4"),
                     (PartSpec("crest", {"orientation": "sideways"}), "must be one of"),
                     (PartSpec("crest", colours={"plume": "country1"}), "unknown colour zone"),
                     (PartSpec("crest", colours={"hair": "country4"}), "country1"),
                     (PartSpec("crest", colours={"hair": [300, 0, 0]}), "0-255"),
                     (PartSpec("dory", blazon="x.png"), "takes no blazon")):
        with pytest.raises(PartError, match=msg):
            bad.check()


def test_resolve_defaults_fitting_overrides_and_point():
    body = measure_body(_skinned_body(joint=REFERENCE_STATURE * 1.1), "body.mesh")
    dory = resolve(PartSpec("dory", {"butt": -10.0}), "infantry", body)
    assert dory.params["tip"] == pytest.approx(15.0 * 1.1, abs=1e-3)  # fitted: scaled to the taller body
    assert dory.params["butt"] == -10.0  # the user's override beats fitting
    assert dory.params["point"] == "up" and dory.shader == "PdxMeshStandard" and dory.bone == "Right_hand_node"
    assert resolve(PartSpec("xiphos"), "rider", body).params["point"] == "down"  # the sabre's way
    assert resolve(PartSpec("xiphos", {"point": "up"}), "rider", body).params["point"] == "up"
    hoplon = resolve(PartSpec("hoplon"), "infantry", None)
    assert hoplon.shader == "PdxMeshColor" and hoplon.zones[0].channel == "country1" and hoplon.fitted == {}
    fixed = resolve(PartSpec("hoplon", colours={"face": [200, 30, 30]}), "crew", None)
    assert fixed.shader == "PdxMeshStandard" and fixed.zones[0].colour == (200, 30, 30)
    with pytest.raises(PartError, match="can't go on the rider"):
        resolve(PartSpec("hoplon"), "rider", None)


# --------------------------------------------------------------------------- fitting

def test_measure_body_and_fit_the_crest():
    fit = measure_body(_skinned_body(top=2.15, half_z=1.27, height=18.0), "body.mesh")
    assert fit.height == pytest.approx(18.0) and fit.scale == pytest.approx(1.0)  # a tall hat doesn't scale parts
    assert fit.head["top"] == pytest.approx(2.15) and fit.head["zmin"] == pytest.approx(-1.27)
    assert fit.head["count"] == 4  # the root-skinned vertex isn't the head's
    crest = fitted_params("crest", fit)
    assert crest["base"] == pytest.approx(1.9) and crest["radius"] == pytest.approx(1.397, abs=1e-3)  # test 7's crest
    assert fitted_params("crest", None) == {}
    bare = measure_body(make_mesh("PdxMeshColor", "d.dds", "n.dds", "s.dds", ["root", "head"]), "bare.mesh")
    assert bare.head is None and bare.stature is None and bare.scale == 1.0 and len(bare.notes) == 1
    assert "base" not in fitted_params("crest", bare)


@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="needs a real EU4 install (EU4_PATH)")
def test_fitting_on_the_vanilla_western_body_reproduces_test_7():
    files = GameFiles(Path(os.environ["EU4_PATH"]), include_dlc=False)
    try:
        rel = "gfx/models/Units/western_generic_01_type1.mesh"
        fit = measure_body(files.read(rel), rel)
    finally:
        files.close()
    assert fit.stature == pytest.approx(REFERENCE_STATURE, abs=0.01) and fit.head["top"] == pytest.approx(2.15, abs=0.01)
    crest = fitted_params("crest", fit)
    assert crest["base"] == pytest.approx(1.9, abs=0.01) and crest["radius"] == pytest.approx(1.4, abs=0.01)


# --------------------------------------------------------------------------- building

def test_build_parts_with_a_fake_runner_and_the_cache():
    runner, cache = FakeRunner(), {}
    builds = [PartBuild(N.part("mac", "dory", 1), 1, resolve(PartSpec("dory"), "infantry")),
              PartBuild(N.part("mac", "hoplon", 1), 1, resolve(PartSpec("hoplon"), "infantry"))]
    built = build_parts(runner, N, "mac", builds, 20, cache)
    (job, args), = runner.calls
    assert job == "build_parts" and [p["builder"] for p in args["parts"]] == ["spear", "shield"]
    assert args["parts"][1]["layout"] == "disc" and args["parts"][1]["shader"] == "PdxMeshColor"
    assert args["parts"][0]["textures"]["diff"] == "hr_mac_dory_1_diffuse.dds"
    paths = {f.path: f for f in built.files}
    mesh = paths["gfx/models/units/hr_mac/hr_mac_dory_1.mesh"]
    assert mesh.procedural and b"boundingsphere" not in mesh.data
    for kind, fmt, size in (("diffuse", "DXT1", 64), ("normal", "DXT1", 64), ("specular", "DXT5", 64)):
        h = read_dds_header(paths[f"gfx/models/units/hr_mac/hr_mac_dory_1_{kind}.dds"].data)
        assert (h["format"], h["width"]) == (fmt, size)
    assert read_dds_header(paths["gfx/models/units/hr_mac/hr_mac_hoplon_1_diffuse.dds"].data)["width"] == 256
    assert built.meshes[0] == {"name": "hr_mac_dory_1_mesh", "file": "gfx/models/units/hr_mac/hr_mac_dory_1.mesh",
                               "animations": {}, "scale": 0.5, "extra": {}, "settings": []}
    again = build_parts(runner, N, "mac", builds, 20, cache)
    assert len(runner.calls) == 1 and again.raw_meshes == built.raw_meshes  # all from the cache


# --------------------------------------------------------------------------- loadouts

LOADOUT = {"schema": 1, "slots": {
    "1/infantry": {"parts": [{"kind": "dory"}, {"kind": "hoplon"}, {"kind": "crest"}]},
    "1/rider": {"parts": [{"kind": "xiphos"}, {"kind": "crest"}]},
    "1/crew": {"parts": [{"kind": "crest", "colours": {"hair": "country2"}}]},
    "2/infantry": {"stance": "polearm_twohanded", "parts": [{"kind": "sarissa"}]},
}}


def test_loadout_round_trip_and_errors(tmp_path):
    lo = Loadout.from_json(LOADOUT, tmp_path)
    assert lo.slots[(1, "infantry")].resolved_stance() == "spear_shield"
    assert lo.slots[(2, "infantry")].resolved_stance() == "polearm_twohanded"
    assert RoleLoadout([PartSpec("sarissa")]).resolved_stance() == "polearm_twohanded"
    assert RoleLoadout([PartSpec("xiphos")]).resolved_stance() == "polearm_onehanded"
    assert RoleLoadout([PartSpec("crest")]).resolved_stance() == "base"
    assert set(lo.active()) == {(1, "infantry"), (1, "rider"), (1, "crew"), (2, "infantry")}
    path = tmp_path / "lo.json"
    lo.save(path)
    assert Loadout.load(path).to_json() == lo.to_json() == Loadout.from_json(lo.to_json()).to_json()
    assert Loadout.from_json({"schema": 1, "slots": {"3/crew": {"parts": []}}}).active()  # a plain crew counts
    assert not Loadout.from_json({"schema": 1, "slots": {"3/infantry": {"parts": []}}}).active()
    for doc, msg in (({"schema": 2}, "schema"),
                     ({"schema": 1, "slots": {"5/infantry": {}}}, "must be <tier>/<role>"),
                     ({"schema": 1, "slots": {"1/horse": {}}}, "must be <tier>/<role>"),
                     ({"schema": 1, "slots": {"1/infantry": {"parts": [{"kind": "dory"}, {"kind": "xiphos"}]}}},
                      "both go on Right_hand_node"),
                     ({"schema": 1, "slots": {"1/rider": {"parts": [{"kind": "hoplon"}]}}}, "can't go on the rider"),
                     ({"schema": 1, "slots": {"1/rider": {"stance": "spear_shield"}}}, "take a stance"),
                     ({"schema": 1, "slots": {"1/infantry": {"stance": "dance"}}}, "stance"),
                     ({"schema": 1, "slots": {"1/infantry": {"parts": [{"kind": "dory", "params": {"x": 1}}]}}},
                      "unknown param")):
        with pytest.raises(LoadoutError, match=msg):
            Loadout.from_json(doc)


def _entities(generated, set_name="mac"):
    """name -> (pdxmesh, attaches) from the generated entity file."""
    from eu4unitgen.library.scanner import _entity_record
    from eu4unitgen.pdxscript.parser import Block, parse_bytes

    doc = parse_bytes(generated.files[N.set_entities_file(set_name)].data)
    out = {}
    for v in doc.get_all("entity"):
        if isinstance(v, Block):
            rec = _entity_record(v, "x", None)
            out[rec["name"]] = rec
    return out


def test_generate_with_a_loadout(project, nation, game):
    files, idx = game
    runner = FakeRunner()
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=runner)
    ents = _entities(gen)
    attach = {n: {k: v for a in (e.get("attach") or []) for k, v in a.items() if k != "name"} for n, e in ents.items()}
    assert attach["MAC_INFANTRY_1"] == {"Right_hand_node": "hr_mac_dory_1", "Left_hand_node": "hr_mac_hoplon_1",
                                        "head": "hr_mac_crest_1"}
    # the rider's crest is the infantry's (identical part at one tier: built once, shared)
    assert attach["MAC_CAVALRY_1_rider"] == {"Right_hand_node": "hr_mac_xiphos_rider_1", "head": "hr_mac_crest_1"}
    # the crew: its own entity, the base infantry's weapon kept, its own (country-coloured) crest
    assert attach["MAC_ARTILLERY_1_crew"] == {"Right_hand_node": "w_halbard", "head": "hr_mac_crest_crew_1"}
    assert attach["MAC_ARTILLERY_1"]["infantry_node"] == "MAC_ARTILLERY_1_crew"
    assert attach["MAC_ARTILLERY_2"]["infantry_node"] == "MAC_INFANTRY_2"
    assert attach["MAC_INFANTRY_2"] == {"Right_hand_node": "hr_mac_sarissa_2"}  # the base's left-hand musket is gone
    assert any("replaces the base's w_musket" in n for n in gen.notes)
    # stances: tier-1 infantry moves to spear_shield on its own pdxmesh; the crew keeps the vanilla one
    assert ents["MAC_INFANTRY_1"]["pdxmesh"] == "hr_mac_infantry_1_mesh"
    assert ents["MAC_ARTILLERY_1_crew"]["pdxmesh"] == "generic_01"
    assert ents["MAC_CAVALRY_1_rider"]["pdxmesh"] == "generic_01"
    assert ents["MAC_INFANTRY_2"]["pdxmesh"] == "hr_mac_infantry_2_mesh"
    gfx = gen.files[N.pdxmesh_file("mac")].text
    assert 'type = "spear_shield_attack_animation"' in gfx and 'type = "polearm_twohanded_idle_animation"' in gfx
    assert 'name = "hr_mac_hoplon_1_mesh"' in gfx and "meshsettings" not in gfx
    assert {r.entity for r in gen.loadout.parts} == {"hr_mac_dory_1", "hr_mac_hoplon_1", "hr_mac_crest_1",
                                                     "hr_mac_xiphos_rider_1", "hr_mac_crest_crew_1", "hr_mac_sarissa_2"}
    assert len(runner.calls) == 1 and len(runner.calls[0][1]["parts"]) == 6
    assert sorted(gen.loadout.slots) == [(1, "crew"), (1, "infantry"), (1, "rider"), (2, "infantry")]
    defaults = gen.files[N.gfx_entities_file("hellenicgfx")].text
    assert "hr_mac_" not in defaults and "spear_shield" not in defaults  # the gfx-culture defaults stay undressed
    report = validate(project, nation, gen, files, idx)
    assert not report.errors, [str(i) for i in report.errors]
    assert not [i for i in report.warnings if i.rule == "loadout"], [str(i) for i in report.warnings]
    with pytest.raises(GenerationError, match="need Blender"):
        generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT))


def test_loadout_rule_catches_mistakes(project, nation, game):
    files, idx = game
    lo = Loadout.from_json({"schema": 1, "slots": {
        "1/infantry": {"stance": "base", "parts": [{"kind": "dory"}, {"kind": "hoplon"}]}}})
    gen = generate(project, nation, files, idx, loadout=lo, runner=FakeRunner())

    def issues(g):
        rep = validate(project, nation, g, files, idx)
        return [str(i) for i in rep.issues if i.rule == "loadout"]
    assert any("doesn't use the spear_shield animations" in m for m in issues(gen))  # shield, no stance: warned
    bad = copy.deepcopy(gen)
    rel = N.set_entities_file("mac")
    bad.files[rel] = GeneratedFile(rel, bad.files[rel].data.replace(b'Left_hand_node = "hr_mac_hoplon_1"',
                                                                    b'head = "hr_mac_hoplon_1"'))
    assert any("belongs on Left_hand_node" in m for m in issues(bad))
    bad = copy.deepcopy(gen)
    spec_rel = "gfx/models/units/hr_mac/hr_mac_hoplon_1_specular.dds"
    blank = Image.new("RGBA", (256, 256), (0, 0, 0, 20))
    from eu4unitgen.textures.dds import DDSFormat, write_dds
    bad.files[spec_rel] = GeneratedFile(spec_rel, write_dds(blank, DDSFormat("DXT5", 256, 256, 9)), procedural=True)
    assert any("no country-colour mask" in m for m in issues(bad))
    bad = copy.deepcopy(gen)
    mesh_rel = "gfx/models/units/hr_mac/hr_mac_dory_1.mesh"
    tex = {"diff": "hr_mac_dory_1_diffuse.dds", "n": "hr_mac_dory_1_normal.dds", "spec": "hr_mac_dory_1_specular.dds"}
    bad.files[mesh_rel] = GeneratedFile(mesh_rel, strip_mesh_props(_static_mesh("dShape", tex, zmin=2.0, zmax=15.0)),
                                        procedural=True)
    assert any("grip (z = 0) isn't inside" in m for m in issues(bad))
    bad.files[mesh_rel] = GeneratedFile(mesh_rel, strip_mesh_props(_static_mesh("dShape", tex, shader="PdxMeshColor")),
                                        procedural=True)
    assert any("isn't the part's PdxMeshStandard" in m for m in issues(bad))


def test_part_mesh_rule_catches_bad_meshes(project, nation, game):
    files, idx = game
    gen = generate(project, nation, files, idx, loadout=Loadout.from_json(LOADOUT), runner=FakeRunner())
    rel = "gfx/models/units/hr_mac/hr_mac_dory_1.mesh"
    tex = {"diff": "hr_mac_dory_1_diffuse.dds", "n": "hr_mac_dory_1_normal.dds", "spec": "hr_mac_dory_1_specular.dds"}
    gen.files[rel] = GeneratedFile(rel, _static_mesh("dShape", tex, tris=5000), procedural=True)
    msgs = [i.message for i in validate(project, nation, gen, files, idx).errors if i.rule == "part_mesh"]
    assert any("boundingsphere" in m for m in msgs) and any("5000 triangles" in m for m in msgs)


def test_stance_edits_a_retextured_mesh_and_the_crew_gets_its_own(project, nation, fake_install):
    from eu4unitgen.core.recipe import TextureRecipe
    from eu4unitgen.library.scanner import build_index

    make_textures_decodable(fake_install)
    files = GameFiles(fake_install)
    try:
        idx = build_index(files)
        recipe = TextureRecipe.from_json({"schema": 1, "slots": {"1/infantry": {"ops": [
            {"op": "hue_shift", "region": "cloth", "degrees": 40}]}}})
        lo = Loadout.from_json({"schema": 1, "slots": {"1/infantry": {"parts": [{"kind": "hoplon"}]},
                                                       "1/crew": {"parts": []}}})
        gen = generate(project, nation, files, idx, recipe=recipe, loadout=lo, runner=FakeRunner())
        ents = _entities(gen)
        assert ents["MAC_INFANTRY_1"]["pdxmesh"] == ents["MAC_CAVALRY_1_rider"]["pdxmesh"] == "hr_mac_infantry_1_mesh"
        assert ents["MAC_ARTILLERY_1_crew"]["pdxmesh"] == "hr_mac_crew_1_mesh"
        gfx = gen.files[N.pdxmesh_file("mac")].text
        crew_block = gfx[gfx.index('name = "hr_mac_crew_1_mesh"'):]
        assert "meshsettings" in crew_block and 'type = "inf_idle_animation"' in crew_block  # retextured, base stance
        inf_block = gfx[gfx.index('name = "hr_mac_infantry_1_mesh"'):gfx.index('name = "hr_mac_crew_1_mesh"')]
        assert 'type = "spear_shield_idle_animation"' in inf_block
        report = validate(project, nation, gen, files, idx)
        assert not report.errors, [str(i) for i in report.errors]
    finally:
        files.close()


def test_parts_test_mod_and_checklist(project, game, tmp_path):
    files, idx = game
    (tmp_path / "mods").mkdir()
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test")
    tm = build_test_mod(project, nation, files, idx, tmp_path / "mods", test_sprite_techs={2: 2, 3: 3, 4: 4},
                        loadout=Loadout.from_json(LOADOUT), runner=FakeRunner())
    paths = {f.path for f in tm.plan.files}
    for p in ("eu4unitgen_test/gfx/models/units/hr_macedonian_test/hr_macedonian_test_hoplon_1.mesh",
              "eu4unitgen_test/gfx/models/units/hr_macedonian_test/hr_macedonian_test_crest_crew_1_specular.dds",
              "eu4unitgen_test/interface/assets/hr_macedonian_test.gfx"):
        assert p in paths, p
    assert not tm.report.errors, [str(i) for i in tm.report.errors]
    text = loadout_checklist(tm.package, "eu4unitgen_test", {2: 2, 3: 3, 4: 4})
    assert "IN-GAME TEST 8" in text and "TIER 1" in text and "TIER 2" in text and "TIER 3" not in text
    assert "[ ] 1C  artillery gun crew: crest, on head, front-to-back, horsehair in the secondary unit colour" in text
    assert "[ ] 2C  artillery gun crew: the infantry entity itself" in text
    assert "[ ] 1R  cavalry rider: xiphos, on Right_hand_node, point -Z (like the sabre); crest, on head" in text
    assert "AAA_INFANTRY_1: spear_shield animations" in text
    assert "2S  AAA_INFANTRY_2: polearm_twohanded animations; both hands" in text


@pytest.mark.skipif(default_blender_path() is None or not default_blender_path().is_file(),
                    reason="pinned Blender not installed")
def test_real_blender_builds_every_kind_in_vanilla_convention(tmp_path):
    specs = [(k, "infantry") for k in KINDS] + [("xiphos", "rider")]
    builds = [PartBuild(N.part("mac", k, 1, r), 1, resolve(PartSpec(k), r)) for k, r in specs]
    built = build_parts(BlenderRunner(default_blender_path(), log_dir=tmp_path), N, "mac", builds, 20)
    boxes = {name: summarize_mesh(parse_pdx_mesh(data))["aabb"] for name, data in built.raw_meshes.items()}
    dory = boxes["hr_mac_dory_1"]
    assert dory["min"][2] == pytest.approx(-12.8, abs=0.01) and dory["max"][2] == pytest.approx(15, abs=0.01)  # test 7's
    assert boxes["hr_mac_sarissa_1"]["max"][2] == pytest.approx(30, abs=0.01)
    assert boxes["hr_mac_xiphos_1"]["max"][2] > 5 and boxes["hr_mac_xiphos_rider_1"]["min"][2] < -5  # point up / down
    crest = boxes["hr_mac_crest_1"]
    assert crest["min"][1] == pytest.approx(0.98, abs=0.01) and crest["max"][1] == pytest.approx(3.1, abs=0.01)
    assert crest["max"][0] - crest["min"][0] == pytest.approx(0.28, abs=0.01)  # test 7's width
    shield = boxes["hr_mac_hoplon_1"]
    assert shield["min"][0] == pytest.approx(-0.1, abs=0.01) and shield["max"][0] > 0.8  # domed towards +X
    assert shield["max"][1] - shield["min"][1] == pytest.approx(7.6, abs=0.05)  # radius 3.8
    for name, data in built.raw_meshes.items():
        summary = summarize_mesh(parse_pdx_mesh(data))
        assert not summary["bones"] and summary["shapes"][0]["shader"] == (
            "PdxMeshColor" if "hoplon" in name else "PdxMeshStandard"), name
    face = read_dds([f for f in built.files if f.path.endswith("hoplon_1_specular.dds")][0].data)
    assert face.getpixel((128, 128))[0] == 255


def test_rigid_bind_drops_scale():
    from eu4unitgen.core.fitting import rigid
    scaled = [1e-12, 0.0, 0.0, -5.9, 0.0, -1e-12, 0.0, 9.9, 0.0, 0.0, -1e-12, 0.2]  # western_generic_03's hand, roughly
    bind, changed = rigid(scaled)
    assert changed and bind == pytest.approx([1, 0, 0, -5.9, 0, -1, 0, 9.9, 0, 0, -1, 0.2])
    assert rigid([1, 0, 0, 0, 0, 1, 0, 14.5, 0, 0, 1, 0]) == ([1, 0, 0, 0, 0, 1, 0, 14.5, 0, 0, 1, 0], False)


# --------------------------------------------------------------------------- after test 8: bodies, weapon point, xyston

def test_loadout_bodies_weapon_point_and_xyston(tmp_path):
    from eu4unitgen.core.loadout import slot_warnings
    lo = Loadout.from_json({"schema": 1, "bodies": {"2": "generic_01"}, "weapon_point": {"rider": "up", "crew": "up"},
                            "slots": {"1/rider": {"parts": [{"kind": "xyston"}]}}})
    assert lo.bodies == {2: "generic_01"} and lo.points == {"rider": "up", "crew": "up"}
    assert lo.to_json()["weapon_point"] == {"rider": "up"}  # defaults aren't written
    assert Loadout.from_json(lo.to_json()).to_json() == lo.to_json()
    assert Loadout.from_json({"schema": 1, "bodies": {"3": "generic_01"}}).changes_anything()
    assert not Loadout.from_json({"schema": 1, "bodies": {"3": "generic_01"}}).needs_blender()
    for doc, msg in (({"schema": 1, "bodies": {"5": "x"}}, "must be a tier"),
                     ({"schema": 1, "bodies": {"1": "bad name"}}, "body pdxmesh"),
                     ({"schema": 1, "weapon_point": {"rider": "sideways"}}, "weapon_point")):
        with pytest.raises(LoadoutError, match=msg):
            Loadout.from_json(doc)
    x = resolve(PartSpec("xyston"), "rider", None)
    assert x.params["point"] == "down" and x.params["tip"] - x.params["butt"] + x.params["spike"] == pytest.approx(40)
    assert resolve(PartSpec("xyston"), "rider", None, lo.points).params["point"] == "up"
    assert resolve(PartSpec("xyston", {"point": "down"}), "rider", None, lo.points).params["point"] == "down"
    assert RoleLoadout([PartSpec("xyston")]).resolved_stance() == "polearm_twohanded"
    assert slot_warnings(RoleLoadout([PartSpec("hoplon"), PartSpec("crest")]))
    assert not slot_warnings(RoleLoadout([PartSpec("hoplon"), PartSpec("dory")]))


def test_generate_with_chosen_bodies(project, nation, fake_install):
    from eu4unitgen.core.recipe import TextureRecipe
    from eu4unitgen.library.scanner import build_index

    make_textures_decodable(fake_install)
    # a made-up body like every non-western base-game body: no artillery_attack id
    anims = " ".join(f'animation = {{ id = "{i}" type = "inf_idle_animation" }}' for i in
                     ("idle", "walk", "attack", "death", "retreat", "horse_idle", "horse_walk", "horse_attack",
                      "horse_death", "horse_retreat"))
    (fake_install / "interface/assets/zz_eastern.gfx").write_text(
        f'objectTypes = {{\n\tpdxmesh = {{ name = "eastern_01" file = "gfx/models/Units/generic_03.mesh" {anims} '
        'scale = 0.5 }\n}\n', encoding="latin-1")
    files = GameFiles(fake_install)
    try:
        idx = build_index(files)
        lo = Loadout.from_json({"schema": 1, "bodies": {"2": "generic_01", "3": "eastern_01", "4": "generic_01"},
                                "slots": {"2/infantry": {"parts": [{"kind": "hoplon"}]}}})
        recipe = TextureRecipe.from_json({"schema": 1, "slots": {"4/infantry": {"ops": [
            {"op": "hue_shift", "region": "cloth", "degrees": 40}]}}})
        gen = generate(project, nation, files, idx, recipe=recipe, loadout=lo, runner=FakeRunner())
        ents = _entities(gen)
        assert ents["MAC_CAVALRY_2_rider"]["pdxmesh"] == "generic_01"
        assert ents["MAC_INFANTRY_2"]["pdxmesh"] == "hr_mac_infantry_2_mesh"  # body generic_01 + spear_shield stance
        gfx = gen.files[N.pdxmesh_file("mac")].text
        assert 'file = "gfx/models/Units/generic_01.mesh"' in gfx[gfx.index('"hr_mac_infantry_2_mesh"'):]
        # the eastern-style body gets artillery_attack (= its attack) for the gun crew; its rider keeps it as it is
        assert ents["MAC_INFANTRY_3"]["pdxmesh"] == "hr_mac_infantry_3_mesh"
        block = gfx[gfx.index('"hr_mac_infantry_3_mesh"'):].split("pdxmesh = {")[0]
        assert 'id = "artillery_attack" type = "inf_idle_animation"' in block
        assert ents["MAC_CAVALRY_3_rider"]["pdxmesh"] == "eastern_01"
        # tier 4: the recipe retextures the chosen body, not the culture's tier-4 one
        block4 = gfx[gfx.index('"hr_mac_infantry_4_mesh"'):]
        assert 'file = "gfx/models/Units/generic_01.mesh"' in block4.split("pdxmesh = {")[0]
        assert ents["MAC_CAVALRY_4_rider"]["pdxmesh"] == "hr_mac_infantry_4_mesh"
        report = validate(project, nation, gen, files, idx)
        assert not report.errors, [str(i) for i in report.errors]
        warns = [str(i) for i in report.warnings if i.rule == "loadout"]
        assert any("MAC_INFANTRY_2 carries a shield and nothing in its right hand" in w for w in warns), warns
        with pytest.raises(GenerationError, match="isn't a base-game pdxmesh"):
            generate(project, nation, files, idx, loadout=Loadout.from_json({"schema": 1, "bodies": {"1": "nope"}}))
    finally:
        files.close()


def test_body_candidates_on_the_fake_install(fake_install, tmp_path):
    from eu4unitgen.export.pipeline import Session, Settings
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path))
    try:
        assert [b["pdxmesh"] for b in s.body_candidates()] == ["generic_01", "generic_02", "generic_03", "generic_04"]
    finally:
        s.close()
