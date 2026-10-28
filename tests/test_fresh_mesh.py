"""Test 15, step 9: a fresh mesh written onto a base mesh's skeleton (made-up 3-influence base,
like the base-game horse's skin)."""
import pytest

from eu4unitgen.core.fresh_mesh import FreshGeometry, FreshMeshError, check_fresh, influences, write_fresh
from eu4unitgen.library.scanner import parse_pdx_mesh, write_pdx_mesh

BONES = ["Root", "HorseSpineTop", "HorseHead", "Saddle_Node"]


def horse_like() -> bytes:
    ident = [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    bones = [{"name": b, "props": {"ix": [i], **({"pa": [0]} if i else {}), "tx": ident}, "children": []}
             for i, b in enumerate(BONES)]
    mesh = {"name": "mesh", "props": {"p": [0.0, 8, 0, 1, 8, 0, 0, 9, 0], "n": [0.0, 0, -1] * 3,
                                      "ta": [1.0, 0, 0, 1] * 3, "u0": [0.0, 0, 1, 0, 0, 1], "tri": [0, 2, 1]},
            "children": [{"name": "aabb", "props": {"min": [0.0, 8, 0], "max": [1.0, 9, 0]}, "children": []},
                         {"name": "material", "props": {"shader": ["PdxMeshColor"], "diff": ["h_d.dds"],
                                                        "n": ["h_n.dds"], "spec": ["h_s.dds"]}, "children": []},
                         {"name": "skin", "props": {"bones": [3], "ix": [1, -1, -1, -1] * 3, "w": [1.0, 0, 0, 0] * 3},
                          "children": []}]}
    return write_pdx_mesh({"name": "root", "props": {"pdxasset": [1, 0]}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": "horseShape", "props": {}, "children": [mesh, {"name": "skeleton", "props": {}, "children": bones}]}]},
        {"name": "locator", "props": {}, "children": []}]})


def quad(weights=None) -> FreshGeometry:
    """Two triangles, normals -Z (like the base's), wound the other way round to test reorienting."""
    return FreshGeometry(p=[0.0, 0, 0, 2, 0, 0, 2, 3, 0, 0, 3, 0], n=[0.0, 0, -1] * 4,
                         u0=[0.0, 1, 1, 1, 1, 0, 0, 0], tri=[0, 1, 2, 0, 2, 3],
                         weights=weights or [{"HorseSpineTop": 1.0}, {"HorseSpineTop": 0.5, "HorseHead": 0.5},
                                             {"HorseHead": 1.0}, {"Root": 0.2, "HorseHead": 0.3, "HorseSpineTop": 0.5}])


def test_fresh_geometry_on_the_base_skeleton():
    base = horse_like()
    out = write_fresh(base, quad(), {"diffuse": "hr_x_horse_1_diffuse.dds"})
    assert check_fresh(base, out) == []
    tree = parse_pdx_mesh(out)
    mesh = tree["children"][0]["children"][0]["children"][0]
    props = mesh["props"]
    assert len(props["p"]) == 12 and len(props["ta"]) == 16 and len(props["tri"]) == 6
    skin = next(c for c in mesh["children"] if c["name"] == "skin")["props"]
    assert skin["bones"] == [3] and skin["ix"][:8] == [1, -1, -1, -1, 1, 2, -1, -1]
    assert skin["ix"][12:16] == [1, 2, 0, -1] and skin["w"][12:16] == pytest.approx([0.5, 0.3, 0.2, 0.0])
    box = next(c for c in mesh["children"] if c["name"] == "aabb")["props"]
    assert box["min"] == [0.0, 0.0, 0.0] and box["max"] == [2.0, 3.0, 0.0]
    material = next(c for c in mesh["children"] if c["name"] == "material")["props"]
    assert material["diff"] == ["hr_x_horse_1_diffuse.dds"] and material["n"] == ["h_n.dds"]
    # the base's winding: its triangle (0,2,1) faces along its -Z normals, so ours are turned to match
    assert props["tri"] == [0, 2, 1, 0, 3, 2]
    assert influences(base) == 3


def test_fresh_mesh_refuses_bad_geometry_and_check_catches_tampering():
    base = horse_like()
    with pytest.raises(FreshMeshError, match="isn't in the skeleton"):
        write_fresh(base, quad([{"tail": 1.0}] * 4))
    with pytest.raises(FreshMeshError, match="base mesh allows 3"):
        write_fresh(base, quad([{"Root": 0.25, "HorseSpineTop": 0.25, "HorseHead": 0.25, "Saddle_Node": 0.25}] * 4))
    with pytest.raises(FreshMeshError, match="sum"):
        write_fresh(base, quad([{"Root": 0.5}] * 4))
    with pytest.raises(FreshMeshError, match="outside"):
        write_fresh(base, FreshGeometry(**{**quad().__dict__, "tri": [0, 1, 9]}))

    tree = parse_pdx_mesh(write_fresh(base, quad()))
    skel = tree["children"][0]["children"][0]["children"][1]
    skel["children"][2]["props"]["tx"] = [2.0] * 12
    assert any("skeleton differs" in p for p in check_fresh(base, write_pdx_mesh(tree)))


def test_a_fresh_horse_through_the_generator_and_validator(fake_install, tmp_path):
    """Test 15 step 10: a loadout's fresh horse becomes the tier's cavalry pdxmesh (the base horse's
    animations and scale), with its .mesh and procedural DDS textures in the set's model folder;
    rule fresh_mesh passes; the checklist has its lines. Tampering is caught."""
    import json

    from PIL import Image

    from eu4unitgen.core.loadout import Loadout, LoadoutError
    from eu4unitgen.export.pipeline import Session, Settings, SetSpec
    from eu4unitgen.export.testmod import loadout_checklist
    from fakes import make_textures_decodable

    make_textures_decodable(fake_install)
    (fake_install / "gfx/models/Units/horse.mesh").write_bytes(horse_like())  # textures h_d/h_n/h_s, as the fake
    geo = quad()
    (tmp_path / "geo.json").write_text(json.dumps(geo.__dict__), encoding="utf-8")
    for kind in ("diffuse", "specular", "normal"):
        Image.new("RGBA", (64, 64), (120, 30, 30, 255)).save(tmp_path / f"{kind}.png")
    doc = {"schema": 1, "horses": {"1": {"base": "horse", "geometry": "geo.json", "rights": "original (test)",
                                         "textures": {k: f"{k}.png" for k in ("diffuse", "specular", "normal")}}}}
    lo = Loadout.from_json(doc, tmp_path)
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out"))
    try:
        res = s.validate(SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=lo))[0]
    finally:
        s.close()
    assert not res.report.errors, [str(i) for i in res.report.errors]
    files = res.generated.files
    assert "gfx/models/units/hr_mac/hr_mac_fresh_horse_1.mesh" in files
    assert all(f"gfx/models/units/hr_mac/hr_mac_fresh_horse_1_{k}.dds" in files for k in ("diffuse", "specular", "normal"))
    text = "".join(f.text for p, f in files.items() if p.endswith(".asset"))
    cav = text.split('name = "MAC_CAVALRY_1"', 1)[1]
    assert cav.split('pdxmesh = "', 1)[1].split('"', 1)[0] == "hr_mac_fresh_horse_1_mesh"
    gfx = "".join(f.text for p, f in files.items() if p.endswith(".gfx"))
    block = gfx.split('name = "hr_mac_fresh_horse_1_mesh"', 1)[1].split("pdxmesh = {", 1)[0]
    assert 'id = "horse_walk"' in block and "scale = 0.48" in block
    assert any(i.rule == "fresh_mesh" and "rights: original (test)" in i.message for i in res.report.issues)
    text = loadout_checklist(res.generated, "eu4unitgen_test", None)
    assert "1H0 cavalry horse: a FRESH mesh" in text and "1H2 rider seating" in text and "skeleton, bone" in text

    with pytest.raises(LoadoutError, match="rights"):
        Loadout.from_json({"schema": 1, "horses": {"1": {**doc["horses"]["1"], "rights": ""}}}, tmp_path)


def body_like() -> bytes:
    """The fake install's infantry body (generic_01: bones root, head, the hand nodes) with one weighted triangle."""
    from fakes import INF_BONES
    tree = parse_pdx_mesh(horse_like())
    shape = tree["children"][0]["children"][0]
    shape["children"][1]["children"] = [
        {"name": b, "props": {"ix": [i], **({"pa": [0]} if i else {}), "tx": [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]},
         "children": []} for i, b in enumerate(INF_BONES)]
    mat = shape["children"][0]["children"][1]["props"]
    mat.update({"diff": ["g1_d.dds"], "n": ["g_n.dds"], "spec": ["g_s.dds"]})
    return write_pdx_mesh(tree)


def test_a_fresh_soldier_on_foot(fake_install, tmp_path):
    """A soldier from a picture as infantry: its body on the infantry entity (so also the gun crew), the infantry's
    animation ids kept, its textures and .mesh named by naming.py; rule fresh_mesh passes."""
    import json

    from PIL import Image

    from eu4unitgen.core.loadout import Loadout
    from eu4unitgen.export.pipeline import Session, Settings, SetSpec
    from fakes import make_textures_decodable

    make_textures_decodable(fake_install)
    (fake_install / "gfx/models/Units/generic_01.mesh").write_bytes(body_like())
    geo = quad([{"head": 1.0}, {"head": 0.5, "root": 0.5}, {"root": 1.0}, {"root": 1.0}])
    (tmp_path / "geo.json").write_text(json.dumps(geo.__dict__), encoding="utf-8")
    for kind in ("diffuse", "specular", "normal"):
        Image.new("RGBA", (64, 64), (40, 60, 160, 255)).save(tmp_path / f"{kind}.png")
    lo = Loadout.from_json({"schema": 1, "soldiers": {"1": {"base": "generic_01", "geometry": "geo.json",
                                                            "rights": "original (test)",
                                                            "textures": {k: f"{k}.png" for k in ("diffuse", "specular", "normal")}}}},
                           tmp_path)
    assert lo.to_json()["soldiers"]["1"]["base"] == "generic_01" and lo.changes_anything()
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out"))
    try:
        res = s.validate(SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=lo))[0]
    finally:
        s.close()
    assert not res.report.errors, [str(i) for i in res.report.errors]
    files = res.generated.files
    assert "gfx/models/units/hr_mac/hr_mac_fresh_soldier_1.mesh" in files
    assert all(f"gfx/models/units/hr_mac/hr_mac_fresh_soldier_1_{k}.dds" in files for k in ("diffuse", "specular", "normal"))
    text = "".join(f.text for p, f in files.items() if p.endswith(".asset"))
    inf = text.split('name = "MAC_INFANTRY_1"', 1)[1]
    assert inf.split('pdxmesh = "', 1)[1].split('"', 1)[0] == "hr_mac_fresh_soldier_1_mesh"
    gfx = "".join(f.text for p, f in files.items() if p.endswith(".gfx"))
    block = gfx.split('name = "hr_mac_fresh_soldier_1_mesh"', 1)[1].split("pdxmesh = {", 1)[0]
    assert 'id = "walk"' in block and 'id = "horse_idle"' in block
    assert any(i.rule == "fresh_mesh" and "fresh soldier" in i.message for i in res.report.issues)



def test_a_fresh_soldier_keeps_its_stance(fake_install, tmp_path):
    """The stance's pdxmesh is replaced by the fresh soldier's: the stance (its animation ids and the validator's
    record of it) moves with it (the cataphract on foot, 2026-10-28)."""
    import json

    from PIL import Image

    from eu4unitgen.core.loadout import Loadout
    from eu4unitgen.export.pipeline import Session, Settings, SetSpec
    from fakes import make_textures_decodable

    make_textures_decodable(fake_install)
    (fake_install / "gfx/models/Units/generic_01.mesh").write_bytes(body_like())
    geo = quad([{"head": 1.0}, {"head": 0.5, "root": 0.5}, {"root": 1.0}, {"root": 1.0}])
    (tmp_path / "geo.json").write_text(json.dumps(geo.__dict__), encoding="utf-8")
    for kind in ("diffuse", "specular", "normal"):
        Image.new("RGBA", (64, 64), (40, 60, 160, 255)).save(tmp_path / f"{kind}.png")
    lo = Loadout.from_json({"schema": 1, "slots": {"1/infantry": {"parts": [], "stance": "spear_shield"}},
                            "soldiers": {"1": {"base": "generic_01", "geometry": "geo.json", "rights": "original (test)",
                                               "textures": {k: f"{k}.png" for k in ("diffuse", "specular", "normal")}}}},
                           tmp_path)
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out"))
    try:
        res = s.validate(SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=lo))[0]
    finally:
        s.close()
    assert not res.report.errors, [str(i) for i in res.report.errors]
    gfx = "".join(f.text for p, f in res.generated.files.items() if p.endswith(".gfx"))
    block = gfx.split('name = "hr_mac_fresh_soldier_1_mesh"', 1)[1].split("pdxmesh = {", 1)[0]
    assert "spear_shield_idle_animation" in block and "spear_shield_attack_animation" in block
