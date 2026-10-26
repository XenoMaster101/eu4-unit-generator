"""Track P2: the app's flow (ref3d.flow), edits (ref3d.edits), Add to mod's checksum and filters (export.modexport),
plain errors (core.errors), setup steps (core.setup), the input guide (core.guide) and the licences list."""
import json

from PIL import Image, ImageDraw

from eu4unitgen.core import errors, guide, licences, setup
from eu4unitgen.export import modexport
from eu4unitgen.ref3d import edits, flow


def test_edits_route_to_the_cheapest_stage():
    assert edits.stage_for({}, {}) is None
    assert edits.stage_for({}, {"brightness": 1.1}) == "finish"
    assert edits.stage_for({}, {"plume": False}) == "finish"
    assert edits.stage_for({}, {"hem": 0.8, "cloth_colour": [10, 20, 30]}) == "finish"
    assert edits.stage_for({}, {"tassels": False}) == "build"
    assert edits.stage_for({"brightness": 1.2}, {"brightness": 1.2, "shield_side": "right"}) == "build"
    assert edits.build_options({"tassels": False, "country_mask": True}, "horse") == {"cfg": {"tassels": False},
                                                                                       "country_mask": True}
    assert edits.build_options({"shield_side": "left"}, "rider")["shield_side"] == "left"


def _version(tmp_path, kind="rider"):
    v = tmp_path / "v001"
    v.mkdir()
    d = Image.new("RGBA", (16, 16), (100, 100, 100, 255))
    d.paste((150, 30, 30, 255), (0, 0, 8, 16))
    d.save(v / "d.png")
    Image.new("RGBA", (16, 16), (0, 0, 0, 5)).save(v / "s.png")
    Image.new("RGBA", (16, 16), (128, 128, 255, 255)).save(v / "n.png")
    # one triangle all on the red cloth (u under 0.5), one reaching the grey (a leg): only the first's lift
    geo = {"p": [0.0, 5.0, 0.0, 0.0, 12.0, 0.0, 1.0, 5.0, 0.0, 2.0, 5.0, 0.0], "tri": [0, 1, 2, 2, 1, 3],
           "u0": [0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.9, 0.5]}
    (v / "g.json").write_text(json.dumps(geo))
    parts = {"plume": {"kind": "mesh_head", "mesh": {"geometry": str(v / "g.json"), "diffuse": str(v / "d.png")}},
             "lance": {"kind": "mesh_weapon", "mesh": {"geometry": str(v / "g.json"), "diffuse": str(v / "d.png")}}}
    m = {"textures": {"diffuse": str(v / "d.png"), "specular": str(v / "s.png"), "normal": str(v / "n.png")},
         "geometry": str(v / "g.json"), "parts": parts, "palette": {"cloth": {"colour": [150, 30, 30]}}, "colour": {}}
    (v / "manifest.json").write_text(json.dumps(m))
    return v


def test_finish_applies_brightness_plume_hem_and_colour(tmp_path):
    v = _version(tmp_path)
    out = edits.finish(v, {"brightness": 1.2, "plume": False, "hem": 0.5, "cloth_colour": [30, 60, 160]}, "rider",
                       tmp_path / "fin")
    m = json.loads((out / "manifest.json").read_text())
    assert "plume" not in m["parts"] and "lance" in m["parts"]
    d = Image.open(m["textures"]["diffuse"]).convert("RGB")
    assert d.getpixel((12, 4)) == (120, 120, 120)  # brightened
    r, g, b = d.getpixel((2, 4))
    assert b > r  # the red cloth moved to the blue asked for
    p = json.loads(open(m["geometry"]).read())["p"]
    assert p[1] == 9.0 - (9.0 - 5.0) * 0.5 and p[4] == 12.0  # the cloth under the hips lifted, the top untouched
    assert p[7] == 5.0 and p[10] == 5.0  # a vertex whose triangle reaches the leg stays (no shards)
    assert json.loads((v / "manifest.json").read_text())["parts"].get("plume")  # the built version untouched


def test_requests_status_and_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path))
    img = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(img)
    f = flow.new_request("my_horse", img, "mine", "horse", 2)
    assert flow.request(f)["tier"] == 2 and (f / "subject.json").is_file()
    try:
        flow.new_request("my_horse", img, "mine", "horse")
    except FileExistsError:
        pass
    else:
        raise AssertionError("a name used twice")
    (f / "status.json").write_text(json.dumps({"stages": {s: {"state": "done", "seconds": 10} for s in flow.STAGES},
                                               "done": True, "heartbeat": 0}))
    assert flow.set_edits(f, {"brightness": 0.9}) == "finish"
    st = json.loads((f / "status.json").read_text())
    assert st["stages"]["build"]["state"] == "done" and st["stages"]["finish"]["state"] == "waiting" and not st["done"]
    assert [r["name"] for r in flow.runs()] == ["my_horse"]
    assert flow.estimates("horse")["build"] == 10  # this machine's own runs
    assert flow.set_edits(f, {"brightness": 0.9, "tassels": False}) == "build"


def test_checksum_rules_and_files(tmp_path):
    (tmp_path / "checksum_manifest.txt").write_text('directory = {\n\tname = "common"\n\tsub_directories = yes\n'
                                                    '\tfile_extension = ".txt"\n}\ndirectory = {\n\tname = "map"\n'
                                                    '\tsub_directories = no\n\tfile_extension = ".csv"\n}\n')
    rules = modexport.checksum_rules(tmp_path)
    assert rules == [("common", True, ".txt"), ("map", False, ".csv")]
    files = ["common/units/a.txt", "gfx/entities/x.asset", "map/definition.csv", "map/sub/b.csv", "localisation/l.yml"]
    assert modexport.checksum_files(files, rules) == ["common/units/a.txt", "map/definition.csv"]


def test_achievement_safe_keeps_graphics_only():
    rules = [("common", True, ".txt")]
    paths = ["m_units.mod", "m_units/descriptor.mod", "m_units/common/units/u.txt", "m_units/gfx/entities/e.asset",
             "m_units/gfx/sprite_packs/p.txt", "m_units/interface/assets/x.gfx", "m_units/localisation/l.yml",
             "m_units/gfx/models/units/x/h.mesh"]
    keep, _, _ = modexport.keep_graphics(paths, "m_units", "westerngfx", 1, rules)
    assert keep == {"m_units.mod", "m_units/descriptor.mod", "m_units/gfx/entities/e.asset",
                    "m_units/interface/assets/x.gfx", "m_units/gfx/models/units/x/h.mesh"}


def test_entities_only_renames_the_tags_unit_to_the_cultures():
    asset = ('entity = {\n\tname = "BYZ_CAVALRY_1_rider"\n\tpdxmesh = "ours_rider"\n\tattach = { head = "plume_1" }\n}\n'
             'entity = {\n\tname = "BYZ_CAVALRY_1"\n\tpdxmesh = "ours_horse"\n\tversion = 20\n'
             '\tattach = { name = "rider" Saddle_Node = "BYZ_CAVALRY_1_rider" }\n}\n'
             'entity = {\n\tname = "plume_1"\n\tpdxmesh = "plume_mesh"\n}\n'
             'entity = {\n\tname = "westerngfx_CAVALRY_1"\n\tpdxmesh = "cavalry_horse"\n}\n'
             'entity = {\n\tname = "BYZ_INFANTRY_1"\n\tpdxmesh = "x"\n}\n')
    data, kept, of = modexport.entities_only(asset.encode(), "westerngfx", 1, "BYZ")
    text = data.decode()
    assert of == 5 and kept == 3
    assert 'name = "westerngfx_CAVALRY_1"' in text and 'pdxmesh = "ours_horse"' in text
    assert 'Saddle_Node = "westerngfx_CAVALRY_1_rider"' in text and 'name = "plume_1"' in text
    assert "cavalry_horse" not in text and "BYZ_" not in text


def test_errors_in_plain_words():
    e = errors.explain("RuntimeError: TRELLIS.2 failed (exit 1); see raw.log", "trellis")
    assert "3D model" in e["what"] and "Setup" in e["todo"] and "raw.log" in e["details"]
    assert "graphics card" in errors.explain("torch.cuda.OutOfMemoryError: CUDA out of memory")["what"]
    assert "building the game model" in errors.explain("weird", "build")["what"]


def test_setup_plans_need_no_terminal(tmp_path):
    for st in setup.steps():
        if st.key == "ai_folder":
            continue
        cmds = setup.plan(st.key, token="hf_x", root=tmp_path / "ai")
        assert cmds and all(c.download or c.argv for c in cmds)
        for c in cmds:  # no step waits on a prompt in a console: no -NoExit, no sudo
            assert "-NoExit" not in c.argv and not any("sudo" in a for a in c.argv)
    ub = setup.plan("ubuntu", root=tmp_path)
    assert any(c.argv[-3:] == ["root", "-e", "true"] for c in ub)  # set up as root: no Linux user to make
    assert any(c.admin for c in setup.plan("wsl")) and any(c.admin for c in setup.plan("blender"))
    w = setup.plan("weights", token="hf_x", root=tmp_path)
    assert w[0].download[0] == setup.BIREFNET and w[1].env["HF_TOKEN"] == "hf_x"
    assert setup.TRELLIS_COMMIT in setup.plan("trellis_env", root=tmp_path)[0].argv[-1]
    assert setup.drives() and setup.default_ai_root().name


def test_guide_warns_on_the_wrong_view(tmp_path):
    im = Image.new("RGB", (900, 400), (240, 240, 240))
    ImageDraw.Draw(im).rectangle([100, 100, 800, 300], fill=(80, 50, 30))  # wide: a horse's shape
    im.save(tmp_path / "wide.png")
    assert any("front view" in w for w in guide.check(tmp_path / "wide.png", "rider"))
    assert not any("side view" in w for w in guide.check(tmp_path / "wide.png", "horse"))


def test_licences_list():
    md = licences.markdown()
    assert "TRELLIS.2-4B weights | MIT" in md and "DINOv3" in md and "RMBG-2.0" in md
    assert "<table" in licences.html()


def test_setup_detects_and_saves_paths(tmp_path, monkeypatch):
    from eu4unitgen.core import config, install
    from eu4unitgen.gui import setup_page
    monkeypatch.setenv("EU4UNITGEN_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("EU4_PATH", raising=False)
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    monkeypatch.setattr(install, "find_eu4", lambda: tmp_path / "eu4")
    monkeypatch.setattr(install, "find_blender", lambda: (tmp_path / "blender.exe", "found"))
    monkeypatch.setattr("eu4unitgen.core.setup.adopt_ai_env", lambda: None)
    found = setup_page.detect()
    assert found["eu4_path"] == str(tmp_path / "eu4") and found["blender_path"] == str(tmp_path / "blender.exe")
    c = config.load()
    assert c.eu4() == tmp_path / "eu4"
    assert setup_page.detect() == {}  # set already: nothing overwritten


def test_inventory_lists_every_piece_and_what_is_safe():
    from eu4unitgen.core import inventory
    names = " ".join(i.name for i in inventory.ITEMS)
    for must in ("TRELLIS.2-4B", "DINOv3", "SAM 2.1", "BiRefNet", "TripoSG", "SDXL", "SigLIP", "LPIPS", "ControlNet",
                 "IP-Adapter", "DINOv2", "cma", "Blender", "io_pdx_mesh", "WSL2", "CUDA"):
        assert must in names, must
    assert all(not i.safe_to_remove for i in inventory.ITEMS if i.status == "required")
    assert 20 < inventory.freed_gb() < 80
    assert any(i.where.startswith("<root>") for i in inventory.ITEMS if i.status == "required")
    assert "| Status |" in inventory.markdown() and "<table" in inventory.html()


def test_candidates_pick_and_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path))
    img = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(img)
    f = flow.new_request("two", img, "mine", "horse")
    for s, body in ((42, b"model-42"), (43, b"model-43")):
        (f / "trellis" / f"raw_s{s}.glb").write_bytes(body)
    (f / "trellis" / "candidates.json").write_text(json.dumps({"candidates": [
        {"seed": 42, "glb": str(f / "trellis" / "raw_s42.glb"), "iou": 0.70},
        {"seed": 43, "glb": str(f / "trellis" / "raw_s43.glb"), "iou": 0.74}], "best": 1}))
    flow.use_candidate(f, flow.candidates(f)["best"])
    assert (f / "trellis" / "raw.glb").read_bytes() == b"model-43"
    flow.use_candidate(f, 0)
    assert (f / "trellis" / "raw.glb").read_bytes() == b"model-42"
    assert edits.stage_for({}, {"candidate": 1}) == "build"
