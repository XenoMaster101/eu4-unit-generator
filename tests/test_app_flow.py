"""Track P2: the app's flow (ref3d.flow), edits (ref3d.edits), Add to mod's checksum and filters (export.modexport),
plain errors (core.errors), setup steps (core.setup), the input guide (core.guide) and the licences list."""
import json
from pathlib import Path

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
    data, kept, _ = modexport.entities_only(asset.encode(), "westerngfx", 1, "BYZ", "infantry")  # a soldier on foot
    assert kept == 1 and 'name = "westerngfx_INFANTRY_1"' in data.decode() and "CAVALRY" not in data.decode()


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


def test_cavalry_joins_a_finished_horse_and_soldier(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path))
    img = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(img)
    flow.new_request("bay", img, "my horse picture", "horse")
    s = flow.new_request("guard", img, "my soldier picture", "soldier", 2)
    flow.set_edits(s, {"shield_side": "left", "candidate": 2})
    try:
        flow.new_request("half", None, "", "cavalry", horse="bay")
    except ValueError:
        pass
    else:
        raise AssertionError("cavalry without a soldier")
    c = flow.new_request("guard_on_bay", None, "", "cavalry", 2, horse="bay", soldier="guard")
    req = flow.request(c)
    assert req["rights"] == "horse: my horse picture; soldier: my soldier picture"
    assert req["edits"]["shield_side"] == "left" and "candidate" not in req["edits"]  # the soldier's look, its own model
    assert flow.stages_of("cavalry") == ("build", "finish", "preview") and flow.stages_of("soldier") == flow.STAGES
    assert flow.soldier_glb(req) == flow.folder_of("guard") / "trellis" / "raw.glb"
    assert flow.FOOT_TYPE["soldier"] == "infantry" and flow.body_kind("cavalry") == "rider"
    assert flow.estimates("soldier")["build"] == flow.DEFAULT_SECONDS["build_rider"]


def test_pick_weighs_outline_colour_and_stretch():
    from eu4unitgen.ref3d import trellis
    neat = {"iou": 0.80, "colour_de": 12.0, "worst_stretch": 1.7, "vanilla_worst": 1.6}
    torn = {"iou": 0.83, "colour_de": 12.0, "worst_stretch": 3.4, "vanilla_worst": 1.6}  # a better outline, but it tears
    off = {"iou": 0.82, "colour_de": 30.0, "worst_stretch": 1.6, "vanilla_worst": 1.6}  # a better outline, wrong colours
    assert trellis.pick_score(neat) > trellis.pick_score(torn)
    assert trellis.pick_score(neat) > trellis.pick_score(off)
    assert trellis.pick_score({"iou": 0.7}) == 0.7  # scored by outline alone (an older run)
    assert flow.candidate_text(neat) == "outline 80%, colour distance 12, worst stretch 1.70"


def test_colour_distance_reads_the_subjects_colours():
    from eu4unitgen.ref3d import trellis
    ref = Image.new("RGBA", (80, 120), (0, 0, 0, 0))
    ImageDraw.Draw(ref).rectangle([20, 10, 60, 110], fill=(160, 30, 30, 255))
    same = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # bigger and placed elsewhere: framed by its box
    ImageDraw.Draw(same).rectangle([10, 20, 70, 170], fill=(160, 30, 30, 255))
    blue = same.copy()
    ImageDraw.Draw(blue).rectangle([10, 20, 70, 170], fill=(30, 40, 160, 255))
    assert trellis.colour_distance(same, ref) < 1.0
    assert trellis.colour_distance(blue, ref) > 40
    render = Image.new("RGBA", (200, 200), (225, 225, 228, 255))  # a render on its grey background, no alpha
    ImageDraw.Draw(render).rectangle([10, 20, 70, 170], fill=(160, 30, 30, 255))
    assert trellis.colour_distance(render, ref) < 1.0
    assert trellis.colour_distance(Image.new("RGBA", (10, 10)), ref) is None


# the hands' frames measured with rider_frames on western_generic_01_type1 (2026-10-28): seated (cavalry_rider_idle)
# and on foot (spear_shield_idle); world out, up and forward in each hand node's own axes
SEATED = {"Right_hand_node": {"out": [-0.95, -0.3, 0.08], "up": [-0.27, 0.92, 0.27], "forward": [-0.15, 0.24, -0.96]},
          "Left_hand_node": {"out": [-0.44, 0.26, 0.86], "up": [-0.2, 0.9, -0.38], "forward": [0.88, 0.34, 0.34]}}
FOOT = {"Right_hand_node": {"out": [-0.86, -0.51, 0.03], "up": [0.08, -0.08, 0.99], "forward": [-0.51, 0.85, 0.11]},
        "Left_hand_node": {"out": [0.8, 0.1, 0.59], "up": [0.01, 0.98, -0.18], "forward": [0.6, -0.16, -0.79]}}


def test_a_soldiers_hand_parts_follow_the_base_game_on_foot():
    """The cataphract on foot (2026-10-28): its lance (point forward when seated: -z) hung point down and its kite
    sat behind the body. On foot the point is along +z (muslim_spear's, straight up in spear_shield_idle), the
    shield faces +x with its back at the hand (indian_generic_01_type1_shield's)."""
    from eu4unitgen.ref3d import trellis
    s, f = SEATED["Right_hand_node"], FOOT["Right_hand_node"]
    # a lance through the hand, its point ahead of the seated rider (along forward) and a longer butt behind
    lance = [s["forward"][k] * t for t in range(-8, 11) for k in range(3)]
    q = trellis.foot_part(lance, "mesh_weapon", s, f)
    zs = q[2::3]
    assert max(zs) > 9.5 and min(zs) < -7.5  # the point (10 ahead) now up along +z, the butt below, the grip at 0
    assert max(abs(x) for x in q[0::3] + q[1::3]) < 0.1
    up = [sum(q[i + j] * f["up"][j] for j in range(3)) for i in range(0, len(q), 3)]
    assert max(up) > 9.5  # straight up in the standing pose
    s, f = SEATED["Left_hand_node"], FOOT["Left_hand_node"]
    # a flat kite 5 tall, 3 wide, facing out from the seated rider, 1 out from the forearm
    plate = []
    for a in (-1.5, 1.5):
        for b in (-2.5, 2.5):
            plate += [s["out"][k] * 1.0 + s["forward"][k] * a + s["up"][k] * b for k in range(3)]
    q = trellis.foot_part(plate, "mesh_shield", s, f)
    xs, ys, zs = q[0::3], q[1::3], q[2::3]
    assert max(xs) - min(xs) < 0.05 and abs(min(xs) - trellis.FOOT["shield_back_x"]) < 1e-6  # its face along +x
    assert abs((min(ys) + max(ys)) / 2 - trellis.FOOT["shield_centre_y"]) < 1e-6 and max(ys) - min(ys) > 4.5
    assert abs(min(zs) + max(zs)) < 1e-6


def test_the_regression_set_expects_no_shield_and_a_bare_coat():
    """The English soldier (no shield, a poleaxe) and horse 2 with its country mask on (the bay coat stays out) are in
    the regression set; their rules hold whatever the baseline says."""
    from eu4unitgen.ref3d import regress
    assert regress.SET["horse2_mask"]["from"] == "horse2" and regress.source("horse2_mask").name == "horse2"
    assert regress.expectations("english_soldier", {"parts": ["lance"]}) == []
    assert regress.expectations("english_soldier", {"parts": ["lance", "shield"]}) == ["parts: has shield, must not"]
    assert regress.expectations("horse2_mask", {"mask_share": 0.4}) == ["mask_share: 0.4, must be at most 0.05"]
    assert regress.expectations("horse2_mask", {}) == []  # no mask at all: nothing was taken for cloth


def test_an_ai_step_waits_for_ram_and_names_what_to_close():
    from eu4unitgen.ref3d import flow
    assert flow.ram_short("trellis", free=20.0) is None and flow.ram_short("preview", free=0.5) is None
    s = flow.ram_short("trellis", free=3.2, top=[("chrome", 4.1), ("vmmemWSL", 9.0), ("blender.exe", 2.0),
                                                 ("Discord.exe", 1.2), ("code", 0.9)])
    assert s["need"] == 14.0 and [n for n, _ in s["close"]] == ["chrome", "Discord.exe", "code"]  # not the app's own


def test_achievement_safe_takes_every_tier_of_a_unit():
    asset = "".join(f'entity = {{\n\tname = "BYZ_CAVALRY_{t}_rider"\n\tpdxmesh = "r{t}"\n}}\n'
                    f'entity = {{\n\tname = "BYZ_CAVALRY_{t}"\n\tpdxmesh = "h{t}"\n'
                    f'\tattach = {{ name = "rider" Saddle_Node = "BYZ_CAVALRY_{t}_rider" }}\n}}\n' for t in (1, 2, 3))
    data, kept, of = modexport.entities_only(asset.encode(), "westerngfx", [1, 3], "BYZ")
    text = data.decode()
    assert kept == 4 and of == 6 and "westerngfx_CAVALRY_3_rider" in text and "CAVALRY_2" not in text


def test_a_soldier_rides_with_its_seated_parts(tmp_path):
    """On foot a hand part is the seated one turned and moved (same triangles): the rider gets the seated file back,
    from the record (new builds) or by its triangles (older ones)."""
    from eu4unitgen.ref3d import assemble
    seated = {"p": [0, 0, 0, 1, 0, 0, 0, 1, 0], "tri": [0, 1, 2]}
    (tmp_path / "lance_part.json").write_text(json.dumps(seated), encoding="utf-8")
    (tmp_path / "kite_part.json").write_text(json.dumps({"p": [0] * 12, "tri": [0, 1, 2, 0, 2, 3]}), encoding="utf-8")
    (tmp_path / "lance_foot.json").write_text(json.dumps({"p": [5, 5, 5, 6, 5, 5, 5, 6, 5], "tri": [0, 1, 2]}),
                                              encoding="utf-8")
    foot = {"kind": "mesh_weapon", "mesh": {"geometry": str(tmp_path / "lance_foot.json")}}
    assert assemble.seated_geometry(foot) == str(tmp_path / "lance_part.json")
    rec = {"kind": "mesh_weapon", "mesh": {"geometry": "x_foot.json", "seated_geometry": "kept.json"}}
    assert assemble.seated_geometry(rec) == "kept.json"


def test_a_gun_has_its_own_names_and_loadout_entry():
    from eu4unitgen.core import loadout
    from eu4unitgen.core.naming import Naming
    n = Naming()
    assert n.fresh_gun_pdxmesh("mac", 1) == "hr_mac_fresh_gun_1_mesh"
    assert n.fresh_gun_file("mac", 2).endswith("/hr_mac_fresh_gun_2.mesh")
    assert n.cannon_entity("MAC", 1) == "MAC_ARTILLERY_1_cannon"
    lo = loadout.Loadout.from_json({"schema": 1, "guns": {"1": {
        "geometry": "g.json", "textures": {"diffuse": "d.png", "specular": "s.png", "normal": "n.png"}, "rights": "mine"}}})
    assert lo.guns[1]["base"] == "artillery_mesh" and lo.changes_anything()
    assert lo.to_json()["guns"]["1"]["base"] == "artillery_mesh"


def test_a_starter_part_is_packed_raw_and_fitted_on_first_use(tmp_path, monkeypatch):
    """Packed: the raw model, the cut-out, the crops, the picture, tags and rights; nothing fitted to the game. Fitted:
    a run of its own with the AI steps marked done (only Blender left), the unit waiting for it."""
    from eu4unitgen.core import library
    from eu4unitgen.ref3d import flow
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    started = []
    monkeypatch.setattr(flow, "start", lambda f: started.append(Path(f).name))
    img = tmp_path / "horse.png"
    Image.new("RGB", (64, 40), (120, 80, 40)).save(img)
    f = flow.new_request("bay", img, "mine; bundling allowed", "horse", 1)
    tr = f / "trellis"
    (tr / "raw.glb").write_bytes(b"glTF raw model")
    (tr / "input").mkdir()
    Image.new("RGBA", (8, 8)).save(tr / "input" / "input_rgba.png")
    v = tr / "game" / "v001"
    (v / "compare").mkdir(parents=True)
    Image.new("RGBA", (8, 8)).save(v / "compare" / "raw_side.png")
    (f / "finished").mkdir()
    (f / "finished" / "manifest.json").write_text(json.dumps({"geometry": "x", "textures": {}, "compare": {
        "images": {"raw": {"side": str(v / "compare" / "raw_side.png")}}}}), encoding="utf-8")
    (f / "status.json").write_text(json.dumps({"done": True, "state": "done", "stages": {}}), encoding="utf-8")
    out = library.pack(f, tmp_path / "starter" / "bay", tags={"style": "light"})
    d = json.loads((out / "part.json").read_text(encoding="utf-8"))
    assert d["kind"] == "horse" and d["tags"] == {"style": "light"} and "CC BY 4.0" in d["rights"]
    assert (out / "raw.glb").is_file() and (out / "render.png").is_file() and (out / d["image"]).is_file()
    assert not list(out.rglob("*.mesh")) and not list(out.rglob("*.dds")) and not (out / "low_geometry.json").exists()
    import shutil
    shutil.rmtree(f)  # a PC without the author's run
    monkeypatch.setattr(library, "starter_root", lambda: tmp_path / "starter")
    assert [p.ref for p in library.parts()] == ["starter:bay"]
    g = library.fit("bay")
    st = flow.status(g)
    assert started == ["bay"] and st["stages"]["trellis"]["state"] == "done" and not st["done"]
    assert (g / "trellis" / "raw.glb").read_bytes() == b"glTF raw model"
    assert flow.request(g)["tags"] == {"style": "light"} and "CC BY 4.0" in flow.request(g)["rights"]
    assert [p.ref for p in library.parts()] == ["run:bay"]  # now a part of your own: the starter is hidden
    u = library.new_unit("bays", "cavalry", {1: {"horse": "starter:bay", "soldier": "run:bay"}})
    assert flow.request(u)["tiers"]["1"]["horse"] == "run:bay" and flow.waiting_for(flow.request(u)) == ["bay"]
