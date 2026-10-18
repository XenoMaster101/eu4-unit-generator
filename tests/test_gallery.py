"""F9: whole-unit gallery previews. The entity tree the game assembles (horse -> rider at
Saddle_Node, artillery base -> crew and cannon, parts on bones), staged with tinted textures for
the render_unit Blender job; one job renders every unit. Real Blender: test_real_blender_gallery."""
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from eu4unitgen.blender.runner import JobResult
from eu4unitgen.export.gallery import entity_tree, stage
from eu4unitgen.export.pipeline import Session, Settings, SetSpec
from eu4unitgen.validate.graph import World
from fakes import make_textures_decodable
from test_parts import FakeRunner


class GalleryRunner(FakeRunner):
    """Builds parts like test_parts' fake; 'renders' each unit as a flat tile and keeps its args."""

    def run_job(self, job, args, timeout=None):
        if job != "render_unit":
            return super().run_job(job, args, timeout)
        self.calls.append((job, args))
        tiles, thumbs = {}, {}
        for u in args["units"]:
            path = Path(args["out_dir"]) / f"{u['name']}.png"
            Image.new("RGB", (args["size"], args["size"]), (90, 120, 150)).save(path)
            tiles[u["name"]] = str(path)
            todo = [u["root"]]
            while todo and args.get("thumbs"):
                n = todo.pop()
                for a in n["attach"]:
                    todo.append(a["node"])
                    if not a["node"]["anim"]:
                        t = Path(args["out_dir"]) / f"thumb_{a['node']['name']}.png"
                        Image.new("RGB", (args["thumbs"], args["thumbs"]), (200, 180, 90)).save(t)
                        thumbs[a["node"]["name"]] = str(t)
        return JobResult(job, 0, {"tiles": tiles, "thumbs": thumbs,
                                  "seconds": {u["name"]: 0.1 for u in args["units"]}}, "fake log")


@pytest.fixture
def session(fake_install, tmp_path):
    make_textures_decodable(fake_install)
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out"))
    yield s
    s.close()


def _loadout(tmp_path):
    from eu4unitgen.core.loadout import Loadout

    return Loadout.from_json({"schema": 1, "slots": {
        "1/infantry": {"parts": [{"kind": "dory"}, {"kind": "hoplon"}]},
        "1/rider": {"parts": [{"kind": "xyston"}, {"kind": "crest"}]}}}, tmp_path)


def test_entity_trees_follow_attaches(session, tmp_path, monkeypatch):
    runner = GalleryRunner()
    monkeypatch.setattr(session, "runner", lambda: runner)
    spec = SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=_loadout(tmp_path))
    res = session.validate(spec)[0]
    game, idx = session.game()
    world = World(game, idx, None, res.generated)

    inf = entity_tree(world, "MAC_INFANTRY_1")
    assert inf.anim_file and {b for b, _ in inf.attach} == {"Right_hand_node", "Left_hand_node"}
    assert {c.entity for _, c in inf.attach} == {"hr_mac_dory_1", "hr_mac_hoplon_1"}
    assert all(c.anim_file is None and c.scale == 0.5 for _, c in inf.attach)  # rigid parts at vanilla scale

    cav = entity_tree(world, "MAC_CAVALRY_1")
    (bone, rider), = cav.attach
    assert bone == "Saddle_Node" and rider.entity == "MAC_CAVALRY_1_rider" and rider.anim_file
    assert {c.entity for _, c in rider.attach} == {"hr_mac_xyston_rider_1", "hr_mac_crest_rider_1"}

    art = entity_tree(world, "MAC_ARTILLERY_1")
    assert {b for b, _ in art.attach} == {"infantry_node", "cannon_node"}

    staged = stage(cav, world, [(200, 0, 0), (0, 200, 0), (0, 0, 200)], tmp_path / "stage")
    assert Path(staged["mesh"]).is_file() and staged["attach"][0]["bone"] == "Saddle_Node"
    rider_dir = Path(staged["attach"][0]["node"]["mesh"]).parent
    assert any(p.suffix == ".dds" for p in rider_dir.iterdir())  # the rider's tinted diffuse beside its mesh
    json.dumps(staged)  # the job's args are plain JSON


def test_gallery_renders_every_unit_in_one_job(session, tmp_path, monkeypatch):
    runner = GalleryRunner()
    monkeypatch.setattr(session, "runner", lambda: runner)
    spec = SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=_loadout(tmp_path))
    g = session.gallery(spec, size=64)
    assert len(g.images) == 12 and not [n for n in g.notes if " tier " in n], g.notes  # every unit assembled
    assert g.images[("cavalry", 4)].size == (64, 64) and not g.result.report.errors
    renders = [a for job, a in runner.calls if job == "render_unit"]
    assert len(renders) == 1 and len(renders[0]["units"]) == 12  # one Blender run for the lot
    some = session.gallery(spec, [("infantry", 1), ("cavalry", 1)], size=32, result=g.result, thumbs=24)
    assert list(some.images) == [("infantry", 1), ("cavalry", 1)]  # no second validation
    assert some.parts[("infantry", 1)] == {"Right_hand_node": "hr_mac_dory_1", "Left_hand_node": "hr_mac_hoplon_1"}
    assert some.parts[("cavalry", 1)] == {"Right_hand_node": "hr_mac_xyston_rider_1", "head": "hr_mac_crest_rider_1"}
    assert some.thumbs["hr_mac_dory_1"].size == (24, 24) and "hr_mac_crest_rider_1" in some.thumbs


def test_fingerprints_change_only_for_units_an_edit_changed(session, tmp_path, monkeypatch):
    """Track C, C-1: a unit's fingerprint covers what its tile shows, so an edit re-renders only
    the units it changed."""
    from eu4unitgen.core.loadout import Loadout

    runner = GalleryRunner()
    monkeypatch.setattr(session, "runner", lambda: runner)
    spec = SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=_loadout(tmp_path))
    before = session.gallery_prints(spec, session.validate(spec)[0])
    assert len(before) == 12 and not [v for v in before.values() if v.startswith("error")]
    assert session.gallery_prints(spec, session.validate(spec)[0]) == before  # stable

    edited = Loadout.from_json({"schema": 1, "slots": {
        "1/infantry": {"parts": [{"kind": "dory"}, {"kind": "hoplon"}]},
        "1/rider": {"parts": [{"kind": "xiphos"}, {"kind": "crest"}]}}}, tmp_path)  # the rider's weapon only
    after = session.gallery_prints(SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=edited),
                                   session.validate(SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=edited))[0])
    assert [k for k in before if before[k] != after[k]] == [("cavalry", 1)]

    recoloured = SetSpec(tags=["MAC"], gfx="hellenicgfx", loadout=_loadout(tmp_path),
                         unit_colours=[(10, 20, 30), (200, 200, 0), (0, 0, 0)])
    again = session.gallery_prints(recoloured, session.validate(recoloured)[0])
    assert all(again[k] != before[k] for k in before)  # unit colours tint every tile


@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="needs the real install (EU4_PATH) and Blender")
def test_real_blender_gallery(tmp_path):
    from eu4unitgen.core.character import CharacterSpec
    from eu4unitgen.core.look import Look, compile_look

    s = Session(Settings(eu4_path=Path(os.environ["EU4_PATH"]), output_dir=tmp_path / "out"))
    try:
        base = {"schema": 1, "set": {"tags": ["ATH"], "gfx": "hellenicgfx", "set_name": "ath"}}
        spec = CharacterSpec.from_json(compile_look(Look.from_template("hoplite"), base), tmp_path).set_spec()
        g = s.gallery(spec, [("infantry", 4), ("cavalry", 1), ("artillery", 1)], size=128)
        assert len(g.images) == 3 and not g.notes, g.notes
        for img in g.images.values():  # something drawn on the plain background
            assert len(img.getcolors(1 << 16) or []) > 50
    finally:
        s.close()
