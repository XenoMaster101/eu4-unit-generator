"""Track P1.5: the quality report (ref3d/quality.py) and the unit staging helpers (ref3d/unitstage.py)."""
import json

from PIL import Image, ImageDraw

from eu4unitgen.ref3d import quality, unitstage


def _shape(size, box, mirror=False, bg=None):
    im = Image.new("RGBA" if bg is None else "RGB", size, (0, 0, 0, 0) if bg is None else bg)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = box
    pts = [(x0, y1), (x1, y1), (x1, y0), ((x0 + x1) / 2, y0 + (y1 - y0) * 0.4)]  # an asymmetric outline
    if mirror:
        pts = [(x0 + x1 - x, y) for x, y in pts]
    d.polygon(pts, fill=(120, 60, 40, 255) if bg is None else (120, 60, 40))
    return im


def test_silhouette_iou_frames_and_mirrors():
    ref = _shape((400, 300), (50, 40, 350, 260))
    small = _shape((100, 100), (10, 20, 85, 75), mirror=True)  # smaller, elsewhere, mirrored, opaque background
    assert quality.silhouette_iou(small, ref) > 0.9
    other = Image.new("RGBA", (100, 100))
    ImageDraw.Draw(other).ellipse([10, 10, 90, 30], fill=(1, 1, 1, 255))
    assert quality.silhouette_iou(other, ref) < 0.5


def test_map_zoom_brightness_and_contrast(tmp_path):
    for name, col in (("ours", (90, 90, 90)), ("van", (100, 100, 100)), ("bright", (220, 220, 220))):
        im = Image.new("RGB", (80, 80), (200, 210, 230))
        d = ImageDraw.Draw(im)
        d.rectangle([30, 20, 50, 60], fill=col)
        d.rectangle([30, 40, 50, 60], fill=tuple(int(c * 0.6) for c in col))  # some contrast
        im.save(tmp_path / f"{name}.png")
    r = quality.map_zoom(tmp_path / "ours.png", tmp_path / "van.png")
    assert r["verdict"] == "good" and 0.85 < r["brightness_ratio"] < 0.95
    assert quality.map_zoom(tmp_path / "bright.png", tmp_path / "van.png")["verdict"] == "check"
    assert "Not measured" in quality.map_zoom(None, None)["text"]


def test_deformation_verdicts():
    van = {a: {"flips": 12, "max": 7.0} for a in ("idle", "moving", "attack")}
    good = quality.deformation({"ours": {a: {"flips": 15, "max": 3.0} for a in van}, "vanilla": van}, "rider")
    assert good["verdict"] == "good" and "15, 15, 15" in good["text"]
    bad = quality.deformation({"ours": {a: {"flips": 200, "max": 3.0} for a in van}, "vanilla": van}, "rider")
    assert bad["verdict"] == "check"


def test_report_and_markdown(tmp_path):
    ref = _shape((400, 300), (50, 40, 350, 260))
    ref.getchannel("A").save(tmp_path / "fg.png")
    _shape((200, 200), (20, 30, 180, 147)).save(tmp_path / "game.png")
    manifest = {"deform": {"ours": {"idle": {"flips": 11, "max": 4.0}}, "vanilla": {"idle": {"flips": 13, "max": 3.9}}},
                "colour": {"covered": 0.64, "reached": 0.99}, "fresh": {"hair": {"tube": True, "mane": True}}}
    rep = quality.report(manifest, "horse", tmp_path / "fg.png", tmp_path / "game.png")
    assert rep["measures"]["silhouette"]["iou"] > 0.9
    assert "tail" in rep["measures"]["parts"]["text"] and "reins" in rep["measures"]["parts"]["text"]
    md = quality.markdown(rep)
    assert "Outline vs the reference" in md and "Deformation vs vanilla" in md
    quality.save(rep, tmp_path)
    assert json.loads((tmp_path / "quality.json").read_text())["summary"]


def test_unit_names_and_spec():
    names = unitstage.unit_names("cataphract_rider")
    assert names["cavalry"] == "Cataphract Rider Cavalry"
    test = {"tag": "HDA", "gfx": "hellenicgfx", "set_name": "t", "base_gfx": "westerngfx", "sprite_techs": [2, 3, 4],
            "unit_colours": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    horse = {"base": "cavalry_horse", "geometry": "g", "textures": {}, "rights": "r", "unit_colours": [[255, 77, 70], [1, 1, 1], [2, 2, 2]]}
    sp = unitstage.spec("x", 1, horse, ({"base": "western_generic_01_type1"}, [{"kind": "mesh_shield"}]), names, test)
    assert sp["loadout"]["horses"]["1"] is horse and sp["loadout"]["riders"]["1"]["base"] == "western_generic_01_type1"
    assert sp["loadout"]["slots"]["1/rider"]["parts"][0]["kind"] == "mesh_shield"
    assert sp["set"]["unit_colours"]["HDA"][0] == [255, 77, 70]
    assert sp["set"]["unit_names"]["cavalry"] == "Cataphract Rider Cavalry"
    only_rider = unitstage.spec("x", 2, None, ({"base": "b"}, []), names, test)
    assert "horses" not in only_rider["loadout"] and "slots" not in only_rider["loadout"]


def _cube(c, h):
    x, y, z = c
    p = [x + dx * h for dx in (-1, 1) for _ in (0, 1) for _ in (0, 1)]
    pts = [[x + sx * h, y + sy * h, z + sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
             (1, 5, 7), (1, 7, 3)]
    del p
    return [v for q in pts for v in q], [i for f in faces for i in f]


def test_inside_count_on_a_closed_box():
    p, tri = _cube((0, 0, 0), 1.0)
    pts = [[0, 0, 0], [0.5, 0.3, -0.2], [2, 0, 0], [0, 3, 0], [-0.99, 0.99, 0.5]]
    assert quality.inside_count(pts, p, tri) == 3


def test_legs_clear_and_verdict(tmp_path):
    hp, ht = _cube((0, 10, 0), 2.0)  # the horse's barrel: y 8-12, the saddle's top at 12
    legs_in = [[0.5, 11.0, 0.0], [1.5, 10.0, 0.5], [3.0, 10.0, 0.0], [-3.0, 9.0, 0.0]]
    d = {"cavalry_horse": {"p": hp, "tri": ht}, "x_rider": {"p": [v for q in legs_in for v in q], "tri": [0, 1, 2]}}
    (tmp_path / "posed.json").write_text(json.dumps(d))
    r = quality.legs_clear(tmp_path / "posed.json")
    assert r["inside"] == 2 and r["points"] == 4 and r["share"] == 0.5
    assert quality.clearance(r, {"share": 0.05})["verdict"] == "check"
    assert quality.clearance({"share": 0.06, "inside": 3, "points": 50}, {"share": 0.05})["verdict"] == "good"
