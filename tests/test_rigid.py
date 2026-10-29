"""Rigid parts (blender/jobs/_rigid.py): a weapon's straightness, its straightening, and a part's change of shape;
the review's flag and the regression's measures."""
import json
import math

from eu4unitgen.blender.jobs import _rigid
from eu4unitgen.ref3d import regress, review


def _shaft(wave=0.0, length=18.0, r=0.06, rings=60, blade=False):
    """A lance along y: rings of 8 points, its middle off the line by ``wave`` (one and a half waves along it);
    ``blade``: a wide head over its top tenth."""
    pts = []
    for i in range(rings + 1):
        y = length * i / rings
        off = wave * math.sin(3 * math.pi * y / length)
        rad = 0.6 if blade and y > 0.9 * length else r
        for k in range(8):
            a = 2 * math.pi * k / 8
            pts.append([off + rad * math.cos(a), y, rad * math.sin(a)])
    return pts


def test_a_straight_shaft_has_no_bow_even_with_a_wide_head():
    assert _rigid.bow(_shaft()) < 1e-6
    assert _rigid.bow(_shaft(blade=True)) < 1e-3  # the blade is the head: not a bend


def test_a_wavy_shaft_is_straightened_and_its_ends_stay():
    pts = _shaft(wave=0.3)
    assert _rigid.bow(pts) > 0.01
    out, info = _rigid.straighten(pts)
    assert info["bow_before"] > 0.01 and info["bow_after"] < _rigid.BOW_MAX
    for i in (0, -1):  # the end rings where they were
        assert math.dist(out[i], pts[i]) < 0.02


def test_distortion_zero_when_moved_whole_and_not_when_bent():
    pts = _shaft()
    c, s = math.cos(0.7), math.sin(0.7)
    moved = [[c * x - s * y + 3, s * x + c * y - 1, z + 2] for x, y, z in pts]
    assert _rigid.distortion(pts, moved) < 1e-9
    bent = [[x + 0.02 * y * y, y, z] for x, y, z in pts]
    assert _rigid.distortion(pts, bent) > _rigid.RIGID_MAX


def test_review_flags_a_part_that_changes_shape():
    body = {"p": [0, 0, 0, 1, 0, 0, 0, 1, 0], "tri": [0, 1, 2]}
    flat = lambda pts: {"p": [c for p in pts for c in p], "tri": [0, 1, 2]}  # noqa: E731
    pts = _shaft()
    rest = {"body": body, "lance": flat(pts)}
    bent = {"body": body, "lance": flat([[x + 0.02 * y * y, y, z] for x, y, z in pts])}
    m = review.shot_measures(rest, bent, set())
    assert m["rigid"]["lance"] > _rigid.RIGID_MAX
    assert any(f.startswith("bent: the weapon") for f in review.flags(m, {}))
    assert review.shot_measures(rest, rest, set())["rigid"]["lance"] == 0.0


def test_regression_measures_the_weapons_bow(tmp_path):
    pts = _shaft(wave=0.3)
    tri = [i for k in range(len(pts) - 2) for i in (k, k + 1, k + 2)]
    geo = tmp_path / "lance_foot.json"
    geo.write_text(json.dumps({"p": [c for p in pts for c in p], "tri": tri}), encoding="utf-8")
    m = regress.metrics({"parts": {"lance": {"kind": "mesh_weapon", "mesh": {"geometry": str(geo)}}}}, "rider")
    assert m["weapon_bow"] > _rigid.BOW_MAX
    assert any(x.startswith("weapon_bow") for x in regress.expectations("cataphract_rider", m))
