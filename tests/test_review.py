"""The animation review (ref3d/review.py): its measures and flags from posed dumps (no Blender here)."""
from eu4unitgen.ref3d import quality, review


def _box(x0, y0, z0, x1, y1, z1):
    """A closed box as a dump mesh."""
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tri = [i for a, b, c, d in q for i in (a, b, c, a, c, d)]
    return {"p": [c for p in v for c in p], "tri": tri}


def test_point_triangle_regions():
    t = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert review.point_triangle([0.2, 0.2, 1.0], t) == 1.0  # over the face
    assert abs(review.point_triangle([2, 0, 0], t) - 1.0) < 1e-9  # past a corner
    assert abs(review.point_triangle([0.5, -1, 0], t) - 1.0) < 1e-9  # off an edge


def test_a_long_shaft_through_the_hand_is_not_floating():
    body = _box(-1, 0, -1, 1, 10, 1)
    shaft = _box(-3.05, -2, -0.05, -2.95, 20, 0.05)  # vertices only at its ends, like western_halbard's
    dump = {"body": body, "western_halbard": shaft, "bones": {"body": {"Right_hand_node": [-3.0, 9.0, 0.0]}}}
    m = review.shot_measures(dump, dump, {"western_halbard"})
    assert m["bends"] == 0.0 and m["clipping"]["western_halbard"] == 0.0
    assert m["floating"]["western_halbard"] < 0.1


def test_squash_and_clipping_flags():
    body0 = _box(-1, 0, -1, 1, 10, 1)
    flat = _box(-1, 0, -0.3, 1, 10, 0.3)  # squashed in depth
    spear = _box(-0.2, 2, -0.2, 0.2, 8, 0.2)  # inside the body
    bones = {"body": {"Right_hand_node": [0.0, 5.0, 0.0]}}
    ours = review.shot_measures({"body": body0}, {"body": flat, "spear": spear, "bones": bones}, {"spear"})
    game = review.shot_measures({"body": body0}, {"body": body0, "bones": bones}, set())
    assert ours["bends"] > 0.5 and ours["clipping"]["spear"] == 1.0
    f = review.flags(ours, game)
    assert any(x.startswith("bad bends") for x in f) and any(x.startswith("clipping") for x in f)
    assert review.flags(game, game) == []


def test_the_animation_measure():
    assert quality.animation(None) is None
    ok = {"against": "westerngfx_INFANTRY_1", "summary": {"flags": 0}, "states": {}}
    assert quality.animation(ok)["verdict"] == quality.GOOD
    bad = {"against": "x", "summary": {"flags": 13},
           "states": {"walk": {"flags": [{"flag": "clipping: 24% of the weapon inside the body"}] * 13}}}
    assert quality.animation(bad)["verdict"] == quality.CHECK
    assert "In its animations" in quality.TITLES.values()
