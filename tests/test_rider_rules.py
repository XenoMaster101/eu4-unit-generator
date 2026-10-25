"""Track P1: the rider's general rules in ref3d/trellis.py (legs clear of the horse, the shield strapped to the
forearm, the head part's animation scale, sword textures)."""
import json

from PIL import Image

from eu4unitgen.ref3d import trellis


def _barrel(half_widths):
    """Points of a horse barrel: at each seated-thigh height a ring of the given half-width."""
    p = []
    for y, w in zip(trellis.SEAT_BAND, half_widths):
        for k in range(40):
            p += [w * (1 if k % 2 else -1), y + 0.2 * ((k // 2) % 3 - 1), -2.0 + 0.1 * k]
    return p


def test_leg_spread_is_the_extra_width_at_the_same_height():
    van = _barrel([2.4, 1.75])
    assert trellis.leg_spread(_barrel([3.05, 2.8]), van) == 1.05  # the cataphract: widest where vanilla narrows
    assert trellis.leg_spread(_barrel([2.0, 1.5]), van) == 0.0  # slimmer: nothing moves
    assert trellis.leg_spread(_barrel([9.0, 9.0]), van) == 1.5  # capped


def test_spread_legs_moves_only_leg_weighted_vertices_outward(tmp_path):
    g = {"p": [1.0, 5.0, 0.0, -1.0, 5.0, 0.0, 0.5, 12.0, 0.0, 0.8, 8.0, 0.0],
         "weights": [{"L_knee": 1.0}, {"R_knee": 1.0}, {"top_spine": 1.0}, {"L_thigh": 0.5, "hip": 0.5}]}
    (tmp_path / "g.json").write_text(json.dumps(g))
    assert trellis.spread_legs(tmp_path / "g.json", 0.6) == 3
    p = json.loads((tmp_path / "g.json").read_text())["p"]
    assert p[0] == 1.6 and p[3] == -1.6 and p[6] == 0.5 and abs(p[9] - 1.1) < 1e-9


def test_shield_strapped_half_way_to_the_elbow_out_and_up():
    frame = {"elbow": [2.0, 0.0, 0.0], "out": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0], "forward": [1.0, 0.0, 0.0]}
    centre, axes = trellis.strap(frame)
    assert centre == [1.0, 0.3, 1.2]  # half way to the elbow, 1.2 out, 0.3 up (test 32's kite placement)
    assert axes["out"] == frame["out"] and axes["across"] == frame["forward"]


def test_head_part_scaled_for_the_rider_animation(tmp_path):
    info = {"finial_base": [0.0, 2.0, 0.0], "helmet_radius": 1.0, "line": [[0, 3.5, 0.5], [0, 3.6, 1.2], [0, 3.2, 2.0]],
            "half_width": [0.3, 0.3, 0.2]}
    a = trellis.plume_part(tmp_path / "a", info, scale=1.0)
    b = trellis.plume_part(tmp_path / "b", info, scale=trellis.RIDER_ANIM_SCALE)
    pa = json.loads(open(a["mesh"]["geometry"]).read())["p"]
    pb = json.loads(open(b["mesh"]["geometry"]).read())["p"]
    assert abs(max(pb[1::3]) - trellis.RIDER_ANIM_SCALE * max(pa[1::3])) < 1e-3
    assert trellis.RIDER_ANIM_SCALE == 0.805  # test 35 tier 1, approved in game


def test_plume_takes_the_units_gold_and_a_ridged_normal(tmp_path):
    info = {"finial_base": [0.0, 2.0, 0.0], "helmet_radius": 1.0, "line": [[0, 3.5, 0.5], [0, 3.2, 2.0]], "half_width": [0.3, 0.2]}
    e = trellis.plume_part(tmp_path, info, gold=(100, 90, 40))
    d = Image.open(e["mesh"]["diffuse"]).convert("RGB")
    n = Image.open(e["mesh"]["normal"]).convert("RGB")
    S = d.width
    assert abs(d.getpixel((int(S * 0.875), 10))[0] - 108) <= 2  # the gold strip's centre: the unit's gold, lit 1.08
    assert n.getpixel((int(S * 0.77), 10))[0] < 128 < n.getpixel((int(S * 0.98), 10))[0]  # a ridge, not flat


def test_sword_is_textured_not_flat(tmp_path):
    frame = {"out": [1.0, 0, 0], "up": [0, 1.0, 0], "forward": [0, 0, -1.0]}
    e = trellis.sword_part(tmp_path, frame, 1.9, gold=(150, 120, 80))
    d = Image.open(e["mesh"]["diffuse"]).convert("RGB")
    n = Image.open(e["mesh"]["normal"]).convert("RGB")
    assert len(set(d.crop((0, 0, 128, 128)).getdata())) > 20  # worn, not one flat colour
    assert len(set(n.getdata())) > 20  # a normal map with detail
