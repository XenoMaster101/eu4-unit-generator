"""Procedural chain weights and tangents for derived bodies (SPEC Track R)."""
from __future__ import annotations

import pytest

from eu4unitgen.core.chain_weights import Chain, WeightError, check_weights, weights
from eu4unitgen.core.derived_body import tangent_agreement, tangents

SKELETON = ["root", "lower_spine", "mid_spine", "top_spine", "head", "hip", "L_thigh"]


def test_rigid_chain_is_all_one_bone():
    rows = weights(Chain("rigid", ("head",)), [0, 1, 2, 3])
    assert rows == [{"head": 1.0}] * 4


def test_length_gradient_tail():
    chain = Chain("length", ("hip", "lower_spine"), {"lower_spine": 0.5})
    chain.check(SKELETON)
    rows = weights(chain, [0, 0, 1, 2, 4])
    assert rows[0] == rows[1] == {"hip": 1.0}  # root ring locked to the root bone
    assert rows[2] == pytest.approx({"hip": 0.875, "lower_spine": 0.125})
    assert rows[-1] == pytest.approx({"hip": 0.5, "lower_spine": 0.5})
    assert check_weights(rows, SKELETON, {"hip", "lower_spine"}) == []


@pytest.mark.parametrize("chain, msg", [
    (Chain("length", ("head", "top_spine"), {"top_spine": 0.7}), "rotates too much"),
    (Chain("rigid", ("head", "top_spine")), "exactly one bone"),
    (Chain("rigid", ("tail_1",)), "isn't in the skeleton"),
    (Chain("length", ("hip", "lower_spine"), {"lower_spine": 1.0}), "sum < 1"),
    (Chain("length", ("hip", "lower_spine"), {"mid_spine": 0.3}), "outside"),
    (Chain("heat", ("hip",)), "rigid or length"),
])
def test_chain_checks(chain, msg):
    with pytest.raises(WeightError, match=msg):
        chain.check(SKELETON)


def test_check_weights_problems():
    rows = [{"hip": 0.6}, {"hip": 0.5, "L_thigh": 0.5}, {"tail": 1.0},
            {"hip": 0.2, "lower_spine": 0.2, "mid_spine": 0.2, "top_spine": 0.2, "head": 0.2}]
    problems = check_weights(rows, SKELETON, {"hip", "lower_spine"})
    assert any("vertex 0: weights sum" in p for p in problems)
    assert any("vertex 1: weight on 'L_thigh', outside" in p for p in problems)
    assert any("vertex 2: bone 'tail'" in p for p in problems)
    assert any("vertex 3: 5 influences" in p for p in problems)


def test_tangents_on_a_quad():
    # quad in the XY plane facing +Z; image UVs (v down): u grows with x, v shrinks with y
    p = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    n = [0, 0, 1] * 4
    u0 = [0, 1, 1, 1, 1, 0, 0, 0]
    tri = [0, 1, 2, 0, 2, 3]
    ta = tangents([float(x) for x in p], [float(x) for x in n], [float(x) for x in u0], tri)
    for i in range(4):
        assert ta[i * 4:i * 4 + 3] == pytest.approx([1, 0, 0])
        assert ta[i * 4 + 3] == -1.0  # (n x t) = +Y, bitangent (dP/dv) = -Y
    assert tangent_agreement(ta, ta) == {"within_10deg": 1.0, "median_deg": 0.0, "w_agrees": 1.0}


# --------------------------------------------------------------------------- derived body writer

from eu4unitgen.core.derived_body import Addition, DerivedBodyError, append_additions, check_derived  # noqa: E402
from eu4unitgen.library.scanner import parse_pdx_mesh, write_pdx_mesh  # noqa: E402


def _body() -> bytes:
    """A tiny one-shape skinned body in the base game's layout: one triangle on 'hip', 3 bones,
    a locator (attach_free's bind matrix all zeros, as in vanilla)."""
    bones = [{"name": n, "props": {"ix": [i], **({"pa": [pa]} if pa is not None else {}), "tx": tx}, "children": []}
             for i, (n, pa, tx) in enumerate((("root", None, [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
                                              ("hip", 0, [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, -8.3, 0]),
                                              ("attach_free", 0, [0.0] * 12)))]
    mesh = {"name": "mesh", "props": {"p": [0.0, 8, 0, 1, 8, 0, 0, 9, 0], "n": [0.0, 0, -1] * 3,
                                      "ta": [1.0, 0, 0, 1] * 3, "u0": [0.0, 0, 1, 0, 0, 1], "tri": [0, 1, 2]},
            "children": [{"name": "aabb", "props": {"min": [0.0, 8, 0], "max": [1.0, 9, 0]}, "children": []},
                         {"name": "material", "props": {"shader": ["PdxMeshColor"], "diff": ["b_diffuse.dds"]},
                          "children": []},
                         {"name": "skin", "props": {"bones": [4], "ix": [1, -1, -1, -1] * 3,
                                                    "w": [1.0, 0, 0, 0] * 3}, "children": []}]}
    tree = {"name": "root", "props": {}, "children": [
        {"name": "object", "props": {}, "children": [
            {"name": "bodyShape", "props": {}, "children": [mesh, {"name": "skeleton", "props": {}, "children": bones}]}]},
        {"name": "locator", "props": {}, "children": [
            {"name": "attack_free_ctrl", "props": {"p": [0.0, 0, 0], "q": [0.0, 0, 0, 1], "pa": ["root"]},
             "children": []}]}]}
    return write_pdx_mesh(tree)


def _tail() -> Addition:
    return Addition("tail", p=[0.0, 8, 1, 0.5, 8, 1, 0, 7, 3], n=[0.0, 0, 1] * 3, u0=[0.2, 0.06, 0.21, 0.06, 0.2, 0.07],
                    tri=[0, 1, 2], weights=[{"hip": 1.0}, {"hip": 1.0}, {"hip": 0.5, "root": 0.5}],
                    chain=("hip", "root"))


def test_append_keeps_every_base_block():
    base = _body()
    derived = append_additions(base, [_tail()])
    problems, counts = check_derived(base, derived)
    assert problems == [], problems
    assert counts == {"base_vertices": 3, "added_vertices": 3, "base_triangles": 1, "added_triangles": 1}
    mesh = next(c for c in parse_pdx_mesh(derived)["children"][0]["children"][0]["children"] if c["name"] == "mesh")
    assert mesh["props"]["tri"] == [0, 1, 2, 3, 4, 5]
    skin = next(c for c in mesh["children"] if c["name"] == "skin")["props"]
    assert skin["ix"][12:] == [1, -1, -1, -1, 1, -1, -1, -1, 1, 0, -1, -1]
    box = next(c for c in mesh["children"] if c["name"] == "aabb")["props"]
    assert box["min"] == [0.0, 7.0, 0.0] and box["max"] == [1.0, 9.0, 3.0]
    assert append_additions(base, []) == base  # nothing appended: byte-identical


def test_check_derived_catches_changes():
    base = _body()
    tree = parse_pdx_mesh(append_additions(base, [_tail()]))
    shape = tree["children"][0]["children"][0]
    mesh, skeleton = shape["children"]
    mesh["props"]["p"][0] = 0.25  # a base vertex moved
    skeleton["children"][2]["props"]["tx"] = [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # attach_free -> identity
    mesh["props"]["boundingsphere"] = [0.0, 0, 0, 1]
    problems, _ = check_derived(base, write_pdx_mesh(tree))
    assert "the base's p values changed" in problems
    assert "skeleton differs from the base body's" in problems
    assert any("boundingsphere" in p for p in problems)


@pytest.mark.parametrize("change, msg", [
    (lambda a: a.weights.__setitem__(0, {"tail_1": 1.0}), "isn't in the skeleton"),
    (lambda a: a.weights.__setitem__(0, {"hip": 0.5}), "sum"),
    (lambda a: a.tri.__setitem__(0, 9), "outside its vertices"),
    (lambda a: a.u0.pop(), "array lengths"),
])
def test_append_refuses_bad_additions(change, msg):
    add = _tail()
    change(add)
    with pytest.raises(DerivedBodyError, match=msg):
        append_additions(_body(), [add])


def test_additions_follow_the_base_winding():
    from eu4unitgen.core.derived_body import oriented, winding_sign

    p = [0.0, 0, 0, 1, 0, 0, 0, 1, 0]
    n = [0.0, 0, 1] * 3
    assert winding_sign(p, n, [0, 1, 2]) == 1 and winding_sign(p, n, [0, 2, 1]) == -1
    assert oriented(p, n, [0, 2, 1], 1) == [0, 1, 2]
    assert oriented(p, n, [0, 1, 2], -1) == [0, 2, 1]
    # appended triangles take the base's convention (the base's one triangle faces its normal)
    tree = parse_pdx_mesh(append_additions(_body(), [_tail()]))
    mesh = tree["children"][0]["children"][0]["children"][0]["props"]
    assert winding_sign(mesh["p"][9:], mesh["n"][9:], [i - 3 for i in mesh["tri"][3:]]) == \
        winding_sign(mesh["p"][:9], mesh["n"][:9], mesh["tri"][:3])
