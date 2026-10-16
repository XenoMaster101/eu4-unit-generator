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
