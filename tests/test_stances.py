"""The base game's animation sets and weapon conventions (core/stances.py): the pure measures (the catalogue itself
reads the install, so it isn't run here)."""
from eu4unitgen.core import stances


def _shaft(z0, z1, n=40, head=None):
    pts = [[0.05 * ((i % 3) - 1), 0.05 * ((i % 2) - 0.5), z0 + (z1 - z0) * i / (n - 1)] for i in range(n)]
    if head is not None:  # a broad blade near the +z end
        pts += [[0.0, y, head] for y in (-1.0, -0.5, 0.5, 1.0)]
    return pts


def test_grip_measure_like_western_halbard():
    m = stances.grip_measure(_shaft(-10.8, 13.9))  # western_halbard's z extent: gripped at 0.44 from the butt
    assert abs(m["length"] - 24.7) < 0.01 and abs(m["grip"] - 0.437) < 0.01 and m["axis"][2] > 0.99


def test_slide_to_grip_moves_the_english_bill_to_the_halberds_place():
    pts = _shaft(-11.11, 6.55, head=5.5)  # the bill: gripped at 0.63, its head 6.55 over the hand
    out, rec = stances.slide_to_grip(pts, 0.438)
    zs = [p[2] for p in out]
    lo, hi = min(zs), max(zs)
    assert abs(rec["grip_before"] - 0.629) < 0.01
    assert abs(-lo / (hi - lo) - 0.438) < 1e-6 and abs((hi - lo) - 17.66) < 1e-6  # length kept
    assert hi > 9.0  # its head now well over the hand
    assert all(abs(p[0]) < 0.2 for p in out)  # the shaft on the hand's axis


def test_stance_rules_match_the_catalogue():
    # every stance our soldiers get is a base game set that holds a weapon in that hand (catalogue, 1.37)
    from eu4unitgen.ref3d.trellis import foot_stance
    assert foot_stance({"parts": {"s": {"kind": "mesh_shield"}, "l": {"kind": "mesh_weapon"}}}) == "spear_shield"
    assert foot_stance({"parts": {"l": {"kind": "mesh_weapon"}}}) == "polearm_twohanded"
    assert stances.WEAPON_HAND.get("polearm_twohanded", "Right_hand_node") == "Right_hand_node"
