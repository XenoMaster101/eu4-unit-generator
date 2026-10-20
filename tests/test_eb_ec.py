"""Track M spikes E-B (plates) and E-C (cloth): the app-side gates and corrections (the Blender and AI
jobs themselves run only in their environments)."""
from eu4unitgen.ref3d import eb, ec

G = {"plate_period_tol": 0.15, "coverage_iou_min": 0.7, "landmark_max": 0.025}


def _eb_result(pp=0, pb=0, vis=0.0, removed=None):
    return {"penetrations": {"plate_plate_pairs_over": pp, "plate_body_over": pb, "plate_plate_max_mm": 0.5,
                             "plate_body_max_mm": 0.2},
            "gaps": {"body_visible": vis}, "removed": removed or {"in_body": 0, "crossing": 0, "of": 500}}


def test_eb_gates_pass_and_fail_each_on_its_own():
    m = {"plate_period": {"ratio": 1.1}, "coverage_iou": 0.75}
    cfg = {"body_visible_max": 0.01}
    g = eb.gates(_eb_result(), m, cfg, G)
    assert all(x["pass"] for x in g.values())
    assert not eb.gates(_eb_result(pp=1), m, cfg, G)["penetrations"]["pass"]
    assert not eb.gates(_eb_result(vis=0.02), m, cfg, G)["gaps"]["pass"]
    assert not eb.gates(_eb_result(), {**m, "plate_period": {"ratio": 1.2}}, cfg, G)["plate_period"]["pass"]
    assert not eb.gates(_eb_result(), {**m, "plate_period": {"ratio": None}}, cfg, G)["plate_period"]["pass"]
    assert not eb.gates(_eb_result(), {**m, "coverage_iou": 0.6}, cfg, G)["coverage_iou"]["pass"]


def test_eb_correct_scales_plates_by_period_and_lifts_only_on_many_removals():
    plate = {"pitch_m": 0.05, "spacing_m": 0.035, "tilt_clear": 1.5, "standoff_m": 0.004}
    m = {"axes": {"y": {"ratio": 1.25}, "x": {"ratio": 1.02}}}
    did = eb.correct(plate, _eb_result(), m, {"period_target": 0.05})
    assert abs(plate["pitch_m"] - 0.04) < 1e-12 and plate["spacing_m"] == 0.035 and len(did) == 1
    plate2 = dict(plate)
    eb.correct(plate2, _eb_result(removed={"in_body": 40, "crossing": 40, "of": 500}), {}, {"period_target": 0.05})
    assert plate2["tilt_clear"] > plate["tilt_clear"] and plate2["standoff_m"] > plate["standoff_m"]
    plate3 = dict(plate)
    assert eb.correct(plate3, _eb_result(removed={"in_body": 5, "crossing": 5, "of": 500}), {}, {"period_target": 0.05}) == []


def _ec_result(errs, inter=0, stretch=1.05):
    return {"metres_per_unit": 0.1, "intersections": {"over_limit": inter, "max_mm": 0.0},
            "stretch": {"max": stretch}, "hem": {"L": {k: {"error": e, "dv_px": 10.0, "column": 0.0} for k, e in errs.items()}}}


def test_ec_gates():
    cfg = {"stretch_max": 1.2, "rerun_tol": 1e-5, "seam_p95_max": 8.0}
    ok = ec.gates(_ec_result({"hem_side": 0.01}), cfg, G, {"runs": 5, "max_diff": 0.0}, {"p95": 5.0, "mean": 3.0})
    assert all(x["pass"] for x in ok.values())
    bad = ec.gates(_ec_result({"hem_side": 0.03}, inter=2, stretch=1.5), cfg, G, {"runs": 5, "max_diff": 1e-3},
                   {"p95": 9.0, "mean": 3.0})
    assert not any(x["pass"] for x in bad.values())


def test_ec_correct_drops_lengthens_near_the_landmark_only():
    panel = {"cols": 48, "drop_m": [0.7] * 7}
    a = {"views": {"L": {"camera": {"focal": 1000.0, "dist": 20.0}}}}
    ec.correct_drops(panel, _ec_result({"hem_side": 0.05}), a)  # 10 px low at column 0
    d = panel["drop_m"]
    assert d[0] > 0.7 + 0.015  # 10 px / 50 px per unit * 0.1 m = 2 cm, mostly at the first control column
    assert abs(d[-1] - 0.7) < 1e-9  # far columns untouched (weight under 0.2)
