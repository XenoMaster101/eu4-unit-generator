"""Track M spike E-D (2D): the app-side pieces (body length, the calibration table, the verdict).
The metrics themselves run in the AI environment and are exercised by the E-D run."""
import json

from eu4unitgen.ref3d import ed2d


def test_body_length_from_landmarks_and_scaled_by_height():
    s = {"views": {"L": {"landmarks": {"chest_front": [100, 50], "hem_rear": [600, 300]},
                         "body_length": {"from": "chest_front", "to": "hem_rear", "axis": "x"}},
                   "R": {"landmarks": {}, "body_length": {"like": "L", "by": "fg_height"}}}}
    bl = ed2d.body_lengths(s, {"L": 700, "R": 350})
    assert bl == {"L": 500, "R": 250}


def _res(unbroken_failed=(), break_failed=("landmarks",)):
    gate = lambda v, ok: {"pass": ok, "worst": "x", "value": v, "limit": 0.5}
    return {"views": {"L": {
        "unbroken_0": {"failed": list(unbroken_failed), "gates": {"landmarks": gate(0.01, not unbroken_failed),
                                                                  "region_iou": gate(0.95, True)}},
        "unbroken_1": {"failed": [], "gates": {"landmarks": gate(0.02, True), "region_iou": gate(0.9, True)}},
        "head_5deg": {"failed": list(break_failed), "gates": {"landmarks": gate(0.04, not break_failed),
                                                              "region_iou": gate(0.93, True)}}}}}


def test_calibration_keeps_the_worst_unbroken_value_per_direction():
    cal = ed2d.calibration(_res())
    assert cal["landmarks"]["unbroken_worst"]["L"][0] == 0.02  # higher is worse
    assert cal["region_iou"]["unbroken_worst"]["L"][0] == 0.9  # lower is worse
    assert cal["landmarks"]["breaks"]["head_5deg"]["L"] == (0.04, "x", False)


def test_verdict_needs_clean_unbroken_and_every_break_caught():
    assert ed2d.verdict(_res())["pass"]
    assert not ed2d.verdict(_res(break_failed=()))["pass"]
    v = ed2d.verdict(_res(unbroken_failed=("landmarks",)))
    assert not v["pass"] and v["unbroken_failures"] == {"L/unbroken_0": ["landmarks"]}


def test_gates_file_has_every_gate_the_metrics_use():
    g = json.loads(ed2d.GATES.read_text(encoding="utf-8"))["gates"]
    for k in ("region_iou_min", "landmark_max", "coverage_iou_min", "coverage_min_share", "delta_e_max",
              "detail_ncc_min", "plate_period_tol", "region_min_px", "hue_max_deg", "label_scale"):
        assert k in g
