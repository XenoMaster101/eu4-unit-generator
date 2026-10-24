"""ref3d.trellis.pin_head: the low body's helmet rigid on the head bone (SPEC M23, test 35)."""
import json

from eu4unitgen.ref3d import trellis


def test_pin_head(tmp_path):
    g = {"p": [0, 14.0, 0, 0, 14.6, 0, 0, 15.5, 0],
         "weights": [{"top_spine": 1.0}, {"top_spine": 0.4, "head": 0.6}, {"top_spine": 0.3, "head": 0.7}]}
    f = tmp_path / "low_geometry.json"
    f.write_text(json.dumps(g))
    assert trellis.pin_head(f) == 2
    w = json.loads(f.read_text())["weights"]
    assert w[0] == {"top_spine": 1.0}  # under the pin: untouched
    assert 0.6 < w[1]["head"] < 1.0 and abs(sum(w[1].values()) - 1) < 1e-6  # faded in
    assert w[2] == {"head": 1.0}  # the dome: rigid
