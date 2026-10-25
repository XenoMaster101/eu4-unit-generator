"""The Blender jobs' numpy rule helpers (_colour, _part_view), run inside the pinned Blender (its numpy; the app has
none): tests/blender_rules_check.py prints each check's outcome."""
import json
import subprocess
from pathlib import Path

import pytest

from eu4unitgen.core.project import default_blender_path

BLENDER = default_blender_path()
pytestmark = pytest.mark.skipif(BLENDER is None or not Path(BLENDER).is_file(), reason="pinned Blender not installed")


@pytest.fixture(scope="module")
def results():
    script = Path(__file__).with_name("blender_rules_check.py")
    r = subprocess.run([str(BLENDER), "--background", "--factory-startup", "--python", str(script)],
                       capture_output=True, text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("name", ["cloth_hue_red", "cloth_hue_blue", "cloth_hue_green", "gold_is_never_cloth",
                                  "default_window_is_the_old_red", "coat_kept_apart_from_cloth", "crest_any_colour",
                                  "flat_part_faces_its_painted_side", "flat_part_wide_end_up", "crop_profile_wider_top",
                                  "reins_two_straps_within_budget", "reins_weights_head_to_rem",
                                  "reins_found_when_the_raw_shows_them", "reins_not_built_when_the_low_kept_them",
                                  "red_cloth_window_skips_copper", "orange_cloth_keeps_orange"])
def test_rule(results, name):
    assert results[name]["ok"], results[name]
