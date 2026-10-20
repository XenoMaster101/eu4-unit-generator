"""Track M S1 helpers (app side: Pillow only; the AI jobs run in their own environment)."""
import json

import pytest
from PIL import Image

from eu4unitgen.ref3d import s1


def _subject(tmp_path, **over):
    Image.new("RGB", (200, 100), (10, 20, 30)).save(tmp_path / "sheet.png")
    s = {"name": "t", "template": "horse", "original": "sheet.png",
         "views": {"L": {"crop": [10, 20, 110, 90], "remove": {"rider": [[20, 30], [60, 30], [60, 70]]},
                         "zones": {"legs": [[[10, 60], [50, 60], [50, 90]], [[60, 60], [100, 60], [100, 90]]],
                                   "tail": [[70, 20], [80, 20], [80, 40]]},
                         "landmarks": {"eye": [30, 40]}},
                   "close": {"crop": [120, 0, 200, 100]}}}
    s.update(over)
    p = tmp_path / "subject.json"
    p.write_text(json.dumps(s))
    return p


def test_load_validates_crops_and_landmarks(tmp_path):
    s = s1.load_subject(_subject(tmp_path))
    assert s["_template"]["template"] == "horse"
    with pytest.raises(ValueError, match="crop"):
        s1.load_subject(_subject(tmp_path, views={"L": {"crop": [10, 20, 5, 90]}}))
    with pytest.raises(ValueError, match="not in the horse template"):
        s1.load_subject(_subject(tmp_path, views={"L": {"crop": [0, 0, 5, 5], "landmarks": {"elbow_x": [1, 1]}}}))


def test_crops_masks_and_zones_are_in_crop_pixels(tmp_path):
    s = s1.load_subject(_subject(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    crops = s1.crop_views(s, out)
    assert Image.open(crops["L"]).size == (100, 70)
    masks = s1.removal_masks(s, crops, out)
    assert set(masks) == {"L"}  # close-ups have nothing to remove
    m = Image.open(masks["L"])
    assert m.getpixel((40, 15)) == 255 and m.getpixel((5, 5)) == 0  # sheet (50, 35) inside the triangle
    z = s1.zones_in_crop(s["views"]["L"])
    assert len(z["legs"]) == 2 and len(z["tail"]) == 1 and z["tail"][0][0] == (60.0, 0.0)


def test_versions_never_overwrite(tmp_path):
    a, b = s1.next_version(tmp_path), s1.next_version(tmp_path)
    assert (a.name, b.name) == ("v001", "v002") and a.is_dir() and b.is_dir()


def test_original_is_untouched(tmp_path):
    p = _subject(tmp_path)
    before = s1.sha256(tmp_path / "sheet.png")
    s = s1.load_subject(p)
    out = tmp_path / "o"
    out.mkdir()
    s1.removal_masks(s, s1.crop_views(s, out), out)
    assert s1.sha256(tmp_path / "sheet.png") == before
