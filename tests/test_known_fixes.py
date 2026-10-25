"""Every known fix (core/known_fixes.py) names code that exists and tests that exist."""
import re
from pathlib import Path

import pytest

from eu4unitgen.core.known_fixes import FIXES
from eu4unitgen.ref3d.regress import METRICS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "eu4unitgen"


@pytest.mark.parametrize("fix", FIXES, ids=lambda f: f.key)
def test_fix_points_at_real_code_and_tests(fix):
    path, symbol = fix.where.split(":")
    text = (SRC / path).read_text(encoding="utf-8")
    assert re.search(rf"^\s*def {symbol}\b", text, re.M), f"{symbol} not defined in {path}"
    assert fix.tests, "no test"
    for t in fix.tests:
        if t.startswith("regress:"):
            assert t[8:] in METRICS, t
            continue
        file, name = t.split("::")
        body = (ROOT / file).read_text(encoding="utf-8")
        base = name.split("[")[0]
        assert re.search(rf"^def {base}\(", body, re.M), t
        if "[" in name:  # a parametrised case: its id appears in the file
            assert name.split("[")[1].rstrip("]") in body, t


def test_keys_unique():
    keys = [f.key for f in FIXES]
    assert len(keys) == len(set(keys))
