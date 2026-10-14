"""Golden-file helper. Set UPDATE_GOLDEN=1 to rewrite the files after reviewing a change."""
from __future__ import annotations

import os
from pathlib import Path

GOLDEN = Path(__file__).with_name("golden")


def check_golden(rel: str, actual: str | bytes) -> None:
    path = GOLDEN / rel
    data = actual.encode("utf-8") if isinstance(actual, str) else actual
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert path.exists(), f"missing golden file {path}; run with UPDATE_GOLDEN=1 and review it"
    expected = path.read_bytes()
    if data != expected:
        raise AssertionError(
            f"{rel} differs from golden file.\n--- expected\n{expected.decode('utf-8', 'replace')}"
            f"\n--- actual\n{data.decode('utf-8', 'replace')}")
