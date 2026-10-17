"""The app's data libraries (SPEC Track R, data-driven characters): body facts, skinned additions
and the painted-style material library, as JSON beside this module. Code holds builders and
checks; the numbers live here. Loaded once; tests may change the returned dicts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
LIBRARIES = ("bodies", "additions", "materials", "templates", "vocabulary")


@lru_cache(maxsize=None)
def library(name: str) -> dict:
    if name not in LIBRARIES:
        raise ValueError(f"unknown data library {name!r}; known: {', '.join(LIBRARIES)}")
    doc = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    if doc.get("schema") != 1:
        raise ValueError(f"{name}.json: schema {doc.get('schema')!r} isn't 1")
    return doc
