"""Area finishes: the Character page's "paint this body area as skin / bronze / linen" (SPEC Track R).

A finish (``library/materials.json`` "finishes") becomes ordinary texture-recipe operations on
one region (``area:<name>`` or ``mask:<png>``): recolour (``protect`` off, ``detail`` from the
finish), clear every unit-colour mask there, gloss, and flatten the normal map. Each operation
carries ``"finish": <name>`` so the page can read its choices back from a saved recipe.
"""
from __future__ import annotations

from eu4unitgen.library.data import library


class FinishError(ValueError):
    pass


def finishes() -> dict[str, dict]:
    return library("materials")["finishes"]


def finish_ops(region: str, finish: str, skin: tuple, colour: tuple | None = None) -> list[dict]:
    """The recipe operations that give ``region`` the ``finish``. ``skin``: the body's own skin tone;
    ``colour``: the user's colour (required by a finish whose colour is null, overrides others)."""
    f = finishes().get(finish)
    if f is None:
        raise FinishError(f"unknown finish {finish!r}; known: {', '.join(finishes())}")
    c = colour or (skin if f["colour"] == "skin" else f["colour"])
    if c is None:
        raise FinishError(f"{region}: finish {finish!r} needs a colour")
    tag = {"finish": finish}
    ops = [{"op": "recolour", "region": region, "colour": [int(v) for v in c], "detail": f["detail"],
            "protect": False, **tag},
           {"op": "spec_paint", "channel": "all", "region": region, "mode": "clear", **tag},
           {"op": "gloss", "region": region, "preset": f["gloss"], **tag}]
    if f["flatten"]:
        ops.append({"op": "flatten", "region": region, "strength": f["flatten"], **tag})
    return ops


def split(ops: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """A slot's operations -> (the ones no finish made, {region: {"finish", "colour"}})."""
    rest, chosen = [], {}
    for op in ops:
        if "finish" not in op:
            rest.append(op)
        elif op.get("op") == "recolour":
            chosen[op["region"]] = {"finish": op["finish"], "colour": tuple(op["colour"])}
    return rest, chosen


def with_finishes(ops: list[dict], chosen: dict[str, dict], skin: tuple, order: list[str]) -> list[dict]:
    """``ops`` with its finish operations replaced by ``chosen`` ({region: {"finish", "colour"}}),
    appended after the rest in ``order`` (broad areas before narrow ones, then masks)."""
    rest, _ = split(ops)
    rank = {r: i for i, r in enumerate(order)}
    out = list(rest)
    for region in sorted(chosen, key=lambda r: (rank.get(r, len(rank)), r)):
        c = chosen[region]
        custom = c.get("colour") if finishes()[c["finish"]]["colour"] is None else None
        out += finish_ops(region, c["finish"], skin, custom)
    return out
