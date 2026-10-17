"""Area finishes: the Character page's "paint this body area as skin / bronze / linen" (SPEC Track R).

A finish (``library/materials.json`` "finishes") becomes ordinary texture-recipe operations on
one region (``area:<name>`` or ``mask:<png>``): recolour (``protect`` off, ``detail`` from the
finish), clear every unit-colour mask there, gloss, and flatten the normal map. Each operation
carries ``"finish": <name>`` so the page can read its choices back from a saved recipe, and the
row's ``grow`` and ``feather`` (pixels; left out when 0).
"""
from __future__ import annotations

from eu4unitgen.library.data import library


class FinishError(ValueError):
    pass


def finishes() -> dict[str, dict]:
    return library("materials")["finishes"]


def finish_ops(region: str, finish: str, skin: tuple, colour: tuple | None = None, grow: int = 0,
               feather: int = 0) -> list[dict]:
    """The recipe operations that give ``region`` the ``finish``. ``skin``: the body's own skin tone;
    ``colour``: the user's colour (required by a finish whose colour is null, overrides others);
    ``grow`` / ``feather``: the region's edge, in pixels."""
    f = finishes().get(finish)
    if f is None:
        raise FinishError(f"unknown finish {finish!r}; known: {', '.join(finishes())}")
    c = colour or (skin if f["colour"] == "skin" else f["colour"])
    if c is None:
        raise FinishError(f"{region}: finish {finish!r} needs a colour")
    tag = {"finish": finish, **({"grow": int(grow)} if grow else {}), **({"feather": int(feather)} if feather else {})}
    ops = [{"op": "recolour", "region": region, "colour": [int(v) for v in c], "detail": f["detail"],
            "protect": False, **tag},
           {"op": "spec_paint", "channel": "all", "region": region, "mode": "clear", **tag},
           {"op": "gloss", "region": region, "preset": f["gloss"], **tag}]
    if f["flatten"]:
        ops.append({"op": "flatten", "region": region, "strength": f["flatten"], **tag})
    return ops


def default_colour(finish: str, skin: tuple) -> tuple | None:
    """The colour a finish paints when the user picks none (None: the finish needs one)."""
    c = finishes()[finish]["colour"]
    return None if c is None else tuple(skin if c == "skin" else c)


def paint_mask_name(finish: str, colour: tuple | None, skin: tuple) -> str:
    """The file a brush painting ``finish`` (in ``colour``, if it isn't the finish's own) paints into."""
    if colour is None or tuple(colour) == default_colour(finish, skin):
        return f"paint_{finish}.png"
    return f"paint_{finish}_{''.join(f'{int(v):02x}' for v in colour)}.png"


def split(ops: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """A slot's operations -> (the ones no finish made, {region: {"finish", "colour"}})."""
    rest, chosen = [], {}
    for op in ops:
        if "finish" not in op:
            rest.append(op)
        elif op.get("op") == "recolour":
            chosen[op["region"]] = {"finish": op["finish"], "colour": tuple(op["colour"]),
                                    "grow": op.get("grow", 0), "feather": op.get("feather", 0)}
    return rest, chosen


def with_finishes(ops: list[dict], chosen: dict[str, dict], skin: tuple, order: list[str]) -> list[dict]:
    """``ops`` with its finish operations replaced by ``chosen`` ({region: {"finish", "colour",
    optional "grow", "feather"}}; a colour other than the finish's own tints it),
    appended after the rest in ``order`` (broad areas before narrow ones, then masks)."""
    rest, _ = split(ops)
    rank = {r: i for i, r in enumerate(order)}
    out = list(rest)
    for region in sorted(chosen, key=lambda r: (rank.get(r, len(rank)), r)):
        c = chosen[region]
        own = default_colour(c["finish"], skin)
        custom = c.get("colour") if c.get("colour") and tuple(c["colour"]) != own else None
        out += finish_ops(region, c["finish"], skin, custom, c.get("grow", 0), c.get("feather", 0))
    return out
