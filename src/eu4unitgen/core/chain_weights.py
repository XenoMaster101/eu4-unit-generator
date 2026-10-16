"""Procedural chain weights for skinned additions to a derived body (SPEC Track R).

Automatic (heat) weights are rejected (spike R0: they pull a tail onto the thighs and a ponytail
onto the shoulders). Weights here come from the part's own structure: its vertices sit on rings
numbered from the root (0) to the tip, and a declared bone chain says how the part follows the
skeleton.

Modes:
  * ``rigid``: every vertex 100% on one bone (R0b/P1: the ponytail tuft on ``head``).
  * ``length``: along-length gradient for parts that grow away from the chain: the root ring is
    100% the first bone, and each later bone's share rises linearly to its ``tip`` share at the
    last ring (R0 C: tail ``hip`` -> ``lower_spine``, tip 50/50).

Never blend one span across a high-rotation joint (``HIGH_ROTATION``): a long span weighted
across ``head``/``top_spine`` kinks and stretches 3x (R0b E). Root rings are always 100% the root
bone, at most ``MAX_INFLUENCES`` bones per vertex, weights normalised.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_INFLUENCES = 4
HIGH_ROTATION = frozenset({("head", "top_spine")})  # bone pairs whose relative rotation is large


class WeightError(ValueError):
    pass


@dataclass(frozen=True)
class Chain:
    mode: str  # "rigid" | "length"
    bones: tuple[str, ...]
    tip: dict = field(default_factory=dict)  # length mode: bone (after the first) -> share at the last ring

    def check(self, skeleton: list[str]) -> None:
        if self.mode not in ("rigid", "length"):
            raise WeightError(f"chain mode {self.mode!r} must be rigid or length")
        if not self.bones:
            raise WeightError("a chain needs at least one bone")
        for b in self.bones:
            if b not in skeleton:
                raise WeightError(f"chain bone {b!r} isn't in the skeleton")
        if self.mode == "rigid" and len(self.bones) != 1:
            raise WeightError("a rigid chain has exactly one bone")
        if len(self.bones) > MAX_INFLUENCES:
            raise WeightError(f"a chain has at most {MAX_INFLUENCES} bones")
        if self.mode == "length":
            extra = set(self.tip) - set(self.bones[1:])
            if extra:
                raise WeightError(f"tip shares for bones outside the chain's tail: {sorted(extra)}")
            total = sum(self.tip.values())
            if not 0 < total < 1 or any(v < 0 for v in self.tip.values()):
                raise WeightError("tip shares must be positive and leave the first bone a share (sum < 1)")
            for a, b in zip(self.bones, self.bones[1:]):
                if (a, b) in HIGH_ROTATION or (b, a) in HIGH_ROTATION:
                    raise WeightError(f"don't blend across {a}/{b}: it rotates too much (spike R0b); "
                                      "keep the part rigid on one side")


def weights(chain: Chain, rings: list[int]) -> list[dict[str, float]]:
    """Per vertex: bone -> weight (sums to 1). ``rings``: each vertex's ring, 0 = root."""
    last = max(rings) if rings else 0
    out = []
    for r in rings:
        if chain.mode == "rigid" or r == 0 or last == 0:
            out.append({chain.bones[0]: 1.0})
            continue
        t = r / last
        w = {b: chain.tip.get(b, 0.0) * t for b in chain.bones[1:]}
        w = {b: x for b, x in w.items() if x > 0}
        w[chain.bones[0]] = 1.0 - sum(w.values())
        out.append(w)
    return out


def check_weights(rows: list[dict[str, float]], skeleton: list[str], allowed: set[str] | None = None) -> list[str]:
    """Problems with per-vertex weights: unknown or undeclared bones, too many influences, sums."""
    problems = []
    for i, w in enumerate(rows):
        if len(w) > MAX_INFLUENCES:
            problems.append(f"vertex {i}: {len(w)} influences (max {MAX_INFLUENCES})")
        for b in w:
            if b not in skeleton:
                problems.append(f"vertex {i}: bone {b!r} isn't in the skeleton")
            elif allowed is not None and b not in allowed:
                problems.append(f"vertex {i}: weight on {b!r}, outside its declared chain")
        if abs(sum(w.values()) - 1.0) > 1e-4:
            problems.append(f"vertex {i}: weights sum to {sum(w.values()):.5f}")
        if len(problems) > 20:
            break
    return problems
