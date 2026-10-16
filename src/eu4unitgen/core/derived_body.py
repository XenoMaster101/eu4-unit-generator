"""Derived bodies: a base-game body .mesh with skinned additions (tail, hair) appended (SPEC Track R).

This is the standard path for every derived body: io_pdx_mesh's export is not used for skinned
bodies (test 10 pre-check P2: it writes every face corner as its own vertex and turns
``attach_free``'s all-zero bind matrix into identity). Instead the vanilla file is parsed and
every block is kept as it is (skeleton, locators, material, vertices, skin rows, triangles); only
the additions' vertices, skin rows and triangles are appended, and the ``aabb`` grows to hold
them. ``scanner.write_pdx_mesh`` writes the result (byte-identical on every vanilla mesh).

Spaces: everything here is PDX mesh space (Y up, the unit faces -Z), UVs in image convention
(v down), as the .mesh stores them.
"""
from __future__ import annotations

import math


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length) if length > 1e-12 else (0.0, 0.0, 0.0)


def tangents(p: list[float], n: list[float], u0: list[float], tri: list[int]) -> list[float]:
    """Per-vertex tangents as the .mesh stores them (``ta``: x, y, z, handedness w = +-1), from
    flattened positions, normals, UVs and triangle indices. Per-triangle UV tangents are summed
    per vertex, made orthogonal to the normal, and w is the sign of (n x t) . bitangent.
    Checked against vanilla's stored tangents (``tangent_agreement``)."""
    count = len(p) // 3
    tsum = [[0.0, 0.0, 0.0] for _ in range(count)]
    bsum = [[0.0, 0.0, 0.0] for _ in range(count)]
    for k in range(0, len(tri), 3):
        i0, i1, i2 = tri[k], tri[k + 1], tri[k + 2]
        p0, p1, p2 = (p[i * 3:i * 3 + 3] for i in (i0, i1, i2))
        (s0, t0), (s1, t1), (s2, t2) = (u0[i * 2:i * 2 + 2] for i in (i0, i1, i2))
        e1, e2 = _sub(p1, p0), _sub(p2, p0)
        du1, dv1, du2, dv2 = s1 - s0, t1 - t0, s2 - s0, t2 - t0
        det = du1 * dv2 - du2 * dv1
        if abs(det) < 1e-12:
            continue
        r = 1.0 / det
        t = tuple((e1[j] * dv2 - e2[j] * dv1) * r for j in range(3))
        b = tuple((e2[j] * du1 - e1[j] * du2) * r for j in range(3))
        for i in (i0, i1, i2):
            for j in range(3):
                tsum[i][j] += t[j]
                bsum[i][j] += b[j]
    out = []
    for i in range(count):
        nv = tuple(n[i * 3:i * 3 + 3])
        t = _norm(_sub(tuple(tsum[i]), tuple(x * _dot(nv, tsum[i]) for x in nv)))
        if t == (0.0, 0.0, 0.0):  # degenerate UVs: any unit vector orthogonal to the normal
            t = _norm(_cross(nv, (0.0, 1.0, 0.0) if abs(nv[1]) < 0.9 else (1.0, 0.0, 0.0)))
        w = -1.0 if _dot(_cross(nv, t), tuple(bsum[i])) < 0 else 1.0
        out += [t[0], t[1], t[2], w]
    return out


def tangent_agreement(stored: list[float], computed: list[float]) -> dict:
    """How well ``computed`` tangents match ``stored`` ones (both ``ta`` arrays): the share of
    vertices within 10 degrees, the median angle, and the share whose handedness agrees."""
    angles, same_w = [], 0
    for i in range(0, len(stored), 4):
        a, b = stored[i:i + 3], computed[i:i + 3]
        c = max(-1.0, min(1.0, _dot(_norm(a), _norm(b))))
        angles.append(math.degrees(math.acos(c)))
        same_w += (stored[i + 3] < 0) == (computed[i + 3] < 0)
    count = len(angles) or 1
    ordered = sorted(angles)
    return {"within_10deg": sum(1 for x in angles if x <= 10) / count, "median_deg": ordered[len(ordered) // 2],
            "w_agrees": same_w / count}
