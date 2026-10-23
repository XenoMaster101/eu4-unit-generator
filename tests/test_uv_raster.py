"""_uv_raster (the TRELLIS.2 bake's rasteriser, SPEC M21) against a per-texel brute force."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "eu4unitgen" / "ai" / "jobs"))
import _uv_raster  # noqa: E402


def _mesh():
    g = torch.Generator().manual_seed(3)
    uv = torch.rand(40, 2, generator=g)
    tri = torch.randint(0, 40, (30, 3), generator=g)
    pos = torch.cat([uv * 2 - 1, torch.zeros(40, 1), torch.ones(40, 1)], 1)[None]
    return uv, tri, pos


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_matches_brute_force(device):
    uv, tri, pos = _mesh()
    h, w = 24, 32
    rast, _ = _uv_raster.rasterize(None, pos.to(device), tri.to(device), [h, w])
    rast = rast.cpu()
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    q = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h], -1)  # texel centres in uv
    covered = torch.zeros(h, w, dtype=torch.bool)
    for f in range(len(tri)):
        a, b, c = uv[tri[f]]
        area = (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]
        if area.abs() < 1e-9:
            continue
        u = ((b - q)[..., 0] * (c - q)[..., 1] - (b - q)[..., 1] * (c - q)[..., 0]) / area
        v = ((c - q)[..., 0] * (a - q)[..., 1] - (c - q)[..., 1] * (a - q)[..., 0]) / area
        covered |= (u > 1e-4) & (v > 1e-4) & (1 - u - v > 1e-4)  # strictly inside: no edge ties
    got = rast[0, ..., 3] > 0
    assert torch.all(got[covered])
    # each written texel lies in the triangle it names, and interpolating uv gives the texel centre back
    out, _ = _uv_raster.interpolate(uv[None], rast, tri)
    assert torch.allclose(out[0][got], q[got], atol=1e-4)


def test_install_registers_module():
    _uv_raster.install()
    import nvdiffrast.torch as dr
    assert dr.rasterize is _uv_raster.rasterize
