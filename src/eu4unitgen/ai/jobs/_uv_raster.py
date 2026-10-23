"""UV-space triangle rasteriser for TRELLIS.2's texture bake (``o_voxel.postprocess.to_glb``), in place of
nvdiffrast (NVIDIA Source Code License: research or evaluation use only; SPEC M21). ``install()`` registers
it as ``nvdiffrast.torch`` before o_voxel is imported, so the real package is never loaded.

Matches what the bake reads from nvdiffrast:
  * ``rasterize(ctx, pos, tri, resolution)`` -> ([1, H, W, 4] of (u, v, z, face id + 1; 0 = empty), None);
    pixel centres at clip ((x + 0.5) / W * 2 - 1, (y + 0.5) / H * 2 - 1), row 0 at clip y = -1;
  * ``interpolate(attr, rast, tri)`` -> ([1, H, W, C] = u * a0 + v * a1 + (1 - u - v) * a2, None).
Where triangles overlap or share an edge, the highest face id wins. z is left at 0.
"""
import sys
import types

import torch

CHUNK = 1 << 24  # candidate texels per pass


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def rasterize(ctx, pos, tri, resolution, **_):
    h, w = int(resolution[0]), int(resolution[1])
    p = pos[0]
    xy = p[:, :2] / p[:, 3:4]
    xy = xy.double()  # float32 barycentrics are off by > 1e-4 on slivers
    px = (xy[:, 0] + 1) * 0.5 * w - 0.5  # integer = a pixel centre
    py = (xy[:, 1] + 1) * 0.5 * h - 0.5
    t = tri.long()
    vx, vy = px[t], py[t]  # [F, 3]
    x0 = vx.min(1).values.ceil().clamp(0, w - 1).long()
    x1 = vx.max(1).values.floor().clamp(-1, w - 1).long()
    y0 = vy.min(1).values.ceil().clamp(0, h - 1).long()
    y1 = vy.max(1).values.floor().clamp(-1, h - 1).long()
    nx, ny = (x1 - x0 + 1).clamp(min=0), (y1 - y0 + 1).clamp(min=0)
    n = nx * ny
    area = _cross(vx[:, 1] - vx[:, 0], vy[:, 1] - vy[:, 0], vx[:, 2] - vx[:, 0], vy[:, 2] - vy[:, 0])
    n = torch.where(area.abs() > 1e-12, n, torch.zeros_like(n))
    out = torch.zeros(h * w, 4, device=p.device, dtype=torch.float32)
    ends = torch.cumsum(n, 0)
    f_lo = 0
    while f_lo < len(n):
        base = int(ends[f_lo - 1]) if f_lo else 0
        f_hi = int(torch.searchsorted(ends, torch.tensor(base + CHUNK, device=ends.device), right=True))
        f_hi = max(f_hi, f_lo + 1)
        nc = n[f_lo:f_hi]
        total = int(nc.sum())
        if total:
            fid = torch.repeat_interleave(torch.arange(f_lo, f_hi, device=p.device), nc)
            local = torch.arange(total, device=p.device) - torch.repeat_interleave(torch.cumsum(nc, 0) - nc, nc)
            ix = x0[fid] + local % nx[fid]
            iy = y0[fid] + local // nx[fid]
            qx, qy = ix.double(), iy.double()
            ax, ay, bx, by, cx, cy = vx[fid, 0], vy[fid, 0], vx[fid, 1], vy[fid, 1], vx[fid, 2], vy[fid, 2]
            ar = area[fid]
            u = _cross(bx - qx, by - qy, cx - qx, cy - qy) / ar  # weight of vertex 0
            v = _cross(cx - qx, cy - qy, ax - qx, ay - qy) / ar  # weight of vertex 1
            eps = -1e-6
            keep = (u >= eps) & (v >= eps) & (1 - u - v >= eps)
            idx, fk = (iy * w + ix)[keep], fid[keep]
            # one face per texel (the highest id), or a row could mix u, v and id from different faces
            win = torch.full((h * w,), -1, device=p.device, dtype=fk.dtype).scatter_reduce_(0, idx, fk, "amax")
            sel = fk == win[idx]
            uk, vk, fk = u[keep][sel], v[keep][sel], fk[sel]
            out[idx[sel]] = torch.stack([uk, vk, torch.zeros_like(uk), (fk + 1).double()], 1).float()
        f_lo = f_hi
    return out.view(1, h, w, 4), None


def interpolate(attr, rast, tri, **_):
    a = attr[0] if attr.dim() == 3 else attr
    r = rast[0]
    out = torch.zeros(r.shape[0], r.shape[1], a.shape[1], device=a.device, dtype=a.dtype)
    mask = r[..., 3] > 0
    f = r[..., 3][mask].long() - 1
    u, v = r[..., 0][mask][:, None], r[..., 1][mask][:, None]
    t = tri.long()[f]
    out[mask] = u * a[t[:, 0]] + v * a[t[:, 1]] + (1 - u - v) * a[t[:, 2]]
    return out.unsqueeze(0), None


def install() -> None:
    """Register this module as ``nvdiffrast.torch`` (call before importing o_voxel)."""
    pkg, mod = types.ModuleType("nvdiffrast"), types.ModuleType("nvdiffrast.torch")
    mod.RasterizeCudaContext = lambda *a, **k: None
    mod.rasterize, mod.interpolate = rasterize, interpolate
    pkg.torch = mod
    sys.modules["nvdiffrast"], sys.modules["nvdiffrast.torch"] = pkg, mod
