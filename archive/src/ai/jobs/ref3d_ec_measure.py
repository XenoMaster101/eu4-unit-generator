"""Track M spike E-C: the texture seam metric (SPEC M9 'texture consistency on unseen surfaces': the
seam at the observed/unseen border). Runs ONLY in the AI environment (numpy, scikit-image).

In the panel's pattern texture, every unseen (prior) texel next to an observed one is a border
texel. Around each: the mean colour (Lab) of the observed texels and of the prior texels within
``radius``; CIEDE2000 (kL = 2, textile weighting, as the M5 colour metric) between the two.
Reports the mean and 95th percentile over the border, and the border's length.

args: {"texture", "evidence", "radius", "result_path"}
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(args_path):
    from PIL import Image
    from skimage.color import deltaE_ciede2000, rgb2lab
    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    tex = np.array(Image.open(a["texture"]).convert("RGB")).astype(np.float64) / 255.0
    evc = np.array(Image.open(a["evidence"]).convert("RGB")).astype(int)
    prior = (evc[..., 0] > 150) & (evc[..., 1] < 120)
    obs = ~prior
    lab = rgb2lab(tex)
    r = int(a.get("radius", 6))
    H, W = prior.shape
    pad = np.pad(obs, 1)
    border = prior & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
    ys, xs = np.nonzero(border)
    d = []
    for y, x in zip(ys[::2], xs[::2]):
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        o, p = obs[y0:y1, x0:x1], prior[y0:y1, x0:x1]
        if o.sum() < 4 or p.sum() < 4:
            continue
        L = lab[y0:y1, x0:x1]
        d.append(float(deltaE_ciede2000(L[o].mean(axis=0)[None], L[p].mean(axis=0)[None], kL=2)[0]))
    d = np.array(d)
    res = {"border_texels": int(border.sum()), "samples": len(d),
           "mean": float(d.mean()) if len(d) else None, "p95": float(np.percentile(d, 95)) if len(d) else None,
           "prior_share": float(prior.mean())}
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
