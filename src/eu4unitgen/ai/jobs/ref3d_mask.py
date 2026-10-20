"""Track M S1 job: foreground masks with rembg's BiRefNet-general (MIT). Runs ONLY in the AI
environment (eu4unitgen.ai.env.AIJobRunner; U2NET_HOME points at the model folder).

args: {"images": {name: path}, "out_dir", "result_path"}
Writes ``<name>_fg.png`` (8-bit alpha). Result: {name: {"path", "share"}}.
"""
import json
import sys
from pathlib import Path


def main(args_path):
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove

    a = json.loads(Path(args_path).read_text(encoding="utf-8"))
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    session = new_session("birefnet-general")
    res = {}
    for name, path in a["images"].items():
        alpha = remove(Image.open(path).convert("RGB"), session=session, only_mask=True)
        p = out / f"{name}_fg.png"
        alpha.save(p)
        res[name] = {"path": str(p), "share": round(float((np.array(alpha) > 127).mean()), 4)}
        print(name, res[name], flush=True)
    Path(a["result_path"]).write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
