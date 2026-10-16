"""Spike R0 driver (app venv): run the Blender job, then build contact sheets and the metrics
table with Pillow. Output: build/spike_r0/ only. Throwaway (SPEC Track R)."""
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "spike_r0"
BLENDER = os.environ.get("BLENDER_PATH") or r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
ANIMS = ["polearm_twohanded_idle", "polearm_twohanded_moving", "polearm_twohanded_attack"]


def run_job(args: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "result.json"
        ap = Path(tmp) / "args.json"
        ap.write_text(json.dumps({**args, "result_path": str(res), "eu4": os.environ["EU4_PATH"],
                                  "out_dir": str(OUT)}), encoding="utf-8")
        p = subprocess.run([BLENDER, "--background", "--python-exit-code", "1", "--python",
                            str(Path(__file__).with_name("spike_job.py")), "--", str(ap)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        (OUT / "blender.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        if p.returncode != 0 or not res.is_file():
            sys.exit("Blender job failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        return json.loads(res.read_text(encoding="utf-8"))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if "--map-only" in sys.argv:
        res = run_job({"anims": ANIMS, "map_only": True, "contact_frames": 8, "size": 220, "worst_size": 420, "map_px": 48})
        (OUT / "result_map.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        sys.exit(0)
    args = {"anims": ["polearm_twohanded_moving"] if quick else ANIMS, "only_metrics": quick,
            "contact_frames": 8, "size": 220, "worst_size": 420, "map_px": 48, "save_rest_render": not quick}
    result = run_job(args)
    (OUT / ("quick.json" if quick else "result.json")).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("ok", list(result["anims"]))
