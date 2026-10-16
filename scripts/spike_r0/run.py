"""Spike R0 driver (app venv): run the Blender job, then build contact sheets and the metrics
table with Pillow. Output: build/spike_r0/ only. Throwaway (SPEC Track R)."""
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "spike_r0"
BLENDER = os.environ.get("BLENDER_PATH") or r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
ANIMS = ["polearm_twohanded_idle", "polearm_twohanded_moving", "polearm_twohanded_attack"]


def run_job(args: dict, job: str = "spike_job.py") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "result.json"
        ap = Path(tmp) / "args.json"
        ap.write_text(json.dumps({**args, "result_path": str(res), "eu4": os.environ["EU4_PATH"],
                                  "out_dir": args.get("out_dir_override") or str(OUT)}), encoding="utf-8")
        p = subprocess.run([BLENDER, "--background", "--python-exit-code", "1", "--python",
                            str(Path(job) if "/" in job or "\\" in job else Path(__file__).with_name(job)), "--", str(ap)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        (OUT / "blender.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        if p.returncode != 0 or not res.is_file():
            sys.exit("Blender job failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        return json.loads(res.read_text(encoding="utf-8"))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if "--p2" in sys.argv:  # test 10 pre-check P2: export + re-import the weighted body
        sub = OUT / "p2_export"
        sub.mkdir(parents=True, exist_ok=True)
        res = run_job({"out_dir_override": str(sub), "patch_px": [105, 30, 115, 40], "tex": 512}, job="p2_job.py")
        (sub / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "out_dir"}, indent=1)[:3000])
        sys.exit(0)
    if "--p1-metrics" in sys.argv:  # P1 metrics again with the facing-frame C5 and per-ring penetration
        res = run_job({"anims": ["spear_shield_idle", "spear_shield_moving", "spear_shield_attack"],
                       "parts": ["tail_flared", "pony_short", "pony_tuft15", "pony_bun"], "only_metrics": True,
                       "out_dir_override": str(OUT / "p1_metrics")})
        (OUT / "p1_spear_shield" / "metrics2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        sys.exit(0)
    if "--p1-rings" in sys.argv:  # metrics only: which rings of the flared tail penetrate
        res = run_job({"anims": ["spear_shield_idle", "spear_shield_moving", "spear_shield_attack"],
                       "parts": ["tail_flared"], "only_metrics": True, "out_dir_override": str(OUT / "p1_rings")})
        for a, r in res["anims"].items():
            rings = sorted({x for m in r["per_frame"]["tail_flared_C"] for x in m["pen_rings"]})
            print(a, "penetrating rings:", rings, "rest:", r["rest"]["tail_flared_C"]["pen_rings"])
        sys.exit(0)
    if "--p1" in sys.argv:  # test 10 pre-check: spear_shield anims, flared tail, last ponytail round
        sub = OUT / "p1_spear_shield"
        res = run_job({"anims": ["spear_shield_idle", "spear_shield_moving", "spear_shield_attack"],
                       "parts": ["tail_flared", "pony_short", "pony_tuft15", "pony_bun"],
                       "out_dir_override": str(sub), "contact_frames": 8, "size": 220, "worst_size": 420,
                       "map_px": 48})
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("ok p1", list(res["anims"]))
        sys.exit(0)
    if "--ponytail-rerun" in sys.argv:  # R0b: alternatives 1 (D) and 2 (E), same criteria and renders
        sub = OUT / "rerun_ponytail"
        res = run_job({"anims": ANIMS, "parts": ["pony_short", "pony_long"], "out_dir_override": str(sub),
                       "contact_frames": 8, "size": 220, "worst_size": 420, "map_px": 48})
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("ok rerun", list(res["anims"]))
        sys.exit(0)
    if "--map-only" in sys.argv:
        res = run_job({"anims": ANIMS, "map_only": True, "contact_frames": 8, "size": 220, "worst_size": 420, "map_px": 48})
        (OUT / "result_map.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        sys.exit(0)
    args = {"anims": ["polearm_twohanded_moving"] if quick else ANIMS, "only_metrics": quick,
            "contact_frames": 8, "size": 220, "worst_size": 420, "map_px": 48, "save_rest_render": not quick}
    result = run_job(args)
    (OUT / ("quick.json" if quick else "result.json")).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("ok", list(result["anims"]))
