"""Long AI jobs: resume after an interruption, and stop cleanly when the AI drive drops out
(2026-10-20: the drive holding the AI environment vanished mid-run, four times in five days).

DriveWatch: a thread reads a file of the AI environment every few seconds; once that fails,
``check()`` raises DriveLost. Jobs call ``check()`` at every evaluation, save their state and exit
with EXIT_DRIVE_LOST. Every library a job needs is imported before the run starts, so nothing is
loaded from the AI drive mid-run.

Checkpoints: JSON files in ``<out_dir>/checkpoints`` (the project drive), written atomically
(temp file, then replace). A job rerun into the same out_dir skips the work they record.

Stdlib only: imported by the app's tests as well as by jobs in the AI environment.
"""
import json
import os
import sys
import threading
from pathlib import Path

EXIT_DRIVE_LOST = 75  # EX_TEMPFAIL: the job stopped cleanly; rerun it into the same out_dir to resume
PROBE_ENV = "EU4UNITGEN_DRIVE_PROBE"  # a file to probe instead of the environment's python (tests)


class DriveLost(RuntimeError):
    pass


def probe_path(probe=None) -> Path:
    return Path(probe or os.environ.get(PROBE_ENV) or sys.executable)


def drive_ok(probe=None) -> bool:
    try:
        with open(probe_path(probe), "rb") as f:
            f.read(4096)
        return True
    except OSError:
        return False


class DriveWatch:
    def __init__(self, probe=None, interval: float = 5.0):
        self.probe = probe_path(probe)
        self.interval = interval
        self.lost = threading.Event()
        self._stop = threading.Event()
        if not drive_ok(self.probe):
            self.lost.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            if not drive_ok(self.probe):
                self.lost.set()
                return

    def check(self):
        if self.lost.is_set():
            raise DriveLost(f"the AI drive is gone ({self.probe} can't be read)")

    def stop(self):
        self._stop.set()


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


class Checkpoints:
    def __init__(self, out_dir):
        self.dir = Path(out_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def load(self, key: str):
        p = self.path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def save(self, key: str, data) -> None:
        p = self.path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=_jsonable), encoding="utf-8")
        os.replace(tmp, p)
