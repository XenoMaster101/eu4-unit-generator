"""A run's status file (the app's detached runs, ref3d.flow): rewritten whole on each change and every ``BEAT``
seconds (the heartbeat), so the app can tell a running run from one that was stopped (closed, crashed, the PC
switched off). Standard library only.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

BEAT = 15.0  # seconds between heartbeats; a status older than 4 beats is a stopped run
STATUS = "status.json"


def read_status(folder: Path) -> dict | None:
    p = Path(folder) / STATUS
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(folder: Path) -> dict | None:
    """The status file plus ``alive`` (its process beat recently) and ``state``: running, done, failed, stopped
    (closed or killed: resume it)."""
    st = read_status(folder)
    if st is None:
        return None
    st["alive"] = time.time() - st.get("heartbeat", 0) < 4 * BEAT
    st["state"] = "done" if st.get("done") else "failed" if st.get("error") else "running" if st["alive"] else "stopped"
    return st


class Status:
    """The status file of a running process, with its heartbeat thread."""

    def __init__(self, folder: Path, base: dict):
        self.path, self.d = Path(folder) / STATUS, base
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.write()
        self.t = threading.Thread(target=self._beat, daemon=True)
        self.t.start()

    def write(self, **kw):
        with self.lock:
            self.d.update(kw)
            self.d["heartbeat"] = time.time()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def stage(self, key: str, state: str, **kw):
        s = self.d["stages"].setdefault(key, {})
        if state == "running":
            s["started"] = time.time()
            s.pop("seconds", None)
        if state == "done" and "started" in s:
            s.setdefault("seconds", round(time.time() - s["started"], 1))
        s.update(state=state, **kw)
        self.write(stage=key)

    def _beat(self):
        while not self.stop.wait(BEAT):
            self.write()

    def close(self):
        self.stop.set()
        self.t.join()
