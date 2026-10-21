"""The machine's RAM for Track M's long runs (SPEC M18 step 5: a run was stopped when the PC ran low on
memory): total and free, a sampler of the lowest free RAM during a step, and the fit's worker count.
Standard library only (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo).
"""
from __future__ import annotations

import sys
import threading

# assumed until measured: one E-A fit process's peak RAM, and what stays free for the system and other programs
FIT_WORKER_GB = 2.5
RESERVE_GB = 6.0
# the least free RAM to start each step with (the TripoSG job loads its model in its own process)
STEP_MIN_GB = {"shells": 10.0, "fit": RESERVE_GB + FIT_WORKER_GB, "refine": 8.0, "master": 8.0, "game": 8.0}


def ram_gb() -> tuple[float, float]:
    """(total, free) RAM in GB."""
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
    info = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) / 2 ** 20
    return info["MemTotal"], info.get("MemAvailable", info["MemFree"])


def fit_workers(free_gb: float, cap: int) -> int:
    """As many fit processes as the free RAM holds after the reserve (at least 1, at most ``cap``)."""
    return max(1, min(cap, int((free_gb - RESERVE_GB) // FIT_WORKER_GB)))


class LowWatermark:
    """The lowest free RAM seen while a step runs (sampled every ``every`` s in a thread)."""

    def __init__(self, every: float = 5.0):
        self.every, self.low = every, None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            free = ram_gb()[1]
            self.low = free if self.low is None else min(self.low, free)
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        return False


if __name__ == "__main__":
    t, f = ram_gb()
    print(f"RAM: {t:.1f} GB total, {f:.1f} GB free; the fit would run {fit_workers(f, 6)} process(es)")
