"""A run's progress (Make a part's last step and the Library): each of its stages with what it does and why, its state
and time, an overall bar and the time left; a stopped or failed run says what happened and what to do, in plain
words (core.errors), with the details behind a button."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QMessageBox, QProgressBar, QVBoxLayout

from eu4unitgen.gui.base import clear_layout, ghost, label, minutes

STATE_TEXT = {"running": "being made", "done": "ready", "failed": "stopped with an error", "stopped": "paused",
              "new": "not started"}


def state_text(r: dict) -> str:
    st = r.get("status") or {}
    if r.get("state") == "running":
        ram = next((s for s in (st.get("stages") or {}).values() if s.get("state") == "waiting_ram"), None)
        if ram:
            close = ", ".join(n for n, _ in ram.get("close") or [])
            return f"waiting for memory" + (f": close {close}" if close else "")
        left = time_left(r["folder"])
        return f"being made: about {minutes(left)} left" if left else "being made"
    return STATE_TEXT.get(r.get("state"), r.get("state", ""))


def time_left(folder: str | Path) -> float:
    from eu4unitgen.ref3d import flow
    st = flow.status(Path(folder)) or {}
    req = flow.request(Path(folder))
    est = flow.estimates(req["kind"], req)
    left, now = 0.0, time.time()
    for key in flow.stages_of(req["kind"], req):
        s = (st.get("stages") or {}).get(key) or {}
        if s.get("state") == "done":
            continue
        took = now - s.get("started", now) if s.get("state") == "running" else 0.0
        left += max(0.0, est[key] - took)
    return left


class StageList(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.grid = QGridLayout()
        self.lay.addLayout(self.grid)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setToolTip("How far the whole run is, by the time each step usually takes on this computer")
        self.lay.addWidget(self.bar)
        self.left = label("", "explain")
        self.lay.addWidget(self.left)
        self.error = label("", "bad", rich=True)
        self.error.hide()
        self.lay.addWidget(self.error)
        self.details = ghost("Details", self._details, "The technical details (send them with a problem report)")
        self.details.hide()
        self.lay.addWidget(self.details, 0, Qt.AlignLeft)
        self.rows: dict[str, tuple[QLabel, QLabel]] = {}
        self.kind = None
        self._st: dict = {}

    def _build(self, kind: str, req: dict | None = None) -> None:
        from eu4unitgen.ref3d import flow
        clear_layout(self.grid)
        self.rows = {}
        for r, key in enumerate(flow.stages_of(kind, req)):
            name = label(f"<b>{r + 1}. {flow.LABELS[key]}</b><br><span style='font-size:12px'>{flow.EXPLAIN[key]}</span>",
                         rich=True)
            state = label("–", "muted", wrap=False)
            state.setAlignment(Qt.AlignRight | Qt.AlignTop)
            self.grid.addWidget(name, r, 0)
            self.grid.addWidget(state, r, 1)
            self.rows[key] = (name, state)
        self.grid.setColumnStretch(0, 1)
        self.kind = kind

    def show_run(self, folder: str | Path) -> dict:
        """Fill from the run's status; -> the status."""
        from eu4unitgen.core import errors
        from eu4unitgen.ref3d import flow
        req = flow.request(Path(folder))
        plan = flow.stages_of(req["kind"], req)
        if self.kind != (req["kind"], plan):
            self._build(req["kind"], req)
            self.kind = (req["kind"], plan)
        st = flow.status(Path(folder)) or {"state": "new", "stages": {}}
        self._st = st
        state = st.get("state", "new")
        est = flow.estimates(req["kind"], req)
        now, done_s, total_s = time.time(), 0.0, 0.0
        for key, (_, lab) in self.rows.items():
            s = (st.get("stages") or {}).get(key) or {}
            ss = s.get("state", "waiting")
            total_s += est[key]
            if ss == "done":
                text = f"✓ done in {minutes(s['seconds'])}" if s.get("seconds") else "✓ done"
                done_s += est[key]
                name = "good"
                if s.get("warning"):  # a stage that finished with a caution (an imported model not shaped as asked)
                    text, name = f"⚠ done: {s['warning']}", "warn"
            elif ss == "running" and state == "running":
                took = now - s.get("started", now)
                text = f"⏳ {minutes(took)} of about {minutes(est[key])}"
                done_s += min(took, est[key] * 0.95)
                name = "warn"
            elif ss == "waiting_eu4":
                text, name = "waiting: close EU4 (it holds the graphics card)", "bad"
            elif ss == "waiting_gpu":
                text, name = "waiting its turn: another unit is using the graphics card", "warn"
            elif ss == "waiting_ram":
                close = ", ".join(f"{n} ({gb:.1f} GB)" for n, gb in s.get("close") or [])
                text = (f"waiting for memory: {s.get('free', '?')} GB free, it needs {s.get('need', '?'):.0f} GB"
                        + (f"; close {close}" if close else "; close programs you don't need"))
                name = "bad"
            elif ss == "waiting_units":
                text, name = "waiting for its horse and soldier to be ready", "warn"
            else:
                text, name = f"about {minutes(est[key])}", "muted"
            lab.setText(text)
            lab.setObjectName(name)
            lab.style().unpolish(lab)
            lab.style().polish(lab)
        self.bar.setValue(100 if st.get("done") else int(100 * done_s / max(total_s, 1)))
        if state == "running":
            self.left.setText(f"About {minutes(max(0.0, total_s - done_s))} left. You can close the app or use it "
                              "for something else: the run keeps going, and the Library shows it when you come back.")
        elif state == "stopped":
            self.left.setText("Paused: the app or the computer was closed while it ran. Resume carries on from the "
                              "last finished step.")
        elif st.get("done"):
            self.left.setText("Ready.")
        else:
            self.left.setText("")
        if st.get("error") and state in ("failed", "stopped"):
            running = next((k for k, v in (st.get("stages") or {}).items() if v.get("state") in ("running", "waiting")), None)
            x = errors.explain(st.get("error") or "", running)
            self.error.setText(f"{x['what']}<br>What to do: {x['todo']}")
            self.error.show()
            self.details.show()
        else:
            self.error.hide()
            self.details.hide()
        return st

    def _details(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Details")
        box.setText("The technical details of the error (send them with a problem report).")
        box.setDetailedText(self._st.get("trace") or self._st.get("error") or "")
        box.exec()
