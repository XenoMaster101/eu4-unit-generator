"""The Horse screen (SPEC M19 step 1): a horse from one image. Drop in a side view, give it a name and the
rights statement, press Generate; the stages show with their times and what's left, then the final sheet,
and "Add to mod" stages the game version in a test mod.

Thin by design: the run is ``ref3d.job`` in its own detached process (it keeps going when the window
closes; reopening the app shows it again, and a stopped run resumes). This screen only starts it, reads
its status file every few seconds and shows it.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from eu4unitgen.gui.home_page import css
from eu4unitgen.gui.pages import Page, open_folder
from eu4unitgen.gui.theme import tokens
from eu4unitgen.ref3d import job

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
POLL_MS = 3000
STATE_TEXT = {"running": "running", "done": "done", "failed": "failed", "stopped": "stopped: press Resume"}


def minutes(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{sec:.0f} s" if sec < 90 else f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


class DropImage(QLabel):
    """The image well: drop a picture on it, or click to choose one."""

    def __init__(self, on_image):
        super().__init__("Drop a side view of the horse here\n(or click to choose)")
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 220)
        self.setCursor(Qt.PointingHandCursor)
        self.on_image = on_image

    def dragEnterEvent(self, e):  # noqa: N802 - Qt name
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802 - Qt name
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.on_image(Path(urls[0]))

    def mousePressEvent(self, e):  # noqa: N802 - Qt name
        path, _ = QFileDialog.getOpenFileName(self, "Choose the horse image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.on_image(Path(path))


class HorsePage(Page):
    title = ""

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("front")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.image: Path | None = None
        self.current: str | None = None  # the run shown (its folder)
        self.shown_sheet: str | None = None

        top = QHBoxLayout()
        back = QPushButton("← Home")
        back.setObjectName("ghost")
        back.clicked.connect(lambda: window.show_home())
        top.addWidget(back)
        h = QLabel("A horse from one image")
        h.setObjectName("h1")
        top.addWidget(h)
        top.addStretch(1)
        self.layout_.addLayout(top)

        body = QHBoxLayout()
        # --- left: the inputs
        left = QVBoxLayout()
        self.drop = DropImage(self.set_image)
        left.addWidget(self.drop)
        form = QGridLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. light_cavalry_horse")
        self.name.setToolTip("Lower case letters, digits and _; also the folder's name")
        self.rights = QLineEdit()
        self.rights.setPlaceholderText("who made the image, and that you may use it")
        self.rights.setToolTip("Rule 8: say who made the image and that you may use it in your mod. Required; "
                               "it is recorded with the horse.")
        self.ai = QLineEdit()
        self.ai.setPlaceholderText("the AI environment's python.exe")
        self.ai.setToolTip("The python of the AI environment (TripoSG, SAM 2, SigLIP 2); the models are found "
                           "beside it")
        pick_ai = QPushButton("Browse…")
        pick_ai.setObjectName("ghost")
        pick_ai.clicked.connect(self._pick_ai)
        for r, (label, w) in enumerate((("Name", self.name), ("Rights", self.rights), ("AI python", self.ai))):
            form.addWidget(QLabel(label), r, 0)
            form.addWidget(w, r, 1)
        form.addWidget(pick_ai, 2, 2)
        left.addLayout(form)
        self.warnings = QLabel()
        self.warnings.setObjectName("warn")
        self.warnings.setWordWrap(True)
        left.addWidget(self.warnings)
        self.generate = QPushButton("Generate")
        self.generate.setObjectName("generate")
        self.generate.setToolTip("Read the image and build the horse (about 2 hours; you can close the app)")
        self.generate.clicked.connect(self.go)
        left.addWidget(self.generate, 0, Qt.AlignLeft)
        left.addWidget(QLabel("Horses made here:"))
        self.list = QListWidget()
        self.list.setMaximumHeight(150)
        self.list.currentItemChanged.connect(lambda cur, _: self.show_run(cur.data(Qt.UserRole) if cur else None))
        left.addWidget(self.list)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(460)
        body.addWidget(lw)

        # --- right: progress, sheet, actions
        right = QVBoxLayout()
        self.run_title = QLabel("No run yet.")
        self.run_title.setObjectName("h2")
        right.addWidget(self.run_title)
        self.stage_box = QFrame()
        self.stage_box.setObjectName("card")
        grid = QGridLayout(self.stage_box)
        self.stage_rows: dict[str, tuple[QLabel, QLabel]] = {}
        for r, key in enumerate(job.STAGES):
            name, state = QLabel(job.LABELS[key]), QLabel("–")
            state.setObjectName("muted")
            grid.addWidget(name, r, 0)
            grid.addWidget(state, r, 1)
            self.stage_rows[key] = (name, state)
        self.left_time = QLabel()
        grid.addWidget(self.left_time, len(job.STAGES), 0, 1, 2)
        right.addWidget(self.stage_box)
        self.sheet = QLabel("The final sheet shows here when the run is done.")
        self.sheet.setObjectName("cardpic")
        self.sheet.setAlignment(Qt.AlignCenter)
        self.sheet.setMinimumHeight(240)
        right.addWidget(self.sheet, 1)
        acts = QHBoxLayout()
        self.open_sheet = QPushButton("Open sheet")
        self.open_sheet.setObjectName("ghost")
        self.open_sheet.clicked.connect(lambda: self.current and QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(self.current) / "final_sheet.png"))))
        self.open_dir = QPushButton("Open folder")
        self.open_dir.setObjectName("ghost")
        self.open_dir.clicked.connect(lambda: self.current and open_folder(Path(self.current)))
        self.add = QPushButton("Add to mod")
        self.add.setObjectName("generate")
        self.add.setToolTip("Stage the game version in a test mod (tier 1 this horse, tier 2 the vanilla horse)")
        self.add.clicked.connect(self.add_to_mod)
        for b in (self.open_sheet, self.open_dir):
            acts.addWidget(b)
        acts.addStretch(1)
        acts.addWidget(self.add)
        right.addLayout(acts)
        right.addWidget(self.message)
        body.addLayout(right, 1)
        self.layout_.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.refresh)
        self.restyle()

    # ------------------------------------------------------------------ settings and paths
    def workspace(self) -> Path:
        return Path(self.window_.horse_workspace)

    def root(self) -> Path:
        return Path("assets_in")

    def _pick_ai(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "The AI environment's python", self.ai.text(), "python (python*)")
        if path:
            self.ai.setText(path)

    def restyle(self) -> None:
        t = tokens()
        self.setStyleSheet(css(t) + f"""
QLabel#drop {{ background: {t['surface']}; color: {t['muted']}; border: 2px dashed {t['border']}; border-radius: 12px; }}
QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QLabel#cardpic {{ background: {t['alt']}; border-radius: 6px; color: {t['muted']}; }}
QLabel#h2 {{ font-size: 17px; font-weight: 600; }}
QLabel#bad {{ color: {t['error']}; font-weight: 600; }}
QLabel#good {{ color: {t['ok']}; }}
QListWidget {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px; }}
""")

    def showEvent(self, e):  # noqa: N802 - Qt name
        super().showEvent(e)
        self.refresh_list()
        self.timer.start()

    def hideEvent(self, e):  # noqa: N802 - Qt name
        super().hideEvent(e)
        self.timer.stop()

    # ------------------------------------------------------------------ input
    def set_image(self, path: Path) -> None:
        self.image = Path(path)
        pix = QPixmap(str(path))
        if pix.isNull():
            self.image = None
            return self.show_message(f"{path.name} isn't an image this app can read.", error=True)
        self.drop.setPixmap(pix.scaled(self.drop.width() - 8, self.drop.height() - 8, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
        if not self.name.text().strip():
            self.name.setText(re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_")[:40] or "horse")
        found = job.precheck(path)
        self.warnings.setText("\n".join("⚠ " + w for w in found))
        self.show_message("")

    def problems(self) -> list[str]:
        """What's missing before Generate (each a sentence)."""
        out = []
        if self.image is None:
            out.append("Drop in an image of the horse.")
        name = self.name.text().strip()
        if not NAME.match(name):
            out.append("The name: lower case letters, digits and _, starting with a letter.")
        elif (self.workspace() / self.root() / name).exists():
            out.append(f"There is already a horse called {name}: pick another name (or resume it from the list).")
        if not self.rights.text().strip():
            out.append("The rights statement: who made the image and that you may use it (rule 8).")
        if not Path(self.ai.text().strip()).is_file():
            out.append("The AI environment's python wasn't found.")
        eu4 = self.window_.settings_page.eu4.text().strip()
        if not eu4 or not (Path(eu4) / "gfx").is_dir():
            out.append("Set the EU4 install in Advanced editor → Settings (the vanilla horse's skeleton is read from it).")
        return out

    # ------------------------------------------------------------------ actions
    def go(self) -> None:
        st = job.status(self.workspace() / self.current) if self.current else None
        resume = st is not None and st["state"] in ("stopped", "failed")
        if resume:
            image, name, rights = Path(st["image"]), st["name"], st["rights"]
        else:
            bad = self.problems()
            if bad:
                return self.show_message(" ".join(bad), error=True)
            image, name, rights = self.image, self.name.text().strip(), self.rights.text().strip()
        warn = [] if resume else job.precheck(image)
        ram, info = job.ram_check()
        if ram:
            warn.append(ram)
        if warn and not self.window_.confirm(
                "Before you start", "Start anyway?\n\n" + "\n\n".join(warn),
                f"Free RAM {info['free_gb']} of {info['total_gb']} GB; the fit would run {info['fit_workers']} process(es)."):
            return
        self.window_._sync_settings()
        s = self.window_.session.settings
        job.start(image, name, rights, self.ai.text().strip(), s.eu4_path, s.blender_path, self.root(), self.workspace())
        self.window_.store_horse(ai_python=self.ai.text().strip())
        self.current = str(self.root() / name)
        self.show_message(("Resumed" if resume else "Started") + f" {name}. You can close the app; it keeps running.")
        QTimer.singleShot(1500, self.refresh_list)

    def add_to_mod(self) -> None:
        if not self.current:
            return
        folder = self.workspace() / self.current

        def work():
            r = subprocess.run([sys.executable, "-m", "eu4unitgen.ref3d.job", "stage", str(Path(self.current))],
                               cwd=str(self.workspace()), capture_output=True, text=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode:
                raise OSError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "staging failed")
            return r.stdout

        def done(out: str) -> None:
            import json
            res = json.loads(out)
            where = self.workspace() / res["stage"]
            self.show_message(f"Staged {folder.name} ({res['game']}) at tier 1 of a test mod: {where}"
                              + ("" if res["exit"] == 0 else f" (the staging tool exited {res['exit']}; see its log.txt)"),
                              error=res["exit"] != 0)
            open_folder(where)
        self.window_.run(self, "Adding the horse to a test mod", work, done)

    # ------------------------------------------------------------------ the runs
    def refresh_list(self) -> None:
        runs = job.runs(self.workspace() / self.root())
        keep = self.current
        self.list.blockSignals(True)
        self.list.clear()
        for r in runs:
            rel = str(Path(r["folder"]).relative_to(self.workspace()))
            it = QListWidgetItem(f"{r['name']}  ·  {STATE_TEXT[r['state']]}")
            it.setData(Qt.UserRole, rel)
            self.list.addItem(it)
            if rel == keep or (keep is None and self.list.count() == 1):
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        cur = self.list.currentItem()
        self.show_run(cur.data(Qt.UserRole) if cur else None)

    def refresh(self) -> None:
        if self.current:
            self.show_run(self.current)

    def show_run(self, rel: str | None) -> None:
        self.current = rel
        st = job.status(self.workspace() / rel) if rel else None
        self.generate.setText("Resume" if st and st["state"] in ("stopped", "failed") else "Generate")
        self.generate.setEnabled(not (st and st["state"] == "running"))
        self.add.setEnabled(bool(st and st["state"] == "done"))
        self.open_sheet.setEnabled(bool(st and st.get("sheet")))
        self.open_dir.setEnabled(st is not None)
        if st is None:
            self.run_title.setText("No run yet.")
            for _, state in self.stage_rows.values():
                state.setText("–")
            self.left_time.setText("")
            return
        self.run_title.setText(f"{st['name']}: {STATE_TEXT[st['state']]}")
        workers = (st["stages"].get("fit") or {}).get("workers")
        est = job.estimates(self.workspace() / self.root(), workers)
        left, now = 0.0, time.time()
        for key in job.STAGES:
            s = st["stages"].get(key) or {}
            state = s.get("state", "waiting")
            label = self.stage_rows[key][1]
            if state == "done":
                text = f"done in {minutes(s['seconds'])}" if s.get("seconds") else "done"
            elif state == "running" and st["state"] == "running":
                took = now - s.get("started", now)
                text = f"running: {minutes(took)} of about {minutes(est[key])}"
                left += max(0.0, est[key] - took)
            elif state == "skipped":
                text = "not needed (the light path)"
            elif state == "low_ram":
                text = f"stopped: {s.get('free_gb')} GB RAM free, {s.get('need_gb'):.0f} GB needed; close programs, Resume"
            else:
                text = f"waiting (about {minutes(est[key])})"
                left += est[key]
            label.setText(text)
            label.setObjectName("bad" if state == "low_ram" else "good" if state == "done" else "muted")
            label.style().unpolish(label)
            label.style().polish(label)
        if st["state"] == "running":
            self.left_time.setText(f"About {minutes(left)} left. You can close the app; the run continues.")
        elif st["state"] == "failed":
            self.left_time.setText("Failed: " + (st.get("error") or "")[-300:] + "\nResume continues from the last finished stage.")
        elif st["state"] == "stopped":
            self.left_time.setText("Stopped (the app or PC was closed). Resume continues from the last finished stage.")
        else:
            self.left_time.setText(f"Done. Parts found: {', '.join(k.replace('_', ' ') for k, v in (st.get('parts') or {}).items() if v)}.")
        sheet = st.get("sheet")
        if sheet and sheet != self.shown_sheet:
            p = Path(sheet) if Path(sheet).is_absolute() else self.workspace() / sheet
            pix = QPixmap(str(p))
            if not pix.isNull():
                self.sheet.setPixmap(pix.scaled(max(400, self.sheet.width() - 8), max(240, self.sheet.height() - 8),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.shown_sheet = sheet
        elif not sheet:
            self.sheet.setPixmap(QPixmap())
            self.sheet.setText("The final sheet shows here when the run is done.")
            self.shown_sheet = None
