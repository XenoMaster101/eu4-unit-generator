"""The first-launch setup wizard (Track P4): the big pieces, one step at a time, each with a button, a progress bar,
its live output and plain instructions. Nothing is typed in a terminal: downloads run inside the app, commands run
hidden with their output shown here, and steps that need administrator rights go through Windows' own prompt.
The steps and their commands are ``core.setup``'s; this window only runs them and shows them.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar,
    QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

OVERVIEW = ("Everything setup downloads, and why: what each piece is, why the app needs it, its size, its licence, "
            "where it comes from and where it goes. Every step is needed; nothing starts until you press its button. "
            "You need an NVIDIA graphics card (8 GB or more), 32 GB of RAM and about 60 GB of disk; it takes 1 to 2 "
            "hours in all, mostly TRELLIS.2.")


class Downloader(QThread):
    progress = Signal(int, int)  # bytes so far, total (0: unknown)
    failed = Signal(str)

    def __init__(self, url: str, dest: str):
        super().__init__()
        self.url, self.dest = url, Path(dest)

    def run(self):
        try:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.dest.with_suffix(self.dest.suffix + ".part")
            with urllib.request.urlopen(self.url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    self.progress.emit(done, total)
            tmp.replace(self.dest)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class StepRunner(QObject):
    """Runs one step's commands in order (``core.setup.Cmd``); reports lines, progress and the end."""
    line = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool, str)

    def __init__(self, cmds):
        super().__init__()
        self.cmds, self.i = list(cmds), -1
        self.proc: QProcess | None = None
        self.dl: Downloader | None = None

    def start(self):
        self.next()

    def next(self):
        self.i += 1
        if self.i >= len(self.cmds):
            return self.finished.emit(True, "")
        c = self.cmds[self.i]
        self.line.emit(f"\n== {c.title}")
        self.progress.emit(0, 0)
        if c.download:
            url, dest = c.download
            if Path(dest).is_file() and Path(dest).stat().st_size > 0:
                self.line.emit(f"already downloaded: {dest}")
                return self.next()
            self.dl = Downloader(url, dest)
            self.dl.progress.connect(self.progress.emit)
            self.dl.failed.connect(lambda m: self.finished.emit(False, m))
            self.dl.finished.connect(lambda: self.next() if Path(dest).is_file() else None)
            self.dl.start()
            return
        argv = c.argv
        if c.admin and sys.platform == "win32":  # through Windows' own prompt; only the exit code comes back
            args = ",".join("'" + a.replace("'", "''") + "'" for a in argv[1:])
            argv = ["powershell", "-NoProfile", "-Command",
                    f"$p = Start-Process -FilePath '{argv[0]}' -ArgumentList {args} -Verb RunAs -Wait -PassThru "
                    f"-WindowStyle Hidden; exit $p.ExitCode"]
            self.line.emit("(Windows asks for permission; the step runs hidden and reports when done)")
        self.proc = QProcess()
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        if c.env:
            env = QProcessEnvironment.systemEnvironment()
            for k, v in c.env.items():
                env.insert(k, v)
            self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(lambda e: self.finished.emit(False, f"couldn't start {argv[0]} ({e})")
                                        if e == QProcess.FailedToStart else None)
        self.proc.start(argv[0], argv[1:])

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput())
        text = data.decode("utf-16-le", errors="ignore") if b"\x00" in data[:40] else data.decode(errors="replace")
        for ln in text.replace("\r", "\n").splitlines():
            if ln.strip():
                self.line.emit(ln)

    def _done(self, code, _status):
        if code != 0:
            return self.finished.emit(False, f"{self.cmds[self.i].title} stopped (exit {code})")
        self.next()


class SetupWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from eu4unitgen.core import setup
        self.setWindowTitle("EU4 Unit Generator: setup")
        self.resize(1050, 680)
        from eu4unitgen.core.setup import Step
        self.steps = [Step("overview", "What gets downloaded, and why", OVERVIEW, "", 0)] + setup.steps()
        self.root = setup.default_ai_root()
        self.runner: StepRunner | None = None
        lay = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.setMaximumWidth(300)
        self.list.currentRowChanged.connect(self.show_step)
        lay.addWidget(self.list)
        right = QVBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.what = QLabel()
        self.what.setWordWrap(True)
        self.needs = QLabel()
        self.needs.setWordWrap(True)
        self.needs.setStyleSheet("font-style: italic;")
        for w in (self.title, self.what, self.needs):
            right.addWidget(w)
        self.downloads = QTextBrowser()  # this step's downloads (the overview: all of them)
        self.downloads.setOpenExternalLinks(True)
        self.downloads.setMinimumHeight(140)
        right.addWidget(self.downloads, 2)
        # the AI folder step: a folder and the drives' free space
        self.folder_box = QWidget()
        fb = QHBoxLayout(self.folder_box)
        fb.setContentsMargins(0, 0, 0, 0)
        self.folder = QLineEdit(str(self.root))
        pick = QPushButton("Choose…")
        pick.clicked.connect(self._pick_folder)
        fb.addWidget(self.folder, 1)
        fb.addWidget(pick)
        right.addWidget(self.folder_box)
        self.drives = QLabel()
        right.addWidget(self.drives)
        # the weights step: the Hugging Face token
        self.token_box = QWidget()
        tb = QHBoxLayout(self.token_box)
        tb.setContentsMargins(0, 0, 0, 0)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("paste a Hugging Face read token (used for this download only, never stored)")
        open_page = QPushButton("Open the DINOv3 licence page")
        open_page.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(setup.GATED["dinov3-vitl16-pretrain-lvd1689m"])))
        tokens = QPushButton("Make a token")
        tokens.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens")))
        tb.addWidget(self.token, 1)
        tb.addWidget(open_page)
        tb.addWidget(tokens)
        right.addWidget(self.token_box)
        row = QHBoxLayout()
        self.go = QPushButton("Run this step")
        self.go.clicked.connect(self.run_step)
        self.state = QLabel()
        row.addWidget(self.go)
        row.addWidget(self.state, 1)
        right.addLayout(row)
        self.bar = QProgressBar()
        right.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom.addWidget(close)
        right.addLayout(bottom)
        lay.addLayout(right, 1)
        self.refresh()
        self.list.setCurrentRow(0)  # the overview first: every download shown before any starts

    def refresh(self):
        from eu4unitgen.core import setup
        try:
            self.done_steps = setup.state()
        except Exception:  # noqa: BLE001 - shows as not done
            self.done_steps = {}
        cur = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        n = 0
        for s in self.steps:
            if s.key == "overview":
                self.list.addItem(QListWidgetItem(f"ⓘ  {s.title}"))
                continue
            n += 1
            self.list.addItem(QListWidgetItem(f"{'✔' if self.done_steps.get(s.key) else '○'}  {n}. {s.title}"))
        self.list.blockSignals(False)
        if cur >= 0:
            self.list.setCurrentRow(cur)
        d = setup.drives()
        self.drives.setText("Free space: " + ", ".join(f"{p.drive or p} {gb:.0f} GB" for p, gb in d[:6]))

    def show_step(self, i):
        if i < 0:
            return
        s = self.steps[i]
        self.title.setText(s.title)
        self.what.setText(s.what)
        self.needs.setText(("Needs: " + s.needs) if s.needs else "")
        self.folder_box.setVisible(s.key == "ai_folder")
        self.drives.setVisible(s.key == "ai_folder")
        self.token_box.setVisible(s.key == "weights")
        from eu4unitgen.core import setup
        over = s.key == "overview"
        self.downloads.setHtml(("<p><b>Downloads in this step:</b></p>" if not over else "")
                               + setup.downloads_html(None if s.key == "overview" else s.key))
        self.downloads.setVisible(s.key != "ai_folder")
        self.log.setVisible(not over)
        self.bar.setVisible(not over)
        self.go.setText("Start with the first step not done" if s.key == "overview"
                        else "Use this folder" if s.key == "ai_folder"
                        else ("Run again" if self.done_steps.get(s.key) else "Download and run this step"))
        self.state.setText("done" if self.done_steps.get(s.key) else "")
        if s.key == "blender" and self.done_steps.get("blender"):
            self.state.setText("Blender 4.2 was found: nothing to install.")

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Where the AI parts go", self.folder.text())
        if d:
            self.folder.setText(str(Path(d) / "eu4unitgen_ai") if Path(d).name != "eu4unitgen_ai" else d)

    def run_step(self):
        from eu4unitgen.core import config, setup
        s = self.steps[self.list.currentRow()]
        if s.key == "overview":  # to the first step not done
            start = self.list.currentRow() + 1
            return self.list.setCurrentRow(next((i for i, t in enumerate(self.steps) if i >= start
                                                 and t.key != "overview" and not self.done_steps.get(t.key)),
                                                start))
        self.root = Path(self.folder.text().strip() or self.root)
        if s.key == "ai_folder":
            self.root.mkdir(parents=True, exist_ok=True)
            config.update(ai_root=str(self.root))
            self.log.appendPlainText(f"The AI parts go in {self.root}.")
            return self._after(True, "")
        if s.key == "weights" and not self.token.text().strip() and not self.done_steps.get("weights"):
            self.log.appendPlainText("DINOv3 needs a Hugging Face token: open its licence page, accept it, make a read "
                                     "token and paste it above. (The other weights download without one.)")
        try:
            cmds = setup.plan(s.key, self.token.text().strip() or None, self.root)
        except setup.SetupError as exc:
            self.state.setText(str(exc))
            self.log.appendPlainText(f"Can't run this step yet: {exc}")
            return
        self.go.setEnabled(False)
        self.state.setText("running…")
        self.runner = StepRunner(cmds)
        self.runner.line.connect(self.log.appendPlainText)
        self.runner.line.connect(self._to_file)  # settings\logs\setup.log: what a step printed, to read afterwards
        self._to_file(f"\n######## {s.title} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        self.runner.progress.connect(self._progress)
        self.runner.finished.connect(self._after)
        self.runner.start()

    def _to_file(self, line: str) -> None:
        from eu4unitgen.core import config
        try:
            d = config.logs_dir()
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "setup.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _progress(self, done, total):
        if total:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(1000 * done / total))
            self.bar.setFormat(f"{done / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB")
        else:
            self.bar.setRange(0, 0)  # busy

    def _after(self, ok: bool, msg: str):
        from eu4unitgen.core import setup
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if ok else 0)
        self.go.setEnabled(True)
        s = self.steps[self.list.currentRow()]
        self._to_file(f"== {'finished' if ok else 'stopped: ' + msg}")
        if ok:
            setup.adopt(self.root)
            self.state.setText("done" + (": restart Windows, then open the app again; setup continues here." if s.restart else ""))
            self.log.appendPlainText("Done." if not s.restart else "Done. Restart Windows now, then open the app again.")
        else:
            from eu4unitgen.core import errors
            x = errors.explain(msg, "setup")
            self.state.setText(f"stopped: {x['what']}")
            self.log.appendPlainText(f"\nStopped: {msg}\nWhat to do: {x['todo']}")
        self.refresh()


def open_wizard(parent=None) -> int:
    return SetupWizard(parent).exec()


if __name__ == "__main__":  # a look at the window without the app
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = SetupWizard()
    w.show()
    sys.exit(app.exec())
