"""The tour (Help → Take the tour): the app's real screens, captured from a hidden copy of the window driven through a
unit's steps (``grab()``: what the screen shows), each with a caption; played here, and saved as an animated GIF
with the captions on it to share.

Made on this computer the first time it's opened (a few seconds) and kept in the settings folder for this version:
nothing is bundled, so the pictures in it are the ones on this computer (the example's, or the guide's drawings)."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QVBoxLayout

from eu4unitgen import __version__
from eu4unitgen.gui.base import ghost, label, open_file, pixmap, primary

SIZE = (1240, 820)
SECONDS = 5


def tour_dir() -> Path:
    from eu4unitgen.core import config
    return config.config_dir() / "tour" / __version__


def _picture(tmp: Path) -> Path:
    """The picture the tour drops in: the example's soldier when it is on this computer, else the guide's drawing."""
    from eu4unitgen.core import example
    from eu4unitgen.ref3d import flow
    if example.available():
        img = flow.request(flow.folder_of(example.RUNS["soldier"])).get("image")
        if img and Path(img).is_file():
            return Path(img)
    from eu4unitgen.core.guide_pictures import soldier
    p = tmp / "tour_soldier.png"
    soldier().resize((720, 720)).save(p)
    return p


def make(out: Path | None = None) -> list[tuple[str, str]]:
    """Capture the screens -> [(png, caption)], also ``frames.json`` and ``tour.gif`` in ``out``."""
    from eu4unitgen.core import example
    from eu4unitgen.gui.app import MainWindow
    from eu4unitgen.ref3d import flow
    out = Path(out or tour_dir())
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance()
    w = MainWindow(run_async=False)
    w.resize(*SIZE)
    w.status_timer.stop()
    w.refresh_status()
    frames: list[tuple[str, str]] = []

    def shot(caption: str) -> None:
        app.processEvents()
        p = out / f"frame_{len(frames):02d}.png"
        w.grab().save(str(p))
        frames.append((str(p), caption))

    try:
        w.go("home")
        shot("Home: pick what to make. A soldier is one person from a front picture; cavalry is a horse with one of "
             "your soldiers riding it. Or press “Show me an example” first.")
        pic = _picture(out)
        w.go("new", kind="soldier")
        shot("New unit, step 1: the picture. The guide on the right shows what works, with good and bad examples.")
        w.new_unit.set_picture(pic)
        shot("Drop your picture in. The app checks it right away (a front view, arms out, a plain background) and says "
             "what may go wrong.")
        w.new_unit.go_next()
        w.new_unit.name.setText("royal_guard")
        w.new_unit.rights.setText("Original character I made with ChatGPT; I own the output")
        shot("Step 2: a name (it becomes its name in game) and where the picture comes from. The example wording is "
             "one click away.")
        w.new_unit.go_next()
        shot("Step 3: when it appears in game (the four tiers) and whether each country paints its cloth in its own "
             "colour.")
        w.new_unit.go_next()
        shot("Step 4: the app checks your computer, then “Make my soldier” starts it. It takes 15 to 30 minutes and "
             "keeps going if you close the app.")
        w.go("units")
        shot("My units: everything you've made, with its picture and state.")
        if example.available():
            w.my_units.open(str(flow.folder_of(example.RUNS["cavalry"])))
            app.processEvents()
            shot("A unit's page: it turns round as the game draws it, with the quality report in plain words. Edit the "
                 "look, put a soldier on a horse, or add it to your mod from here.")
            w.my_units.detail.add_box.show()
            shot("Add to mod: a normal mod for one country, or an achievement-safe one (graphics only). If Windows "
                 "blocks the mod folder, the app tells you exactly what to copy.")
        w.go("help")
        shot("Help: this tour, the example, the picture guide, answers to common questions, and Check for updates.")
    finally:
        w.close()
        w.deleteLater()
    (out / "frames.json").write_text(json.dumps(frames, indent=1), encoding="utf-8")
    gif(frames, out / "tour.gif")
    return frames


def gif(frames: list[tuple[str, str]], path: Path, width: int = 960) -> Path:
    """The frames as an animated GIF, each caption on a band under its screen."""
    import textwrap

    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    out = []
    for p, caption in frames:
        im = Image.open(p).convert("RGB")
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        lines = textwrap.wrap(caption, 90)
        band = 16 + 28 * len(lines)
        f = Image.new("RGB", (width, im.height + band), (30, 30, 34))
        f.paste(im, (0, 0))
        d = ImageDraw.Draw(f)
        for i, ln in enumerate(lines):
            d.text((14, im.height + 8 + 28 * i), ln, fill=(240, 240, 240), font=font)
        out.append(f.convert("P", palette=Image.ADAPTIVE, colors=128))
    out[0].save(path, save_all=True, append_images=out[1:], duration=SECONDS * 1000, loop=0)
    return path


def frames() -> list[tuple[str, str]]:
    p = tour_dir() / "frames.json"
    if p.is_file():
        got = [tuple(x) for x in json.loads(p.read_text(encoding="utf-8"))]
        if all(Path(f).is_file() for f, _ in got):
            return got
    return make()


class TourDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("The tour")
        self.resize(1080, 820)
        lay = QVBoxLayout(self)
        self.step = label("", "step")
        lay.addWidget(self.step)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignCenter)
        self.pic.setMinimumSize(960, 640)
        lay.addWidget(self.pic, 1)
        self.caption = label("")
        self.caption.setStyleSheet("font-size: 15px;")
        lay.addWidget(self.caption)
        row = QHBoxLayout()
        self.prev = ghost("← Back", lambda: self.show_frame(self.i - 1))
        self.play = ghost("Pause", self.toggle)
        self.next = primary("Next →", lambda: self.show_frame(self.i + 1))
        self.save = ghost("Save as a video (GIF)…", self._save, "An animated GIF with the captions, to share or watch later")
        for b in (self.prev, self.play, self.next):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.save)
        row.addWidget(ghost("Close", self.accept))
        lay.addLayout(row)
        self.caption.setText("Making the tour from the app's screens (a few seconds)…")
        QApplication.processEvents()
        self.frames = frames()
        self.i = 0
        self.timer = QTimer(self)
        self.timer.setInterval(SECONDS * 1000)
        self.timer.timeout.connect(lambda: self.show_frame(self.i + 1))
        self.show_frame(0)
        self.timer.start()

    def show_frame(self, i: int) -> None:
        if not self.frames:
            return
        self.i = i % len(self.frames)
        p, caption = self.frames[self.i]
        self.pic.setPixmap(pixmap(p, self.pic.width(), self.pic.height()))
        self.caption.setText(caption)
        self.step.setText(f"{self.i + 1} OF {len(self.frames)}")

    def toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        else:
            self.timer.start()
            self.play.setText("Pause")

    def _save(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        dst, _ = QFileDialog.getSaveFileName(self, "Save the tour", str(Path.home() / "eu4unitgen_tour.gif"), "GIF (*.gif)")
        if dst:
            src = tour_dir() / "tour.gif"
            Path(dst).write_bytes(src.read_bytes())
            open_file(Path(dst))
