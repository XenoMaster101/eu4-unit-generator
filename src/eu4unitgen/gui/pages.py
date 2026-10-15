"""The wizard's screens. Each page only collects input and shows results; the work is done by
export.pipeline.Session, called through the main window's ``run``."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from eu4unitgen.core.project import BLENDER_VERSION, IO_PDX_MESH_VERSION
from eu4unitgen.export.pipeline import SetSpec, Settings
from eu4unitgen.validate.rules import ERROR, INFO, WARNING

SEVERITY_COLOUR = {ERROR: "#c62828", WARNING: "#b26a00", INFO: "#546e7a"}


def open_folder(path: Path) -> bool:
    """Open ``path`` in the system file manager (Explorer, Files, ...). False if it doesn't exist."""
    if not Path(path).is_dir():
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))


class Page(QWidget):
    title = ""

    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.layout_ = QVBoxLayout(self)
        heading = QLabel(f"<h2>{self.title}</h2>")
        self.layout_.addWidget(heading)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def show_message(self, text: str, error: bool = False) -> None:
        self.message.setText(f'<span style="color:{SEVERITY_COLOUR[ERROR]}">{text}</span>' if text and error else text)

    def set_busy(self, busy: bool) -> None:
        for b in self.findChildren(QPushButton):
            b.setEnabled(not busy)


def _path_row(edit: QLineEdit, pick_dir: bool = True) -> QWidget:
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit)
    browse = QPushButton("Browse…")

    def pick() -> None:
        start = edit.text() or str(Path.cwd())
        if pick_dir:
            path = QFileDialog.getExistingDirectory(row, "Choose folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(row, "Choose file", start)
        if path:
            edit.setText(path)
    browse.clicked.connect(pick)
    lay.addWidget(browse)
    return row


# --------------------------------------------------------------------------- settings

class SettingsPage(Page):
    title = "1. Settings"

    def __init__(self, window, settings: Settings):
        super().__init__(window)
        form = QFormLayout()
        self.eu4 = QLineEdit(str(settings.eu4_path or ""))
        self.eu4.setPlaceholderText("EU4 install folder (the one with common/ and gfx/)")
        self.output = QLineEdit(str(settings.output_dir))
        self.mod_folder = QLineEdit(str(settings.mod_folder or ""))
        self.mod_folder.setPlaceholderText("optional: your mod's folder, read for name collisions and its mil.txt")
        self.stage = QLineEdit(str(settings.stage_dir))
        self.user_mods = QLineEdit(str(settings.user_mod_dir()))
        self.user_mods.setToolTip("Where the launcher looks for mods; test mod descriptors point here. Read only.")
        self.prefix = QLineEdit(settings.prefix)
        self.blender = QLineEdit(str(settings.blender_path or ""))
        self.blender.setPlaceholderText("blender.exe / blender (4.2 LTS); only run as a background subprocess")
        self.blender.setToolTip(f"Pinned: Blender {BLENDER_VERSION} LTS with io_pdx_mesh {IO_PDX_MESH_VERSION}")
        form.addRow("EU4 install", _path_row(self.eu4))
        form.addRow("Output folder", _path_row(self.output))
        form.addRow("Mod folder (checks)", _path_row(self.mod_folder))
        form.addRow("Test mod staging", _path_row(self.stage))
        form.addRow("EU4 mod directory", _path_row(self.user_mods))
        form.addRow("Name prefix", self.prefix)
        form.addRow(f"Blender {BLENDER_VERSION} LTS", _path_row(self.blender, pick_dir=False))
        self.layout_.addLayout(form)

        self.load = QPushButton("Load EU4 install")
        self.load.clicked.connect(window.load_install)
        self.layout_.addWidget(self.load)
        self.info = QLabel("Not loaded.")
        self.layout_.addWidget(self.info)

        box = QGroupBox("Owned DLC (the dlc/ folder doesn't prove ownership; tick what you own)")
        bl = QVBoxLayout(box)
        self.dlc = QListWidget()
        bl.addWidget(self.dlc)
        self.layout_.addWidget(box, 1)
        self._owned = set(settings.owned_dlc)
        self.layout_.addWidget(self.message)

    def fill_dlc(self, names: list[str]) -> None:
        self._owned |= set(self.owned())
        self.dlc.clear()
        for n in names:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if n in self._owned else Qt.Unchecked)
            self.dlc.addItem(item)

    def owned(self) -> list[str]:
        if self.dlc.count() == 0:
            return sorted(self._owned)
        return [self.dlc.item(i).text() for i in range(self.dlc.count()) if self.dlc.item(i).checkState() == Qt.Checked]

    def settings(self) -> Settings:
        return Settings(eu4_path=Path(self.eu4.text().strip()) if self.eu4.text().strip() else None,
                        output_dir=Path(self.output.text().strip() or "output"), owned_dlc=self.owned(),
                        mod_folder=Path(self.mod_folder.text().strip()) if self.mod_folder.text().strip() else None,
                        stage_dir=Path(self.stage.text().strip() or "build/test_mod"),
                        mod_dir=Path(self.user_mods.text().strip()) if self.user_mods.text().strip() else None,
                        prefix=self.prefix.text().strip() or "hr_",
                        blender_path=Path(self.blender.text().strip()) if self.blender.text().strip() else None)


# --------------------------------------------------------------------------- unit set

SAME_AS_BASE = "(same as base culture)"


class UnitSetPage(Page):
    title = "2. Unit set"

    def __init__(self, window):
        super().__init__(window)
        form = QFormLayout()
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("e.g. MAC, or several: MAC EPI (one set per tag)")
        self.set_name = QLineEdit()
        self.set_name.setPlaceholderText("default: the tag in lowercase; leave blank with several tags")
        self.gfx = QComboBox()
        self.gfx.setEditable(True)
        self.gfx.setEditText("hellenicgfx")
        self.gfx.setToolTip("Pick an existing graphical culture or type a new one (lowercase, ends in gfx)")
        self.base = QComboBox()
        self.base.addItem("westerngfx")
        self.gfx_base = QComboBox()
        self.gfx_base.addItem(SAME_AS_BASE)
        self.adjective = QLineEdit()
        self.adjective.setPlaceholderText("used in unit names, e.g. Macedonian")
        self.tech = QLineEdit()
        self.tech.setPlaceholderText("unit_type; blank: the tag's 1444 tech group")
        self.colour_on = QCheckBox("Set the map colour")
        self.colour_on.setToolTip("The country file's color (map). Units are tinted with common/country_colors "
                                  "color1-3 instead; the Textures preview uses those.")
        self.colour_btn = QPushButton()
        self._colour = QColor(120, 30, 140)
        self.colour_btn.clicked.connect(self._pick_colour)
        self._paint_colour()
        crow = QWidget()
        cl = QHBoxLayout(crow)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.colour_on)
        cl.addWidget(self.colour_btn)
        cl.addStretch(1)
        form.addRow("Tag(s)", self.tags)
        form.addRow("Set name", self.set_name)
        form.addRow("Graphical culture", self.gfx)
        form.addRow("Copy units from", self.base)
        form.addRow("Default set copies", self.gfx_base)
        form.addRow("Adjective", self.adjective)
        form.addRow("Tech group", self.tech)
        form.addRow("Colour", crow)
        self.layout_.addLayout(form)
        self.layout_.addWidget(self.message)
        self.layout_.addStretch(1)

    def _paint_colour(self) -> None:
        self.colour_btn.setText(f"{self._colour.red()} {self._colour.green()} {self._colour.blue()}")
        self.colour_btn.setStyleSheet(f"background:{self._colour.name()}; color:{'#fff' if self._colour.lightness() < 128 else '#000'}")

    def _pick_colour(self) -> None:
        c = QColorDialog.getColor(self._colour, self, "Country colour")
        if c.isValid():
            self._colour = c
            self.colour_on.setChecked(True)
            self._paint_colour()

    def fill(self, cultures: list[str], bases: list[str]) -> None:
        current_gfx, current_base = self.gfx.currentText(), self.base.currentText()
        self.gfx.clear()
        self.gfx.addItems(cultures)
        self.gfx.setEditText(current_gfx)
        for combo, keep, extra in ((self.base, current_base, []), (self.gfx_base, self.gfx_base.currentText(), [SAME_AS_BASE])):
            combo.clear()
            combo.addItems(extra + bases)
            i = combo.findText(keep)
            combo.setCurrentIndex(i if i >= 0 else 0)

    def spec(self) -> SetSpec:
        c = self._colour
        gfx_base = self.gfx_base.currentText()
        return SetSpec(tags=SetSpec.parse_tags(self.tags.text()), gfx=self.gfx.currentText().strip(),
                       set_name=self.set_name.text().strip(), base_gfx=self.base.currentText(),
                       adjective=self.adjective.text().strip(),
                       colour=(c.red(), c.green(), c.blue()) if self.colour_on.isChecked() else None,
                       tech_group=self.tech.text().strip(),
                       gfx_set_base=None if gfx_base in ("", SAME_AS_BASE) else gfx_base)


# --------------------------------------------------------------------------- results helpers

def fill_issues(tree: QTreeWidget, results) -> None:
    """results: [(label, Report)]"""
    tree.clear()
    for label, report in results:
        top = QTreeWidgetItem([label, "", report.summary(), ""])
        top.setFirstColumnSpanned(False)
        tree.addTopLevelItem(top)
        for issue in report.issues:
            item = QTreeWidgetItem([issue.severity.upper(), issue.rule, issue.message, issue.path or ""])
            item.setForeground(0, QBrush(QColor(SEVERITY_COLOUR[issue.severity])))
            item.setToolTip(2, issue.message)
            top.addChild(item)
        top.setExpanded(True)
    for i in range(3):
        tree.resizeColumnToContents(i)


def _issue_tree() -> QTreeWidget:
    tree = QTreeWidget()
    tree.setHeaderLabels(["Severity", "Rule", "Message", "File"])
    tree.setWordWrap(True)
    tree.setAlternatingRowColors(True)
    return tree


# --------------------------------------------------------------------------- validate

class ValidatePage(Page):
    title = "4. Validate"

    def __init__(self, window):
        super().__init__(window)
        self.run = QPushButton("Run validator")
        self.run.clicked.connect(window.validate)
        self.layout_.addWidget(self.run)
        self.layout_.addWidget(self.message)
        self.tree = _issue_tree()
        self.layout_.addWidget(self.tree, 1)

    def show_results(self, results) -> None:
        fill_issues(self.tree, [(f"{r.nation.tag} / {r.nation.set_name}", r.report) for r in results])
        errors = sum(len(r.report.errors) for r in results)
        self.show_message("No errors: ready to generate." if not errors else
                          f"{errors} error(s): fix them before generating.", error=bool(errors))


# --------------------------------------------------------------------------- generate

class GeneratePage(Page):
    title = "5. Generate"

    def __init__(self, window):
        super().__init__(window)
        row = QHBoxLayout()
        self.run = QPushButton("Generate package")
        self.run.clicked.connect(window.generate)
        self.open = QPushButton("Open output folder")
        self.open.clicked.connect(lambda: self._open(window.session.settings.output_dir))
        row.addWidget(self.run)
        row.addWidget(self.open)
        row.addStretch(1)
        self.layout_.addLayout(row)
        self.layout_.addWidget(self.message)
        split = QSplitter(Qt.Vertical)
        self.files = QTreeWidget()
        self.files.setHeaderLabels(["Package files", "Action"])
        self.issues = _issue_tree()
        self.integration = QPlainTextEdit()
        self.integration.setReadOnly(True)
        self.integration.setPlaceholderText("INTEGRATION.md appears here after generating.")
        split.addWidget(self.files)
        split.addWidget(self.integration)
        split.addWidget(self.issues)
        self.layout_.addWidget(split, 1)

    def _open(self, path: Path) -> None:
        if not open_folder(path):
            self.show_message(f"{path} doesn't exist yet; generate first.", error=True)

    def show_results(self, results) -> None:
        self.files.clear()
        for r in results:
            top = QTreeWidgetItem([str(r.package_dir), r.plan.summary() if r.plan else ""])
            self.files.addTopLevelItem(top)
            nodes: dict[str, QTreeWidgetItem] = {}
            for pf in sorted(r.plan.files if r.plan else [], key=lambda f: f.path):
                parts = pf.path.split("/")[1:]  # drop the <set>/ folder
                parent = top
                for depth, part in enumerate(parts[:-1]):
                    key = "/".join(parts[:depth + 1])
                    if key not in nodes:
                        nodes[key] = QTreeWidgetItem([part, ""])
                        parent.addChild(nodes[key])
                    parent = nodes[key]
                parent.addChild(QTreeWidgetItem([parts[-1], pf.action + (f" ({pf.reason})" if pf.reason else "")]))
            top.setExpanded(True)
            for item in nodes.values():
                item.setExpanded(True)
        self.files.resizeColumnToContents(0)
        self.integration.setPlainText("\n\n".join(r.integration_text for r in results))
        fill_issues(self.issues, [(f"{r.nation.tag} / {r.nation.set_name}", r.report) for r in results])
        written = sum(len(r.written) for r in results)
        if written:
            self.show_message(f"Wrote {written} files. Copy each package folder's contents into your mod's root, "
                              "then merge INTEGRATION.md by hand.")
        else:
            blocked = [f"{r.nation.set_name}: {r.plan.summary()}" for r in results if r.plan and not r.plan.can_apply]
            self.show_message("Nothing written. " + ("; ".join(blocked) or "Up to date."), error=bool(blocked))


# --------------------------------------------------------------------------- test mod

class TestModPage(Page):
    title = "6. Test mod"

    def __init__(self, window):
        super().__init__(window)
        form = QFormLayout()
        self.suite = QCheckBox("In-game test suite (two variant mods: pack priority + fallback cases)")
        self.textures = QCheckBox("Texture test (tier 1 retextured through meshsettings)")
        self.suite.toggled.connect(lambda on: on and self.textures.setChecked(False))
        self.textures.toggled.connect(lambda on: on and self.suite.setChecked(False))
        self.rival = QComboBox()
        self.rival.addItem("asiangfx")
        self.sprites_on = QCheckBox("Move sprite tiers 2-4 to early MIL techs (test only)")
        self.sprites_on.setChecked(True)
        self.sprite_techs = [QSpinBox() for _ in range(3)]
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.sprites_on)
        for tier, (box, default) in enumerate(zip(self.sprite_techs, (2, 3, 4)), start=2):
            box.setRange(1, 32)
            box.setValue(default)
            box.setPrefix(f"tier {tier}: MIL ")
            sl.addWidget(box)
        sl.addStretch(1)
        self.clear = QCheckBox("Empty the staging folder first (only if it holds nothing but staged test mods)")
        self.clear.setChecked(True)
        form.addRow(self.suite)
        form.addRow(self.textures)
        form.addRow("Rival pack copies", self.rival)
        form.addRow(srow)
        form.addRow(self.clear)
        self.layout_.addLayout(form)
        row = QHBoxLayout()
        self.run = QPushButton("Build test mod")
        self.run.clicked.connect(window.build_test_mod)
        self.open = QPushButton("Open staging folder")
        self.open.clicked.connect(self._open)
        row.addWidget(self.run)
        row.addWidget(self.open)
        row.addStretch(1)
        self.layout_.addLayout(row)
        self.layout_.addWidget(self.message)
        split = QSplitter(Qt.Vertical)
        self.issues = _issue_tree()
        self.checklist = QPlainTextEdit()
        self.checklist.setReadOnly(True)
        self.checklist.setPlaceholderText("The test suite's or texture test's checklist appears here.")
        split.addWidget(self.issues)
        split.addWidget(self.checklist)
        self.layout_.addWidget(split, 1)

    def _open(self) -> None:
        path = self.window_.session.settings.stage_dir
        if not open_folder(path):
            self.show_message(f"{path} doesn't exist yet; build first.", error=True)

    def options(self) -> dict:
        techs = dict(zip((2, 3, 4), (b.value() for b in self.sprite_techs))) if self.sprites_on.isChecked() else None
        return {"suite": self.suite.isChecked(), "sprite_techs": techs, "rival_gfx": self.rival.currentText(),
                "clear": self.clear.isChecked(), "textures": self.textures.isChecked()}

    def fill(self, bases: list[str]) -> None:
        keep = self.rival.currentText()
        self.rival.clear()
        self.rival.addItems(bases)
        i = self.rival.findText(keep)
        self.rival.setCurrentIndex(i if i >= 0 else 0)

    def show_results(self, result) -> None:
        fill_issues(self.issues, result.reports)
        self.checklist.setPlainText(result.checklist)
        mod_dir = self.window_.session.settings.user_mod_dir()
        lines = [f"WARNING: {w}" for w in result.warnings]
        if result.written:
            lines.append(f"Staged {len(result.written)} files in {result.stage_dir}. Copy everything inside it into "
                         f"{mod_dir} with Explorer, then enable the mod in the launcher.")
        else:
            lines.append("Nothing staged: the validator or the export plan blocked it (see the list).")
        self.show_message("<br>".join(lines), error=not result.written)
