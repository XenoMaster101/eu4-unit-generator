"""The app's screens (the UI update, then the part library): the top bar on every screen, Home leading into the
picture flow (New part) and Build a unit, the guided steps (Next only when a step is done, the reason beside it), the
older cavalry steps, the Library (parts, units, tags, starter parts), Build a unit (tiers, suggestions), Help (the
example, the guide, updates), the readability of every screen in light and dark."""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PIL import Image, ImageDraw  # noqa: E402

from eu4unitgen import __version__  # noqa: E402
from eu4unitgen.core import checks  # noqa: E402


@pytest.fixture
def win(tmp_path, monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("EU4UNITGEN_CONFIG_DIR", str(tmp_path / "cfg"))
    ok = [checks.Check(k, True, f"{k} fine") for k in ("eu4_path", "blender", "ai_env", "weights", "eu4")]
    monkeypatch.setattr(checks, "run_all", lambda quick=False: list(ok))
    monkeypatch.setattr("eu4unitgen.ref3d.flow.start", lambda folder: None)  # no detached runs from tests
    # no starter parts unless a test adds them (the installer build packs the author's into the app's folder)
    monkeypatch.setattr("eu4unitgen.core.library.starter_root", lambda: tmp_path / "no_starter")
    from eu4unitgen.gui.app import MainWindow
    w = MainWindow(run_async=False)
    w.confirm = lambda *a, **k: True
    yield w
    w.close()


def _soldier_png(path):
    im = Image.new("RGB", (512, 768), (226, 226, 230))
    d = ImageDraw.Draw(im)
    d.rectangle([200, 100, 312, 500], fill=(150, 150, 160))
    d.line([200, 140, 90, 380], fill=(150, 150, 160), width=24)
    d.line([312, 140, 422, 380], fill=(150, 150, 160), width=24)
    d.rectangle([220, 500, 290, 720], fill=(40, 40, 40))
    im.save(path)
    return path


def test_top_bar_on_every_screen(win):
    assert [b.text() for b in win.bar.buttons.values()] == ["Home", "Build a unit", "Make a part", "Library", "Setup",
                                                            "Help"]
    assert all(not b.icon().isNull() for b in win.bar.buttons.values())  # one icon set
    assert all(b.toolTip() for b in win.bar.buttons.values())
    assert __version__ in win.windowTitle()
    win.refresh_status()
    assert [t.text() for t in win.bar.ticks.values()] == ["✓ EU4", "✓ Blender", "✓ AI ready"]
    for key in ("home", "build", "new", "units", "setup", "help"):
        win.go(key)
        assert win.bar.buttons[key].isChecked() and win.stack.currentWidget() is win.screens[key]
        assert win.bar.isVisibleTo(win)


def test_home_leads_into_the_picture_flow(win, tmp_path):
    home = win.home
    assert set(home.cards) == {"build", "new"} and not hasattr(home, "describe") and not hasattr(home, "drop")
    home.cards["new"].click()
    assert win.stack.currentWidget() is win.new_unit
    win.go("home")
    home.build_card.click()
    assert win.stack.currentWidget() is win.builder
    win.go("home")
    win.go("home")
    home.picture(_soldier_png(tmp_path / "guard.png"))  # a tall picture: a soldier
    assert win.new_unit.kind == "soldier" and win.new_unit.image.name == "guard.png"


def test_soldier_steps_open_one_at_a_time(win, tmp_path):
    nu = win.new_unit
    win.go("new", kind="soldier")
    assert "STEP 1 OF 4" in nu.step_label.text() and not nu.next.isEnabled() and "picture" in nu.next_why.text()
    assert nu.examples.count() == 5  # the guide's good and bad pictures
    win.go("home")
    win.go("new", kind="soldier")  # opened again: the old cells are gone, not left drawn over the drop box
    cells = [c for c in nu.steps["picture"].findChildren(QtWidgets.QWidget) if c.layout() is not None
             and c.objectName() != "card"  # the kind choice above the picture
             and c.layout().count() == 2 and isinstance(c.layout().itemAt(0).widget(), QtWidgets.QLabel)]
    assert nu.examples.count() == 5 and len(cells) == 5
    nu.kind_cards["gun"].click()  # the kind chosen on the first step: its guide follows
    assert nu.kind == "gun" and "NEW PART: GUN" in nu.step_label.text() and "three-quarter" in nu.rules.text()
    nu.kind_cards["soldier"].click()
    nu.set_picture(_soldier_png(tmp_path / "Royal Guard.png"))
    assert nu.next.isEnabled() and "✓" in nu.checks_out.text()
    nu.go_next()
    assert "STEP 2 OF 4" in nu.step_label.text() and nu.name.text() == "royal_guard"
    assert not nu.next.isEnabled() and "rights" in nu.next_why.text()
    nu.name.setText("Bad Name")
    assert "lower-case" in nu.name_hint.text().lower() and nu.name.property("bad") == "true"
    nu.name.setText("royal_guard")
    assert "Royal Guard Infantry" in nu.name_hint.text()  # its name in game: on foot, the infantry
    assert "ChatGPT" in nu.rights.placeholderText()
    [b for b in nu.steps["about"].findChildren(QtWidgets.QPushButton) if b.text() == "Use this wording"][0].click()
    assert nu.rights.text().startswith("Original character") and nu.next.isEnabled()
    nu.go_next()
    assert nu.tier_group.checkedId() == 1 and len(nu.tier_group.buttons()) == 4 and nu.mask.isChecked()
    nu.tier_group.button(2).click()
    nu.go_next()
    assert "STEP 4 OF 4" in nu.step_label.text() and nu.go.text() == "Make my soldier" and nu.go.isEnabled()
    assert "tier 2" in nu.summary.text()
    nu.start()
    from eu4unitgen.ref3d import flow
    req = flow.request(flow.folder_of("royal_guard"))
    assert req["kind"] == "soldier" and req["tier"] == 2 and req["rights"].startswith("Original character")
    assert not nu.go.isEnabled() and nu.stages.isVisibleTo(nu) and nu.see.isVisibleTo(nu)


def _finished(name, kind, image, **extra):
    from eu4unitgen.ref3d import flow
    f = flow.new_request(name, image, "mine", kind, **extra)
    (f / "status.json").write_text(json.dumps({"stages": {s: {"state": "done", "seconds": 5} for s in flow.STAGES},
                                               "done": True, "state": "done", "heartbeat": 0}), encoding="utf-8")
    return f


def test_cavalry_from_a_horse_and_a_soldier(win, tmp_path):
    img = _soldier_png(tmp_path / "s.png")
    _finished("bay", "horse", img)
    _finished("guard", "soldier", img)
    nu = win.new_unit
    win.go("new", kind="cavalry")
    assert "NEW CAVALRY UNIT" in nu.step_label.text() and not nu.next.isEnabled()
    pick = nu.steps["horse"]
    assert len(pick._group.buttons()) == 1
    pick._group.buttons()[0].click()
    assert nu.picks["horse"] == "bay" and nu.next.isEnabled()
    nu.go_next()
    nu.steps["rider"]._group.buttons()[0].click()
    nu.go_next()
    assert nu.cav_name.text() == "guard_on_bay" and nu.next.isHidden() is False
    nu.go_next()
    nu.start()
    from eu4unitgen.ref3d import flow
    req = flow.request(flow.folder_of("guard_on_bay"))
    assert req["kind"] == "cavalry" and req["horse"] == "bay" and req["soldier"] == "guard"


def test_a_horse_made_inside_the_cavalry_steps_comes_back(win, tmp_path):
    nu = win.new_unit
    win.go("new", kind="cavalry")
    nu._make_child("horse")
    assert nu.kind == "horse" and "FOR A CAVALRY UNIT" in nu.step_label.text()
    assert nu.back.isEnabled() and "cavalry" in nu.back.text()
    nu._return_to_parent("new_bay")
    assert nu.kind == "cavalry" and nu.picks["horse"] == "new_bay" and "STEP 2 OF 4" in nu.step_label.text()


def test_my_units_gallery_and_page(win, tmp_path):
    img = _soldier_png(tmp_path / "s.png")
    f = _finished("guard", "soldier", img)
    win.go("units")
    mu = win.my_units
    assert mu.flow_layout.count() == 1 and not mu.empty.isVisibleTo(mu)
    mu.open(str(f))
    d = mu.detail
    assert d.title.text() == "Guard" and d.ride_btn.text() == "Use in a unit"  # shown capitalised; the folder stays guard
    assert d.ride_btn.objectName() == "generate" and d.del_btn.objectName() == "danger"  # one primary; Delete danger
    assert not d.add_btn.isVisibleTo(d)  # a part goes into a unit first; units go into mods
    assert "Made by you" in d.state.text() and "mine" not in d.state.text() or "mine" in d.rights_full.text()
    assert not d.details_box.isVisibleTo(d)  # the numbers behind Details
    assert not d.stages.isVisibleTo(d)  # finished: no progress
    d.ride_btn.click()
    b = win.builder
    assert win.stack.currentWidget() is b and b.utype == "infantry" and b.slots[(1, "soldier")].currentData() == "run:guard"
    win.go("units", folder=str(f))
    d.del_btn.click()
    assert not f.exists() and mu.stack.currentWidget() is mu.gallery_page



def test_the_library_shows_parts_units_and_starter_parts(win, tmp_path, monkeypatch):
    """Parts (the user's and the starter ones) and units in their own sections; a part's tags are saved; a starter
    part opens read-only with Use in a unit."""
    from eu4unitgen.core import library
    img = _soldier_png(tmp_path / "s.png")
    h = _finished("bay", "horse", img)
    _finished("guard", "soldier", img)
    st = tmp_path / "starter" / "cataphract_horse"
    st.mkdir(parents=True)
    img2 = st / "picture.png"
    Image.open(img).save(img2)
    (st / "part.json").write_text(json.dumps({"schema": 1, "name": "cataphract_horse", "kind": "horse",
                                              "tags": {"style": "barded", "regions": ["Byzantine"]}, "rights": "mine",
                                              "image": "picture.png", "quality": {"summary": "Usable"}}), encoding="utf-8")
    monkeypatch.setattr(library, "starter_root", lambda: tmp_path / "starter")
    library.new_unit("guard_cav", "cavalry", {1: {"horse": "run:bay", "soldier": "run:guard"}})
    mu = win.my_units
    win.go("units")
    assert mu.flow_layout.count() == 3 and mu.units_layout.count() == 1 and not mu.ai_note.isVisibleTo(mu)
    mu._set_filter("horse")
    assert mu.flow_layout.count() == 2 and mu.units_layout.count() == 0
    mu._set_filter("all")
    mu.open(str(h))
    d = mu.detail
    d.tags_btn.click()
    d.t_style.setCurrentIndex(d.t_style.findData("light"))
    d.t_regions.setText("Western, European")
    d._save_tags()
    from eu4unitgen.ref3d import flow
    assert flow.request(h)["tags"] == {"regions": ["Western", "European"], "style": "light"}
    mu.open_part("starter:cataphract_horse")
    assert d.copy_btn.isVisibleTo(d) and not d.del_btn.isVisibleTo(d) and "starter" in d.state.text()
    d.ride_btn.click()
    assert win.builder.utype == "cavalry" and win.builder.slots[(1, "horse")].currentData() == "starter:cataphract_horse"
    mu.open(str(h))  # a run again: its own buttons back
    assert d.del_btn.isVisibleTo(d) and not d.copy_btn.isVisibleTo(d)


def test_build_a_unit_per_tier_with_suggestions(win, tmp_path, monkeypatch):
    from eu4unitgen.core import library
    img = _soldier_png(tmp_path / "s.png")
    for name, style, regions in (("bay", "light", "Western"), ("cata_horse", "barded", "Byzantine, Greek")):
        library.set_tags(_finished(name, "horse", img), style, regions)
    library.set_tags(_finished("cata", "soldier", img), None, "Byzantine")
    monkeypatch.setattr(library, "starter_root", lambda: tmp_path / "none")
    monkeypatch.setattr(library, "country", lambda tag, eu4: library.Country(
        "BYZ", "ByzantineEmpire", "easterngfx", "eastern", "greek", "byzantine"))
    b = win.builder
    win.go("build")
    b.set_type("cavalry")
    assert not b.build_btn.isEnabled() and "at least one tier" in b.why.text()
    b.tag.setText("BYZ")
    b.look_up()
    assert "Byzantine Empire" in b.country_line.text() and b.gfx.currentData() in ("easterngfx", "")
    horses = b.slots[(1, "horse")]
    assert horses.itemData(1) == "run:cata_horse" and horses.itemText(1).startswith("★")  # its regions match
    horses.setCurrentIndex(horses.findData("run:bay"))
    assert "needs a rider" in b.why.text()
    b.slots[(1, "soldier")].setCurrentIndex(1)
    b.slots[(3, "horse")].setCurrentIndex(b.slots[(3, "horse")].findData("run:cata_horse"))
    b.slots[(3, "soldier")].setCurrentIndex(1)
    b.name.setText("byz_cav")
    assert b.build_btn.isEnabled()
    b.build()
    from eu4unitgen.ref3d import flow
    req = flow.request(flow.folder_of("byz_cav"))
    assert req["kind"] == "unit" and req["type"] == "cavalry" and req["country"] == "BYZ"
    assert req["tiers"] == {"1": {"horse": "run:bay", "soldier": "run:cata"}, "3": {"horse": "run:cata_horse",
                                                                                     "soldier": "run:cata"}}
    assert win.stack.currentWidget() is win.my_units and "Not built yet" in win.my_units.detail.summary.text()
    b.set_type("artillery")
    assert "no gun" in b.no_part.text()


def test_library_suggestions_follow_the_country():
    from eu4unitgen.core import library
    pool = [library.Part(f"run:{n}", n, "horse", None, tags={"style": s, "regions": r}, done=True)
            for n, s, r in (("bay", "light", ["Western"]), ("barded", "barded", []), ("steppe", "steppe", []),
                            ("arab", "arab", []), ("cata", "barded", ["Byzantine"]))]
    names = lambda **kw: [p.name for p, _ in library.suggest("horse", pool=pool, **kw)]  # noqa: E731
    assert names(gfx="westerngfx")[:2] == ["barded", "cata"]
    assert names(gfx="muslimgfx")[0] == "arab"
    assert names(gfx="muslimgfx", tech_group="nomad_group")[0] == "steppe"  # a nomad: steppe first
    assert names(gfx="easterngfx", culture="greek", culture_group="byzantine")[0] == "cata"  # its regions match
    with pytest.raises(ValueError):
        library.new_unit("x", "cavalry", {1: {"horse": "run:bay"}})  # a rider too


def test_builder_name_follows_the_type_and_country(win, monkeypatch):
    from eu4unitgen.core import library
    monkeypatch.setattr(library, "country", lambda tag, eu4: library.Country(
        "BYZ", "ByzantineEmpire", "easterngfx", "eastern", "greek", "byzantine"))
    b = win.builder
    win.go("build")
    b.set_type("cavalry")
    assert b.name.text() == "my_cavalry"
    b.tag.setText("BYZ")
    b.look_up()
    assert b.name.text() == "byzantine_cavalry"
    for utype in ("infantry", "artillery"):  # the default follows the type (it said byzantine_cavalry everywhere)
        b.set_type(utype)
        assert b.name.text() == f"byzantine_{utype}"
        assert b.copy_btn.isVisibleTo(b) and b.clear_btn.isVisibleTo(b)
    b.name.setText("my_own")
    b.set_type("cavalry")
    assert b.name.text() == "my_own"  # typed by the user: kept


def test_plain_words_for_rights_and_quality():
    from eu4unitgen.core import library
    r = ("generated with ChatGPT; the user owns the output and authorises its use as a reference (declared by the "
         "user, 2026-10-21; CLAUDE.md rule 8); bundling with the app allowed under CC BY 4.0 (declared 2026-10-28)")
    assert library.rights_text(r) == "Made by you with ChatGPT · CC BY 4.0 · bundling allowed"
    assert library.rights_text(r, starter=True).startswith("Made by the app's author with ChatGPT")
    assert "rule" not in library.rights_text(r) and "2026" not in library.rights_text(r)
    assert library.summary({"verdict": "ok"}) == ("warn", "Usable: a few things to look at in game")
    assert library.summary(None)[0] == "none"


def test_thumbnails_lift_a_render_onto_the_backdrop(tmp_path):
    from eu4unitgen.core import thumbs
    im = Image.new("RGB", (200, 200), (189, 189, 189))  # a render's flat grey ground
    ImageDraw.Draw(im).rectangle([80, 40, 120, 180], fill=(30, 30, 34))  # a dark model
    im.save(tmp_path / "r.png")
    p = thumbs.cache(tmp_path / "r.png", (120, 100), "#f6eedd", "#c9b38a", tmp_path / "cache")
    out = Image.open(p).convert("RGB")
    assert out.size == (120, 100) and out.getpixel((4, 4)) != (189, 189, 189)  # the grey is gone
    assert thumbs.cache(tmp_path / "r.png", (120, 100), "#f6eedd", "#c9b38a", tmp_path / "cache") == p  # cached
    assert thumbs.cache(None, (10, 10), "#000000", "#000000", tmp_path) is None

def test_every_option_has_a_tooltip(win):
    missing = []
    for key in ("home", "new", "build", "units", "help"):
        kinds = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QLineEdit, QtWidgets.QSlider)
        for w in win.screens[key].findChildren(QtWidgets.QWidget):
            if not isinstance(w, kinds) or (isinstance(w, QtWidgets.QLineEdit) and isinstance(w.parent(), QtWidgets.QComboBox)):
                continue
            text = w.text() if isinstance(w, QtWidgets.QAbstractButton) else ""
            plain = text in ("← Back", "Next →", "Close", "← Library")  # says what it does
            if not w.toolTip() and not plain:
                missing.append(f"{key}: {type(w).__name__} {text!r}")
    assert not missing, missing


def test_updates_check_reads_github_releases(monkeypatch):
    from eu4unitgen.core import updates
    assert updates.parse_version("v0.10.1") > updates.parse_version("0.9") == (0, 9, 0)
    u = updates.check("me/app", "0.2.0", fetch=lambda url: {"tag_name": "v0.3.0", "html_url": "https://x/rel"})
    assert u.ok and u.newer and u.url == "https://x/rel" and "0.3.0" in u.text
    u = updates.check("me/app", "0.3.0", fetch=lambda url: {"tag_name": "v0.3.0"})
    assert u.ok and not u.newer and "newest" in u.text
    assert not updates.check("", "0.2.0").ok  # no repository yet: says so

    def offline(url):
        raise updates.urllib.error.URLError("no network")
    assert "Couldn't reach" in updates.check("me/app", "0.2.0", fetch=offline).text


def test_example_needs_its_runs(tmp_path):
    from eu4unitgen.core import example
    assert example.steps(tmp_path) is None and not example.available(tmp_path)


def test_example_walks_a_finished_cavalry(win, tmp_path, monkeypatch):
    from eu4unitgen.core import example
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "ws"))
    img = _soldier_png(tmp_path / "s.png")
    _finished("cataphract_horse", "horse", img)
    _finished("cataphract", "soldier", img)
    from eu4unitgen.ref3d import flow
    c = _finished("cataphract_cavalry", "cavalry", None, horse="cataphract_horse", soldier="cataphract")
    (c / "preview").mkdir()
    (c / "preview" / "quality.json").write_text(json.dumps({"summary": "Usable; see the notes.", "measures": {
        "silhouette": {"verdict": "ok", "text": "Its outline overlaps the reference's by 60%."}}}), encoding="utf-8")
    s = example.steps(flow.root())
    assert len(s) == 10 and s[0].images and "Usable" in s[7].lines[0] and "Add to mod" in s[9].title
    from eu4unitgen.gui.help_page import ExampleDialog
    monkeypatch.setattr(example, "steps", lambda *a: s)
    d = ExampleDialog(win)
    for i in range(len(s)):  # each step shows its own pictures only, none left over from the one before
        d.show_step(i)
        pics = [w for w in d.findChildren(QtWidgets.QLabel) if w.pixmap() is not None and not w.pixmap().isNull()]
        assert len(pics) == len(s[i].images) + (1 if s[i].frames else 0), i
    d.close()



def test_the_example_pictures_come_with_the_app(win, tmp_path, monkeypatch):
    """The three example pictures are package data, pass their own kind's check, and start New unit with their rights
    filled in; without the runs the Example shows them and says how to make it."""
    from eu4unitgen.core import example, guide
    names = sorted(p.name for k in ("soldier", "horse") for _, p in example.pictures(k))
    assert names == ["cataphract_rider.png", "cataphract_side_clean.png", "horse2_light_cavalry.png"]
    for k in ("soldier", "horse"):
        for _, p in example.pictures(k):
            assert guide.check(p, k) == [], p.name
    assert "allows" in (example.PICTURES_DIR / "RIGHTS.txt").read_text(encoding="utf-8")
    nu = win.new_unit
    win.go("new", kind="horse")
    buttons = [b for b in nu.steps["picture"].findChildren(QtWidgets.QPushButton) if b.text().startswith("Example")]
    assert len(buttons) == 2
    win.go("new", kind="soldier")
    b = [b for b in nu.steps["picture"].findChildren(QtWidgets.QPushButton)
         if b.text() == "Use the example picture" and b.isVisibleTo(nu)]
    assert len(b) == 1
    b[0].click()
    assert nu.image.name == "cataphract_rider.png" and nu.rights.text() == example.RIGHTS
    assert nu.name.text() == "cataphract_rider"
    from eu4unitgen.gui.help_page import ExampleDialog
    monkeypatch.setattr(example, "steps", lambda *a: None)
    d = ExampleDialog(win)
    pics = [w for w in d.findChildren(QtWidgets.QLabel) if w.pixmap() is not None and not w.pixmap().isNull()]
    assert len(pics) == 3 and "Use the example picture" in d.text.text() and not d.next.isVisibleTo(d)
    d.close()

def test_guide_checks_adapt_to_the_kind(tmp_path):
    from eu4unitgen.core import guide, guide_pictures
    for kind in ("soldier", "horse", "artillery"):
        pics = guide_pictures.examples(kind)
        assert pics[0][0] and not any(g for g, _, _ in pics[1:])  # one good, the rest bad
        p = tmp_path / f"{kind}.png"
        pics[0][2].resize((768, 768)).save(p)
        assert guide.check(p, kind) == [], kind  # the good one passes its own kind's check
    narrow = tmp_path / "down.png"
    guide_pictures.soldier(arms="down").resize((768, 768)).save(narrow)
    assert any("A-pose" in w for w in guide.check(narrow, "soldier"))
    assert guide.VIEW["horse"] == "side view" and guide.WHY["soldier"]


@pytest.mark.parametrize("choice", ["campaign", "admiralty"])
def test_screens_are_readable(win, tmp_path, choice):
    """Every text on Home, the first New unit step and Help reads against its background (WCAG 4.5:1)."""
    from test_theme import MIN, _text_contrast, _texts, _window_image
    win.set_theme(choice)
    win.resize(1240, 820)
    win.show()
    bad, checked = [], 0
    for key in ("home", "new", "help"):
        win.go(key)
        QtWidgets.QApplication.processEvents()
        shot = _window_image(win)
        for widget, text in _texts(win.screens[key]):
            if not widget.isEnabled() or widget.visibleRegion().isEmpty():
                continue  # the coming-soon card and the off text box are dimmed on purpose; below the fold: not drawn
            ratio = _text_contrast(widget, shot)
            checked += 1
            if ratio < MIN:
                bad.append(f"{key}: {type(widget).__name__} {text[:30]!r}: {ratio:.1f}")
    assert checked >= 30 and not bad, bad
    win.hide()
