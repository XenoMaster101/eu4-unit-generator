"""Rule colour_scripts: mod scripts that restore or change country colours (in-game test 9)."""
from __future__ import annotations

from pathlib import Path

from eu4unitgen.core.generator import generate
from eu4unitgen.library.scanner import GameFiles
from eu4unitgen.validate import validate

ON_ACTIONS = """on_startup = {
	random_country = { limit = { ai = no } country_event = { id = tst.1 } }
}
"""

EVENTS = """namespace = tst
country_event = {
	id = tst.1
	is_triggered_only = yes
	immediate = {
		every_country = {
			limit = { is_colonial_nation = no%s }
			restore_country_color = yes
		}
	}
	option = {
		name = tst.1a
		every_country = { limit = { is_subject = no } country_event = { id = tst.2 } }
	}
}
country_event = {
	id = tst.2
	is_triggered_only = yes
	hidden = yes
	immediate = { tst_colour_subjects = yes }
}
country_event = {
	id = tst.3
	is_triggered_only = yes
	option = { name = tst.3a every_country = { restore_country_color = yes } }
}
"""

EFFECTS = """tst_colour_subjects = {
	every_subject_country = { change_country_color = { country = PREV } }
}
"""


def _mod(tmp_path: Path, limit: str = "") -> GameFiles:
    root = tmp_path / "colour_mod"
    for rel, text in (("common/on_actions/tst_on_actions.txt", ON_ACTIONS), ("events/tst.txt", EVENTS % limit),
                      ("common/scripted_effects/tst_effects.txt", EFFECTS)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return GameFiles(root, include_dlc=False)


def _issues(project, nation, game, mod):
    files, idx = game
    gen = generate(project, nation, files, idx, mod=mod)
    return validate(project, nation, gen, files, idx, mod, rules=["colour_scripts"]).issues


def test_startup_restore_on_every_country_warns(project, nation, game, tmp_path):
    issues = _issues(project, nation, game, _mod(tmp_path))
    warn = [i for i in issues if i.severity == "warning"]
    assert len(warn) == 1, [str(i) for i in issues]
    assert "on_startup (common/on_actions/tst_on_actions.txt) -> event tst.1 (events/tst.txt) -> " \
           "restore_country_color on every country" in warn[0].message and "MAC" in warn[0].message
    # the subject recolour through an event and a scripted effect is only a note
    notes = [i.message for i in issues if i.severity == "info"]
    assert any("event tst.2 (events/tst.txt) -> scripted effect tst_colour_subjects "
               "(common/scripted_effects/tst_effects.txt) -> change_country_color on subjects" in m for m in notes), notes
    # outside startup: tst.3 restores every country when it runs
    assert any(m.startswith("tst.3 changes or restores country colours on every country") for m in notes), notes
    assert not any(m.startswith("tst.1 ") for m in notes), notes  # already warned through startup


def test_restore_limited_to_subjects_is_a_note(project, nation, game, tmp_path):
    issues = _issues(project, nation, game, _mod(tmp_path, " is_subject = yes"))
    assert not [i for i in issues if i.severity == "warning"], [str(i) for i in issues]
    assert any("restore_country_color on subjects at game start" in i.message for i in issues)


def test_needs_mod_checks(project, nation, game):
    assert _issues(project, nation, game, None) == []
