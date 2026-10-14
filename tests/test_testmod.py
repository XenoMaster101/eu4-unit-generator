"""Tests for export/testmod.py and the test-mod / pick-test-tag CLI commands."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from eu4unitgen.cli import main as cli_main
from eu4unitgen.core.generator import GenerationError
from eu4unitgen.core.nation import Nation
from eu4unitgen.core.project import Project
from eu4unitgen.export.testmod import build_test_mod, descriptor_text, supported_version
from eu4unitgen.export.writer import BLOCKED, CREATE, UNCHANGED, apply_export, read_manifest, stage_plan
from eu4unitgen.library.scanner import GameFiles, build_index
from eu4unitgen.pdxscript.parser import parse_bytes
from eu4unitgen.validate.rules import check_descriptor

T = "eu4unitgen_test"


@pytest.fixture
def mod_dir(tmp_path) -> Path:
    d = tmp_path / "eu4_user" / "mod"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def built(project, game, mod_dir):
    files, idx = game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test")
    return build_test_mod(project, nation, files, idx, mod_dir)


def test_layout(built):
    paths = {f.path for f in built.plan.files}
    for p in (f"{T}.mod", f"{T}/descriptor.mod", f"{T}/common/countries/Alpha.txt",
              f"{T}/gfx/entities/hr_macedonian_test_units.asset", f"{T}/gfx/entities/hr_hellenicgfx_units.asset",
              f"{T}/gfx/sprite_packs/hr_macedonian_test_pack.txt", f"{T}/common/technologies/mil.txt",
              f"{T}/common/graphicalculturetype.txt", f"{T}/localisation/hr_macedonian_test_units_l_english.yml",
              f"{T}/INTEGRATION.md", f"{T}/common/units/hr_macedonian_test_cavalry_4.txt"):
        assert p in paths, p
    assert all(p == f"{T}.mod" or p.startswith(f"{T}/") for p in paths)
    assert built.plan.manifest_rel == f"{T}/.eu4unitgen/manifest.json"
    assert built.report.errors == [] and built.report.warnings == [], "\n".join(map(str, built.report.issues))


def test_country_file_gets_integration_applied_and_keeps_the_rest(built, fake_install):
    pf = next(f for f in built.plan.files if f.path == f"{T}/common/countries/Alpha.txt")
    doc = parse_bytes(pf.data)
    assert doc.get_str("graphical_culture") == "hellenicgfx"
    assert doc.get_block("color").bare() == ["10", "20", "30"]  # no colour given: vanilla colour kept
    assert doc.get_block("historical_units").bare()[0] == "hr_macedonian_test_infantry_1"
    vanilla = (fake_install / "common/countries/Alpha.txt").read_bytes()
    assert pf.data.startswith(b"#Country file\ngraphical_culture = hellenicgfx\n\ncolor = { 10 20 30 }\n")
    assert pf.data.endswith(vanilla[vanilla.index(b"\n\nmonarch_names"):])
    assert "--- vanilla/common/countries/Alpha.txt" in built.plan.diff_text()


def test_sprite_pack_targets_the_test_tag(built):
    pf = next(f for f in built.plan.files if f.path.endswith("hr_macedonian_test_pack.txt"))
    pack = parse_bytes(pf.data).get_block("hr_macedonian_test_units")
    assert pack.get_block("country_trigger").scalar_map() == {"tag": "AAA"}
    assert pack.get_block("sprite_infantry").scalar_map()["0"] == "AAA_INFANTRY_1"


def test_descriptors(built, mod_dir):
    outer = next(f for f in built.plan.files if f.path == f"{T}.mod").data.decode()
    inner = next(f for f in built.plan.files if f.path == f"{T}/descriptor.mod").data.decode()
    doc = parse_bytes(outer.encode())
    assert doc.get_str("name") == T and doc.get_str("supported_version") == "v9.99.*"
    assert doc.get_str("path") == (mod_dir / T).as_posix() and "\\" not in doc.get_str("path")
    assert doc.get_block("tags").bare() == ["Graphics", "Military"]
    assert parse_bytes(inner.encode()).get_str("path") is None
    assert check_descriptor(outer, (mod_dir / T).as_posix()) == [] and check_descriptor(inner) == []


def test_descriptor_rule_catches_problems():
    bad = check_descriptor('name="x"\npath="C:\\\\mods\\\\x"\n', "C:/mods/x")
    text = " ".join(i.message for i in bad)
    assert "missing version" in text and "forward slashes" in text and "expected" in text


def test_supported_version(game):
    _, idx = game
    assert supported_version(idx) == "v9.99.*"
    assert descriptor_text("n", "v1.37.*", None).splitlines()[-1] == 'supported_version="v1.37.*"'


def test_apply_then_rerun_is_unchanged_and_hand_edits_block(built, project, game, mod_dir):
    apply_export(built.plan)
    assert (mod_dir / f"{T}.mod").is_file() and (mod_dir / T / "descriptor.mod").is_file()
    manifest = read_manifest(mod_dir, f"{T}/.eu4unitgen/manifest.json")
    assert manifest["files"][f"{T}.mod"]["sets"] == ["macedonian_test"]
    files, idx = game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test")
    again = build_test_mod(project, nation, files, idx, mod_dir)
    assert {f.action for f in again.plan.files} == {UNCHANGED}
    (mod_dir / T / "common/countries/Alpha.txt").write_text("# my edit\n")
    third = build_test_mod(project, nation, files, idx, mod_dir)
    assert [f.path for f in third.plan.by(BLOCKED)] == [f"{T}/common/countries/Alpha.txt"]


def test_never_touches_other_mods(built, mod_dir):
    (mod_dir / "ugc_1.mod").write_text('name="other"\n')
    (mod_dir / "other").mkdir()
    (mod_dir / "other" / "x.txt").write_text("keep")
    apply_export(built.plan)
    assert (mod_dir / "ugc_1.mod").read_text() == 'name="other"\n'
    assert (mod_dir / "other" / "x.txt").read_text() == "keep"
    assert {f.action for f in built.plan.files} == {CREATE}


def test_launcher_stripping_the_final_newline_is_not_an_edit(built, project, game, mod_dir):
    """The EU4 launcher rewrites <name>.mod without its final newline."""
    apply_export(built.plan)
    desc = mod_dir / f"{T}.mod"
    desc.write_bytes(desc.read_bytes().rstrip(b"\n"))
    (mod_dir / T / "descriptor.mod").write_bytes((mod_dir / T / "descriptor.mod").read_bytes().rstrip(b"\n"))
    files, idx = game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test")
    again = build_test_mod(project, nation, files, idx, mod_dir)
    by_path = {f.path: f.action for f in again.plan.files}
    assert by_path[f"{T}.mod"] != BLOCKED
    assert by_path[f"{T}/descriptor.mod"] == BLOCKED  # only the launcher descriptor gets the allowance
    desc.write_bytes(desc.read_bytes() + b'\nuser_dir="x"')
    third = build_test_mod(project, nation, files, idx, mod_dir)
    assert f"{T}.mod" in [f.path for f in third.plan.by(BLOCKED)]


def test_foreign_descriptor_is_blocked(project, game, mod_dir):
    (mod_dir / f"{T}.mod").write_text('name="someone else"\n')
    files, idx = game
    tm = build_test_mod(project, Nation("AAA", "hellenicgfx", set_name="macedonian_test"), files, idx, mod_dir)
    assert [f.path for f in tm.plan.by(BLOCKED)] == [f"{T}.mod"] and not tm.plan.can_apply


def test_tag_without_country_file_is_refused(project, game, mod_dir):
    files, idx = game
    with pytest.raises(GenerationError):
        build_test_mod(project, Nation("MAC", "hellenicgfx"), files, idx, mod_dir)


def test_test_mod_ignores_mod_collision_setting(project, game, mod_dir):
    project.check_mod_collisions = True  # the test mod is self-contained either way
    files, idx = game
    tm = build_test_mod(project, Nation("AAA", "hellenicgfx", set_name="macedonian_test"), files, idx, mod_dir)
    assert tm.report.ok


def test_unwritable_target_fails_before_any_write(built, mod_dir, monkeypatch):
    import tempfile
    from eu4unitgen.export import writer

    real = tempfile.mkstemp

    def blocked(*a, dir=None, **kw):  # what Controlled Folder Access looks like from Python
        if Path(dir) == mod_dir:
            raise FileNotFoundError(2, "No such file or directory")
        return real(*a, dir=dir, **kw)

    monkeypatch.setattr(writer.tempfile, "mkstemp", blocked)
    with pytest.raises(writer.ExportError, match="can't write to"):
        apply_export(built.plan)
    assert list(mod_dir.iterdir()) == []


def test_stage_writes_elsewhere_with_real_descriptor_path(built, mod_dir, tmp_path):
    from eu4unitgen.export.writer import ExportError

    stage = tmp_path / "stage"
    written = apply_export(stage_plan(built.plan, stage))
    assert f"{T}.mod" in written and not (mod_dir / T).exists()
    descriptor = (stage / f"{T}.mod").read_text()
    assert f'path="{(mod_dir / T).as_posix()}"' in descriptor
    assert read_manifest(stage, f"{T}/.eu4unitgen/manifest.json")["files"]
    with pytest.raises(ExportError, match="isn't empty"):
        stage_plan(built.plan, stage)


def test_copied_stage_is_recognised_as_tool_owned(built, project, game, mod_dir, tmp_path):
    import shutil
    stage = tmp_path / "stage"
    apply_export(stage_plan(built.plan, stage))
    shutil.copytree(stage, mod_dir, dirs_exist_ok=True)  # what the user does in Explorer
    files, idx = game
    again = build_test_mod(project, Nation("AAA", "hellenicgfx", set_name="macedonian_test",
                                           adjective="Macedonian Test"), files, idx, mod_dir)
    assert {f.action for f in again.plan.files} == {UNCHANGED}


# --------------------------------------------------------------------------- CLI

def test_cli_pick_test_tag(fake_install, capsys):
    assert cli_main(["pick-test-tag", "--eu4", str(fake_install)]) == 0
    out = capsys.readouterr().out
    assert "3 tags own land at 1444.11.11" in out and "  AAA " in out and "  TST " not in out


def test_cli_test_mod_dry_run_and_apply(fake_install, mod_dir, capsys):
    args = ["test-mod", "--tag", "AAA", "--gfx", "hellenicgfx", "--set", "macedonian_test",
            "--eu4", str(fake_install), "--mod-dir", str(mod_dir)]
    assert cli_main(args) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and not (mod_dir / T).exists()
    assert "test mod for AAA (alphan, western, was westerngfx)" in out
    assert cli_main(args + ["--apply"]) == 0
    assert (mod_dir / T / "common/countries/Alpha.txt").is_file()


def test_cli_test_mod_warns_on_covered_tag(fake_install, mod_dir, capsys):
    cli_main(["test-mod", "--tag", "BBB", "--gfx", "hellenicgfx", "--eu4", str(fake_install), "--mod-dir", str(mod_dir)])
    assert "WARNING: BBB is matched by ['dlc999_grp_sprite_pack']" in capsys.readouterr().out


def test_cli_test_mod_refuses_tag_without_land(fake_install, mod_dir, capsys):
    assert cli_main(["test-mod", "--tag", "CCC", "--gfx", "hellenicgfx", "--eu4", str(fake_install),
                     "--mod-dir", str(mod_dir)]) == 1


@pytest.mark.vanilla
@pytest.mark.skipif(not os.environ.get("EU4_PATH"), reason="EU4_PATH not set")
def test_real_install_test_mod_validates(tmp_path):
    project = Project(mod_root=tmp_path / "unused", owned_dlc=["dlc116_scandinavian_unit_pack"])
    files = GameFiles(project.eu4_path)
    try:
        idx = build_index(files)
        nation = Nation("HDA", "hellenicgfx", set_name="macedonian_test", adjective="Macedonian Test",
                        tech_group="north_american")
        tm = build_test_mod(project, nation, files, idx, tmp_path)
    finally:
        files.close()
    assert tm.report.errors == [] and tm.report.warnings == [], "\n".join(map(str, tm.report.issues))
    assert tm.plan.can_apply


def test_test_sprite_techs_only_in_the_test_mod(project, game, mod_dir):
    files, idx = game
    nation = Nation("AAA", "hellenicgfx", set_name="macedonian_test")
    tm = build_test_mod(project, nation, files, idx, mod_dir, test_sprite_techs={2: 1, 3: 2, 4: 4})
    mil = next(f for f in tm.plan.files if f.path == f"{T}/common/technologies/mil.txt")
    techs = parse_bytes(mil.data).get_all("technology")
    assert {v.value: i for i, t in enumerate(techs) for v in t.get_all("sprite_level")} == {"1": 0, "2": 1, "3": 2, "4": 4}
    assert tm.report.errors == [], "\n".join(map(str, tm.report.issues))


def test_cli_test_sprite_techs_warns_at_or_below_start_level(fake_install, mod_dir, capsys):
    args = ["test-mod", "--tag", "AAA", "--gfx", "hellenicgfx", "--set", "macedonian_test",
            "--eu4", str(fake_install), "--mod-dir", str(mod_dir), "--test-sprite-techs", "1", "2", "4"]
    (fake_install / "common/technology.txt").write_text("groups = { western = { start_level = 1 } }\n")
    assert cli_main(args) == 0
    assert "starts at MIL 1" in capsys.readouterr().out
