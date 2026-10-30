"""U2: finding the EU4 install and the owned DLC through Steam's files (made-up Steam folders)."""
from pathlib import Path

from eu4unitgen.core.install import find_blender, find_eu4, owned_dlc, parse_vdf


def _steam(tmp: Path, installed_ids: list[str]) -> tuple[Path, Path]:
    root = tmp / "Steam"
    lib = tmp / "Library2"
    (root / "steamapps").mkdir(parents=True)
    (lib / "steamapps" / "common" / "Europa Universalis IV" / "dlc").mkdir(parents=True)
    (root / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
        % (str(root).replace("\\", "\\\\"), str(lib).replace("\\", "\\\\")), encoding="utf-8")
    depots = "".join(f'\t\t"{i}"\n\t\t{{\n\t\t\t"manifest"\t\t"1"\n\t\t\t"dlcappid"\t\t"{i}"\n\t\t}}\n' for i in installed_ids)
    (lib / "steamapps" / "appmanifest_236850.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"236850"\n\t"installdir"\t\t"Europa Universalis IV"\n\t"InstalledDepots"\n\t{\n'
        '\t\t"236851"\n\t\t{\n\t\t\t"manifest"\t\t"9"\n\t\t}\n' + depots + '\t}\n}\n', encoding="utf-8")
    game = lib / "steamapps" / "common" / "Europa Universalis IV"
    for folder, sid in (("dlc116_scandinavian_unit_pack", "2012010"), ("dlc999_not_owned", "555")):
        (game / "dlc" / folder).mkdir()
        (game / "dlc" / folder / "x.dlc").write_text(f'name = "x"\nsteam_id = "{sid}"\n', encoding="utf-8")
    return root, game


def test_parse_vdf_nested_and_escaped():
    doc = parse_vdf('"a"\n{\n\t"path"\t\t"D:\\Steam"\n\t"b"\n\t{\n\t\t"c"\t"1"\n\t}\n}\n')
    assert doc == {"a": {"path": "D:\Steam", "b": {"c": "1"}}}


def test_find_eu4_in_a_second_library_and_owned_dlc(tmp_path, monkeypatch):
    monkeypatch.delenv("EU4_PATH", raising=False)
    root, game = _steam(tmp_path, ["2012010"])
    assert find_eu4([root]) == game
    owned, note = owned_dlc(game)
    assert owned == ["dlc116_scandinavian_unit_pack"] and "Steam" in note


def test_no_manifest_means_tick_by_hand(tmp_path):
    (tmp_path / "eu4" / "dlc").mkdir(parents=True)
    owned, note = owned_dlc(tmp_path / "eu4")
    assert owned == [] and "by hand" in note


def test_f2_find_blender_prefers_the_pinned_series(tmp_path, monkeypatch):
    """F2: of several Blenders, the pinned 4.2 series wins; another version is never taken; BLENDER_PATH wins."""
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    new, old = tmp_path / "5.2" / "blender.exe", tmp_path / "4.2" / "blender.exe"
    exe, note = find_blender([(new, "5.2.2"), (old, "4.2.23")], pinned="4.2")
    assert exe == old and note == "found Blender 4.2.23"
    exe, note = find_blender([(new, "5.2.2")], pinned="4.2")  # never another version (1.0.3)
    assert exe is None and "can't use" in note
    assert find_blender([], pinned="4.2")[0] is None
    assert find_blender([(new, "4.20.1")], pinned="4.2")[0] is None
    monkeypatch.setenv("BLENDER_PATH", str(new))
    assert find_blender([(old, "4.2.23")])[0] == new
