"""F6: the nation picker's list: vanilla tags and names, plus a mod's own tags and localisation."""
from pathlib import Path

from eu4unitgen.export.pipeline import Session, Settings
from eu4unitgen.library.scanner import GameFiles
from eu4unitgen.library.tags import country_names


def _write(root: Path, rel: str, text: str, bom: bool = False) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))


def test_country_names_game_then_mod(tmp_path):
    game, mod = tmp_path / "game", tmp_path / "mod"
    _write(game, "common/country_tags/00_countries.txt", 'ATH = "countries/Athens.txt"\nNOL = "countries/X.txt"\n')
    _write(game, "localisation/countries_l_english.yml", 'l_english:\n ATH:0 "Athens"\n ATH_ADJ:0 "Athenian"\n', bom=True)
    _write(game, "localisation/countries_l_french.yml", 'l_french:\n ATH:0 "Athènes"\n')
    _write(mod, "common/country_tags/hr_tags.txt", 'MAC = "countries/Macedon.txt"\n')
    _write(mod, "localisation/hr_l_english.yml", 'l_english:\n MAC:0 "Macedon"\n ATH:0 "Athenai"\n', bom=True)
    _write(mod, "localisation/replace/zz_l_english.yml", 'l_english:\n ATH:0 "Athens of the Hellenes"\n')
    g, m = GameFiles(game, include_dlc=False), GameFiles(mod, include_dlc=False)
    assert country_names(g) == {"ATH": ("Athens", False), "NOL": ("NOL", False)}  # no name: the tag itself
    both = country_names(g, m)
    assert both["MAC"] == ("Macedon", True) and both["ATH"] == ("Athens of the Hellenes", False)


def test_session_country_list_sorted_by_name(fake_install, tmp_path):
    s = Session(Settings(eu4_path=fake_install, output_dir=tmp_path / "out"))
    try:
        rows = s.country_list()
        assert rows and rows == sorted(rows, key=lambda r: (r[1].lower(), r[0]))
        assert all(len(t) == 3 and not only for t, _, only in rows)
    finally:
        s.close()
