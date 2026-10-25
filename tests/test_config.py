from pathlib import Path

from eu4unitgen.core import config


def test_round_trip_and_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("EU4_PATH", raising=False)
    monkeypatch.delenv("EU4UNITGEN_WORKSPACE", raising=False)
    assert config.load().eu4() is None
    config.update(eu4_path="X:/eu4", workspace=str(tmp_path / "w"))
    c = config.load()
    assert c.eu4() == Path("X:/eu4") and c.work() == tmp_path / "w"
    monkeypatch.setenv("EU4_PATH", "Y:/other")
    assert config.load().eu4() == Path("Y:/other")


def test_unknown_keys_ignored_and_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text('{"eu4_path": "Z:/e", "old_key": 1}', encoding="utf-8")
    assert config.load().eu4_path == "Z:/e"
    try:
        config.update(nope=1)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown key was accepted")


def test_workspace_defaults_to_the_checkout(monkeypatch):
    monkeypatch.delenv("EU4UNITGEN_WORKSPACE", raising=False)
    assert config.default_workspace() == config.checkout_root()


def test_installed_layout_is_in_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("EU4UNITGEN_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "Documents" / "eu4unitgen"))
    c = config.load()
    assert c.projects() == tmp_path / "Documents" / "eu4unitgen" / "projects"
    assert c.references().name == "references" and c.sheets().name == "result sheets"
    assert c.exports().name == "exported mods"
    assert config.logs_dir() == tmp_path / "cfg" / "logs"


def test_a_reference_is_kept_once(tmp_path, monkeypatch):
    from PIL import Image

    from eu4unitgen.ref3d import flow
    monkeypatch.setenv("EU4UNITGEN_WORKSPACE", str(tmp_path / "w"))
    img = tmp_path / "horse.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(img)
    a = flow.keep_reference(img)
    assert a == tmp_path / "w" / "references" / "horse.png" and flow.keep_reference(img) == a
    Image.new("RGB", (8, 8), (9, 9, 9)).save(img)  # another image, same name
    assert flow.keep_reference(img).name == "horse_1.png"
