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
