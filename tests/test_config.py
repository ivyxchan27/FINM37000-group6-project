"""Behavior of Databento API key selection."""

import pytest

from sofr_data import config


@pytest.mark.parametrize("contents", ["", "\n"])
def test_empty_api_key_file_uses_environment(tmp_path, monkeypatch, contents):
    key_file = tmp_path / "api_key"
    key_file.write_text(contents)
    monkeypatch.setattr(config, "API_KEY_FILE", key_file)
    monkeypatch.setenv(config.API_KEY_ENV, "db-from-environment")

    assert config.resolve_api_key() == "db-from-environment"


def test_api_key_file_takes_precedence(tmp_path, monkeypatch):
    key_file = tmp_path / "api_key"
    key_file.write_text(" db-from-file \nignored-second-line")
    monkeypatch.setattr(config, "API_KEY_FILE", key_file)
    monkeypatch.setenv(config.API_KEY_ENV, "db-from-environment")

    assert config.resolve_api_key() == "db-from-file"
