from pathlib import Path

import pytest

from forge.core.config import ForgeConfig


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "forge.yaml"
    p.write_text(content)
    return p


def test_load_parses_valid_yaml(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "northstar: 'do the thing'\nworkspace: ./ws\nconcurrency: 4\nverbose: true\n")
    config = ForgeConfig.load(p)
    assert config.northstar == "do the thing"
    assert config.concurrency == 4
    assert config.verbose is True


def test_load_raises_on_missing_northstar(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "workspace: ./ws\n")
    with pytest.raises(ValueError, match="northstar"):
        ForgeConfig.load(p)


def test_load_raises_on_missing_workspace(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "northstar: 'goal'\n")
    with pytest.raises(ValueError, match="workspace"):
        ForgeConfig.load(p)


def test_load_resolves_workspace_to_absolute(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n")
    config = ForgeConfig.load(p)
    assert config.workspace.is_absolute()


def test_load_defaults_concurrency_and_verbose(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n")
    config = ForgeConfig.load(p)
    assert config.concurrency == 1
    assert config.verbose is False
