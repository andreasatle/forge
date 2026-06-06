"""Tests for ForgeConfig loading and validation from YAML files."""

from pathlib import Path

import pytest

from forge.core.config import ArtifactConfig, ForgeConfig


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "forge.yaml"
    p.write_text(content)
    return p


_ARTIFACTS_YAML = "artifacts:\n  - name: codebase\n    type: coding\n    language: python\n"


def test_load_parses_valid_yaml(tmp_path: Path) -> None:
    """load() correctly parses all fields from a valid YAML config file."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'do the thing'\nworkspace: ./ws\nconcurrency: 4\nverbose: true\n" + _ARTIFACTS_YAML,
    )
    config = ForgeConfig.load(p)
    assert config.northstar == "do the thing"
    assert config.concurrency == 4
    assert config.verbose is True


def test_load_raises_on_missing_northstar(tmp_path: Path) -> None:
    """load() raises ValueError when northstar is absent from the YAML file."""
    p = _write_yaml(tmp_path, "workspace: ./ws\n" + _ARTIFACTS_YAML)
    with pytest.raises(ValueError, match="northstar"):
        ForgeConfig.load(p)


def test_load_raises_on_missing_workspace(tmp_path: Path) -> None:
    """load() raises ValueError when workspace is absent from the YAML file."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\n" + _ARTIFACTS_YAML)
    with pytest.raises(ValueError, match="workspace"):
        ForgeConfig.load(p)


def test_load_resolves_workspace_to_absolute(tmp_path: Path) -> None:
    """load() resolves the workspace path to an absolute path."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.workspace.is_absolute()


def test_load_defaults_concurrency_and_verbose(tmp_path: Path) -> None:
    """load() defaults concurrency to 1 and verbose to False when omitted."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.concurrency == 1
    assert config.verbose is False


def test_load_parses_artifacts_list(tmp_path: Path) -> None:
    """load() parses multiple artifacts into ArtifactConfig instances."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: codebase\n    type: coding\n    language: python\n  - name: docs\n    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert len(config.artifacts) == 2
    assert config.artifacts[0].name == "codebase"
    assert config.artifacts[0].type == "coding"
    assert config.artifacts[0].language == "python"
    assert config.artifacts[1].name == "docs"
    assert config.artifacts[1].type == "document"


def test_load_raises_on_missing_artifacts_key(tmp_path: Path) -> None:
    """load() raises ValueError when artifacts key is absent from the YAML file."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n")
    with pytest.raises(ValueError, match="artifacts"):
        ForgeConfig.load(p)


def test_load_raises_on_empty_artifacts_list(tmp_path: Path) -> None:
    """load() raises ValueError when artifacts list is present but empty."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\nartifacts: []\n")
    with pytest.raises(ValueError, match="artifacts"):
        ForgeConfig.load(p)


def test_artifact_config_has_name_and_type_fields() -> None:
    """ArtifactConfig stores name and type fields correctly."""
    artifact = ArtifactConfig(name="codebase", type="coding")
    assert artifact.name == "codebase"
    assert artifact.type == "coding"


def test_coding_artifact_without_language_raises(tmp_path: Path) -> None:
    """load() raises ValueError when a coding artifact has no language declared."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: codebase\n    type: coding\n",
    )
    with pytest.raises(ValueError, match="artifact 'codebase' has type 'coding' but no language declared"):
        ForgeConfig.load(p)


def test_non_coding_artifact_without_language_is_valid(tmp_path: Path) -> None:
    """load() succeeds when a non-coding artifact has no language declared."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: docs\n    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert config.artifacts[0].language is None


def test_language_is_parsed_correctly_from_yaml(tmp_path: Path) -> None:
    """load() sets language on ArtifactConfig when declared in YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.artifacts[0].language == "python"
