"""Tests for ForgeConfig loading and validation from YAML files."""

from pathlib import Path

import pytest

from forge.core.config import ArtifactConfig, ForgeConfig, ModelsConfig


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


def test_max_retries_defaults_to_three(tmp_path: Path) -> None:
    """load() defaults max_retries to 3 when not present in YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_retries == 3


def test_max_retries_parsed_from_yaml(tmp_path: Path) -> None:
    """load() reads max_retries from YAML when explicitly declared."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\nmax_retries: 5\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_retries == 5


def test_max_tokens_defaults_to_8192(tmp_path: Path) -> None:
    """load() defaults max_tokens to 8192 when not present in YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_tokens == 8192


def test_max_tokens_parsed_from_yaml(tmp_path: Path) -> None:
    """load() reads max_tokens from YAML when explicitly declared."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\nmax_tokens: 4096\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_tokens == 4096


def test_models_defaults_to_ollama_when_absent(tmp_path: Path) -> None:
    """load() defaults all model strings to ollama/gemma4:e4b when models section is absent."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.models.planner == "ollama/gemma4:e4b"
    assert config.models.worker == "ollama/gemma4:e4b"
    assert config.models.integrator == "ollama/gemma4:e4b"


def test_models_section_parsed_correctly(tmp_path: Path) -> None:
    """load() reads model strings from the models section when declared."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n  planner: claude/claude-sonnet-4-20250514\n  worker: openai/gpt-4o\n  integrator: ollama/gemma4:e4b\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner == "claude/claude-sonnet-4-20250514"
    assert config.models.worker == "openai/gpt-4o"
    assert config.models.integrator == "ollama/gemma4:e4b"


def test_models_config_defaults() -> None:
    """ModelsConfig defaults all fields to ollama/gemma4:e4b when constructed without args."""
    m = ModelsConfig()
    assert m.planner == "ollama/gemma4:e4b"
    assert m.worker == "ollama/gemma4:e4b"
    assert m.integrator == "ollama/gemma4:e4b"
