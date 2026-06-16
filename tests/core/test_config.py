"""Tests for ForgeConfig loading and validation from YAML files."""

from pathlib import Path

import pytest

from forge.core.config import (
    ArtifactConfig,
    ForgeConfig,
    ForgeConfigLoader,
    ModelsConfig,
    load_config,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "forge.yaml"
    p.write_text(content)
    return p


_ARTIFACTS_YAML = "artifacts:\n  - name: codebase\n    type: coding\n    language: python\n"


def test_load_parses_valid_yaml(tmp_path: Path) -> None:
    """load() correctly parses all fields from a valid YAML config file."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'do the thing'\nworkspace: ./ws\nconcurrency: 4\nverbose: true\n"
        + _ARTIFACTS_YAML,
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
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n"
        "  - name: codebase\n"
        "    type: coding\n"
        "    language: python\n"
        "    description: Python package.\n"
        "  - name: docs\n"
        "    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert len(config.artifacts) == 2
    assert config.artifacts[0].name == "codebase"
    assert config.artifacts[0].type == "coding"
    assert config.artifacts[0].language == "python"
    assert config.artifacts[0].description == "Python package."
    assert config.artifacts[1].name == "docs"
    assert config.artifacts[1].type == "document"
    assert config.artifacts[1].description is None


def test_artifact_description_parses(tmp_path: Path) -> None:
    """load() parses optional artifact descriptions."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n"
        "  - name: codebase\n"
        "    type: coding\n"
        "    language: python\n"
        "    description: Python package containing implementation and tests.\n",
    )
    config = ForgeConfig.load(p)
    assert config.artifacts[0].description == "Python package containing implementation and tests."


def test_missing_artifact_description_remains_valid(tmp_path: Path) -> None:
    """Artifact configs without descriptions remain valid."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.artifacts[0].description is None


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
    with pytest.raises(
        ValueError, match="artifact 'codebase' has type 'coding' but no language declared"
    ):
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
    p = _write_yaml(
        tmp_path, "northstar: 'goal'\nworkspace: ./ws\nmax_retries: 5\n" + _ARTIFACTS_YAML
    )
    config = ForgeConfig.load(p)
    assert config.max_retries == 5


def test_max_tokens_defaults_to_8192(tmp_path: Path) -> None:
    """load() defaults max_tokens to 8192 when not present in YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_tokens == 8192


def test_max_tokens_parsed_from_yaml(tmp_path: Path) -> None:
    """load() reads max_tokens from YAML when explicitly declared."""
    p = _write_yaml(
        tmp_path, "northstar: 'goal'\nworkspace: ./ws\nmax_tokens: 4096\n" + _ARTIFACTS_YAML
    )
    config = ForgeConfig.load(p)
    assert config.max_tokens == 4096


def test_models_defaults_to_ollama_when_absent(tmp_path: Path) -> None:
    """load() defaults planner and worker to full PWC when models section is absent."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.models.planner.producer == "ollama/gemma4:e4b"
    assert config.models.planner.critic == "ollama/gemma4:e4b"
    assert config.models.planner.referee == "ollama/gemma4:e4b"
    assert config.models.worker.producer == "ollama/gemma4:e4b"
    assert config.models.worker.critic == "ollama/gemma4:e4b"
    assert config.models.worker.referee == "ollama/gemma4:e4b"
    assert not hasattr(config.models, "integrator")


def test_compact_planner_string_expands_to_full_pwc(tmp_path: Path) -> None:
    """A compact planner string configures planner producer, critic, and referee."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner: anthropic/claude-haiku\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.producer == "anthropic/claude-haiku"
    assert config.models.planner.critic == "anthropic/claude-haiku"
    assert config.models.planner.referee == "anthropic/claude-haiku"


def test_compact_worker_string_expands_to_full_pwc(tmp_path: Path) -> None:
    """A compact worker string configures worker producer, critic, and referee."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  worker: anthropic/claude-haiku\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.worker.producer == "anthropic/claude-haiku"
    assert config.models.worker.critic == "anthropic/claude-haiku"
    assert config.models.worker.referee == "anthropic/claude-haiku"


def test_old_flat_models_section_parsed_correctly(tmp_path: Path) -> None:
    """load() keeps accepting old flat planner/worker/critic/referee model fields."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner: claude/claude-sonnet-4-20250514\n"
        + "  worker: openai/gpt-4o\n"
        + "  critic: claude/planner-critic\n"
        + "  referee: openai/referee\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.producer == "claude/claude-sonnet-4-20250514"
    assert config.models.worker.producer == "openai/gpt-4o"
    assert config.models.planner.critic == "claude/planner-critic"
    assert config.models.planner.referee == "openai/referee"
    assert config.models.worker.critic == "claude/planner-critic"
    assert config.models.worker.referee == "openai/referee"


def test_new_nested_models_section_parsed_correctly(tmp_path: Path) -> None:
    """load() reads producer/critic/referee nested under scheduler roles."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner:\n"
        + "    producer: claude/planner\n"
        + "    critic: claude/planner-critic\n"
        + "    referee: claude/planner-referee\n"
        + "  worker:\n"
        + "    producer: openai/worker\n"
        + "    critic: openai/worker-critic\n"
        + "    referee: openai/worker-referee\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.producer == "claude/planner"
    assert config.models.planner.critic == "claude/planner-critic"
    assert config.models.planner.referee == "claude/planner-referee"
    assert config.models.worker.producer == "openai/worker"
    assert config.models.worker.critic == "openai/worker-critic"
    assert config.models.worker.referee == "openai/worker-referee"


def test_models_config_defaults() -> None:
    """ModelsConfig defaults planner and worker producers when constructed without args."""
    m = ModelsConfig()
    assert m.planner.producer == "ollama/gemma4:e4b"
    assert m.planner.critic == "ollama/gemma4:e4b"
    assert m.planner.referee == "ollama/gemma4:e4b"
    assert m.worker.producer == "ollama/gemma4:e4b"
    assert m.worker.critic == "ollama/gemma4:e4b"
    assert m.worker.referee == "ollama/gemma4:e4b"
    assert not hasattr(m, "integrator")


def test_nested_explicit_null_critic_and_referee_disables_review(tmp_path: Path) -> None:
    """Nested null critic/referee disables review for that PWC role."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner:\n"
        + "    producer: ollama/planner\n"
        + "    critic: null\n"
        + "    referee: null\n"
        + "  worker:\n"
        + "    producer: ollama/worker\n"
        + "    critic: null\n"
        + "    referee: null\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.critic is None
    assert config.models.planner.referee is None
    assert config.models.worker.critic is None
    assert config.models.worker.referee is None


def test_models_critic_and_referee_default_to_full_pwc() -> None:
    """ModelsConfig defaults nested critic and referee to the default producer."""
    m = ModelsConfig()
    assert m.planner.critic == m.planner.producer
    assert m.planner.referee == m.planner.producer
    assert m.worker.critic == m.worker.producer
    assert m.worker.referee == m.worker.producer


def test_old_flat_critic_and_referee_map_to_planner_and_worker(tmp_path: Path) -> None:
    """load() maps old global critic/referee to both PWC configs."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n  critic: ollama/gemma4:e4b\n  referee: ollama/gemma4:e4b\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.critic == "ollama/gemma4:e4b"
    assert config.models.planner.referee == "ollama/gemma4:e4b"
    assert config.models.worker.critic == "ollama/gemma4:e4b"
    assert config.models.worker.referee == "ollama/gemma4:e4b"


def test_old_flat_global_critic_and_referee_override_compact_expansion(tmp_path: Path) -> None:
    """Old flat global critic/referee override compact planner and worker strings."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner: ollama/planner\n"
        + "  worker: ollama/worker\n"
        + "  critic: ollama/critic\n"
        + "  referee: ollama/referee\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.producer == "ollama/planner"
    assert config.models.planner.critic == "ollama/critic"
    assert config.models.planner.referee == "ollama/referee"
    assert config.models.worker.producer == "ollama/worker"
    assert config.models.worker.critic == "ollama/critic"
    assert config.models.worker.referee == "ollama/referee"


def test_models_critic_and_referee_absent_defaults_to_full_pwc(tmp_path: Path) -> None:
    """load() defaults critic/referee to producer when omitted from the models section."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.models.planner.critic == config.models.planner.producer
    assert config.models.planner.referee == config.models.planner.producer
    assert config.models.worker.critic == config.models.worker.producer
    assert config.models.worker.referee == config.models.worker.producer


def test_pwc_max_attempts_defaults_to_three(tmp_path: Path) -> None:
    """PwcModelConfig max_attempts defaults to 3 when absent from YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.models.planner.max_attempts == 3
    assert config.models.worker.max_attempts == 3


def test_pwc_max_attempts_parsed_from_yaml(tmp_path: Path) -> None:
    """PwcModelConfig max_attempts is read correctly when declared in YAML."""
    yaml = (
        "northstar: 'goal'\nworkspace: ./ws\n"
        + _ARTIFACTS_YAML
        + "models:\n"
        + "  planner:\n"
        + "    producer: ollama/planner\n"
        + "    max_attempts: 7\n"
        + "  worker:\n"
        + "    producer: ollama/worker\n"
        + "    max_attempts: 5\n"
    )
    p = _write_yaml(tmp_path, yaml)
    config = ForgeConfig.load(p)
    assert config.models.planner.max_attempts == 7
    assert config.models.worker.max_attempts == 5


def test_max_tool_iterations_defaults_to_25(tmp_path: Path) -> None:
    """load() defaults max_tool_iterations to 25 when not present in YAML."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfig.load(p)
    assert config.max_tool_iterations == 25


def test_max_tool_iterations_parsed_from_yaml(tmp_path: Path) -> None:
    """load() reads max_tool_iterations from YAML when explicitly declared."""
    p = _write_yaml(
        tmp_path, "northstar: 'goal'\nworkspace: ./ws\nmax_tool_iterations: 50\n" + _ARTIFACTS_YAML
    )
    config = ForgeConfig.load(p)
    assert config.max_tool_iterations == 50


# ForgeConfigLoader direct tests


_MINIMAL_DATA: dict[str, object] = {
    "northstar": "do the thing",
    "workspace": "./ws",
    "artifacts": [{"name": "codebase", "type": "coding", "language": "python"}],
}


def test_loader_load_data_returns_forge_config() -> None:
    """ForgeConfigLoader.load_data parses a minimal valid mapping."""
    config = ForgeConfigLoader().load_data(_MINIMAL_DATA)
    assert config.northstar == "do the thing"
    assert config.workspace.is_absolute()
    assert len(config.artifacts) == 1


def test_loader_load_data_raises_on_missing_northstar() -> None:
    """ForgeConfigLoader.load_data raises when northstar is absent."""
    data: dict[str, object] = {
        "workspace": "./ws",
        "artifacts": [{"name": "codebase", "type": "coding", "language": "python"}],
    }
    with pytest.raises(ValueError, match="northstar"):
        ForgeConfigLoader().load_data(data)


def test_loader_load_data_raises_on_missing_workspace() -> None:
    """ForgeConfigLoader.load_data raises when workspace is absent."""
    data: dict[str, object] = {
        "northstar": "goal",
        "artifacts": [{"name": "codebase", "type": "coding", "language": "python"}],
    }
    with pytest.raises(ValueError, match="workspace"):
        ForgeConfigLoader().load_data(data)


def test_loader_load_artifacts_parses_single_artifact() -> None:
    """ForgeConfigLoader.load_artifacts converts raw list to ArtifactConfig instances."""
    raw = [{"name": "codebase", "type": "coding", "language": "python"}]
    artifacts = ForgeConfigLoader().load_artifacts(raw)
    assert len(artifacts) == 1
    assert artifacts[0].name == "codebase"
    assert artifacts[0].language == "python"


def test_loader_load_artifacts_raises_on_empty_list() -> None:
    """ForgeConfigLoader.load_artifacts raises when the list is empty."""
    with pytest.raises(ValueError, match="artifacts"):
        ForgeConfigLoader().load_artifacts([])


def test_loader_load_artifacts_raises_on_non_list() -> None:
    """ForgeConfigLoader.load_artifacts raises when raw is not a list."""
    with pytest.raises(ValueError, match="artifacts"):
        ForgeConfigLoader().load_artifacts("not-a-list")


def test_loader_load_models_config_returns_models_config() -> None:
    """ForgeConfigLoader.load_models_config parses nested planner/worker mapping."""
    raw: dict[str, object] = {
        "planner": {"producer": "claude/sonnet", "critic": "claude/haiku"},
        "worker": {"producer": "openai/gpt-4o"},
    }
    models = ForgeConfigLoader().load_models_config(raw)
    assert models.planner.producer == "claude/sonnet"
    assert models.planner.critic == "claude/haiku"
    assert models.worker.producer == "openai/gpt-4o"


def test_loader_load_models_config_defaults_when_empty() -> None:
    """ForgeConfigLoader.load_models_config returns defaults for an empty mapping."""
    models = ForgeConfigLoader().load_models_config({})
    assert models.planner.producer == "ollama/gemma4:e4b"
    assert models.worker.producer == "ollama/gemma4:e4b"


def test_loader_load_path_delegates_to_load_data(tmp_path: Path) -> None:
    """ForgeConfigLoader.load_path reads YAML from disk and produces a ForgeConfig."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    config = ForgeConfigLoader().load_path(p)
    assert config.northstar == "goal"
    assert config.workspace.is_absolute()


def test_load_config_wrapper_delegates_correctly(tmp_path: Path) -> None:
    """load_config() is a thin wrapper that returns the same result as ForgeConfig.load()."""
    p = _write_yaml(tmp_path, "northstar: 'goal'\nworkspace: ./ws\n" + _ARTIFACTS_YAML)
    assert load_config(p).northstar == ForgeConfig.load(p).northstar
    assert load_config(p).workspace == ForgeConfig.load(p).workspace


def test_loader_invalid_int_scalar_raises() -> None:
    """ForgeConfigLoader.load_data raises when concurrency is not an integer."""
    data: dict[str, object] = {**_MINIMAL_DATA, "concurrency": "fast"}
    with pytest.raises(ValueError, match="concurrency"):
        ForgeConfigLoader().load_data(data)


def test_loader_invalid_bool_scalar_raises() -> None:
    """ForgeConfigLoader.load_data raises when verbose is not a boolean."""
    data: dict[str, object] = {**_MINIMAL_DATA, "verbose": "yes"}
    with pytest.raises(ValueError, match="verbose"):
        ForgeConfigLoader().load_data(data)


# --- P0: artifact name validation ---


def test_artifact_name_simple_letters_accepted(tmp_path: Path) -> None:
    """Artifact names consisting solely of letters are valid."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: codebase\n    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert config.artifacts[0].name == "codebase"


def test_artifact_name_with_hyphen_and_underscore_accepted(tmp_path: Path) -> None:
    """Artifact names with hyphens and underscores are valid."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: my-app_v2\n    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert config.artifacts[0].name == "my-app_v2"


def test_artifact_name_with_digits_accepted(tmp_path: Path) -> None:
    """Artifact names with digits are valid."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: app2\n    type: document\n",
    )
    config = ForgeConfig.load(p)
    assert config.artifacts[0].name == "app2"


def test_artifact_name_with_space_rejected(tmp_path: Path) -> None:
    """Artifact names containing spaces are rejected at load time."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: 'my app'\n    type: document\n",
    )
    with pytest.raises(ValueError, match="invalid"):
        ForgeConfig.load(p)


def test_artifact_name_with_slash_rejected(tmp_path: Path) -> None:
    """Artifact names containing slashes are rejected at load time."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: foo/bar\n    type: document\n",
    )
    with pytest.raises(ValueError, match="invalid"):
        ForgeConfig.load(p)


def test_artifact_name_with_semicolon_rejected(tmp_path: Path) -> None:
    """Artifact names containing semicolons are rejected at load time."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: foo;bar\n    type: document\n",
    )
    with pytest.raises(ValueError, match="invalid"):
        ForgeConfig.load(p)


def test_artifact_name_with_shell_metachar_rejected(tmp_path: Path) -> None:
    """Artifact names containing shell metacharacters are rejected at load time."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: 'foo$bar'\n    type: document\n",
    )
    with pytest.raises(ValueError, match="invalid"):
        ForgeConfig.load(p)


def test_artifact_name_path_traversal_rejected(tmp_path: Path) -> None:
    """Artifact names containing path traversal sequences are rejected at load time."""
    p = _write_yaml(
        tmp_path,
        "northstar: 'goal'\nworkspace: ./ws\nartifacts:\n  - name: ../secret\n    type: document\n",
    )
    with pytest.raises(ValueError, match="invalid"):
        ForgeConfig.load(p)
