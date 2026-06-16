"""Tests for AdapterRegistry loading and retrieval behaviour."""

from pathlib import Path

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec

ADAPTERS_DIR = Path(__file__).parents[2] / "adapters"


def _write_yaml(tmp_path: Path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content)


def test_registry_loads_all_yamls_from_directory(tmp_path: Path) -> None:
    """Registry loads every *.yaml file in the directory and registers each adapter."""
    _write_yaml(
        tmp_path,
        "coding.yaml",
        """
name: coding
description: Writes code
tools:
  - read_file
prompt_template: "do: {{ objective }}"
""",
    )
    _write_yaml(
        tmp_path,
        "audit.yaml",
        """
name: audit
description: Audits code
tools:
  - read_file
prompt_template: "audit: {{ objective }}"
""",
    )

    registry = AdapterRegistry()
    registry.load(tmp_path)

    assert registry.names() == ["audit", "coding"]


def test_registry_raises_on_unknown_adapter_name(tmp_path: Path) -> None:
    """get() raises KeyError when the requested adapter name is not registered."""
    _write_yaml(
        tmp_path,
        "coding.yaml",
        """
name: coding
description: Writes code
tools:
  - read_file
prompt_template: "do: {{ objective }}"
""",
    )

    registry = AdapterRegistry()
    registry.load(tmp_path)

    with pytest.raises(KeyError, match="unknown adapter: missing"):
        registry.get("missing")


def test_registry_raises_on_malformed_yaml_missing_field(tmp_path: Path) -> None:
    """load() raises ValueError when a YAML file is missing a required field."""
    _write_yaml(
        tmp_path,
        "bad.yaml",
        """
name: bad
description: Missing tools and prompt_template
""",
    )

    registry = AdapterRegistry()

    with pytest.raises(ValueError, match="missing required field"):
        registry.load(tmp_path)


def test_registry_get_returns_correct_adapter_spec(tmp_path: Path) -> None:
    """get() returns an AdapterSpec with all fields from the loaded YAML."""
    _write_yaml(
        tmp_path,
        "coding.yaml",
        """
name: coding
description: Writes and edits code
tools:
  - list_files
  - read_file
  - run_tests
prompt_template: "Complete: {{ objective }}"
""",
    )

    registry = AdapterRegistry()
    registry.load(tmp_path)

    spec = registry.get("coding")

    assert isinstance(spec, AdapterSpec)
    assert spec.name == "coding"
    assert spec.description == "Writes and edits code"
    assert spec.tools == ["list_files", "read_file", "run_tests"]
    assert "{{ objective }}" in spec.prompt_template


def test_validation_adapters_do_not_duplicate_response_schema_examples() -> None:
    """Pydantic models own critic/referee response schemas, not YAML examples."""
    registry = AdapterRegistry()
    registry.load(ADAPTERS_DIR)

    for adapter_name in ("critic", "referee"):
        prompt = registry.get(adapter_name).prompt_template
        assert "Respond with a JSON object exactly matching this structure" not in prompt
        assert "{{" not in prompt


def test_adapter_spec_new_fields_default_correctly() -> None:
    """AdapterSpec defaults mutating_tools, verification_tools, and verification_required."""
    spec = AdapterSpec(
        name="test",
        description="test adapter",
        tools=["read_file"],
        prompt_template="do: {objective}",
    )
    assert spec.mutating_tools == ["write_file", "replace_in_file"]
    assert spec.verification_tools == []
    assert spec.verification_required is None


def test_adapter_spec_new_fields_loaded_from_yaml(tmp_path: Path) -> None:
    """load() reads mutating_tools, verification_tools, and verification_required when declared."""
    _write_yaml(
        tmp_path,
        "custom.yaml",
        """
name: custom
description: Custom adapter with explicit tool semantics
tools:
  - custom_write
  - custom_check
prompt_template: "do: {objective}"
mutating_tools:
  - custom_write
verification_tools:
  - custom_check
verification_required: true
""",
    )
    registry = AdapterRegistry()
    registry.load(tmp_path)
    spec = registry.get("custom")

    assert spec.mutating_tools == ["custom_write"]
    assert spec.verification_tools == ["custom_check"]
    assert spec.verification_required is True


def test_existing_adapter_yaml_loads_without_new_fields(tmp_path: Path) -> None:
    """Adapters without new semantic fields still load and get correct defaults."""
    _write_yaml(
        tmp_path,
        "legacy.yaml",
        """
name: legacy
description: Legacy adapter
tools:
  - read_file
  - write_file
prompt_template: "do: {objective}"
""",
    )
    registry = AdapterRegistry()
    registry.load(tmp_path)
    spec = registry.get("legacy")

    assert spec.mutating_tools == ["write_file", "replace_in_file"]
    assert spec.verification_tools == []
    assert spec.verification_required is None


def test_real_adapter_yamls_load_without_new_fields() -> None:
    """Real adapter YAML files load with correct semantic field values."""
    registry = AdapterRegistry()
    registry.load(ADAPTERS_DIR)

    for name in ("audit", "document"):
        spec = registry.get(name)
        assert spec.mutating_tools == ["write_file", "replace_in_file"]
        assert spec.verification_tools == []
        assert spec.verification_required is None

    coding = registry.get("coding")
    assert coding.mutating_tools == ["write_file", "replace_in_file"]
    assert coding.verification_tools == ["run_tests"]
    assert coding.verification_required is None


def test_coding_adapter_tools_do_not_include_run_tests() -> None:
    """coding.yaml tools list contains only tools available in the base worktree registry."""

    registry = AdapterRegistry()
    registry.load(ADAPTERS_DIR)
    coding = registry.get("coding")

    assert "run_tests" not in coding.tools
    assert "run_tests" in coding.verification_tools


def test_coding_adapter_declared_tools_resolve_without_language_plugin(tmp_path: Path) -> None:
    """All tools in coding.tools are present in the base worktree registry (no language plugin)."""
    from forge.tools.builtin import build_worktree_registry

    registry = AdapterRegistry()
    registry.load(ADAPTERS_DIR)
    coding = registry.get("coding")

    full_registry = build_worktree_registry(str(tmp_path))
    for tool_name in coding.tools:
        full_registry.get(tool_name)
