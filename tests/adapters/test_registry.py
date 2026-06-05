"""Tests for AdapterRegistry loading and retrieval behaviour."""

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec


def _write_yaml(tmp_path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content)


def test_registry_loads_all_yamls_from_directory(tmp_path) -> None:
    """Registry loads every *.yaml file in the directory and registers each adapter."""
    _write_yaml(tmp_path, "coding.yaml", """
name: coding
description: Writes code
tools:
  - read_file
prompt_template: "do: {{ objective }}"
""")
    _write_yaml(tmp_path, "audit.yaml", """
name: audit
description: Audits code
tools:
  - read_file
prompt_template: "audit: {{ objective }}"
""")

    registry = AdapterRegistry()
    registry.load(tmp_path)

    assert registry.names() == ["audit", "coding"]


def test_registry_raises_on_unknown_adapter_name(tmp_path) -> None:
    """get() raises KeyError when the requested adapter name is not registered."""
    _write_yaml(tmp_path, "coding.yaml", """
name: coding
description: Writes code
tools:
  - read_file
prompt_template: "do: {{ objective }}"
""")

    registry = AdapterRegistry()
    registry.load(tmp_path)

    with pytest.raises(KeyError, match="unknown adapter: missing"):
        registry.get("missing")


def test_registry_raises_on_malformed_yaml_missing_field(tmp_path) -> None:
    """load() raises ValueError when a YAML file is missing a required field."""
    _write_yaml(tmp_path, "bad.yaml", """
name: bad
description: Missing tools and prompt_template
""")

    registry = AdapterRegistry()

    with pytest.raises(ValueError, match="missing required field"):
        registry.load(tmp_path)


def test_registry_get_returns_correct_adapter_spec(tmp_path) -> None:
    """get() returns an AdapterSpec with all fields from the loaded YAML."""
    _write_yaml(tmp_path, "coding.yaml", """
name: coding
description: Writes and edits code
tools:
  - read_file
  - write_file
prompt_template: "Complete: {{ objective }}"
""")

    registry = AdapterRegistry()
    registry.load(tmp_path)

    spec = registry.get("coding")

    assert isinstance(spec, AdapterSpec)
    assert spec.name == "coding"
    assert spec.description == "Writes and edits code"
    assert spec.tools == ["read_file", "write_file"]
    assert "{{ objective }}" in spec.prompt_template
