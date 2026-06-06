"""Tests for LanguageRegistry loading and retrieval."""

from pathlib import Path

import pytest
import yaml

from forge.languages.registry import LanguageRegistry


def _write_plugin(dir: Path, name: str) -> None:
    (dir / f"{name}.yaml").write_text(yaml.dump({
        "name": name,
        "package_manager": "test-pm",
        "test_command": "test-cmd",
        "sync_command": "sync-cmd",
        "add_dependency_command": "pm add {package}",
        "project_structure": ["src/", "tests/"],
        "prompt_supplement": f"Use {name} conventions.",
    }))


def test_registry_loads_all_yamls(tmp_path: Path) -> None:
    """load() discovers and loads all *.yaml files from the directory."""
    _write_plugin(tmp_path, "alpha")
    _write_plugin(tmp_path, "beta")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    assert "alpha" in reg.names()
    assert "beta" in reg.names()


def test_registry_raises_on_unknown_language(tmp_path: Path) -> None:
    """get() raises KeyError for an unknown language name."""
    reg = LanguageRegistry()
    reg.load(tmp_path)
    with pytest.raises(KeyError, match="unknown language"):
        reg.get("nonexistent")


def test_registry_raises_on_malformed_yaml(tmp_path: Path) -> None:
    """load() raises ValueError when a YAML file is missing a required field."""
    (tmp_path / "bad.yaml").write_text("name: bad\n")
    reg = LanguageRegistry()
    with pytest.raises(ValueError, match="missing required field"):
        reg.load(tmp_path)


def test_get_returns_correct_plugin(tmp_path: Path) -> None:
    """get() returns the LanguagePlugin matching the given name."""
    _write_plugin(tmp_path, "python")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("python")
    assert plugin.name == "python"


def test_language_plugin_has_all_required_fields(tmp_path: Path) -> None:
    """LanguagePlugin stores all required fields from the YAML file."""
    _write_plugin(tmp_path, "rust")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("rust")
    assert plugin.package_manager == "test-pm"
    assert plugin.test_command == "test-cmd"
    assert plugin.sync_command == "sync-cmd"
    assert plugin.project_structure == ["src/", "tests/"]
    assert "rust" in plugin.prompt_supplement


def test_language_plugin_has_add_dependency_command(tmp_path: Path) -> None:
    """LanguagePlugin stores add_dependency_command from the YAML file."""
    _write_plugin(tmp_path, "python")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("python")
    assert plugin.add_dependency_command == "pm add {package}"


def test_add_dependency_command_loaded_correctly_from_yaml(tmp_path: Path) -> None:
    """add_dependency_command is loaded verbatim from the YAML file."""
    (tmp_path / "custom.yaml").write_text(yaml.dump({
        "name": "custom",
        "package_manager": "custom-pm",
        "test_command": "custom-test",
        "sync_command": "custom-sync",
        "add_dependency_command": "custom install {package}",
        "project_structure": ["src/"],
        "prompt_supplement": "Use custom conventions.",
    }))
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("custom")
    assert plugin.add_dependency_command == "custom install {package}"
