"""Tests for LanguageRegistry loading and retrieval."""

import re
from pathlib import Path

import pytest
import yaml

from forge.languages.registry import LanguageRegistry

LANGUAGES_DIR = Path(__file__).parents[2] / "languages"
TOOL_LIKE_NAME = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
NON_TOOL_SUPPLEMENT_NAMES = {"ini_options"}


def _write_plugin(dir: Path, name: str) -> None:
    (dir / f"{name}.yaml").write_text(
        yaml.dump(
            {
                "name": name,
                "package_manager": "test-pm",
                "init_command": f"init-{name} {{artifact_name}}",
                "test_command": "test-cmd",
                "sync_command": "sync-cmd",
                "add_dependency_command": "pm add {package}",
                "project_structure": ["src/", "tests/"],
                "prompt_supplement": f"Use {name} conventions.",
                "work_output_example": "",
            }
        )
    )


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


def test_init_command_loaded_correctly_from_yaml(tmp_path: Path) -> None:
    """init_command is loaded verbatim from the YAML file."""
    _write_plugin(tmp_path, "python")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("python")
    assert plugin.init_command == "init-python {artifact_name}"


def test_add_dependency_command_loaded_correctly_from_yaml(tmp_path: Path) -> None:
    """add_dependency_command is loaded verbatim from the YAML file."""
    (tmp_path / "custom.yaml").write_text(
        yaml.dump(
            {
                "name": "custom",
                "package_manager": "custom-pm",
                "init_command": "custom init {artifact_name}",
                "test_command": "custom-test",
                "sync_command": "custom-sync",
                "add_dependency_command": "custom install {package}",
                "project_structure": ["src/"],
                "prompt_supplement": "Use custom conventions.",
                "work_output_example": "",
            }
        )
    )
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("custom")
    assert plugin.add_dependency_command == "custom install {package}"


_STALE_WORKER_PHRASES = [
    "after every file change",
    "after each file change",
    "never rewrite the same file",
    "iterate until tests pass",
    "verify progress after",
    "if tests fail",
    "check the current test status",
    "checking test status",
]


def test_language_supplements_have_no_stale_worker_mutation_phrases() -> None:
    """Language supplements must not contain stale worker mutation directives."""
    for path in LANGUAGES_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        supplement = data["prompt_supplement"].lower()
        for phrase in _STALE_WORKER_PHRASES:
            assert phrase not in supplement, f"{path.name}: stale phrase {phrase!r}"


def test_language_prompt_supplements_do_not_name_tools() -> None:
    """Language supplements stay convention-focused instead of naming tool APIs."""
    for path in LANGUAGES_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        supplement = data["prompt_supplement"]
        assert "tool_call" not in supplement, path.name
        tool_like_names = set(TOOL_LIKE_NAME.findall(supplement)) - NON_TOOL_SUPPLEMENT_NAMES
        assert not tool_like_names, path.name


def test_python_supplement_contains_packaging_guidance() -> None:
    """Python packaging policy belongs in the Python language supplement."""
    data = yaml.safe_load((LANGUAGES_DIR / "python.yaml").read_text())
    supplement = data["prompt_supplement"]
    assert "pyproject.toml" in supplement
    assert "uv" in supplement
    assert "requirements.txt" in supplement
    assert "setup.py" in supplement


def test_language_work_output_examples_are_str_format_safe() -> None:
    """work_output_example must be safe to call str.format(base_version=N) on — all literal braces escaped."""
    for path in LANGUAGES_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        example = data["work_output_example"]
        if example:
            example.format(base_version=0)
