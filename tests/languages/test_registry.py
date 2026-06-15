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
                "init_command": f"init-{name} {{artifact_name}}",
                "test_command": "test-cmd",
                "sync_command": "sync-cmd",
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
    assert plugin.test_command == "test-cmd"
    assert plugin.sync_command == "sync-cmd"
    assert "rust" in plugin.prompt_supplement


def test_init_command_loaded_correctly_from_yaml(tmp_path: Path) -> None:
    """init_command is loaded verbatim from the YAML file."""
    _write_plugin(tmp_path, "python")
    reg = LanguageRegistry()
    reg.load(tmp_path)
    plugin = reg.get("python")
    assert plugin.init_command == "init-python {artifact_name}"


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


def test_python_supplement_distinguishes_uv_as_package_manager() -> None:
    """Python supplement must clarify uv is the package manager, not a project dependency."""
    data = yaml.safe_load((LANGUAGES_DIR / "python.yaml").read_text())
    supplement = data["prompt_supplement"]
    assert "uv is the package manager" in supplement
    assert "Do not add uv itself as a dependency" in supplement
    assert "Do not put uv in [project].dependencies" in supplement
    assert "Do not put uv in [build-system].requires" in supplement


def test_python_supplement_import_invariant() -> None:
    """Python supplement must state the src/-on-path invariant and show both incorrect forms."""
    data = yaml.safe_load((LANGUAGES_DIR / "python.yaml").read_text())
    supplement = data["prompt_supplement"]
    assert "Imports behave as if src/ is already on the Python path" in supplement
    assert "from src.mymodule import MyClass" in supplement
    assert "<artifact>.src.mymodule" in supplement


def test_language_work_output_examples_contain_no_format_placeholders() -> None:
    """work_output_example must contain no {identifier} format placeholders."""
    import re

    for path in LANGUAGES_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        example = data["work_output_example"]
        if example:
            assert not re.search(r"\{[a-zA-Z_]\w*\}", example), (
                f"{path.name}: work_output_example contains format placeholder"
            )
