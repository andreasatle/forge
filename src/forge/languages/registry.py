"""Language plugin registry for loading and accessing language configurations."""

from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_FIELDS = (
    "name",
    "init_command",
    "test_command",
    "sync_command",
    "prompt_supplement",
    "work_output_example",
)


@dataclass
class LanguagePlugin:
    """Language plugin loaded from a YAML file defining project structure and tooling."""

    name: str
    init_command: str
    test_command: str
    sync_command: str
    prompt_supplement: str
    work_output_example: str


class LanguageRegistry:
    """Registry for loading and retrieving language plugins by name."""

    def __init__(self) -> None:
        self._plugins: dict[str, LanguagePlugin] = {}

    def register(self, plugin: LanguagePlugin) -> None:
        """Register a language plugin directly."""
        self._plugins[plugin.name] = plugin

    def load(self, languages_dir: Path) -> None:
        """Load all *.yaml language plugins from the given directory."""
        for path in sorted(languages_dir.glob("*.yaml")):
            with path.open() as f:
                data = yaml.safe_load(f)
            for field in _REQUIRED_FIELDS:
                if field not in data:
                    raise ValueError(
                        f"language plugin {path.name!r} missing required field: {field!r}"
                    )
            plugin = LanguagePlugin(
                name=data["name"],
                init_command=data["init_command"],
                test_command=data["test_command"],
                sync_command=data["sync_command"],
                prompt_supplement=data["prompt_supplement"],
                work_output_example=data["work_output_example"],
            )
            self._plugins[plugin.name] = plugin
            print(f"loaded language plugin: {plugin.name}")

    def get(self, name: str) -> LanguagePlugin:
        """Return the language plugin for the given name, raising KeyError if unknown."""
        if name not in self._plugins:
            raise KeyError(f"unknown language: {name}")
        return self._plugins[name]

    def names(self) -> list[str]:
        """Return a sorted list of all registered language names."""
        return sorted(self._plugins)
