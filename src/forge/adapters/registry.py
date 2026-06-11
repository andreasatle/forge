"""Adapter registry for loading and accessing named adapter configurations."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AdapterSpec:
    """Named adapter configuration loaded from a YAML file."""

    name: str
    description: str
    tools: list[str]
    prompt_template: str
    requires_nonempty_output: bool = True
    work_noun: str = "implementation"


_REQUIRED_FIELDS = ("name", "description", "tools", "prompt_template")


class AdapterRegistry:
    """Registry for loading and retrieving adapter configurations by name."""

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterSpec] = {}

    def register(self, spec: AdapterSpec) -> None:
        """Register an adapter spec directly."""
        self._adapters[spec.name] = spec

    def load(self, adapters_dir: Path) -> None:
        """Load all *.yaml adapter files from the given directory."""
        for path in sorted(adapters_dir.glob("*.yaml")):
            with path.open() as f:
                data = yaml.safe_load(f)
            for field in _REQUIRED_FIELDS:
                if field not in data:
                    raise ValueError(f"adapter {path.name!r} missing required field: {field!r}")
            spec = AdapterSpec(
                name=data["name"],
                description=data["description"],
                tools=data["tools"],
                prompt_template=data["prompt_template"],
                requires_nonempty_output=data.get("requires_nonempty_output", True),
                work_noun=data.get("work_noun", "implementation"),
            )
            self._adapters[spec.name] = spec
            print(f"loaded adapter: {spec.name}")

    def get(self, name: str) -> AdapterSpec:
        """Return the adapter spec for the given name, raising KeyError if unknown."""
        if name not in self._adapters:
            raise KeyError(f"unknown adapter: {name}")
        return self._adapters[name]

    def names(self) -> list[str]:
        """Return a sorted list of all registered adapter names."""
        return sorted(self._adapters)
