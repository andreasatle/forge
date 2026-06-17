"""Adapter registry for loading and accessing named adapter configurations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


def _default_mutating_tools() -> list[str]:
    """Return default mutating tool names."""
    return ["write_file", "replace_in_file"]


def _empty_str_list() -> list[str]:
    """Return an empty string list for dataclass defaults."""
    return []


def _str_list(value: object) -> list[str]:
    """Coerce a YAML value into a list of strings."""
    if value is None:
        return []

    items = cast(list[object], value)

    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"expected str, got {type(item).__name__}")

    return cast(list[str], items)


@dataclass
class AdapterSpec:
    """Named adapter configuration loaded from a YAML file."""

    name: str
    description: str
    tools: list[str]
    prompt_template: str
    requires_nonempty_output: bool = True
    work_noun: str = "implementation"
    review_focus: str = ""
    mutating_tools: list[str] = field(default_factory=_default_mutating_tools)
    verification_tools: list[str] = field(default_factory=_empty_str_list)
    verification_required: bool | None = None


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
                data: dict[str, Any] = yaml.safe_load(f)

            for required_field in _REQUIRED_FIELDS:
                if required_field not in data:
                    raise ValueError(
                        f"adapter {path.name!r} missing required field: {required_field!r}"
                    )

            spec = AdapterSpec(
                name=str(data["name"]),
                description=str(data["description"]),
                tools=_str_list(data["tools"]),
                prompt_template=str(data["prompt_template"]),
                requires_nonempty_output=bool(data.get("requires_nonempty_output", True)),
                work_noun=str(data.get("work_noun", "implementation")),
                review_focus=str(data.get("review_focus", "")),
                mutating_tools=_str_list(data.get("mutating_tools", _default_mutating_tools())),
                verification_tools=_str_list(data.get("verification_tools", [])),
                verification_required=data.get("verification_required"),
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
