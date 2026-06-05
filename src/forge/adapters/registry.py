from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AdapterSpec:
    name: str
    description: str
    tools: list[str]
    prompt_template: str


_REQUIRED_FIELDS = ("name", "description", "tools", "prompt_template")


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, AdapterSpec] = {}

    def load(self, adapters_dir: Path) -> None:
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
            )
            self._adapters[spec.name] = spec
            print(f"loaded adapter: {spec.name}")

    def get(self, name: str) -> AdapterSpec:
        if name not in self._adapters:
            raise KeyError(f"unknown adapter: {name}")
        return self._adapters[name]

    def names(self) -> list[str]:
        return sorted(self._adapters)
