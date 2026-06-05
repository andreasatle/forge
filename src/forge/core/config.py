"""ForgeConfig dataclass for loading and validating the forge YAML config file."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ForgeConfig:
    """Runtime configuration for a forge run loaded from a YAML file."""

    northstar: str
    workspace: Path
    concurrency: int = 1
    verbose: bool = False

    @staticmethod
    def load(path: Path) -> "ForgeConfig":
        """Parse the YAML file at path and return a validated ForgeConfig."""
        data = yaml.safe_load(path.read_text())
        if "northstar" not in data:
            raise ValueError("ForgeConfig: missing required field 'northstar'")
        if "workspace" not in data:
            raise ValueError("ForgeConfig: missing required field 'workspace'")
        return ForgeConfig(
            northstar=data["northstar"],
            workspace=Path(data["workspace"]).resolve(),
            concurrency=data.get("concurrency", 1),
            verbose=data.get("verbose", False),
        )
