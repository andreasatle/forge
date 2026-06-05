from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ForgeConfig:
    northstar: str
    workspace: Path
    concurrency: int = 1
    verbose: bool = False

    @staticmethod
    def load(path: Path) -> "ForgeConfig":
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
