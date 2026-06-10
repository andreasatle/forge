"""ForgeConfig dataclass for loading and validating the forge YAML config file."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ArtifactConfig:
    """Named artifact with a type and optional language declared in forge.yaml."""

    name: str
    type: str  # "coding" | "document" | "audit"
    language: str | None = None


@dataclass
class ModelsConfig:
    """Model configuration per agent type."""

    planner: str = "ollama/gemma4:e4b"
    worker: str = "ollama/gemma4:e4b"
    integrator: str = "ollama/gemma4:e4b"
    critic: str | None = None
    referee: str | None = None


@dataclass
class ForgeConfig:
    """Runtime configuration for a forge run loaded from a YAML file."""

    northstar: str
    workspace: Path
    artifacts: list[ArtifactConfig]
    models: ModelsConfig = field(default_factory=ModelsConfig)
    concurrency: int = 1
    verbose: bool = False
    max_retries: int = 3
    max_tokens: int = 8192
    max_tool_iterations: int = 25

    @staticmethod
    def load(path: Path) -> "ForgeConfig":
        """Parse the YAML file at path and return a validated ForgeConfig."""
        data = yaml.safe_load(path.read_text())
        if "northstar" not in data:
            raise ValueError("ForgeConfig: missing required field 'northstar'")
        if "workspace" not in data:
            raise ValueError("ForgeConfig: missing required field 'workspace'")
        if "artifacts" not in data or not data["artifacts"]:
            raise ValueError("artifacts is required — declare at least one artifact in forge.yaml")
        artifacts = [
            ArtifactConfig(name=a["name"], type=a["type"], language=a.get("language"))
            for a in data["artifacts"]
        ]
        for artifact in artifacts:
            if artifact.type == "coding" and not artifact.language:
                raise ValueError(
                    f"artifact '{artifact.name}' has type 'coding' but no language declared"
                )
        models_data = data.get("models", {})
        models = ModelsConfig(
            planner=models_data.get("planner", "ollama/gemma4:e4b"),
            worker=models_data.get("worker", "ollama/gemma4:e4b"),
            integrator=models_data.get("integrator", "ollama/gemma4:e4b"),
            critic=models_data.get("critic"),
            referee=models_data.get("referee"),
        )
        return ForgeConfig(
            northstar=data["northstar"],
            workspace=Path(data["workspace"]).resolve(),
            artifacts=artifacts,
            models=models,
            concurrency=data.get("concurrency", 1),
            verbose=data.get("verbose", False),
            max_retries=data.get("max_retries", 3),
            max_tokens=data.get("max_tokens", 8192),
            max_tool_iterations=data.get("max_tool_iterations", 25),
        )
