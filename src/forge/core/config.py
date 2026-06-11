"""ForgeConfig dataclass for loading and validating the forge YAML config file."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml


@dataclass
class ArtifactConfig:
    """Named artifact with a type and optional language declared in forge.yaml."""

    name: str
    type: str  # "coding" | "document" | "audit"
    language: str | None = None


@dataclass
class PwcModelConfig:
    """Model configuration for a producer/critic/referee loop."""

    producer: str = "ollama/gemma4:e4b"
    critic: str | None = None
    referee: str | None = None


@dataclass
class IntegratorModelConfig:
    """Model configuration for integrator behavior."""

    producer: str | None = None


@dataclass
class ModelsConfig:
    """Model configuration per scheduler agent type."""

    planner: PwcModelConfig = field(default_factory=PwcModelConfig)
    worker: PwcModelConfig = field(default_factory=PwcModelConfig)
    integrator: IntegratorModelConfig = field(default_factory=IntegratorModelConfig)


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
        raw: object = yaml.safe_load(path.read_text())
        data = _as_mapping(raw, "ForgeConfig")
        if "northstar" not in data:
            raise ValueError("ForgeConfig: missing required field 'northstar'")
        if "workspace" not in data:
            raise ValueError("ForgeConfig: missing required field 'workspace'")
        artifact_data = _required_sequence(data.get("artifacts"), "artifacts")
        if not artifact_data:
            raise ValueError("artifacts is required — declare at least one artifact in forge.yaml")
        artifacts = _load_artifacts(artifact_data)
        for artifact in artifacts:
            if artifact.type == "coding" and not artifact.language:
                raise ValueError(
                    f"artifact '{artifact.name}' has type 'coding' but no language declared"
                )
        models = _load_models_config(data.get("models", {}))
        return ForgeConfig(
            northstar=_required_string(data.get("northstar"), "northstar"),
            workspace=Path(_required_string(data.get("workspace"), "workspace")).resolve(),
            artifacts=artifacts,
            models=models,
            concurrency=_optional_int(data.get("concurrency"), 1, "concurrency"),
            verbose=_optional_bool(data.get("verbose"), False, "verbose"),
            max_retries=_optional_int(data.get("max_retries"), 3, "max_retries"),
            max_tokens=_optional_int(data.get("max_tokens"), 8192, "max_tokens"),
            max_tool_iterations=_optional_int(
                data.get("max_tool_iterations"), 25, "max_tool_iterations"
            ),
        )


def _as_mapping(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    return {str(key): item for key, item in mapping.items()}


def _required_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{field} must be a list")
    return cast(Sequence[object], value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _optional_int(value: object, default: int, field: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_bool(value: object, default: bool, field: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _load_artifacts(artifacts_data: Sequence[object]) -> list[ArtifactConfig]:
    artifacts: list[ArtifactConfig] = []
    for index, item in enumerate(artifacts_data):
        artifact = _as_mapping(item, f"artifacts[{index}]")
        artifacts.append(
            ArtifactConfig(
                name=_required_string(artifact.get("name"), f"artifacts[{index}].name"),
                type=_required_string(artifact.get("type"), f"artifacts[{index}].type"),
                language=_optional_string(artifact.get("language"), f"artifacts[{index}].language"),
            )
        )
    return artifacts


def _load_pwc_model_config(
    value: object,
    *,
    default_producer: str = "ollama/gemma4:e4b",
    fallback_critic: str | None = None,
    fallback_referee: str | None = None,
) -> PwcModelConfig:
    if isinstance(value, str):
        return PwcModelConfig(
            producer=value,
            critic=fallback_critic,
            referee=fallback_referee,
        )
    if value is None:
        return PwcModelConfig(
            producer=default_producer,
            critic=fallback_critic,
            referee=fallback_referee,
        )

    model_data = _as_mapping(value, "models.<role>")
    return PwcModelConfig(
        producer=_optional_string(model_data.get("producer"), "models.<role>.producer")
        or default_producer,
        critic=_optional_string(model_data.get("critic"), "models.<role>.critic")
        if "critic" in model_data
        else fallback_critic,
        referee=_optional_string(model_data.get("referee"), "models.<role>.referee")
        if "referee" in model_data
        else fallback_referee,
    )


def _load_integrator_model_config(value: object) -> IntegratorModelConfig:
    if isinstance(value, str):
        return IntegratorModelConfig(producer=value)
    if value is None:
        return IntegratorModelConfig()

    model_data = _as_mapping(value, "models.integrator")
    return IntegratorModelConfig(
        producer=_optional_string(model_data.get("producer"), "models.integrator.producer")
    )


def _load_models_config(models_data: object) -> ModelsConfig:
    models = _as_mapping(models_data, "models")

    flat_critic = _optional_string(models.get("critic"), "models.critic")
    flat_referee = _optional_string(models.get("referee"), "models.referee")
    return ModelsConfig(
        planner=_load_pwc_model_config(
            models.get("planner"),
            fallback_critic=flat_critic,
            fallback_referee=flat_referee,
        ),
        worker=_load_pwc_model_config(
            models.get("worker"),
            fallback_critic=flat_critic,
            fallback_referee=flat_referee,
        ),
        integrator=_load_integrator_model_config(models.get("integrator")),
    )
