"""ForgeConfig dataclass for loading and validating the forge YAML config file."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from forge.core.task_complexity import TaskComplexity

_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class ArtifactConfig:
    """Named artifact with a type and optional language declared in forge.yaml."""

    name: str
    type: str  # "coding" | "document" | "audit"
    language: str | None = None
    description: str | None = None


@dataclass
class PwcModelConfig:
    """Model configuration for a producer/critic/referee loop."""

    producer: str = "ollama/gemma4:e4b"
    critic: str | None = "ollama/gemma4:e4b"
    referee: str | None = "ollama/gemma4:e4b"
    max_attempts: int = 3


def _empty_complexity_to_profile() -> dict[TaskComplexity, str]:
    return {}


@dataclass
class ComplexityClassifierConfig:
    """Optional model configuration for worker task complexity classification."""

    model: str
    max_tokens: int = 512
    complexity_to_profile: dict[TaskComplexity, str] = field(
        default_factory=_empty_complexity_to_profile
    )


def _empty_worker_profiles() -> dict[str, PwcModelConfig]:
    return {}


@dataclass
class ModelsConfig:
    """Model configuration per scheduler agent type."""

    planner: PwcModelConfig = field(default_factory=PwcModelConfig)
    worker: PwcModelConfig = field(default_factory=PwcModelConfig)
    worker_profiles: dict[str, PwcModelConfig] = field(default_factory=_empty_worker_profiles)
    complexity_classifier: ComplexityClassifierConfig | None = None


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
        return ForgeConfigLoader().load_path(path)


class ForgeConfigLoader:
    """Owns the boundary from untyped YAML to typed ForgeConfig."""

    def load_path(self, path: Path) -> ForgeConfig:
        """Read YAML from disk and return a validated ForgeConfig."""
        raw: object = yaml.safe_load(path.read_text())
        return self.load_data(_as_mapping(raw, "ForgeConfig"))

    def load_data(self, data: Mapping[str, object]) -> ForgeConfig:
        """Parse a raw mapping to a validated ForgeConfig."""
        if "northstar" not in data:
            raise ValueError("ForgeConfig: missing required field 'northstar'")
        if "workspace" not in data:
            raise ValueError("ForgeConfig: missing required field 'workspace'")
        artifacts = self.load_artifacts(data.get("artifacts"))
        for artifact in artifacts:
            if artifact.type == "coding" and not artifact.language:
                raise ValueError(
                    f"artifact '{artifact.name}' has type 'coding' but no language declared"
                )
        models = self.load_models_config(data.get("models", {}))
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

    def load_artifacts(self, raw: object) -> list[ArtifactConfig]:
        """Parse raw artifact data to list[ArtifactConfig]."""
        artifacts_data = _required_sequence(raw, "artifacts")
        if not artifacts_data:
            raise ValueError("artifacts is required — declare at least one artifact in forge.yaml")
        artifacts: list[ArtifactConfig] = []
        for index, item in enumerate(artifacts_data):
            artifact = _as_mapping(item, f"artifacts[{index}]")
            name = _required_string(artifact.get("name"), f"artifacts[{index}].name")
            _validate_artifact_name(name, f"artifacts[{index}].name")
            artifacts.append(
                ArtifactConfig(
                    name=name,
                    type=_required_string(artifact.get("type"), f"artifacts[{index}].type"),
                    language=_optional_string(
                        artifact.get("language"), f"artifacts[{index}].language"
                    ),
                    description=_optional_string(
                        artifact.get("description"), f"artifacts[{index}].description"
                    ),
                )
            )
        return artifacts

    def load_models_config(self, raw: object) -> ModelsConfig:
        """Parse raw models mapping to ModelsConfig."""
        models = _as_mapping(raw, "models")
        flat_critic = _optional_model(models.get("critic"), "models.critic")
        flat_referee = _optional_model(models.get("referee"), "models.referee")
        worker_profiles = self._load_worker_profiles(
            models.get("worker_profiles"),
            fallback_critic=flat_critic,
            fallback_referee=flat_referee,
        )
        return ModelsConfig(
            planner=self._load_pwc_model_config(
                models.get("planner"),
                field="models.planner",
                fallback_critic=flat_critic,
                fallback_referee=flat_referee,
            ),
            worker=self._load_pwc_model_config(
                models.get("worker"),
                field="models.worker",
                fallback_critic=flat_critic,
                fallback_referee=flat_referee,
            ),
            worker_profiles=worker_profiles,
            complexity_classifier=self._load_complexity_classifier_config(
                models.get("complexity_classifier"),
                worker_profiles=worker_profiles,
            ),
        )

    def _load_complexity_classifier_config(
        self,
        value: object,
        *,
        worker_profiles: Mapping[str, PwcModelConfig],
    ) -> ComplexityClassifierConfig | None:
        """Parse optional complexity_classifier mapping."""
        if value is None:
            return None
        raw = _as_mapping(value, "models.complexity_classifier")
        mapping_raw = _as_mapping(
            raw.get("complexity_to_profile"),
            "models.complexity_classifier.complexity_to_profile",
        )
        expected_keys = {complexity.value for complexity in TaskComplexity}
        actual_keys = set(mapping_raw)
        if actual_keys != expected_keys:
            raise ValueError(
                "models.complexity_classifier.complexity_to_profile keys must be exactly "
                "easy, medium, hard"
            )

        valid_profiles = {"default", *worker_profiles.keys()}
        complexity_to_profile: dict[TaskComplexity, str] = {}
        for key, profile_value in mapping_raw.items():
            profile = _required_string(
                profile_value,
                f"models.complexity_classifier.complexity_to_profile.{key}",
            )
            if profile not in valid_profiles:
                raise ValueError(
                    f"models.complexity_classifier.complexity_to_profile.{key} "
                    f"references unknown worker profile {profile!r}"
                )
            complexity_to_profile[TaskComplexity(key)] = profile

        return ComplexityClassifierConfig(
            model=_required_model(raw.get("model"), "models.complexity_classifier.model"),
            max_tokens=_optional_int(
                raw.get("max_tokens"), 512, "models.complexity_classifier.max_tokens"
            ),
            complexity_to_profile=complexity_to_profile,
        )

    def _load_worker_profiles(
        self,
        value: object,
        *,
        fallback_critic: str | None = None,
        fallback_referee: str | None = None,
    ) -> dict[str, PwcModelConfig]:
        """Parse optional worker_profiles mapping to dict[str, PwcModelConfig]."""
        profiles_raw = _as_mapping(value, "models.worker_profiles")
        result: dict[str, PwcModelConfig] = {}
        for name, profile_value in profiles_raw.items():
            result[name] = self._load_pwc_model_config(
                profile_value,
                field=f"models.worker_profiles.{name}",
                fallback_critic=fallback_critic,
                fallback_referee=fallback_referee,
            )
        return result

    def _load_pwc_model_config(
        self,
        value: object,
        *,
        field: str,
        default_producer: str = "ollama/gemma4:e4b",
        fallback_critic: str | None = None,
        fallback_referee: str | None = None,
    ) -> PwcModelConfig:
        if isinstance(value, str):
            producer = _required_model(value, field)
            return PwcModelConfig(
                producer=producer,
                critic=fallback_critic or producer,
                referee=fallback_referee or producer,
            )
        if value is None:
            return PwcModelConfig(
                producer=default_producer,
                critic=fallback_critic or default_producer,
                referee=fallback_referee or default_producer,
            )
        model_data = _as_mapping(value, field)
        producer = _required_model(model_data.get("producer"), f"{field}.producer")
        return PwcModelConfig(
            producer=producer,
            critic=_optional_model(model_data.get("critic"), f"{field}.critic")
            if "critic" in model_data
            else fallback_critic or producer,
            referee=_optional_model(model_data.get("referee"), f"{field}.referee")
            if "referee" in model_data
            else fallback_referee or producer,
            max_attempts=_optional_int(model_data.get("max_attempts"), 3, f"{field}.max_attempts"),
        )


def load_config(path: Path) -> ForgeConfig:
    """Load a ForgeConfig from a YAML file at path."""
    return ForgeConfigLoader().load_path(path)


def _validate_artifact_name(name: str, field: str) -> None:
    if not _ARTIFACT_NAME_RE.match(name):
        raise ValueError(
            f"{field}: artifact name {name!r} is invalid — "
            "only letters, digits, underscore, and hyphen are permitted"
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


def _required_model(value: object, field: str) -> str:
    model = _required_string(value, field)
    if not model:
        raise ValueError(f"{field} must be a non-empty string")
    return model


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _optional_model(value: object, field: str) -> str | None:
    model = _optional_string(value, field)
    if model == "":
        raise ValueError(f"{field} must be a non-empty string or null")
    return model


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
