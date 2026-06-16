"""Profile assignment for worker AgentRequests."""

from collections.abc import Mapping
from typing import Protocol

from forge.core.models import AgentRequest
from forge.core.task_complexity import (
    DefaultTaskComplexityClassifier,
    TaskComplexity,
    TaskComplexityClassifier,
)


class ProfileAssigner(Protocol):
    """Assign a model profile to an AgentRequest."""

    async def assign(self, request: AgentRequest) -> str:
        """Return the model profile name for request."""
        ...


class DefaultProfileAssigner:
    """Default profile assigner that preserves existing routing behavior."""

    async def assign(self, request: AgentRequest) -> str:
        """Return the default worker profile."""
        return "default"


class ComplexityProfileAssigner:
    """Assign profiles by mapping classified task complexity to profile names."""

    def __init__(
        self,
        classifier: TaskComplexityClassifier | None = None,
        complexity_to_profile: Mapping[TaskComplexity, str] | None = None,
    ) -> None:
        self.classifier = classifier or DefaultTaskComplexityClassifier()
        self.complexity_to_profile = dict(
            complexity_to_profile
            or {
                TaskComplexity.EASY: "default",
                TaskComplexity.MEDIUM: "default",
                TaskComplexity.HARD: "default",
            }
        )

    async def assign(self, request: AgentRequest) -> str:
        """Return the profile mapped from the request's classified complexity."""
        complexity = await self.classifier.classify(request)
        try:
            return self.complexity_to_profile[complexity]
        except KeyError as exc:
            raise ValueError(
                f"No model profile configured for task complexity {complexity.value!r}"
            ) from exc
