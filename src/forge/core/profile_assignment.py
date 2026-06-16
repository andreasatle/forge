"""Profile assignment for worker AgentRequests."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from forge.core.models import AgentRequest
from forge.core.task_complexity import (
    DefaultTaskComplexityClassifier,
    TaskComplexity,
    TaskComplexityClassifier,
    TaskComplexityResponse,
)


class ProfileAssignmentResult(BaseModel):
    """Metadata produced when a worker request is assigned to a model profile."""

    model_config = ConfigDict(frozen=True)

    model_profile: str
    complexity: TaskComplexity | None = None
    rationale: str | None = None


class ProfileAssigner(Protocol):
    """Assign a model profile to an AgentRequest."""

    async def assign(self, request: AgentRequest) -> str:
        """Return the model profile name for request."""
        ...


@runtime_checkable
class ProfileMetadataAssigner(Protocol):
    """Profile assigner that can return routing metadata."""

    async def assign_with_metadata(self, request: AgentRequest) -> ProfileAssignmentResult:
        """Return profile assignment metadata for request."""
        ...


@runtime_checkable
class TaskComplexityResponseClassifier(Protocol):
    """Classifier that can return the rich parsed complexity response."""

    async def classify_with_response(self, request: AgentRequest) -> TaskComplexityResponse:
        """Return the parsed complexity response."""
        ...


async def assign_profile_with_metadata(
    assigner: ProfileAssigner, request: AgentRequest
) -> ProfileAssignmentResult:
    """Return profile metadata from assigner, preserving compatibility with simple assigners."""
    if isinstance(assigner, ProfileMetadataAssigner):
        return await assigner.assign_with_metadata(request)
    return ProfileAssignmentResult(model_profile=await assigner.assign(request))


class DefaultProfileAssigner:
    """Default profile assigner that preserves existing routing behavior."""

    async def assign(self, request: AgentRequest) -> str:
        """Return the default worker profile."""
        return (await self.assign_with_metadata(request)).model_profile

    async def assign_with_metadata(self, request: AgentRequest) -> ProfileAssignmentResult:
        """Return the default worker profile metadata."""
        return ProfileAssignmentResult(model_profile="default")


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
        return (await self.assign_with_metadata(request)).model_profile

    async def assign_with_metadata(self, request: AgentRequest) -> ProfileAssignmentResult:
        """Return the mapped profile and complexity classification metadata."""
        rationale: str | None = None
        if isinstance(self.classifier, TaskComplexityResponseClassifier):
            response = await self.classifier.classify_with_response(request)
            complexity = response.complexity
            rationale = response.rationale
        else:
            complexity = await self.classifier.classify(request)
        try:
            profile = self.complexity_to_profile[complexity]
        except KeyError as exc:
            raise ValueError(
                f"No model profile configured for task complexity {complexity.value!r}"
            ) from exc
        return ProfileAssignmentResult(
            model_profile=profile,
            complexity=complexity,
            rationale=rationale,
        )
