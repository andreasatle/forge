"""Tests for worker profile assignment abstractions."""

import pytest

from forge.core.models import (
    AgentRequest,
    AgentType,
    RequestSource,
    WorkSpec,
)
from forge.core.profile_assignment import (
    ComplexityProfileAssigner,
    DefaultProfileAssigner,
    ProfileAssignmentResult,
)
from forge.core.task_complexity import TaskComplexity, TaskComplexityResponse


class FakeTaskComplexityClassifier:
    """Test double that returns a fixed task complexity."""

    def __init__(self, complexity: TaskComplexity) -> None:
        self.complexity = complexity

    async def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the configured complexity."""
        return self.complexity


class FakeTaskComplexityResponseClassifier:
    """Test double that returns a rich task complexity response."""

    def __init__(self, response: TaskComplexityResponse) -> None:
        self.response = response
        self.calls = 0

    async def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the configured complexity."""
        self.calls += 1
        return self.response.complexity

    async def classify_with_response(self, request: AgentRequest) -> TaskComplexityResponse:
        """Return the configured rich response."""
        self.calls += 1
        return self.response


def _work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.USER,
        spec=WorkSpec(
            objective="do something",
            success_condition="it is done",
            adapter="coding",
            artifact="codebase",
        ),
    )


async def test_default_profile_assigner_returns_default() -> None:
    """DefaultProfileAssigner is awaitable and preserves default behavior."""
    assert await DefaultProfileAssigner().assign(_work_request()) == "default"


async def test_default_profile_assigner_returns_default_metadata() -> None:
    """DefaultProfileAssigner exposes metadata without complexity classification."""
    result = await DefaultProfileAssigner().assign_with_metadata(_work_request())

    assert result == ProfileAssignmentResult(model_profile="default")
    assert result.complexity is None
    assert result.rationale is None


async def test_complexity_profile_assigner_default_mapping_returns_default() -> None:
    """Default complexity classification and mapping route to default."""
    assert await ComplexityProfileAssigner().assign(_work_request()) == "default"


@pytest.mark.parametrize(
    ("complexity", "profile"),
    [
        (TaskComplexity.EASY, "fast"),
        (TaskComplexity.MEDIUM, "default"),
        (TaskComplexity.HARD, "strong"),
    ],
)
async def test_complexity_profile_assigner_maps_classifier_result(
    complexity: TaskComplexity, profile: str
) -> None:
    """Injected classifier results map through supplied profile mappings."""
    assigner = ComplexityProfileAssigner(
        classifier=FakeTaskComplexityClassifier(complexity),
        complexity_to_profile={
            TaskComplexity.EASY: "fast",
            TaskComplexity.MEDIUM: "default",
            TaskComplexity.HARD: "strong",
        },
    )

    assert await assigner.assign(_work_request()) == profile


async def test_complexity_profile_assigner_returns_complexity_metadata() -> None:
    """ComplexityProfileAssigner exposes selected profile and complexity metadata."""
    assigner = ComplexityProfileAssigner(
        classifier=FakeTaskComplexityClassifier(TaskComplexity.HARD),
        complexity_to_profile={
            TaskComplexity.EASY: "fast",
            TaskComplexity.MEDIUM: "default",
            TaskComplexity.HARD: "strong",
        },
    )

    result = await assigner.assign_with_metadata(_work_request())

    assert result.model_profile == "strong"
    assert result.complexity is TaskComplexity.HARD
    assert result.rationale is None


async def test_complexity_profile_assigner_preserves_classifier_rationale() -> None:
    """Rich classifiers preserve rationale without a duplicate classification call."""
    classifier = FakeTaskComplexityResponseClassifier(
        TaskComplexityResponse(complexity=TaskComplexity.EASY, rationale="small isolated edit")
    )
    assigner = ComplexityProfileAssigner(
        classifier=classifier,
        complexity_to_profile={
            TaskComplexity.EASY: "fast",
            TaskComplexity.MEDIUM: "default",
            TaskComplexity.HARD: "strong",
        },
    )

    result = await assigner.assign_with_metadata(_work_request())

    assert result.model_profile == "fast"
    assert result.complexity is TaskComplexity.EASY
    assert result.rationale == "small isolated edit"
    assert classifier.calls == 1


async def test_complexity_profile_assigner_missing_mapping_raises_value_error() -> None:
    """Missing profile mappings fail clearly for the classified complexity."""
    assigner = ComplexityProfileAssigner(
        classifier=FakeTaskComplexityClassifier(TaskComplexity.HARD),
        complexity_to_profile={TaskComplexity.EASY: "fast"},
    )

    with pytest.raises(ValueError, match="hard"):
        await assigner.assign(_work_request())


async def test_complexity_profile_assigner_does_not_mutate_request() -> None:
    """Assigning a profile does not mutate the request object."""
    request = _work_request()
    before = request.model_dump()

    await ComplexityProfileAssigner().assign(request)

    assert request.model_dump() == before
