"""Tests for task complexity classification abstractions."""

from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentType,
    RequestSource,
    WorkSpec,
)
from forge.core.task_complexity import (
    DefaultTaskComplexityClassifier,
    TaskComplexity,
)


def _work_request(
    *,
    objective: str = "do something",
    success_condition: str = "it is done",
    adapter: str = "coding",
    artifact: str = "codebase",
    language: str | None = None,
    acceptance_criteria: list[AcceptanceCriterion] | None = None,
) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.USER,
        spec=WorkSpec(
            objective=objective,
            success_condition=success_condition,
            contract=AgentContract(
                objective=objective,
                success_condition=success_condition,
                acceptance_criteria=acceptance_criteria or [],
            ),
            adapter=adapter,
            artifact=artifact,
            language=language,
        ),
    )


def test_task_complexity_enum_values_are_correct() -> None:
    """TaskComplexity exposes the expected stable string values."""
    assert TaskComplexity.EASY == "easy"
    assert TaskComplexity.MEDIUM == "medium"
    assert TaskComplexity.HARD == "hard"


def test_default_task_complexity_classifier_returns_medium() -> None:
    """DefaultTaskComplexityClassifier returns MEDIUM for a basic request."""
    request = _work_request()

    result = DefaultTaskComplexityClassifier().classify(request)

    assert result is TaskComplexity.MEDIUM


def test_default_task_complexity_classifier_ignores_explicit_request_contents() -> None:
    """Explicit task details do not affect the default classifier result."""
    requests = [
        _work_request(
            objective="fix typo",
            success_condition="copy is corrected",
            adapter="docs",
            artifact="readme",
            language="markdown",
        ),
        _work_request(
            objective="design and implement a distributed parser",
            success_condition="all integration tests pass",
            adapter="coding",
            artifact="service",
            language="python",
            acceptance_criteria=[
                AcceptanceCriterion(id="AC1", text="parser handles nested input"),
                AcceptanceCriterion(id="AC2", text="failures are reported clearly"),
            ],
        ),
    ]
    classifier = DefaultTaskComplexityClassifier()

    results = [classifier.classify(request) for request in requests]

    assert results == [TaskComplexity.MEDIUM, TaskComplexity.MEDIUM]


def test_default_task_complexity_classifier_does_not_mutate_request() -> None:
    """Classifying a request leaves the immutable AgentRequest unchanged."""
    request = _work_request(
        objective="implement parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="api",
        language="python",
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="unit tests pass")],
    )
    before = request.model_dump()

    DefaultTaskComplexityClassifier().classify(request)

    assert request.model_dump() == before
