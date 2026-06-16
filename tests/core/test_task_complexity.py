"""Tests for task complexity classification abstractions."""

import json
from typing import cast

import pytest

from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentType,
    PlanSpec,
    RequestSource,
    WorkSpec,
)
from forge.core.task_complexity import (
    DefaultTaskComplexityClassifier,
    LLMTaskComplexityClassifier,
    TaskComplexity,
    TaskComplexityInput,
    parse_task_complexity_response,
    task_complexity_input_from_request,
)
from forge.llm.providers import ChatMessage, LLMProvider


def _work_request(
    *,
    objective: str = "do something",
    success_condition: str = "it is done",
    adapter: str = "coding",
    artifact: str = "codebase",
    language: str | None = None,
    acceptance_criteria: list[AcceptanceCriterion] | None = None,
    constraints: list[str] | None = None,
    non_goals: list[str] | None = None,
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
                constraints=constraints or [],
                non_goals=non_goals or [],
            ),
            adapter=adapter,
            artifact=artifact,
            language=language,
        ),
    )


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="make a plan"),
    )


class _FakeAsyncProvider:
    """Async fake provider for classifier tests."""

    max_tokens = 128

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] | None = None

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


def test_task_complexity_enum_values_are_correct() -> None:
    """TaskComplexity exposes the expected stable string values."""
    assert TaskComplexity.EASY == "easy"
    assert TaskComplexity.MEDIUM == "medium"
    assert TaskComplexity.HARD == "hard"


async def test_default_task_complexity_classifier_returns_medium() -> None:
    """DefaultTaskComplexityClassifier is awaitable and returns MEDIUM."""
    request = _work_request()

    result = await DefaultTaskComplexityClassifier().classify(request)

    assert result is TaskComplexity.MEDIUM


async def test_default_task_complexity_classifier_ignores_explicit_request_contents() -> None:
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

    results = [await classifier.classify(request) for request in requests]

    assert results == [TaskComplexity.MEDIUM, TaskComplexity.MEDIUM]


async def test_default_task_complexity_classifier_does_not_mutate_request() -> None:
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

    await DefaultTaskComplexityClassifier().classify(request)

    assert request.model_dump() == before


def test_task_complexity_input_can_be_built_from_work_request() -> None:
    """A WORK request is converted to the compact classifier DTO."""
    request = _work_request(
        objective="fix parser",
        success_condition="parser tests pass",
        adapter="coding",
        artifact="api",
        language="python",
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="unit tests pass")],
    )

    result = task_complexity_input_from_request(request)

    assert isinstance(result, TaskComplexityInput)
    assert result.objective == "fix parser"
    assert result.success_condition == "parser tests pass"
    assert result.adapter == "coding"
    assert result.artifact == "api"
    assert result.language == "python"
    assert result.acceptance_criteria == [AcceptanceCriterion(id="AC1", text="unit tests pass")]


def test_task_complexity_input_uses_contract_review_metadata() -> None:
    """Extraction uses contract acceptance criteria, constraints, and non-goals."""
    request = _work_request(
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="document behavior")],
        constraints=["keep public API stable"],
        non_goals=["rewrite the module"],
    )

    result = task_complexity_input_from_request(request)

    assert result.acceptance_criteria == [AcceptanceCriterion(id="AC1", text="document behavior")]
    assert result.constraints == ["keep public API stable"]
    assert result.non_goals == ["rewrite the module"]


def test_task_complexity_input_rejects_plan_requests() -> None:
    """Input extraction is only valid for worker tasks."""
    with pytest.raises(ValueError, match="WORK request"):
        task_complexity_input_from_request(_plan_request())


def test_parse_task_complexity_response_accepts_valid_response() -> None:
    """Strict parser accepts the expected classifier object."""
    result = parse_task_complexity_response(
        '{"complexity":"hard","rationale":"requires broad coordination"}'
    )

    assert result.complexity is TaskComplexity.HARD
    assert result.rationale == "requires broad coordination"


def test_parse_task_complexity_response_rejects_invalid_json() -> None:
    """Invalid JSON fails with a clear ValueError."""
    with pytest.raises(ValueError, match="invalid task complexity JSON"):
        parse_task_complexity_response("not json")


def test_parse_task_complexity_response_rejects_invalid_label() -> None:
    """Unknown complexity labels are rejected."""
    with pytest.raises(ValueError, match="invalid task complexity response schema"):
        parse_task_complexity_response('{"complexity":"tiny","rationale":"bad label"}')


def test_parse_task_complexity_response_rejects_missing_rationale() -> None:
    """The rationale field is required."""
    with pytest.raises(ValueError, match="invalid task complexity response schema"):
        parse_task_complexity_response('{"complexity":"easy"}')


def test_parse_task_complexity_response_rejects_extra_fields() -> None:
    """No fields beyond complexity and rationale are accepted."""
    with pytest.raises(ValueError, match="invalid task complexity response schema"):
        parse_task_complexity_response('{"complexity":"medium","rationale":"ok","profile":"fast"}')


async def test_llm_classifier_sends_compact_metadata_json_only() -> None:
    """Classifier sends only the compact DTO JSON as the user message."""
    provider = _FakeAsyncProvider('{"complexity":"easy","rationale":"small edit"}')
    request = _work_request(
        objective="fix typo",
        success_condition="copy is corrected",
        adapter="document",
        artifact="docs",
        language="markdown",
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="typo is gone")],
        constraints=["do not alter meaning"],
        non_goals=["rewrite the page"],
    )

    await LLMTaskComplexityClassifier(cast(LLMProvider, provider)).classify(request)

    assert provider.messages is not None
    payload = json.loads(provider.messages[1]["content"])
    assert set(payload) == {
        "objective",
        "success_condition",
        "acceptance_criteria",
        "constraints",
        "non_goals",
        "adapter",
        "artifact",
        "language",
    }
    assert "id" not in payload
    assert "dependencies" not in payload
    assert "model_profile" not in payload


async def test_llm_classifier_prompt_excludes_fake_file_and_stateview_content() -> None:
    """Prompt construction does not include file or StateView-shaped content."""
    provider = _FakeAsyncProvider('{"complexity":"medium","rationale":"bounded"}')

    await LLMTaskComplexityClassifier(cast(LLMProvider, provider)).classify(_work_request())

    assert provider.messages is not None
    rendered = "\n".join(message["content"] for message in provider.messages)
    assert "SECRET_FAKE_FILE_CONTENT" not in rendered
    assert "StateView(files=[...])" not in rendered
    assert "dispatch_sha" not in rendered
    assert "worktree" not in rendered


async def test_llm_classifier_returns_parsed_complexity() -> None:
    """The classifier returns only the parsed complexity value."""
    provider = _FakeAsyncProvider('{"complexity":"hard","rationale":"many moving parts"}')

    result = await LLMTaskComplexityClassifier(cast(LLMProvider, provider)).classify(
        _work_request()
    )

    assert result is TaskComplexity.HARD


async def test_llm_classifier_prompt_does_not_expose_profile_or_model_names() -> None:
    """Classifier prompts do not include routing profile names or model names."""
    provider = _FakeAsyncProvider('{"complexity":"easy","rationale":"small"}')

    await LLMTaskComplexityClassifier(cast(LLMProvider, provider)).classify(_work_request())

    assert provider.messages is not None
    rendered = "\n".join(message["content"] for message in provider.messages)
    assert "fast" not in rendered
    assert "strong" not in rendered
    assert "gpt-4o" not in rendered
    assert "claude" not in rendered
