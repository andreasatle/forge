"""Tests for AttemptLifecycle, WorkOutputValidator, and PlannerOutputValidator."""

import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.attempt import (
    AttemptLifecycle,
    PlannerOutputValidator,
    RunAgentFailed,
    WorkOutputValidator,
)
from forge.agents.attempt_telemetry import AttemptTelemetryReporter
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentDiagnostic,
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DecompositionNodeSpec,
    FailureKind,
    GraphSplitDecision,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    RevisionItem,
    RevisionRequest,
    StateView,
    TaskSpec,
    WorkDecision,
    WorkOutput,
    WorkSpec,
    render_agent_contract,
)
from forge.core.telemetry import TelemetryEvent


class _MemoryTelemetrySink:
    """In-memory TelemetrySink for attempt tests."""

    def __init__(self) -> None:
        self.run_id = uuid4()
        self.events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class _FailingTelemetrySink:
    """TelemetrySink stub that always raises on append."""

    def __init__(self) -> None:
        self.run_id = uuid4()

    def append(self, event: TelemetryEvent) -> None:
        raise OSError("telemetry unavailable")


def _work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do task",
            success_condition="task done",
            adapter="coding",
            artifact="codebase",
        ),
    )


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


def _state_view() -> StateView:
    return StateView(artifact_name="codebase", language=None, files=[])


def _adapter_spec(
    name: str = "coding",
    work_noun: str = "implementation",
    requires_nonempty: bool = True,
) -> AdapterSpec:
    return AdapterSpec(
        name=name,
        description="test",
        tools=[],
        prompt_template="",
        requires_nonempty_output=requires_nonempty,
        work_noun=work_noun,
    )


def _registry_with(name: str = "coding", work_noun: str = "implementation") -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(_adapter_spec(name, work_noun))
    return registry


def _make_run_fn(
    responses: list[AgentResponse],
) -> tuple[Callable[[str], Awaitable[AgentResponse]], list[str]]:
    """Return (run_fn, captured_prompts)."""
    prompts: list[str] = []
    idx = 0

    async def run_fn(prompt: str) -> AgentResponse:
        nonlocal idx
        prompts.append(prompt)
        resp = responses[idx] if idx < len(responses) else responses[-1]
        idx += 1
        return resp

    return run_fn, prompts


def _engine(
    request: AgentRequest | None = None,
    run_fn: Callable[[str], Awaitable[AgentResponse]] | None = None,
    critic_provider: MagicMock | None = None,
    referee_provider: MagicMock | None = None,
    max_attempts: int = 3,
    work_noun: str = "implementation",
    requires_nonempty: bool = True,
    worktree_path: Path | None = None,
) -> AttemptLifecycle[WorkOutput]:
    req = request or _work_request()
    sv = _state_view()
    if run_fn is None:

        async def _default(prompt: str) -> AgentResponse:
            return AgentResponse(
                request_id=req.id, status=ResponseStatus.COMPLETED, output=WorkOutput()
            )

        run_fn = _default
    return AttemptLifecycle[WorkOutput](
        request=req,
        state_view=sv,
        validator=WorkOutputValidator(
            _adapter_spec(work_noun=work_noun, requires_nonempty=requires_nonempty),
            sv,
            worktree_path=worktree_path,
        ),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        max_attempts=max_attempts,
    )


async def test_accept_on_first_attempt_returns_immediately() -> None:
    """Engine returns the work output immediately when referee accepts on the first attempt."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == work_output
    assert len(_) == 1


async def test_revise_injects_feedback_and_retries() -> None:
    """Engine retries with a structured required-revision block when disposition is REVISE."""
    request = _work_request()
    first_work_output = WorkOutput(summary="Completed worktree changes.")
    improved_work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=first_work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=improved_work_output,
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE, rationale="needs work", hints=["add tests"]
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="agreed", override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == improved_work_output
    assert len(prompts) == 2
    assert "REQUIRED REVISION" in prompts[1]
    assert "Previous disposition: revise" in prompts[1]
    assert "Rationale: agreed" in prompts[1]
    assert "1. Required change: add tests" in prompts[1]
    assert "The next output must address every required change listed below." in prompts[1]
    assert "FINAL OUTPUT FORMAT REMINDER" in prompts[1]
    assert "Return valid JSON only matching WorkOutput." in prompts[1]
    assert "summary must describe the worktree changes made" in prompts[1]
    assert "Do not include full file contents" in prompts[1]


async def test_revise_prompt_preserves_structured_criterion_ids() -> None:
    """Structured revision items supplied by the referee are rendered with criterion ids."""
    request = _work_request()
    first_work_output = WorkOutput(summary="Completed worktree changes.")
    improved_work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=first_work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=improved_work_output,
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="missing tests",
                revision_items=[
                    RevisionItem(
                        criterion_id="AC2",
                        required_change="Add unit tests for parser failures.",
                        rationale="The contract requires parser failure coverage.",
                    )
                ],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale="tests are required",
                override=False,
                revision_items=[
                    RevisionItem(
                        criterion_id="AC2",
                        required_change="Add parser failure tests.",
                        rationale="Acceptance criterion AC2 is unmet.",
                    )
                ],
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert "1. Required change [AC2]: Add parser failure tests." in prompts[1]
    assert "Rationale: Acceptance criterion AC2 is unmet." in prompts[1]


async def test_multiple_revise_rounds_accumulate_required_changes() -> None:
    """Successive revise rounds accumulate all prior RevisionRequests in the prompt."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="missing tests",
                hints=["add tests"],
            ),
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="error handling missing from the file",
                hints=["handle failures"],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="tests missing", override=False
            ),
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale="error handling missing from the file",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 3
    assert "Revision request 1 (after 1 prior attempt(s))" in prompts[2]
    assert "1. Required change: add tests" in prompts[2]
    assert "Revision request 2 (after 2 prior attempt(s))" in prompts[2]
    assert "1. Required change: handle failures" in prompts[2]


async def test_generic_argument_feedback_does_not_become_coding_revision() -> None:
    """Generic essay feedback is rejected instead of becoming coding revision instructions."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="The argument needs more supporting evidence.",
            hints=["Add more supporting evidence."],
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="The argument needs more supporting evidence.",
            override=False,
            revision_items=[
                RevisionItem(
                    required_change="Add more supporting evidence.",
                    rationale="The argument is weak.",
                )
            ],
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "ungrounded generic feedback" in (result.error or "")
    assert len(prompts) == 1


async def test_generic_verbose_feedback_requires_contract_grounding() -> None:
    """Generic verbosity feedback is allowed only when the contract asks for it."""
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write concise docs",
            success_condition="docs are concise",
            contract=AgentContract(
                objective="write concise docs",
                success_condition="docs are concise",
                constraints=["Generated text must not be verbose."],
            ),
            adapter="coding",
            artifact="codebase",
        ),
    )
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="The generated text is too verbose.",
                hints=["Shorten the generated text."],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="concise now"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale="The generated text is too verbose.",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2
    assert "REQUIRED REVISION" in prompts[1]
    assert "too verbose" in prompts[1]


async def test_generic_verbose_feedback_without_contract_grounding_fails() -> None:
    """Generic verbosity feedback without contract support is rejected as ungrounded."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="The generated text is too verbose.",
            hints=["Make it less verbose."],
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="The generated text is too verbose.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "ungrounded generic feedback" in (result.error or "")
    assert len(prompts) == 1


async def test_missing_file_feedback_remains_valid_revision() -> None:
    """Grounded missing-file feedback still becomes a normal revision request."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="The required src/app.py file is missing from the worktree.",
                revision_items=[
                    RevisionItem(
                        required_change="Create src/app.py with the requested implementation.",
                        rationale="The worktree evidence lacks the required file.",
                    )
                ],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="file exists"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale="The required src/app.py file is missing from the worktree.",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2
    assert "1. Required change: Create src/app.py" in prompts[1]


async def test_methodology_feedback_is_rejected_as_ungrounded() -> None:
    """Generic methodology feedback is rejected as ungrounded without phrase matching."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="The text lacks sufficient detail regarding the methodology used.",
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="The text lacks sufficient detail regarding the methodology used.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "ungrounded generic feedback" in (result.error or "")
    assert len(prompts) == 1


async def test_complex_language_feedback_is_rejected_as_ungrounded() -> None:
    """Generic 'complex language / core message' feedback is rejected as ungrounded."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="The use of overly complex language obscures the core message.",
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="The use of overly complex language obscures the core message.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "ungrounded generic feedback" in (result.error or "")
    assert len(prompts) == 1


async def test_criterion_id_reference_is_not_rejected_as_ungrounded() -> None:
    """Feedback with a non-empty criterion_id is not rejected as ungrounded."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="Criterion not satisfied.",
                revision_items=[
                    RevisionItem(
                        criterion_id="AC1",
                        required_change="Satisfy the criterion.",
                        rationale="Criterion AC1 is unmet.",
                    )
                ],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="done"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="agreed", override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2


async def test_document_feedback_grounded_in_contract_section_is_not_rejected() -> None:
    """Feedback referencing contract vocabulary is not rejected as ungrounded."""
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write authentication documentation",
            success_condition="documentation covers authentication flow",
            contract=AgentContract(
                objective="write authentication documentation",
                success_condition="documentation covers authentication flow",
            ),
            adapter="coding",
            artifact="codebase",
        ),
    )
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="The authentication section lacks detail on the token refresh flow.",
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="done"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="agreed", override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2


async def test_style_feedback_allowed_when_contract_requests_style() -> None:
    """Style/verbosity feedback is allowed when the contract explicitly asks for it."""
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write concise release notes",
            success_condition="release notes are concise and clear",
            contract=AgentContract(
                objective="write concise release notes",
                success_condition="release notes are concise and clear",
                constraints=["Output must be concise."],
            ),
            adapter="coding",
            artifact="codebase",
        ),
    )
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="The output is not concise enough.",
                hints=["Shorten to key points only."],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="concise now"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale="Needs to be more concise.",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2


async def test_style_feedback_rejected_when_contract_does_not_request_style() -> None:
    """Style/verbosity feedback without a matching contract term is rejected as ungrounded."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="The style of the output is inappropriate.",
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="The style of the output is inappropriate.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "ungrounded generic feedback" in (result.error or "")
    assert len(prompts) == 1


def test_contract_vocabulary_excludes_stopwords_and_short_words() -> None:
    """_contract_vocabulary filters stopwords and words shorter than 4 characters."""
    from forge.agents.attempt import _contract_vocabulary  # pyright: ignore[reportPrivateUsage]

    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="parse and validate configuration schema",
            success_condition="schema validation is complete",
            adapter="coding",
            artifact="codebase",
        ),
    )
    vocab = _contract_vocabulary(request)

    assert "and" not in vocab  # too short (3 chars)
    assert "is" not in vocab  # too short (2 chars)
    assert "with" not in vocab  # stopword
    assert "parse" in vocab  # content word (5 chars)
    assert "validate" in vocab  # content word
    assert "configuration" in vocab  # content word
    assert "schema" in vocab  # content word
    assert "complete" in vocab  # content word
    assert "coding" in vocab  # adapter
    assert "codebase" in vocab  # artifact


def test_contract_vocabulary_includes_artifact_adapter_language_and_criterion_ids() -> None:
    """_contract_vocabulary includes artifact, adapter, language, and acceptance criterion ids."""
    from forge.agents.attempt import _contract_vocabulary  # pyright: ignore[reportPrivateUsage]

    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="build scraper",
            success_condition="scraper runs successfully",
            contract=AgentContract(
                objective="build scraper",
                success_condition="scraper runs successfully",
                acceptance_criteria=[
                    AcceptanceCriterion(id="PERF", text="scraper must handle timeouts"),
                    AcceptanceCriterion(id="AUTH", text="authentication headers required"),
                ],
            ),
            adapter="coding",
            artifact="codebase",
            language="python",
        ),
    )
    vocab = _contract_vocabulary(request)

    assert "coding" in vocab  # adapter
    assert "codebase" in vocab  # artifact
    assert "python" in vocab  # language
    assert "scraper" in vocab  # from objective/criteria
    assert "timeouts" in vocab  # from criterion text
    assert "authentication" in vocab  # from criterion text
    assert "perf" in vocab  # criterion id (4 chars)
    assert "auth" in vocab  # criterion id (4 chars)


def test_feedback_has_grounding_via_contract_token_overlap() -> None:
    """_feedback_has_grounding returns True when feedback shares tokens with contract vocabulary."""
    from forge.agents.attempt import _feedback_has_grounding  # pyright: ignore[reportPrivateUsage]

    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="implement authentication middleware",
            success_condition="middleware handles tokens correctly",
            adapter="coding",
            artifact="codebase",
        ),
    )

    assert _feedback_has_grounding(
        "the authentication logic is missing from the middleware", request
    )
    assert not _feedback_has_grounding("the output lacks sufficient clarity and precision", request)


async def test_revise_prompt_omits_repeated_contract_and_plugin_guidance() -> None:
    """Reviewer-quoted invariant contract text is not duplicated in REQUIRED REVISION."""
    plugin_guidance = "Language plugin guidance:\n" + "\n".join(
        f"PLUGIN_RULE_{index}: follow this binding language rule" for index in range(80)
    )
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do task",
            success_condition="task done",
            adapter="coding",
            artifact="codebase",
            language="toy",
            contract=AgentContract(
                objective="do task",
                success_condition="task done",
                constraints=[plugin_guidance],
            ),
        ),
    )
    contract_block = render_agent_contract(request)
    base_prompt = f"base prompt\n\n{contract_block}\n\nYour output must satisfy every requirement in this contract."
    quoted_contract = (
        f"The output missed a rule.\n\n{contract_block}\n\nAfter applying the contract, add tests."
    )
    first_work_output = WorkOutput(summary="Completed worktree changes.")
    improved_work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=first_work_output,
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=improved_work_output,
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale=quoted_contract,
                revision_items=[
                    RevisionItem(
                        required_change=f"Respect this rule:\n\n{plugin_guidance}",
                        rationale=f"{contract_block}\n\nThe language rule is binding.",
                    )
                ],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE,
                rationale=quoted_contract,
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run(base_prompt)

    assert result.status == ResponseStatus.COMPLETED
    retry_prompt = prompts[1]
    assert retry_prompt.count("AgentRequest contract:") == 1
    assert retry_prompt.count("Language plugin guidance:") == 1
    assert "same AgentRequest contract above" in retry_prompt
    assert "[omitted repeated AgentRequest contract" in retry_prompt
    assert "[omitted repeated language plugin guidance" in retry_prompt
    assert "After applying the contract, add tests." in retry_prompt


async def test_revise_prompt_growth_excludes_repeated_contract_blocks() -> None:
    """Accumulated revision history grows incrementally, not by repeated full contract text."""
    plugin_guidance = "Language plugin guidance:\n" + "\n".join(
        f"LONG_BINDING_RULE_{index}: keep this language invariant" for index in range(120)
    )
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do task",
            success_condition="task done",
            adapter="coding",
            artifact="codebase",
            language="toy",
            contract=AgentContract(
                objective="do task",
                success_condition="task done",
                constraints=[plugin_guidance],
            ),
        ),
    )
    contract_block = render_agent_contract(request)
    base_prompt = f"base prompt\n\n{contract_block}"
    quoted_contract = f"Fix the missing work.\n\n{contract_block}"
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale=quoted_contract,
                hints=[quoted_contract],
            ),
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale=quoted_contract,
                hints=[quoted_contract],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale=quoted_contract, override=False
            ),
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale=quoted_contract, override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run(base_prompt)

    assert result.status == ResponseStatus.COMPLETED
    assert prompts[2].count("AgentRequest contract:") == 1
    assert prompts[2].count("Language plugin guidance:") == 1
    assert "Revision request 1 (after 1 prior attempt(s))" in prompts[2]
    assert "Revision request 2 (after 2 prior attempt(s))" in prompts[2]
    assert len(prompts[2]) - len(prompts[1]) < len(contract_block)


async def test_rejected_validation_returns_failed_without_output() -> None:
    """Engine fails immediately when validation rejects a worker output."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REJECT,
            rationale="output violates contract",
            hints=["fix contract violations"],
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REJECT,
            rationale="still violates the contract",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert result.output is None
    assert "reject" in (result.error or "")
    assert len(_) == 1


async def test_no_worktree_change_reject_retries_when_attempts_remain(
    git_worktree: Path,
) -> None:
    """Referee REJECT for claimed work with no file changes becomes a retry."""
    request = _work_request()
    attempts = 0
    prompts: list[str] = []

    async def run_fn(prompt: str) -> AgentResponse:
        nonlocal attempts
        attempts += 1
        prompts.append(prompt)
        if attempts == 2:
            (git_worktree / "scraper.py").write_text("class WebScraper:\n    pass\n")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Implemented scraper files."),
        )

    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=2,
        worktree_path=git_worktree,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REJECT,
                rationale="Worker claimed implementation, but no files changed.",
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="files changed"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REJECT,
                rationale="Summary claimed files were implemented, but no worktree changes exist.",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert attempts == 2
    assert len(prompts) == 2


async def test_no_worktree_change_retry_prompt_requires_actual_file_changes(
    git_worktree: Path,
) -> None:
    """No-change retry prompt tells the producer to modify the assigned worktree."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkOutput(summary="Implemented scraper files."),
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkOutput(summary="Implemented scraper files."),
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=2,
        worktree_path=git_worktree,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REJECT,
                rationale="The git diff is empty; no files were changed.",
            ),
            CriticFinding(
                disposition=CriticDisposition.REJECT,
                rationale="The git diff is empty; no files were changed.",
            ),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REJECT,
                rationale="Empty git diff: no actual file changes support the summary.",
                override=False,
            ),
            RefereeDecision(
                disposition=CriticDisposition.REJECT,
                rationale="Empty git diff: no actual file changes support the summary.",
                override=False,
            ),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert len(prompts) == 2
    assert "REQUIRED REVISION" in prompts[1]
    assert (
        "Actually modify the assigned worktree to satisfy the task; the previous attempt "
        "returned WorkOutput metadata but produced no file changes."
    ) in prompts[1]
    assert "Rationale: Empty git diff: no actual file changes support the summary." in prompts[1]


async def test_no_worktree_change_retry_can_accept_after_file_changes(
    git_worktree: Path,
) -> None:
    """A no-change retry can complete once the producer actually changes the worktree."""
    request = _work_request()
    attempts = 0

    async def run_fn(prompt: str) -> AgentResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            (git_worktree / "scraper.py").write_text("class WebScraper:\n    pass\n")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Implemented scraper files."),
        )

    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        worktree_path=git_worktree,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REJECT,
                rationale="No files changed despite the summary.",
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="new file present"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REJECT,
                rationale="No worktree changes are present.",
                override=False,
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert attempts == 2


async def test_no_worktree_change_reject_on_final_attempt_returns_failed(
    git_worktree: Path,
) -> None:
    """No-change REJECT remains terminal when no attempts remain."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkOutput(summary="Implemented scraper files."),
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=1,
        worktree_path=git_worktree,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REJECT,
            rationale="No files changed.",
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REJECT,
            rationale="No worktree changes support the claimed implementation.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert len(prompts) == 1


async def test_unrelated_referee_reject_remains_terminal(
    git_worktree: Path,
) -> None:
    """Referee REJECT unrelated to no-change evidence remains terminal."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkOutput(summary="Implemented scraper files."),
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        worktree_path=git_worktree,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REJECT,
            rationale="The output violates an explicit non-goal.",
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REJECT,
            rationale="Unsafe out-of-scope work violates the contract.",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert len(prompts) == 1


async def test_no_worktree_change_rule_does_not_override_already_done(
    git_worktree: Path,
) -> None:
    """ALREADY_DONE handling remains unchanged when the success condition is already met."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        worktree_path=git_worktree,
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ALREADY_DONE,
            rationale="success condition already met",
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert len(prompts) == 1


async def test_repeated_revise_until_max_attempts_returns_failed_without_output() -> None:
    """Engine fails when all validation attempts are exhausted without acceptance."""
    request = _work_request()
    last_work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=last_work_output
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=2,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE, rationale="keep improving", hints=[]
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE, rationale="not done", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert result.output is None
    assert "maximum validation attempts exhausted" in (result.error or "")
    assert [diagnostic.kind for diagnostic in result.diagnostics] == ["validation_exhausted"]
    assert len(prompts) == 2


async def test_failed_pwc_writes_attempt_and_exhausted_telemetry() -> None:
    """Failed PWC preserves attempt starts, parsed results, revisions, and exhaustion."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    sink = _MemoryTelemetrySink()
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=WorkOutputValidator(_adapter_spec(), _state_view()),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=3,
        telemetry_sink=sink,
        run_id=sink.run_id,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE,
            rationale="still missing work",
            revision_items=[
                RevisionItem(
                    criterion_id="AC1",
                    required_change="Add the missing behavior.",
                    rationale="The contract requires it.",
                )
            ],
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE,
            rationale="not accepted",
            override=False,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("pwc.attempt.started") == 3
    assert event_types.count("producer.response.parsed") == 3
    assert event_types.count("critic.finding.parsed") == 3
    assert event_types.count("referee.decision.parsed") == 3
    assert event_types.count("pwc.revision.appended") == 3
    assert event_types.count("pwc.exhausted") == 1
    assert all(event.run_id == sink.run_id for event in sink.events)
    assert all(event.node_id == request.id for event in sink.events)
    assert all(event.request_id == request.id for event in sink.events)
    assert {
        event.attempt_number for event in sink.events if event.event_type == "critic.finding.parsed"
    } == {1, 2, 3}
    revision_events = [
        event for event in sink.events if event.event_type == "pwc.revision.appended"
    ]
    assert revision_events[0].data["revision_request"]["items"][0]["criterion_id"] == "AC1"
    assert (
        revision_events[0].data["revision_request"]["items"][0]["required_change"]
        == "Add the missing behavior."
    )
    exhausted = [event for event in sink.events if event.event_type == "pwc.exhausted"][0]
    assert exhausted.data["attempt_count"] == 3


def test_producer_telemetry_includes_diagnostics_for_failed_response() -> None:
    """producer_response_parsed emits diagnostics in telemetry data when the response has them."""
    sink = _MemoryTelemetrySink()
    reporter = AttemptTelemetryReporter(
        sink=sink,
        run_id=sink.run_id,
        node_id=uuid4(),
        agent_type=AgentType.WORK,
    )
    response = AgentResponse(
        request_id=uuid4(),
        status=ResponseStatus.FAILED,
        failure_kind=FailureKind.INVALID_JSON,
        error="agent failed after 3 retries: response is not valid JSON",
        diagnostics=[
            AgentDiagnostic(
                kind="invalid_structured_output",
                message="response is not valid JSON",
                raw_response_excerpt="this is definitely not json",
            )
        ],
    )

    reporter.producer_response_parsed(1, response)

    assert sink.events
    event = sink.events[0]
    assert event.event_type == "producer.response.parsed"
    diagnostics = event.data.get("diagnostics")
    assert diagnostics
    assert diagnostics[0]["raw_response_excerpt"] == "this is definitely not json"


async def test_telemetry_append_failure_does_not_change_pwc_outcome() -> None:
    """Telemetry is best-effort and cannot fail an otherwise accepted PWC run."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _FailingTelemetrySink()
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=WorkOutputValidator(_adapter_spec(), _state_view()),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        telemetry_sink=sink,
        run_id=sink.run_id,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == work_output


async def test_critic_parse_failure_returns_failed_without_output() -> None:
    """Engine fails when critic validation cannot be parsed."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = ValueError("critic_agent failed after 3 retries: invalid json")
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.INVALID_JSON
    assert result.output is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_referee.assert_not_called()


async def test_referee_parse_failure_returns_failed_without_output() -> None:
    """Engine fails when referee validation cannot be parsed."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.side_effect = ValueError("referee_agent failed after 3 retries: invalid json")
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.INVALID_JSON
    assert result.output is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_referee.assert_called_once()


async def test_no_providers_skips_validation() -> None:
    """Engine skips critic/referee entirely when both providers are None."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(request=request, run_fn=run_fn, critic_provider=None, referee_provider=None)

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == work_output
    assert len(_) == 1
    mock_critic.assert_not_called()


async def test_run_agent_failure_raises_run_agent_failed() -> None:
    """Engine raises RunAgentFailed when run_fn returns a non-COMPLETED response."""
    request = _work_request()
    failed_response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error="provider error",
        failure_kind=FailureKind.PROVIDER_ERROR,
    )
    run_fn, _ = _make_run_fn([failed_response])
    engine = _engine(request=request, run_fn=run_fn)

    try:
        await engine.run("base prompt")
        assert False, "expected RunAgentFailed"
    except RunAgentFailed as e:
        assert e.response is failed_response


async def test_empty_work_output_no_critic_first_attempt_retries_not_already_done() -> None:
    """Engine retries with correction feedback on a non-final empty output when no critic is configured."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=None,
        referee_provider=None,
        max_attempts=2,
    )

    result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == work_output
    assert len(prompts) == 2
    assert "produced no implementation" in prompts[1]


async def test_initial_revision_is_rendered_in_first_producer_prompt() -> None:
    """AttemptLifecycle includes initial RevisionRequest in the first producer prompt."""
    request = _work_request()
    revision = RevisionRequest(
        rationale="Integration tests failed after merge.\nSummary: FAILED test_scraper",
        prior_attempts=1,
        items=[
            RevisionItem(
                required_change="Fix the implementation so tests pass.",
                rationale="pytest failed",
            )
        ],
    )
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=work_output,
            )
        ]
    )
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=WorkOutputValidator(_adapter_spec(), _state_view()),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=None,
        referee_provider=None,
        initial_revision=revision,
    )

    result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 1
    assert "REQUIRED REVISION" in prompts[0]
    assert "Summary: FAILED test_scraper" in prompts[0]
    assert "1. Required change: Fix the implementation so tests pass." in prompts[0]


async def test_empty_work_output_no_critic_last_attempt_requires_nonempty_false_returns_already_done() -> (
    None
):
    """Engine returns ALREADY_DONE on the last attempt with empty output when requires_nonempty is False."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=None,
        referee_provider=None,
        max_attempts=1,
        requires_nonempty=False,
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.output == WorkOutput()
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_work_output_no_critic_last_attempt_requires_nonempty_true_returns_failed() -> (
    None
):
    """Engine returns FAILED on the last attempt with empty output when requires_nonempty is True."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=None,
        referee_provider=None,
        max_attempts=1,
        requires_nonempty=True,
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_work_output_ran_tests_and_passed_returns_already_done() -> None:
    """Engine returns ALREADY_DONE for empty output when ran_tests_and_passed=True, ignoring requires_nonempty."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
                ran_tests_and_passed=True,
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=None,
        referee_provider=None,
        max_attempts=1,
        requires_nonempty=True,
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.output == WorkOutput()
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_work_output_ran_tests_not_passed_uses_existing_behavior() -> None:
    """Engine falls through to normal retry/FAILED behavior when ran_tests_and_passed=False."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
                ran_tests_and_passed=False,
            )
        ]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=None,
        referee_provider=None,
        max_attempts=1,
        requires_nonempty=True,
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_work_output_critic_already_done_accepts() -> None:
    """Engine accepts empty output when critic confirms the success condition is already met."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            )
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ALREADY_DONE, rationale="success condition already met"
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.output == WorkOutput()
    assert len(_) == 1
    mock_critic.assert_called_once()


async def test_empty_work_output_critic_parse_failure_returns_failed_without_output() -> None:
    """Engine does not treat an unparsable empty-output critic result as ALREADY_DONE."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            )
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        mock_critic.side_effect = ValueError("critic_agent failed after 3 retries: invalid json")
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.INVALID_JSON
    assert result.output is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_critic.assert_called_once()


async def test_empty_work_output_critic_revise_triggers_retry_with_feedback() -> None:
    """Engine retries with feedback when critic returns REVISE on an empty output attempt."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="needs implementation",
                hints=["add some code"],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="done"),
        ]
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == work_output
    assert len(prompts) == 2
    assert "REQUIRED REVISION" in prompts[1]
    assert "Revise your implementation now." in prompts[1]
    assert "1. Required change: add some code" in prompts[1]


async def test_empty_work_output_critic_accept_returns_already_done() -> None:
    """Critic ACCEPT on empty output yields ALREADY_DONE instead of silently retrying."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="nothing to do"
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE


async def test_empty_work_output_critic_decompose_propagates() -> None:
    """Critic DECOMPOSE on empty output propagates DECOMPOSE instead of silently retrying."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                output=WorkOutput(),
                error="empty output",
            ),
        ]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.DECOMPOSE, rationale="task too large"
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.DECOMPOSE


async def test_referee_already_done_on_non_empty_output_returns_already_done() -> None:
    """Referee ALREADY_DONE returns ALREADY_DONE instead of silently retrying (bug fix)."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ALREADY_DONE, rationale="already done"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ALREADY_DONE, rationale="already done", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.output == work_output


async def test_work_noun_comes_from_adapter_spec() -> None:
    """Engine uses adapter.work_noun in retry feedback rather than a hardcoded string."""
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write docs",
            success_condition="docs complete",
            adapter="document",
            artifact="docs",
        ),
    )
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
            ),
        ]
    )
    registry = AdapterRegistry()
    registry.register(
        AdapterSpec(
            name="document",
            description="docs",
            tools=[],
            prompt_template="",
            requires_nonempty_output=True,
            work_noun="document",
        )
    )
    sv = StateView(artifact_name="docs", language=None, files=[])
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=sv,
        validator=WorkOutputValidator(registry.get("document"), sv),
        run_fn=run_fn,
        registry=registry,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="the docs need more detail",
                hints=["expand docs coverage"],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="agreed", override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert "REQUIRED REVISION" in prompts[1]
    assert "Revise your document now." in prompts[1]
    assert "Revise your implementation" not in prompts[1]


# ── PlannerOutputValidator tests ─────────────────────────────────────────────


async def test_planner_output_validator_work_noun_is_plan() -> None:
    """PlannerOutputValidator.work_noun returns 'plan'."""
    assert PlannerOutputValidator().work_noun() == "plan"


async def test_planner_output_validator_requires_nonempty() -> None:
    """PlannerOutputValidator.requires_nonempty returns True."""
    assert PlannerOutputValidator().requires_nonempty() is True


async def test_planner_output_validator_review_context_is_contract_bounded() -> None:
    """Planner validation wording is bounded to the planning contract."""
    context = PlannerOutputValidator().review_context()

    assert (
        context.review_focus == "whether the decomposition decision satisfies the planning contract"
    )
    assert "fully covers" not in context.review_focus
    assert "northstar goal" not in context.review_focus


async def test_planner_output_validator_review_context_excludes_optimization_guidance() -> None:
    """ReviewContext.topology_rules must not contain optimization preferences."""
    context = PlannerOutputValidator().review_context()
    rules = context.topology_rules

    assert "Maximize safe concurrency" not in rules
    assert "When in doubt" not in rules
    assert "not a goal" not in rules
    assert "more parallel work" not in rules


async def test_planner_output_validator_review_context_includes_structural_validation() -> None:
    """ReviewContext.topology_rules contains structural validation authority only."""
    context = PlannerOutputValidator().review_context()
    rules = context.topology_rules

    assert "genuine ordering constraint" in rules
    assert "information flow" in rules
    assert "Convention, symmetry" in rules


async def test_work_output_validator_review_context_has_no_topology_rules() -> None:
    """WorkOutputValidator review context carries no topology rules — topology is planner-only."""
    adapters_dir = Path(__file__).parents[2] / "adapters"
    registry = AdapterRegistry()
    registry.load(adapters_dir)
    state_view = StateView(artifact_name="codebase", language=None, files=[])
    context = WorkOutputValidator(registry.get("coding"), state_view).review_context()

    assert context.topology_rules == ""


async def test_planner_output_validator_extracts_only_typed_output() -> None:
    """PlannerOutputValidator returns None when response has no WorkDecision/GraphSplitDecision."""
    request = _plan_request()
    response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
    )

    assert PlannerOutputValidator().extract_from_response(response) is None


async def test_plan_engine_goes_through_full_pwc_loop() -> None:
    """AttemptLifecycle with PlannerOutputValidator calls critic and referee for plan output."""
    request = _plan_request()
    decision = WorkDecision(
        task=WorkSpec(
            objective="scrape pages",
            success_condition="tests pass",
            adapter="coding",
            artifact="codebase",
        )
    )
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=decision)]
    )
    registry = _registry_with()
    engine = AttemptLifecycle(
        request=request,
        state_view=_state_view(),
        validator=PlannerOutputValidator(),
        run_fn=run_fn,
        registry=registry,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good plan"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.output == decision
    mock_critic.assert_called_once()
    mock_referee.assert_called_once()
    assert mock_critic.call_args.args[2].startswith("Decision: work")
    assert mock_critic.call_args.kwargs["review_context"].output_noun == "plan"
    assert mock_referee.call_args.kwargs["review_context"].output_noun == "plan"


async def test_planner_output_validator_render_for_critic_graph_split_decision() -> None:
    """PlannerOutputValidator.render_for_critic formats GraphSplitDecision nodes correctly."""
    decision = GraphSplitDecision(
        nodes=[
            DecompositionNodeSpec(
                id="a",
                task=TaskSpec(
                    objective="implement storage",
                    success_condition="tests pass",
                    adapter="coding",
                    artifact="codebase",
                ),
                depends_on=[],
            ),
            DecompositionNodeSpec(
                id="b",
                task=TaskSpec(
                    objective="implement api",
                    success_condition="endpoint works",
                    adapter="coding",
                    artifact="codebase",
                ),
                depends_on=["a"],
            ),
        ]
    )

    text = PlannerOutputValidator().render_for_critic(decision)

    assert text.startswith("Decision: split_graph (mixed topology)")
    assert "Node a:" in text
    assert "implement storage" in text
    assert "Node b (depends_on: a):" in text
    assert "implement api" in text
    assert "Success condition: tests pass" in text


async def test_decompose_disposition_returns_decompose_status_immediately() -> None:
    """Engine returns ResponseStatus.DECOMPOSE immediately when referee disposition is DECOMPOSE."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _MemoryTelemetrySink()
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=WorkOutputValidator(_adapter_spec(), _state_view()),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=3,
        telemetry_sink=sink,
        run_id=sink.run_id,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE, rationale="too broad"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.DECOMPOSE,
            rationale="task has unrelated concerns that cannot converge",
            override=True,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.DECOMPOSE
    assert result.output is None
    decompose_events = [e for e in sink.events if e.event_type == "pwc.decompose.requested"]
    assert len(decompose_events) == 1
    assert decompose_events[0].status == "decompose"
    assert "cannot converge" in (decompose_events[0].summary or "")


async def test_decompose_disposition_does_not_retry() -> None:
    """Engine makes exactly one producer call when referee returns DECOMPOSE — no retry."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, prompts = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    engine = _engine(
        request=request,
        run_fn=run_fn,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=3,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE, rationale="scope too wide"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.DECOMPOSE,
            rationale="break into separate tasks",
            override=True,
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.DECOMPOSE
    assert len(prompts) == 1
    mock_critic.assert_called_once()
    mock_referee.assert_called_once()


async def test_plan_engine_revise_injects_feedback_and_retries() -> None:
    """AttemptLifecycle retries plans with the same structured RevisionRequest mechanism."""
    request = _plan_request()
    decision = WorkDecision(
        task=WorkSpec(
            objective="scrape pages",
            success_condition="tests pass",
            adapter="coding",
            artifact="codebase",
        )
    )
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=decision),
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=decision),
        ]
    )
    registry = _registry_with()
    engine = AttemptLifecycle(
        request=request,
        state_view=_state_view(),
        validator=PlannerOutputValidator(),
        run_fn=run_fn,
        registry=registry,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="missing error handling task",
                hints=["add error handling"],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="good"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REVISE, rationale="agreed", override=False
            ),
            RefereeDecision(disposition=CriticDisposition.ACCEPT, rationale="done", override=False),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert len(prompts) == 2
    assert "REQUIRED REVISION" in prompts[1]
    assert "Revise your plan now." in prompts[1]
    assert "1. Required change: add error handling" in prompts[1]


# ── WorkOutputValidator.render_for_critic worktree evidence tests ─────────────


@pytest.fixture()
def git_worktree(tmp_path: Path) -> Path:
    """Minimal git repo with one initial commit for render_for_critic evidence tests."""
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@forge.local"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Forge Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / ".gitkeep").write_text("")
    subprocess.run(["git", "add", ".gitkeep"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


async def test_render_for_critic_includes_untracked_text_file_content(
    git_worktree: Path,
) -> None:
    """A newly-created untracked file appears in critic evidence with its full content."""
    (git_worktree / "src").mkdir()
    (git_worktree / "src" / "scraper.py").write_text("class WebScraper:\n    pass\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Implemented WebScraper."))

    assert "src/scraper.py" in text
    assert "class WebScraper:" in text


async def test_render_for_critic_includes_multiple_untracked_files(
    git_worktree: Path,
) -> None:
    """All untracked text files in the worktree appear in the critic evidence."""
    (git_worktree / "src").mkdir()
    (git_worktree / "tests").mkdir()
    (git_worktree / "src" / "scraper.py").write_text("class WebScraper:\n    pass\n")
    (git_worktree / "tests" / "test_scraper.py").write_text("def test_scraper():\n    pass\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Created scraper and tests."))

    assert "class WebScraper:" in text
    assert "def test_scraper():" in text


async def test_render_for_critic_excludes_binary_untracked_files(
    git_worktree: Path,
) -> None:
    """Binary untracked files are excluded from the critic evidence."""
    (git_worktree / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Added image."))

    assert "New file: image.png" not in text


async def test_render_for_critic_excludes_oversized_untracked_files(
    git_worktree: Path,
) -> None:
    """Untracked files larger than 64 KB are excluded from the critic evidence."""
    (git_worktree / "huge.txt").write_text("x" * (65 * 1024))

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Generated huge file."))

    assert "New file: huge.txt" not in text


async def test_render_for_critic_without_worktree_omits_git_and_untracked_sections() -> None:
    """When worktree_path is None, git status, git diff, and untracked sections are omitted."""
    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=None)
    text = validator.render_for_critic(WorkOutput(summary="Did something."))

    assert "Git status:" not in text
    assert "Git diff:" not in text
    assert "New file:" not in text


async def test_render_for_critic_excludes_pycache_untracked_files(
    git_worktree: Path,
) -> None:
    """__pycache__ files are not included as 'New file:' content blocks in critic evidence."""
    (git_worktree / "__pycache__").mkdir()
    (git_worktree / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"pyc")
    # A real source file that should appear.
    (git_worktree / "real.py").write_text("x = 1\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Did something."))

    assert "New file: __pycache__/module.cpython-312.pyc" not in text
    assert "New file: real.py" in text


async def test_render_for_critic_excludes_pyc_suffix_files(
    git_worktree: Path,
) -> None:
    """Untracked .pyc files do not appear as 'New file:' content blocks in critic evidence."""
    (git_worktree / "compiled.pyc").write_bytes(b"pyc")
    (git_worktree / "real.py").write_text("x = 1\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Did something."))

    assert "New file: compiled.pyc" not in text
    assert "New file: real.py" in text


async def test_render_for_critic_excludes_noise_file_names(
    git_worktree: Path,
) -> None:
    """EXCLUDED_FILE_NAMES entries (e.g. pyvenv.cfg) do not appear as 'New file:' blocks."""
    (git_worktree / "pyvenv.cfg").write_text("[python]\nversion = 3.12\n")
    (git_worktree / "real.py").write_text("x = 1\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Did something."))

    assert "New file: pyvenv.cfg" not in text
    assert "New file: real.py" in text


async def test_render_for_critic_excludes_venv_untracked_files(
    git_worktree: Path,
) -> None:
    """Files inside .venv do not appear as 'New file:' content blocks in critic evidence."""
    (git_worktree / ".venv").mkdir()
    (git_worktree / ".venv" / "pyvenv.cfg").write_text("[python]\nversion = 3.12\n")
    (git_worktree / "real.py").write_text("x = 1\n")

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Did something."))

    assert "New file: .venv/pyvenv.cfg" not in text
    assert "New file: real.py" in text


async def test_render_for_critic_includes_new_untracked_lock_file(
    git_worktree: Path,
) -> None:
    """A new untracked uv.lock within the byte cap appears in critic evidence."""
    (git_worktree / "uv.lock").write_text('version = 1\n[[package]]\nname = "httpx"\n')

    sv = _state_view()
    validator = WorkOutputValidator(_adapter_spec(), sv, worktree_path=git_worktree)
    text = validator.render_for_critic(WorkOutput(summary="Added dependency."))

    assert "New file: uv.lock" in text
    assert "httpx" in text


async def test_render_for_critic_excludes_pyc_and_pyd_still() -> None:
    """Compiled Python extensions remain excluded from critic evidence after lockfile change."""
    from forge.core.file_filters import CRITIC_EVIDENCE_EXCLUDED_SUFFIXES

    assert ".pyc" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
    assert ".pyo" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
    assert ".pyd" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
    assert ".lock" not in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES


async def test_evidence_snapshot_telemetry_emitted_before_critic() -> None:
    """pwc.evidence.snapshot is emitted before critic.finding.parsed in each attempt."""
    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _MemoryTelemetrySink()
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=WorkOutputValidator(_adapter_spec(), _state_view()),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=1,
        telemetry_sink=sink,
        run_id=sink.run_id,
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    event_types = [e.event_type for e in sink.events]
    assert "pwc.evidence.snapshot" in event_types
    snapshot_idx = event_types.index("pwc.evidence.snapshot")
    critic_idx = event_types.index("critic.finding.parsed")
    assert snapshot_idx < critic_idx


async def test_evidence_snapshot_excerpt_is_bounded() -> None:
    """Evidence snapshot excerpt in telemetry is at most 2000 chars even for large evidence."""
    from unittest.mock import patch as _patch

    request = _work_request()
    work_output = WorkOutput(summary="Completed worktree changes.")
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _MemoryTelemetrySink()
    validator = WorkOutputValidator(_adapter_spec(), _state_view())
    engine = AttemptLifecycle[WorkOutput](
        request=request,
        state_view=_state_view(),
        validator=validator,
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=1,
        telemetry_sink=sink,
        run_id=sink.run_id,
    )

    large_evidence = "evidence line\n" * 500  # ~7000 chars

    with (
        _patch.object(validator, "render_for_critic", return_value=large_evidence),
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        await engine.run("base prompt")

    snapshot_events = [e for e in sink.events if e.event_type == "pwc.evidence.snapshot"]
    assert snapshot_events
    ev = snapshot_events[0]
    assert ev.data["evidence_length"] == len(large_evidence)
    assert len(str(ev.data["evidence_excerpt"])) <= 2000
