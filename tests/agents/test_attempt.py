"""Tests for TaskAttemptEngine."""

from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import RunAgentFailed, TaskAttemptEngine
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FailureKind,
    FileWrite,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.tools.registry import ToolRegistry


def _request() -> AgentRequest:
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


def _state_view() -> StateView:
    return StateView(artifact_name="codebase", language=None, files=[], dependencies=[])


def _engine(
    request: AgentRequest | None = None,
    critic_provider: MagicMock | None = None,
    referee_provider: MagicMock | None = None,
    max_attempts: int = 3,
) -> TaskAttemptEngine:
    if request is None:
        request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    return TaskAttemptEngine(
        request=request,
        state_view=_state_view(),
        provider=provider,
        registry=AdapterRegistry(),
        tools=ToolRegistry(),
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        max_attempts=max_attempts,
    )


async def test_accept_on_first_attempt_returns_immediately() -> None:
    """Engine returns the delta immediately when referee accepts on the first attempt."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT, rationale="good"
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == delta
    mock_run.assert_called_once()


async def test_revise_injects_feedback_and_retries() -> None:
    """Engine retries with feedback injected in the prompt when disposition is REVISE."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REVISE,
                rationale="needs work",
                hints=["add tests"],
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
    assert result.delta == delta
    assert mock_run.call_count == 2
    second_prompt = mock_run.call_args_list[1].args[3]
    assert "Revise your implementation addressing the feedback above" in second_prompt
    assert "add tests" in second_prompt


async def test_reject_retries_with_feedback() -> None:
    """Engine retries on REJECT, injecting feedback rather than failing immediately."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        mock_critic.side_effect = [
            CriticFinding(
                disposition=CriticDisposition.REJECT,
                rationale="bad output",
                hints=["fix everything"],
            ),
            CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="ok now"),
        ]
        mock_referee.side_effect = [
            RefereeDecision(
                disposition=CriticDisposition.REJECT, rationale="still bad", override=False
            ),
            RefereeDecision(
                disposition=CriticDisposition.ACCEPT, rationale="approved", override=False
            ),
        ]
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == delta
    assert mock_run.call_count == 2
    second_prompt = mock_run.call_args_list[1].args[3]
    assert "Revise your implementation addressing the feedback above" in second_prompt
    assert "fix everything" in second_prompt


async def test_max_attempts_exhaustion_returns_last_delta() -> None:
    """Engine returns the last delta when all attempts are exhausted without acceptance."""
    request = _request()
    last_delta = DeltaState(new_files=[FileWrite(path="main.py", content="final")])
    engine = _engine(
        request=request,
        critic_provider=MagicMock(),
        referee_provider=MagicMock(),
        max_attempts=2,
    )

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=last_delta
        )
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REVISE, rationale="keep improving", hints=[]
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REVISE, rationale="not done", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == last_delta
    assert mock_run.call_count == 2


async def test_critic_parse_failure_degrades_gracefully() -> None:
    """Engine returns the last successful delta when critic raises ValueError."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        mock_critic.side_effect = ValueError("critic_agent failed after 3 retries: invalid json")
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == delta
    mock_run.assert_called_once()
    mock_referee.assert_not_called()


async def test_no_providers_skips_validation() -> None:
    """Engine skips critic/referee entirely when both providers are None."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=None, referee_provider=None)

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == delta
    mock_run.assert_called_once()
    mock_critic.assert_not_called()


async def test_run_agent_failure_raises_run_agent_failed() -> None:
    """Engine raises RunAgentFailed when run_agent returns a non-COMPLETED response."""
    request = _request()
    failed_response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error="provider error",
        failure_kind=FailureKind.PROVIDER_ERROR,
    )
    engine = _engine(request=request)

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = failed_response
        try:
            await engine.run("base prompt")
            assert False, "expected RunAgentFailed"
        except RunAgentFailed as e:
            assert e.response is failed_response


async def test_empty_delta_no_critic_returns_already_done() -> None:
    """Engine returns empty delta immediately when run_agent signals empty delta and no critic is configured."""
    request = _request()
    engine = _engine(request=request, critic_provider=None, referee_provider=None)

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            delta=DeltaState(),
            error="empty delta",
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.delta == DeltaState()
    mock_run.assert_called_once()
    mock_critic.assert_not_called()


async def test_empty_delta_critic_already_done_accepts() -> None:
    """Engine accepts empty delta when critic confirms the success condition is already met."""
    request = _request()
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            delta=DeltaState(),
            error="empty delta",
        )
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ALREADY_DONE,
            rationale="success condition already met",
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.ALREADY_DONE
    assert result.delta == DeltaState()
    mock_run.assert_called_once()
    mock_critic.assert_called_once()


async def test_empty_delta_critic_revise_triggers_retry_with_feedback() -> None:
    """Engine retries with feedback when critic returns REVISE on an empty delta attempt."""
    request = _request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    engine = _engine(request=request, critic_provider=MagicMock(), referee_provider=MagicMock())

    with (
        patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run.side_effect = [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
            ),
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                delta=delta,
            ),
        ]
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
    assert result.delta == delta
    assert mock_run.call_count == 2
    second_prompt = mock_run.call_args_list[1].args[3]
    assert "Revise your implementation addressing the feedback above" in second_prompt
    assert "add some code" in second_prompt
