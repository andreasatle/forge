"""Tests for AttemptEngine, DeltaStateValidator, and PlanResponseValidator."""

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.attempt import (
    AttemptEngine,
    DeltaStateValidator,
    PlanResponseValidator,
    RunAgentFailed,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FailureKind,
    FileWrite,
    PlanResponse,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    StateView,
    TaskSpec,
    WorkSpec,
)


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
    return StateView(artifact_name="codebase", language=None, files=[], dependencies=[])


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
) -> AttemptEngine[DeltaState]:
    req = request or _work_request()
    sv = _state_view()
    if run_fn is None:

        async def _default(prompt: str) -> AgentResponse:
            return AgentResponse(
                request_id=req.id, status=ResponseStatus.COMPLETED, delta=DeltaState()
            )

        run_fn = _default
    return AttemptEngine[DeltaState](
        request=req,
        state_view=sv,
        validator=DeltaStateValidator(
            _adapter_spec(work_noun=work_noun, requires_nonempty=requires_nonempty), sv
        ),
        run_fn=run_fn,
        registry=_registry_with(),
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        max_attempts=max_attempts,
    )


async def test_accept_on_first_attempt_returns_immediately() -> None:
    """Engine returns the delta immediately when referee accepts on the first attempt."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta)]
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
    assert result.delta == delta
    assert len(_) == 1


async def test_revise_injects_feedback_and_retries() -> None:
    """Engine retries with feedback injected in the prompt when disposition is REVISE."""
    request = _work_request()
    first_delta = DeltaState(new_files=[FileWrite(path="main.py", content="first")])
    improved_delta = DeltaState(new_files=[FileWrite(path="main.py", content="improved")])
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, delta=first_delta
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, delta=improved_delta
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
    assert result.delta == improved_delta
    assert len(prompts) == 2
    assert "Revise your implementation addressing the feedback above" in prompts[1]
    assert "add tests" in prompts[1]


async def test_rejected_validation_returns_failed_without_delta() -> None:
    """Engine fails immediately when validation rejects a worker delta."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta)]
    )
    engine = _engine(
        request=request, run_fn=run_fn, critic_provider=MagicMock(), referee_provider=MagicMock()
    )

    with (
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.REJECT, rationale="bad output", hints=["fix everything"]
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REJECT, rationale="still bad", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert result.delta is None
    assert "reject" in (result.error or "")
    assert len(_) == 1


async def test_repeated_revise_until_max_attempts_returns_failed_without_delta() -> None:
    """Engine fails when all validation attempts are exhausted without acceptance."""
    request = _work_request()
    last_delta = DeltaState(new_files=[FileWrite(path="main.py", content="final")])
    run_fn, prompts = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=last_delta)]
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
    assert result.delta is None
    assert "maximum validation attempts exhausted" in (result.error or "")
    assert len(prompts) == 2


async def test_critic_parse_failure_returns_failed_without_delta() -> None:
    """Engine fails when critic validation cannot be parsed."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta)]
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
    assert result.delta is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_referee.assert_not_called()


async def test_referee_parse_failure_returns_failed_without_delta() -> None:
    """Engine fails when referee validation cannot be parsed."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta)]
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
    assert result.delta is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_referee.assert_called_once()


async def test_no_providers_skips_validation() -> None:
    """Engine skips critic/referee entirely when both providers are None."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta)]
    )
    engine = _engine(request=request, run_fn=run_fn, critic_provider=None, referee_provider=None)

    with patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic:
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.COMPLETED
    assert result.delta == delta
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


async def test_empty_delta_no_critic_first_attempt_retries_not_already_done() -> None:
    """Engine retries with correction feedback on a non-final empty delta when no critic is configured."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
            ),
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta),
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
    assert result.delta == delta
    assert len(prompts) == 2
    assert "produced no implementation" in prompts[1]


async def test_empty_delta_no_critic_last_attempt_requires_nonempty_false_returns_already_done() -> (
    None
):
    """Engine returns ALREADY_DONE on the last attempt with empty delta when requires_nonempty is False."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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
    assert result.delta == DeltaState()
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_delta_no_critic_last_attempt_requires_nonempty_true_returns_failed() -> None:
    """Engine returns FAILED on the last attempt with empty delta when requires_nonempty is True."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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


async def test_empty_delta_ran_tests_and_passed_returns_already_done() -> None:
    """Engine returns ALREADY_DONE for empty delta when ran_tests_and_passed=True, ignoring requires_nonempty."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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
    assert result.delta == DeltaState()
    assert len(prompts) == 1
    mock_critic.assert_not_called()


async def test_empty_delta_ran_tests_not_passed_uses_existing_behavior() -> None:
    """Engine falls through to normal retry/FAILED behavior when ran_tests_and_passed=False."""
    request = _work_request()
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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


async def test_empty_delta_critic_already_done_accepts() -> None:
    """Engine accepts empty delta when critic confirms the success condition is already met."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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
    assert result.delta == DeltaState()
    assert len(_) == 1
    mock_critic.assert_called_once()


async def test_empty_delta_critic_parse_failure_returns_failed_without_delta() -> None:
    """Engine does not treat an unparsable empty-delta critic result as ALREADY_DONE."""
    request = _work_request()
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
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
    assert result.delta is None
    assert "could not be parsed" in (result.error or "")
    assert len(_) == 1
    mock_critic.assert_called_once()


async def test_empty_delta_critic_revise_triggers_retry_with_feedback() -> None:
    """Engine retries with feedback when critic returns REVISE on an empty delta attempt."""
    request = _work_request()
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                delta=DeltaState(),
                error="empty delta",
            ),
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta),
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
    assert result.delta == delta
    assert len(prompts) == 2
    assert "Revise your implementation addressing the feedback above" in prompts[1]
    assert "add some code" in prompts[1]


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
    delta = DeltaState(new_files=[FileWrite(path="README.md", content="# Hello")])
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta),
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta),
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
    sv = StateView(artifact_name="docs", language=None, files=[], dependencies=[])
    engine = AttemptEngine[DeltaState](
        request=request,
        state_view=sv,
        validator=DeltaStateValidator(registry.get("document"), sv),
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
                rationale="needs more detail",
                hints=["expand section 1"],
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
    assert "Revise your document addressing the feedback above" in prompts[1]
    assert "Revise your implementation" not in prompts[1]


# ── PlanResponseValidator tests ──────────────────────────────────────────────


async def test_plan_response_validator_is_empty_always_false() -> None:
    """PlanResponseValidator.is_empty returns False regardless of task count."""
    v = PlanResponseValidator()
    assert v.is_empty(PlanResponse(tasks=[])) is False
    assert (
        v.is_empty(
            PlanResponse(
                tasks=[
                    TaskSpec(objective="x", success_condition="y", adapter="coding", artifact="a")
                ]
            )
        )
        is False
    )


async def test_plan_response_validator_work_noun_is_plan() -> None:
    """PlanResponseValidator.work_noun returns 'plan'."""
    assert PlanResponseValidator().work_noun() == "plan"


async def test_plan_response_validator_requires_nonempty() -> None:
    """PlanResponseValidator.requires_nonempty returns True."""
    assert PlanResponseValidator().requires_nonempty() is True


async def test_plan_response_validator_render_for_critic_includes_tasks() -> None:
    """PlanResponseValidator.render_for_critic renders objectives and success conditions."""
    v = PlanResponseValidator()
    plan = PlanResponse(
        tasks=[
            TaskSpec(
                objective="write tests",
                success_condition="all tests pass",
                adapter="coding",
                artifact="codebase",
                language="python",
            )
        ]
    )
    text = v.render_for_critic(plan)
    assert "write tests" in text
    assert "all tests pass" in text
    assert "codebase" in text
    assert "python" in text


async def test_plan_engine_goes_through_full_pwc_loop() -> None:
    """AttemptEngine with PlanResponseValidator calls critic and referee for plan output."""
    request = _plan_request()
    follow_ups = [
        AgentRequest(
            agent_type=AgentType.WORK,
            source=RequestSource.PLANNER,
            spec=WorkSpec(
                objective="scrape pages",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            ),
        )
    ]
    run_fn, _ = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=follow_ups
            )
        ]
    )
    registry = _registry_with()
    engine = AttemptEngine(
        request=request,
        state_view=_state_view(),
        validator=PlanResponseValidator(),
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
    assert result.follow_up == follow_ups
    mock_critic.assert_called_once()
    mock_referee.assert_called_once()


async def test_plan_engine_revise_injects_feedback_and_retries() -> None:
    """AttemptEngine retries plan with feedback when critic returns REVISE."""
    request = _plan_request()
    follow_ups = [
        AgentRequest(
            agent_type=AgentType.WORK,
            source=RequestSource.PLANNER,
            spec=WorkSpec(
                objective="scrape pages",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            ),
        )
    ]
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=follow_ups
            ),
            AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=follow_ups
            ),
        ]
    )
    registry = _registry_with()
    engine = AttemptEngine(
        request=request,
        state_view=_state_view(),
        validator=PlanResponseValidator(),
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
    assert "Revise your plan addressing the feedback above" in prompts[1]
    assert "add error handling" in prompts[1]
