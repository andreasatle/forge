"""Tests for AttemptEngine, WorkOutputValidator, and PlanResponseValidator."""

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.attempt import (
    AttemptEngine,
    PlanResponseValidator,
    RunAgentFailed,
    WorkOutputValidator,
)
from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    FailureKind,
    FileContent,
    PlanResponse,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    RevisionItem,
    StateView,
    TaskSpec,
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
) -> AttemptEngine[WorkOutput]:
    req = request or _work_request()
    sv = _state_view()
    if run_fn is None:

        async def _default(prompt: str) -> AgentResponse:
            return AgentResponse(
                request_id=req.id, status=ResponseStatus.COMPLETED, output=WorkOutput()
            )

        run_fn = _default
    return AttemptEngine[WorkOutput](
        request=req,
        state_view=sv,
        validator=WorkOutputValidator(
            _adapter_spec(work_noun=work_noun, requires_nonempty=requires_nonempty), sv
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    first_work_output = WorkOutput(files=[FileContent(path="main.py", content="first")])
    improved_work_output = WorkOutput(files=[FileContent(path="main.py", content="improved")])
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
    assert "files must be an array of JSON objects" in prompts[1]
    assert 'Do not use string entries like "path:...".' in prompts[1]


async def test_revise_prompt_preserves_structured_criterion_ids() -> None:
    """Structured revision items supplied by the referee are rendered with criterion ids."""
    request = _work_request()
    first_work_output = WorkOutput(files=[FileContent(path="main.py", content="first")])
    improved_work_output = WorkOutput(files=[FileContent(path="main.py", content="improved")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
                rationale="missing error handling",
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
                rationale="error handling missing",
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
    base_prompt = f"base prompt\n\n{contract_block}\n\nProduce output satisfying this contract."
    quoted_contract = (
        f"The output missed a rule.\n\n{contract_block}\n\nAfter applying the contract, add tests."
    )
    first_work_output = WorkOutput(files=[FileContent(path="main.toy", content="first")])
    improved_work_output = WorkOutput(files=[FileContent(path="main.toy", content="improved")])
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
    work_output = WorkOutput(files=[FileContent(path="main.toy", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
            disposition=CriticDisposition.REJECT, rationale="bad output", hints=["fix everything"]
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.REJECT, rationale="still bad", override=False
        )
        result = await engine.run("base prompt")

    assert result.status == ResponseStatus.FAILED
    assert result.failure_kind == FailureKind.VALIDATION_REJECTED
    assert result.output is None
    assert "reject" in (result.error or "")
    assert len(_) == 1


async def test_repeated_revise_until_max_attempts_returns_failed_without_output() -> None:
    """Engine fails when all validation attempts are exhausted without acceptance."""
    request = _work_request()
    last_work_output = WorkOutput(files=[FileContent(path="main.py", content="final")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    engine = AttemptEngine[WorkOutput](
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


async def test_telemetry_append_failure_does_not_change_pwc_outcome() -> None:
    """Telemetry is best-effort and cannot fail an otherwise accepted PWC run."""
    request = _work_request()
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _FailingTelemetrySink()
    engine = AttemptEngine[WorkOutput](
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    work_output = WorkOutput(files=[FileContent(path="README.md", content="# Hello")])
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
    sv = StateView(artifact_name="docs", language=None, files=[], dependencies=[])
    engine = AttemptEngine[WorkOutput](
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
    assert "REQUIRED REVISION" in prompts[1]
    assert "Revise your document now." in prompts[1]
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


async def test_plan_response_validator_review_context_is_contract_bounded() -> None:
    """Planner validation wording is bounded to the planning contract."""
    context = PlanResponseValidator().review_context()

    assert context.review_focus == "whether the task decomposition satisfies the planning contract"
    assert "fully covers" not in context.review_focus
    assert "northstar goal" not in context.review_focus


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


async def test_plan_response_validator_extracts_only_typed_output() -> None:
    """PlanResponseValidator returns None when response has no PlanResponse output."""
    request = _plan_request()
    response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
    )

    assert PlanResponseValidator().extract_from_response(response) is None


async def test_plan_engine_goes_through_full_pwc_loop() -> None:
    """AttemptEngine with PlanResponseValidator calls critic and referee for plan output."""
    request = _plan_request()
    plan = PlanResponse(
        tasks=[
            TaskSpec(
                objective="scrape pages",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            )
        ]
    )
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=plan)]
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
    assert result.output == plan
    mock_critic.assert_called_once()
    mock_referee.assert_called_once()
    assert mock_critic.call_args.args[2].startswith("Task 0: scrape pages")
    assert mock_critic.call_args.kwargs["review_context"].output_noun == "plan"
    assert mock_referee.call_args.kwargs["review_context"].output_noun == "plan"


async def test_decompose_disposition_returns_decompose_status_immediately() -> None:
    """Engine returns ResponseStatus.DECOMPOSE immediately when referee disposition is DECOMPOSE."""
    request = _work_request()
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
    run_fn, _ = _make_run_fn(
        [AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output)]
    )
    sink = _MemoryTelemetrySink()
    engine = AttemptEngine[WorkOutput](
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
    work_output = WorkOutput(files=[FileContent(path="main.py", content="code")])
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
    """AttemptEngine retries plans with the same structured RevisionRequest mechanism."""
    request = _plan_request()
    plan = PlanResponse(
        tasks=[
            TaskSpec(
                objective="scrape pages",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            )
        ]
    )
    run_fn, prompts = _make_run_fn(
        [
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=plan),
            AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, output=plan),
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
    assert "REQUIRED REVISION" in prompts[1]
    assert "Revise your plan now." in prompts[1]
    assert "1. Required change: add error handling" in prompts[1]
