"""Tests for CLI runtime assembly."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from pathlib import Path

from pytest import MonkeyPatch

from forge.adapters.registry import AdapterRegistry
from forge.core.config import (
    ArtifactConfig,
    ForgeConfig,
    ModelsConfig,
    PwcModelConfig,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    FailureKind,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    StateView,
    WorkSpec,
)
from forge.core.runtime import format_failed_node
from forge.core.scheduler import SchedulerCallbacks
from forge.core.telemetry import TelemetrySink
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.run import _format_failed_node as _format_failed_node_from_run
from forge.run import _start


class _FakeProvider:
    """Minimal provider stand-in that records the configured producer string."""

    def __init__(self, producer: str) -> None:
        self.producer = producer
        self.max_tokens = 8192


def test_failed_plan_node_output_includes_response_reason() -> None:
    """CLI failure text includes failed plan response details."""
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )
    response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        failure_kind=FailureKind.VALIDATION_REJECTED,
        error="validation rejected work with disposition 'reject': missing tests",
    )
    node = DAGNode(request=request).with_response(response)

    text = format_failed_node(node)

    assert "✗ failed: plan" in text
    assert "status: failed" in text
    assert "failure_kind: validation_rejected" in text
    assert "validation rejected work with disposition 'reject': missing tests" in text


def test_format_failed_node_re_exported_from_run() -> None:
    """_format_failed_node is accessible from forge.run for backward compatibility."""
    assert _format_failed_node_from_run is format_failed_node


async def test_start_wires_nested_planner_and_worker_models(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """ForgeRuntime uses nested producer/critic/referee config for planner and worker handlers."""
    made_models: list[str] = []
    captured: dict[str, object] = {}

    def fake_make_provider(model_string: str, max_tokens: int) -> _FakeProvider:
        made_models.append(model_string)
        return _FakeProvider(model_string)

    async def fake_plan_agent(
        request: AgentRequest,
        artifact_names: list[str],
        artifact_languages: dict[str, str],
        provider: _FakeProvider,
        max_retries: int = 3,
        critic_provider: _FakeProvider | None = None,
        referee_provider: _FakeProvider | None = None,
        registry: AdapterRegistry | None = None,
        artifact_types: dict[str, str] | None = None,
        artifact_descriptions: dict[str, str] | None = None,
        artifact_language_guidance: dict[str, str] | None = None,
        telemetry_sink: TelemetrySink | None = None,
        max_attempts: int = 3,
    ) -> AgentResponse:
        assert critic_provider is not None
        assert referee_provider is not None
        assert registry is not None
        assert artifact_types == {"codebase": "coding"}
        assert artifact_descriptions == {"codebase": "Implementation and tests."}
        assert artifact_language_guidance is not None
        assert "codebase" in artifact_language_guidance
        captured["planner_provider"] = provider.producer
        captured["planner_critic"] = critic_provider.producer
        captured["planner_referee"] = referee_provider.producer
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    async def fake_work_agent(
        request: AgentRequest,
        registry: AdapterRegistry,
        workspace: Workspace,
        language_registry: LanguageRegistry,
        provider: _FakeProvider,
        state_view: StateView,
        max_retries: int = 3,
        max_tool_iterations: int = 25,
        critic_provider: _FakeProvider | None = None,
        referee_provider: _FakeProvider | None = None,
        max_attempts: int = 3,
        telemetry_sink: TelemetrySink | None = None,
        integration_revision: object = None,
    ) -> AgentResponse:
        assert critic_provider is not None
        assert referee_provider is not None
        captured["worker_provider"] = provider.producer
        captured["worker_critic"] = critic_provider.producer
        captured["worker_referee"] = referee_provider.producer
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    class FakeScheduler:
        def __init__(
            self,
            *,
            runner: Callable[[AgentRequest], Awaitable[AgentResponse]],
            state_services: object | None = None,
            callbacks: SchedulerCallbacks | None = None,
            telemetry_sink: TelemetrySink | None = None,
            run_id: object | None = None,
        ) -> None:
            self.runner = runner

        async def run(self, state: SchedulerState, global_planner: AgentRequest) -> SchedulerState:
            await self.runner(global_planner)
            await self.runner(
                AgentRequest(
                    agent_type=AgentType.WORK,
                    source=RequestSource.PLANNER,
                    spec=WorkSpec(
                        objective="write code",
                        success_condition="code exists",
                        adapter="coding",
                        artifact="codebase",
                        language="python",
                    ),
                )
            )
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", FakeScheduler)
    monkeypatch.setattr("forge.core.runner.plan_agent", fake_plan_agent)
    monkeypatch.setattr("forge.core.runner.work_agent", fake_work_agent)

    config = ForgeConfig(
        northstar="build a tool",
        workspace=tmp_path / "ws",
        artifacts=[
            ArtifactConfig(
                name="codebase",
                type="coding",
                language="python",
                description="Implementation and tests.",
            )
        ],
        models=ModelsConfig(
            planner=PwcModelConfig(
                producer="ollama/planner",
                critic="ollama/planner-critic",
                referee="ollama/planner-referee",
            ),
            worker=PwcModelConfig(
                producer="ollama/worker",
                critic="ollama/worker-critic",
                referee="ollama/worker-referee",
            ),
        ),
    )

    await _start(config)

    assert made_models == [
        "ollama/planner",
        "ollama/planner-critic",
        "ollama/planner-referee",
        "ollama/worker",
        "ollama/worker-critic",
        "ollama/worker-referee",
    ]
    assert captured == {
        "planner_provider": "ollama/planner",
        "planner_critic": "ollama/planner-critic",
        "planner_referee": "ollama/planner-referee",
        "worker_provider": "ollama/worker",
        "worker_critic": "ollama/worker-critic",
        "worker_referee": "ollama/worker-referee",
    }
