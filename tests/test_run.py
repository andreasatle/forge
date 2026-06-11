"""Tests for CLI runtime assembly."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from pathlib import Path

from pytest import MonkeyPatch

from forge.adapters.registry import AdapterRegistry
from forge.core.config import (
    ArtifactConfig,
    ForgeConfig,
    IntegratorModelConfig,
    ModelsConfig,
    PwcModelConfig,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    StateView,
    WorkSpec,
)
from forge.core.scheduler import SchedulerCallbacks
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.run import _start


class _FakeProvider:
    """Minimal provider stand-in that records the configured producer string."""

    def __init__(self, producer: str) -> None:
        self.producer = producer
        self.max_tokens = 8192


async def test_start_wires_nested_planner_and_worker_models(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """_start uses nested producer/critic/referee config for planner and worker handlers."""
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
    ) -> AgentResponse:
        assert critic_provider is not None
        assert referee_provider is not None
        assert registry is not None
        assert artifact_types == {"codebase": "coding"}
        assert artifact_descriptions == {"codebase": "Implementation and tests."}
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

    monkeypatch.setattr("forge.run.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.run.Scheduler", FakeScheduler)
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
            integrator=IntegratorModelConfig(producer="ollama/integrator"),
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
