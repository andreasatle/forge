"""Tests for ForgeRuntime composition root."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from pathlib import Path

from pytest import MonkeyPatch

from forge.core.config import (
    ArtifactConfig,
    ComplexityClassifierConfig,
    ForgeConfig,
    ModelsConfig,
    PwcModelConfig,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    NodeState,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkDecision,
    WorkSpec,
)
from forge.core.profile_assignment import ComplexityProfileAssigner, DefaultProfileAssigner
from forge.core.runtime import ForgeRuntime, StartResult
from forge.core.scheduler import SchedulerCallbacks
from forge.core.task_complexity import LLMTaskComplexityClassifier, TaskComplexity
from forge.core.telemetry import TelemetrySink


class _FakeProvider:
    """Minimal provider stand-in that records the configured producer string."""

    def __init__(self, producer: str, response: str = "") -> None:
        self.producer = producer
        self.max_tokens = 8192
        self.response = response

    async def chat(self, messages: object) -> str:
        return self.response


def _fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
    return _FakeProvider(model)


def _minimal_config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        northstar="build a tool",
        workspace=tmp_path / "ws",
        artifacts=[ArtifactConfig(name="codebase", type="coding", language="python")],
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


class _FakeScheduler:
    """Minimal scheduler stand-in that accepts pre-seeded state and returns it unchanged."""

    def __init__(
        self,
        *,
        runner: Callable[[AgentRequest], Awaitable[AgentResponse]],
        state_services: object | None = None,
        callbacks: SchedulerCallbacks | None = None,
        telemetry_sink: TelemetrySink | None = None,
        run_id: object | None = None,
        profile_assigner: object | None = None,
    ) -> None:
        self.runner = runner
        self.run_id = run_id
        self.telemetry_sink = telemetry_sink
        self.profile_assigner = profile_assigner

    async def run(self, state: SchedulerState) -> SchedulerState:
        return state


async def test_runtime_returns_start_result(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """ForgeRuntime.start() returns a StartResult with final_state and save_path."""
    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    result = await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert isinstance(result, StartResult)
    assert isinstance(result.final_state, SchedulerState)
    assert result.save_path.exists()


async def test_runtime_saves_final_state(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """ForgeRuntime.start() persists final state to disk."""
    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    result = await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert result.save_path.is_file()
    assert "northstar" in result.save_path.read_text()


async def test_runtime_builds_planner_and_worker_providers(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """ForgeRuntime creates providers for planner and worker (producer, critic, referee)."""
    made_models: list[str] = []

    def fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
        made_models.append(model)
        return _FakeProvider(model)

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert made_models == [
        "ollama/planner",
        "ollama/planner-critic",
        "ollama/planner-referee",
        "ollama/worker",
        "ollama/worker-critic",
        "ollama/worker-referee",
    ]


async def test_runtime_without_classifier_config_uses_default_profile_assigner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Without complexity_classifier, Scheduler receives DefaultProfileAssigner."""
    captured_assigners: list[object] = []

    class _CapturingScheduler:
        def __init__(self, *, profile_assigner: object | None = None, **kwargs: object) -> None:
            captured_assigners.append(profile_assigner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _CapturingScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(captured_assigners) == 1
    assert isinstance(captured_assigners[0], DefaultProfileAssigner)


async def test_runtime_with_classifier_config_builds_llm_complexity_profile_assigner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """With complexity_classifier, Scheduler receives a ComplexityProfileAssigner."""
    captured_assigners: list[object] = []

    class _CapturingScheduler:
        def __init__(self, *, profile_assigner: object | None = None, **kwargs: object) -> None:
            captured_assigners.append(profile_assigner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _CapturingScheduler)

    config = _config_with_classifier(tmp_path)
    await ForgeRuntime(config).start()

    assigner = captured_assigners[0]
    assert isinstance(assigner, ComplexityProfileAssigner)
    assert isinstance(assigner.classifier, LLMTaskComplexityClassifier)
    assert assigner.complexity_to_profile == {
        TaskComplexity.EASY: "fast",
        TaskComplexity.MEDIUM: "default",
        TaskComplexity.HARD: "strong",
    }


async def test_runtime_classifier_provider_constructed_once(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Runtime constructs the configured classifier provider once."""
    made_models: list[tuple[str, int]] = []

    def fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
        made_models.append((model, max_tokens))
        return _FakeProvider(model)

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_config_with_classifier(tmp_path)).start()

    assert made_models.count(("openai/gpt-4o-mini", 512)) == 1


async def test_runtime_scheduler_receives_profile_assigner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Runtime passes the composed profile assigner into Scheduler."""
    captured_assigners: list[object] = []

    class _CapturingScheduler:
        def __init__(self, *, profile_assigner: object | None = None, **kwargs: object) -> None:
            captured_assigners.append(profile_assigner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _CapturingScheduler)

    await ForgeRuntime(_config_with_classifier(tmp_path)).start()

    assert isinstance(captured_assigners[0], ComplexityProfileAssigner)


async def test_runtime_completes_with_failed_state_when_classifier_output_is_invalid(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Invalid classifier output fails the plan node without raising out of runtime."""

    def fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
        return _FakeProvider(model, response="not json")

    def fake_make_plan_handler(
        *args: object, **kwargs: object
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        async def handler(request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkDecision(
                    task=WorkSpec(
                        objective="implement parser",
                        success_condition="tests pass",
                        adapter="coding",
                        artifact="codebase",
                    )
                ),
            )

        return handler

    def fake_make_work_handler(
        *args: object, **kwargs: object
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        async def handler(request: AgentRequest) -> AgentResponse:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_plan_handler", fake_make_plan_handler)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", fake_make_work_handler)

    result = await ForgeRuntime(_config_with_classifier(tmp_path)).start()

    plan_node = next(
        node
        for node in result.final_state.dag.values()
        if node.request.agent_type is AgentType.PLAN
    )
    assert plan_node.node_state == NodeState.FAILED
    assert plan_node.response is not None
    assert (
        plan_node.response.error
        == "profile assignment failed: invalid task complexity JSON: Expecting value; "
        "raw output excerpt: not json"
    )


async def test_runtime_creates_telemetry_sink(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """ForgeRuntime creates a JsonlTelemetrySink and passes it to the Scheduler."""
    captured_sink: list[object] = []

    class CapturingScheduler:
        def __init__(
            self,
            *,
            runner: object,
            state_services: object = None,
            callbacks: object = None,
            telemetry_sink: object = None,
            run_id: object = None,
            profile_assigner: object = None,
        ) -> None:
            captured_sink.append(telemetry_sink)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", CapturingScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(captured_sink) == 1
    assert captured_sink[0] is not None


async def test_runtime_verbose_prints_dag_summary(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: object
) -> None:
    """ForgeRuntime logs DAG summary when verbose=True."""
    import logging

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    with caplog.at_level(logging.INFO, logger="forge.core.runtime"):  # type: ignore[attr-defined]
        await ForgeRuntime(_minimal_config(tmp_path), verbose=True).start()

    assert "DAG summary:" in caplog.text  # type: ignore[attr-defined]


async def test_runtime_no_dag_summary_when_not_verbose(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: object
) -> None:
    """ForgeRuntime does not log DAG summary when verbose=False."""
    import logging

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    with caplog.at_level(logging.INFO, logger="forge.core.runtime"):  # type: ignore[attr-defined]
        await ForgeRuntime(_minimal_config(tmp_path), verbose=False).start()

    assert "DAG summary:" not in caplog.text  # type: ignore[attr-defined]


async def test_runtime_wires_adapter_and_language_registries(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """ForgeRuntime loads adapter and language registries before creating handlers."""
    from forge.adapters.registry import AdapterRegistry
    from forge.languages.registry import LanguageRegistry

    loaded_adapters: list[Path] = []
    loaded_languages: list[Path] = []

    real_adapter_load = AdapterRegistry.load
    real_language_load = LanguageRegistry.load

    def spy_adapter_load(self: AdapterRegistry, path: Path) -> None:
        loaded_adapters.append(path)
        real_adapter_load(self, path)

    def spy_language_load(self: LanguageRegistry, path: Path) -> None:
        loaded_languages.append(path)
        real_language_load(self, path)

    monkeypatch.setattr(AdapterRegistry, "load", spy_adapter_load)
    monkeypatch.setattr(LanguageRegistry, "load", spy_language_load)
    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(loaded_adapters) == 1
    assert len(loaded_languages) == 1


async def test_runtime_seeds_root_node_into_empty_dag(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Runtime adds a root planner node to the empty DAG before calling Scheduler.run."""
    captured_states: list[SchedulerState] = []

    class _CapturingScheduler:
        def __init__(
            self,
            *,
            runner: object,
            callbacks: object = None,
            telemetry_sink: object = None,
            run_id: object = None,
            state_services: object = None,
            profile_assigner: object = None,
        ) -> None:
            pass

        async def run(self, state: SchedulerState) -> SchedulerState:
            captured_states.append(state)
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _CapturingScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(captured_states) == 1
    state = captured_states[0]
    assert len(state.dag) == 1
    root = next(iter(state.dag.values()))
    assert root.request.agent_type == AgentType.PLAN
    assert root.request.source == RequestSource.USER
    assert isinstance(root.request.spec, PlanSpec)
    assert root.node_state == NodeState.PENDING


async def test_runtime_builds_planner_and_worker_providers_after_profile_refactor(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Provider construction order is unchanged after the profile-loop refactor."""
    made_models: list[str] = []

    def fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
        made_models.append(model)
        return _FakeProvider(model)

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert made_models == [
        "ollama/planner",
        "ollama/planner-critic",
        "ollama/planner-referee",
        "ollama/worker",
        "ollama/worker-critic",
        "ollama/worker-referee",
    ]


async def test_runtime_does_not_seed_root_node_when_resuming(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Runtime does not add a root node when the loaded SchedulerState already has nodes."""
    existing_plan = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a tool"),
    )
    existing_state = SchedulerState(northstar="build a tool", max_concurrency=1).add_nodes(
        [DAGNode(request=existing_plan)]
    )
    captured_states: list[SchedulerState] = []

    class _CapturingScheduler:
        def __init__(
            self,
            *,
            runner: object,
            callbacks: object = None,
            telemetry_sink: object = None,
            run_id: object = None,
            state_services: object = None,
            profile_assigner: object = None,
        ) -> None:
            pass

        async def run(self, state: SchedulerState) -> SchedulerState:
            captured_states.append(state)
            return state

    ws_path = tmp_path / "ws"
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "state.json").write_text(existing_state.model_dump_json())

    from forge.core.workspace import Workspace

    def _fake_load_run(_ws: Workspace) -> SchedulerState:
        return existing_state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _CapturingScheduler)
    monkeypatch.setattr("forge.core.runtime.load_run", _fake_load_run)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(captured_states) == 1
    state = captured_states[0]
    assert len(state.dag) == 1
    assert existing_plan.id in state.dag


# --- worker_profiles runtime wiring ---


def _config_with_fast_profile(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        northstar="build a tool",
        workspace=tmp_path / "ws",
        artifacts=[ArtifactConfig(name="codebase", type="coding", language="python")],
        models=ModelsConfig(
            planner=PwcModelConfig(producer="ollama/planner", critic=None, referee=None),
            worker=PwcModelConfig(producer="ollama/worker", critic=None, referee=None),
            worker_profiles={
                "fast": PwcModelConfig(
                    producer="ollama/fast-worker", critic=None, referee=None, max_attempts=1
                ),
            },
        ),
    )


def _config_with_classifier(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        northstar="build a tool",
        workspace=tmp_path / "ws",
        artifacts=[ArtifactConfig(name="codebase", type="coding", language="python")],
        models=ModelsConfig(
            planner=PwcModelConfig(producer="ollama/planner", critic=None, referee=None),
            worker=PwcModelConfig(producer="ollama/worker", critic=None, referee=None),
            worker_profiles={
                "fast": PwcModelConfig(producer="ollama/fast-worker", critic=None, referee=None),
                "strong": PwcModelConfig(
                    producer="ollama/strong-worker", critic=None, referee=None
                ),
            },
            complexity_classifier=ComplexityClassifierConfig(
                model="openai/gpt-4o-mini",
                max_tokens=512,
                complexity_to_profile={
                    TaskComplexity.EASY: "fast",
                    TaskComplexity.MEDIUM: "default",
                    TaskComplexity.HARD: "strong",
                },
            ),
        ),
    )


def _work_request_with_profile(profile: str) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do work", success_condition="done", adapter="coding", artifact="codebase"
        ),
        model_profile=profile,
    )


async def test_runtime_no_worker_profiles_builds_one_work_handler(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """With no worker_profiles, exactly one make_work_handler call is made."""
    call_count = [0]

    def counting_make_work_handler(
        registry: object,
        workspace: object,
        language_registry: object,
        provider: object,
        **kwargs: object,
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        call_count[0] += 1

        async def handler(request: AgentRequest) -> AgentResponse:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", counting_make_work_handler)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert call_count[0] == 1


async def test_runtime_with_fast_profile_builds_two_work_handlers(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """With a fast worker profile, two make_work_handler calls are made (default + fast)."""
    call_count = [0]

    def counting_make_work_handler(
        registry: object,
        workspace: object,
        language_registry: object,
        provider: object,
        **kwargs: object,
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        call_count[0] += 1

        async def handler(request: AgentRequest) -> AgentResponse:
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", counting_make_work_handler)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_config_with_fast_profile(tmp_path)).start()

    assert call_count[0] == 2


async def test_runtime_explicit_default_profile_overrides_models_worker(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """worker_profiles['default'] overrides models.worker for provider construction."""
    made_models: list[str] = []

    def fake_make_provider(model: str, max_tokens: int) -> _FakeProvider:
        made_models.append(model)
        return _FakeProvider(model)

    monkeypatch.setattr("forge.core.runtime.make_provider", fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    config = ForgeConfig(
        northstar="build a tool",
        workspace=tmp_path / "ws",
        artifacts=[ArtifactConfig(name="codebase", type="coding", language="python")],
        models=ModelsConfig(
            planner=PwcModelConfig(producer="ollama/planner", critic=None, referee=None),
            worker=PwcModelConfig(producer="ollama/worker", critic=None, referee=None),
            worker_profiles={
                "default": PwcModelConfig(producer="ollama/override", critic=None, referee=None),
            },
        ),
    )
    await ForgeRuntime(config).start()

    assert "ollama/override" in made_models
    assert "ollama/worker" not in made_models


async def test_runtime_request_with_fast_profile_routes_to_fast_handler(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A work request with model_profile='fast' is dispatched to the fast-profile handler."""
    dispatched_to: list[str] = []
    captured_runner: list[object] = []

    def make_traceable_work_handler(
        registry: object,
        workspace: object,
        language_registry: object,
        provider: _FakeProvider,
        **kwargs: object,
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        producer = provider.producer

        async def handler(request: AgentRequest) -> AgentResponse:
            dispatched_to.append(producer)
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    class _RunnerCapturingScheduler:
        def __init__(self, *, runner: object, **kwargs: object) -> None:
            captured_runner.append(runner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", make_traceable_work_handler)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _RunnerCapturingScheduler)

    await ForgeRuntime(_config_with_fast_profile(tmp_path)).start()

    runner = captured_runner[0]
    await runner(_work_request_with_profile("fast"))  # type: ignore[operator]

    assert dispatched_to == ["ollama/fast-worker"]


async def test_runtime_request_with_unknown_profile_falls_back_to_default(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: object
) -> None:
    """A work request with an unknown model_profile falls back to the default handler."""
    import logging

    dispatched_to: list[str] = []
    captured_runner: list[object] = []

    def make_traceable_work_handler(
        registry: object,
        workspace: object,
        language_registry: object,
        provider: _FakeProvider,
        **kwargs: object,
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        producer = provider.producer

        async def handler(request: AgentRequest) -> AgentResponse:
            dispatched_to.append(producer)
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    class _RunnerCapturingScheduler:
        def __init__(self, *, runner: object, **kwargs: object) -> None:
            captured_runner.append(runner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", make_traceable_work_handler)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _RunnerCapturingScheduler)

    await ForgeRuntime(_config_with_fast_profile(tmp_path)).start()

    runner = captured_runner[0]
    with caplog.at_level(logging.WARNING, logger="forge.core.runner"):  # type: ignore[attr-defined]
        await runner(_work_request_with_profile("nonexistent"))  # type: ignore[operator]

    assert dispatched_to == ["ollama/worker"]
    assert "nonexistent" in caplog.text  # type: ignore[attr-defined]


async def test_runtime_no_worker_profiles_default_request_routes_to_default_handler(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Legacy configs with no worker_profiles route default requests to the single handler."""
    dispatched_to: list[str] = []
    captured_runner: list[object] = []

    def make_traceable_work_handler(
        registry: object,
        workspace: object,
        language_registry: object,
        provider: _FakeProvider,
        **kwargs: object,
    ) -> Callable[[AgentRequest], Awaitable[AgentResponse]]:
        producer = provider.producer

        async def handler(request: AgentRequest) -> AgentResponse:
            dispatched_to.append(producer)
            return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

        return handler

    class _RunnerCapturingScheduler:
        def __init__(self, *, runner: object, **kwargs: object) -> None:
            captured_runner.append(runner)

        async def run(self, state: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.make_work_handler", make_traceable_work_handler)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _RunnerCapturingScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    runner = captured_runner[0]
    await runner(_work_request_with_profile("default"))  # type: ignore[operator]

    assert dispatched_to == ["ollama/worker"]


# --- runtime summary enum comparisons ---


async def test_runtime_summary_counts_integrated_nodes(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: object
) -> None:
    """Runtime summary counts INTEGRATED nodes using enum comparison."""
    import logging

    plan = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a tool"),
    )
    integrated_node = DAGNode(request=plan).with_state(NodeState.INTEGRATED)
    pre_seeded = SchedulerState(northstar="build a tool", max_concurrency=1).add_nodes(
        [integrated_node]
    )

    class _PreseededScheduler:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def run(self, state: SchedulerState) -> SchedulerState:
            return pre_seeded

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _PreseededScheduler)

    with caplog.at_level(logging.INFO, logger="forge.core.runtime"):  # type: ignore[attr-defined]
        await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert "integrated: 1" in caplog.text  # type: ignore[attr-defined]
    assert "failed: 0" in caplog.text  # type: ignore[attr-defined]
    assert "cancelled: 0" in caplog.text  # type: ignore[attr-defined]


async def test_runtime_summary_counts_failed_and_cancelled_nodes(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: object
) -> None:
    """Runtime summary counts FAILED and CANCELLED nodes using enum comparison."""
    import logging

    def _make_work_request() -> AgentRequest:
        return AgentRequest(
            agent_type=AgentType.WORK,
            source=RequestSource.USER,
            spec=WorkSpec(
                objective="do work",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
        )

    r1, r2 = _make_work_request(), _make_work_request()
    state = (
        SchedulerState(northstar="build a tool", max_concurrency=1)
        .add_nodes([DAGNode(request=r1).with_state(NodeState.FAILED)])
        .add_nodes([DAGNode(request=r2).with_state(NodeState.CANCELLED)])
    )

    class _PreseededScheduler:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def run(self, s: SchedulerState) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _PreseededScheduler)

    with caplog.at_level(logging.INFO, logger="forge.core.runtime"):  # type: ignore[attr-defined]
        await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert "integrated: 0" in caplog.text  # type: ignore[attr-defined]
    assert "failed: 1" in caplog.text  # type: ignore[attr-defined]
    assert "cancelled: 1" in caplog.text  # type: ignore[attr-defined]
