"""Tests for ForgeRuntime composition root."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from pathlib import Path

from pytest import MonkeyPatch

from forge.core.config import (
    ArtifactConfig,
    ForgeConfig,
    ModelsConfig,
    PwcModelConfig,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    SchedulerState,
)
from forge.core.runtime import ForgeRuntime, StartResult
from forge.core.scheduler import SchedulerCallbacks
from forge.core.telemetry import TelemetrySink


class _FakeProvider:
    """Minimal provider stand-in that records the configured producer string."""

    def __init__(self, producer: str) -> None:
        self.producer = producer
        self.max_tokens = 8192


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
    """Minimal scheduler stand-in that dispatches the global planner and returns state unchanged."""

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
        self.run_id = run_id
        self.telemetry_sink = telemetry_sink

    async def run(self, state: SchedulerState, global_planner: AgentRequest) -> SchedulerState:
        await self.runner(global_planner)
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
        ) -> None:
            captured_sink.append(telemetry_sink)

        async def run(self, state: SchedulerState, global_planner: AgentRequest) -> SchedulerState:
            return state

    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", CapturingScheduler)

    await ForgeRuntime(_minimal_config(tmp_path)).start()

    assert len(captured_sink) == 1
    assert captured_sink[0] is not None


async def test_runtime_verbose_prints_dag_summary(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: object
) -> None:
    """ForgeRuntime prints DAG summary when verbose=True."""
    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path), verbose=True).start()

    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "DAG summary:" in out


async def test_runtime_no_dag_summary_when_not_verbose(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: object
) -> None:
    """ForgeRuntime does not print DAG summary when verbose=False."""
    monkeypatch.setattr("forge.core.runtime.make_provider", _fake_make_provider)
    monkeypatch.setattr("forge.core.runtime.Scheduler", _FakeScheduler)

    await ForgeRuntime(_minimal_config(tmp_path), verbose=False).start()

    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "DAG summary:" not in out


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
