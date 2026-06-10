"""Tests for worker agent prompt assembly and response wrapping."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.attempt import RunAgentFailed
from forge.agents.base import _build_system_prompt
from forge.agents.worker import work_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    FailureKind,
    FileWrite,
    RequestSource,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin, LanguageRegistry

BLACKBOARD_TOOL_NAMES = {"read_blackboard", "write_blackboard"}
MUTATING_TOOL_NAMES = {"write_file", "replace_in_file", "add_dependency", "write_blackboard"}


def _tool_names(registry) -> set[str]:
    return {tool.name for tool in registry}


def _available_tool_names(system_prompt: str) -> set[str]:
    return set(re.findall(r"^  ([a-z_]+): ", system_prompt, flags=re.MULTILINE))


def _registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry._adapters["coding"] = AdapterSpec(
        name="coding",
        description="test",
        tools=[],
        prompt_template="do: {objective}\nsuccess: {success_condition}",
    )
    return registry


def _request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="inspect code",
            success_condition="report changes",
            adapter="coding",
            artifact="codebase",
        ),
    )


def _yaml_adapter_registry() -> AdapterRegistry:
    adapters_dir = Path(__file__).parents[2] / "adapters"
    registry = AdapterRegistry()
    registry.load(adapters_dir)
    return registry


def _language_registry_with_tests(name: str = "python") -> LanguageRegistry:
    lr = LanguageRegistry()
    lr._plugins[name] = LanguagePlugin(
        name=name,
        package_manager="uv",
        init_command="uv init",
        test_command="pytest",
        sync_command="uv sync",
        add_dependency_command="uv add {package}",
        project_structure=[],
        prompt_supplement="",
        delta_example="",
    )
    return lr


def _state_view(artifact: str = "codebase", language: str | None = None) -> StateView:
    return StateView(artifact_name=artifact, language=language, files=[], dependencies=[])


def _work_request(adapter: str, language: str | None = None) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task objective",
            success_condition="task done",
            adapter=adapter,
            artifact="codebase",
            language=language,
        ),
    )


async def test_worker_tools_do_not_include_blackboard_tools(tmp_path) -> None:
    """work_agent passes a registry with no blackboard tools — neither read nor write."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools).isdisjoint(BLACKBOARD_TOOL_NAMES)


async def test_worker_prompt_does_not_expose_blackboard_tools(tmp_path) -> None:
    """Neither user prompt nor system prompt seen by a worker mentions any blackboard tool."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = _build_system_prompt(tools, DeltaState)

    for tool_name in BLACKBOARD_TOOL_NAMES:
        assert tool_name not in system_prompt
        assert tool_name not in user_prompt


async def test_worker_prompt_tool_mentions_match_registry(tmp_path) -> None:
    """Worker-facing prompts mention only tools that are actually registered."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = _build_system_prompt(tools, DeltaState)
    tool_names = _tool_names(tools)

    assert _available_tool_names(system_prompt) == tool_names
    for unavailable in MUTATING_TOOL_NAMES - tool_names:
        assert unavailable not in system_prompt
        assert unavailable not in user_prompt


def test_production_adapter_yamls_expose_no_blackboard_tools() -> None:
    """No adapter YAML in the production adapters directory advertises blackboard tools."""
    adapters_dir = Path(__file__).parents[2] / "adapters"
    registry = AdapterRegistry()
    registry.load(adapters_dir)
    for name in registry.names():
        spec = registry.get(name)
        exposed = BLACKBOARD_TOOL_NAMES.intersection(spec.tools)
        assert not exposed, f"adapter '{name}' exposes blackboard tools: {exposed}"


async def test_coding_adapter_receives_exactly_declared_tools(tmp_path) -> None:
    """work_agent passes exactly the tools declared in coding.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("coding").tools)
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request,
            adapter_registry,
            workspace,
            _language_registry_with_tests(),
            provider,
            _state_view(language="python"),
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_document_adapter_receives_exactly_declared_tools(tmp_path) -> None:
    """work_agent passes exactly the tools declared in document.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("document").tools)
    request = _work_request("document")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_audit_adapter_receives_exactly_declared_tools(tmp_path) -> None:
    """work_agent passes exactly the tools declared in audit.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("audit").tools)
    request = _work_request("audit")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_audit_adapter_does_not_receive_list_files_or_run_tests(tmp_path) -> None:
    """audit adapter declares only read_file — list_files and run_tests must not be exposed."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    request = _work_request("audit")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert "list_files" not in _tool_names(tools)
    assert "run_tests" not in _tool_names(tools)


async def test_worker_prompt_warns_against_empty_delta(tmp_path) -> None:
    """work_agent prompt contains an explicit warning that empty DeltaState is always wrong."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "non-empty DeltaState" in user_prompt
    assert "empty DeltaState is always wrong" in user_prompt


async def test_language_supplement_appears_in_worker_prompt(tmp_path) -> None:
    """work_agent injects the language plugin's prompt_supplement into the rendered prompt."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    lr = LanguageRegistry()
    lr._plugins["python"] = LanguagePlugin(
        name="python",
        package_manager="uv",
        init_command="uv init",
        test_command="pytest",
        sync_command="uv sync",
        add_dependency_command="uv add {package}",
        project_structure=[],
        prompt_supplement="UNIQUE_SUPPLEMENT_MARKER",
        delta_example="",
    )

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, lr, provider, _state_view(language="python")
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "UNIQUE_SUPPLEMENT_MARKER" in user_prompt


async def test_language_delta_example_appears_in_worker_prompt(tmp_path) -> None:
    """work_agent renders the language-specific delta_example into the prompt."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    lr = LanguageRegistry()
    lr._plugins["python"] = LanguagePlugin(
        name="python",
        package_manager="uv",
        init_command="uv init",
        test_command="pytest",
        sync_command="uv sync",
        add_dependency_command="uv add {package}",
        project_structure=[],
        prompt_supplement="",
        delta_example='{{"path": "DELTA_EXAMPLE_MARKER"}}',
    )

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            lr,
            provider,
            _state_view(language="python"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "DELTA_EXAMPLE_MARKER" in user_prompt


async def test_unknown_tool_in_adapter_returns_failed_response(tmp_path) -> None:
    """work_agent returns FAILED with a clear message when adapter declares an unknown tool."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = AdapterRegistry()
    adapter_registry._adapters["broken"] = AdapterSpec(
        name="broken",
        description="test",
        tools=["nonexistent_tool"],
        prompt_template="do: {objective}\nsuccess: {success_condition}",
    )
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task",
            success_condition="done",
            adapter="broken",
            artifact="codebase",
        ),
    )
    provider = MagicMock()

    response = await work_agent(
        request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
    )

    assert response.status == ResponseStatus.FAILED
    assert "nonexistent_tool" in (response.error or "")


async def test_run_agent_failure_propagates_as_failed_response(tmp_path) -> None:
    """work_agent returns the failed AgentResponse when TaskAttemptEngine raises RunAgentFailed."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    failed_response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error="provider error",
        failure_kind=FailureKind.PROVIDER_ERROR,
    )

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = failed_response
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.PROVIDER_ERROR
    assert response.error == "provider error"


async def test_successful_engine_result_wrapped_in_completed_response(tmp_path) -> None:
    """work_agent returns AgentResponse(COMPLETED, delta=...) for a successful engine run."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])

    with patch("forge.agents.attempt.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == delta
