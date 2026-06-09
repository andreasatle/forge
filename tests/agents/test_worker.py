"""Tests for worker agent prompt/tool wiring."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.base import _build_system_prompt
from forge.agents.worker import work_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    RequestSource,
    ResponseStatus,
    WorkSpec,
)
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry

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


async def test_worker_tools_do_not_include_blackboard_tools(tmp_path) -> None:
    """work_agent passes a registry with no blackboard tools — neither read nor write."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider)

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

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider)

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

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider)

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
