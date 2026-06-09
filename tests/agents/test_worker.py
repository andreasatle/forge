"""Tests for worker agent prompt/tool wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.worker import work_agent
from forge.core.models import AgentRequest, AgentResponse, AgentType, RequestSource, ResponseStatus, WorkSpec
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry


def _tool_names(registry) -> set[str]:
    return {tool.name for tool in registry}


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


async def test_worker_tools_do_not_include_write_blackboard(tmp_path) -> None:
    """work_agent passes a read-only registry without hidden blackboard mutation."""
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
    assert "write_blackboard" not in _tool_names(tools)
