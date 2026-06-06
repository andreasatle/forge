"""Worker agent that executes a task using an adapter and tool registry."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, WorkSpec
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.tools.builtin import build_default_registry

WORK_MODEL = "gemma4:e4b"


async def work_agent(
    request: AgentRequest,
    registry: AdapterRegistry,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    max_retries: int = 3,
) -> AgentResponse:
    """Run the agentic tool loop for a work request using the specified adapter and artifact."""
    spec = request.spec
    if not isinstance(spec, WorkSpec):
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"expected WorkSpec, got {type(spec).__name__}",
        )

    adapter = registry.get(spec.adapter)
    plugin = language_registry.get(spec.language) if spec.language else None
    tools = build_default_registry(workspace, spec.artifact, plugin.test_command if plugin else None)
    tool_schema = tools.to_ollama_schema(adapter.tools)
    prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
    )
    if plugin:
        prompt += f"\n\n{plugin.prompt_supplement}"

    return await run_agent(
        request,
        WorkSpec,
        WORK_MODEL,
        prompt,
        tools=tools,
        tool_schema=tool_schema,
        max_retries=max_retries,
    )
