"""Worker agent that executes a task using an adapter and tool registry."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, StateView, WorkSpec
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.providers import LLMProvider
from forge.tools.builtin import build_read_registry
from forge.tools.registry import ToolRegistry


async def work_agent(
    request: AgentRequest,
    registry: AdapterRegistry,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    provider: LLMProvider,
    state_view: StateView,
    max_retries: int = 3,
    max_tool_iterations: int = 25,
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
    full_registry = build_read_registry(
        workspace,
        spec.artifact,
        plugin.test_command if plugin else None,
    )
    try:
        tool_list = full_registry.get_many(adapter.tools)
    except KeyError as e:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"adapter '{adapter.name}' declares {e}",
        )
    tools = ToolRegistry()
    for tool in tool_list:
        tools.register(tool)

    prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
    )
    if plugin:
        prompt += f"\n\n{plugin.prompt_supplement}"

    prompt += f"\n\nLanguage: {spec.language or 'not specified'}"
    prompt += f"\n\nState version: {state_view.version}"
    if state_view.files:
        sections = "\n\n".join(
            f"File: {fv.path}\n```\n{fv.content}\n```" for fv in state_view.files
        )
        prompt += f"\n\nExisting files in '{spec.artifact}':\n\n{sections}"
    else:
        prompt += (
            f"\n\nArtifact '{spec.artifact}' has no files yet — create all files from scratch."
        )

    prompt += (
        "\n\nUse the available tools to understand the existing codebase and verify current state."
        "\nProduce ALL new files and edits in your final JSON response."
        "\nDo not attempt to write files via tools — workers are read-only."
    )

    prompt += (
        "\n\nIMPORTANT — your final JSON response MUST be a non-empty DeltaState:"
        "\n- new_files must contain the complete content of every file you create"
        "\n- edits must contain every change to existing files"
        "\n- An empty DeltaState is always wrong for a coding task"
        "\n- Do not summarise what you would do — write the actual file contents"
    )

    return await run_agent(
        request,
        WorkSpec,
        provider,
        prompt,
        tools=tools,
        max_retries=max_retries,
        max_tool_iterations=max_tool_iterations,
    )
