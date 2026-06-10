"""Worker agent that executes a task using an adapter and tool registry."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import RunAgentFailed, TaskAttemptEngine
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.providers import LLMProvider
from forge.tools.builtin import build_read_registry
from forge.tools.registry import ToolRegistry

_FALLBACK_DELTA_EXAMPLE = (
    "No language-specific file layout conventions are configured for this artifact."
)


async def work_agent(
    request: AgentRequest,
    registry: AdapterRegistry,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    provider: LLMProvider,
    state_view: StateView,
    max_retries: int = 3,
    max_tool_iterations: int = 25,
    critic_provider: LLMProvider | None = None,
    referee_provider: LLMProvider | None = None,
    max_attempts: int = 3,
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

    if "{delta_example}" in adapter.prompt_template:
        if plugin:
            delta_example = plugin.delta_example.format(base_version=state_view.version)
        else:
            delta_example = _FALLBACK_DELTA_EXAMPLE.format(base_version=state_view.version)
    else:
        delta_example = ""

    base_prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
        base_version=state_view.version,
        delta_example=delta_example,
    )
    if plugin:
        base_prompt += f"\n\n{plugin.prompt_supplement}"
        base_prompt += f"\n\nLanguage: {spec.language}"

    base_prompt += f"\n\nState version: {state_view.version}"
    if state_view.files:
        sections = "\n\n".join(
            f"File: {fv.path}\n```\n{fv.content}\n```" for fv in state_view.files
        )
        base_prompt += f"\n\nExisting files in '{spec.artifact}':\n\n{sections}"
    else:
        base_prompt += (
            f"\n\nArtifact '{spec.artifact}' has no files yet — create all files from scratch."
        )

    base_prompt += (
        "\n\nWorkers are read-only: do not attempt to write files via tools."
        "\nPropose file creations and edits only through your task result."
    )

    engine = TaskAttemptEngine(
        request=request,
        state_view=state_view,
        provider=provider,
        registry=registry,
        tools=tools,
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        max_attempts=max_attempts,
        max_retries=max_retries,
        max_tool_iterations=max_tool_iterations,
    )

    try:
        return await engine.run(base_prompt)
    except RunAgentFailed as e:
        return e.response
