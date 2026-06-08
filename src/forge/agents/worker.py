"""Worker agent that executes a task using an adapter and tool registry."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, WorkSpec
from forge.core.state_service import StateService
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.providers import LLMProvider
from forge.tools.builtin import build_default_registry


async def work_agent(
    request: AgentRequest,
    registry: AdapterRegistry,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    provider: LLMProvider,
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
    tools = build_default_registry(
        workspace,
        spec.artifact,
        plugin.test_command if plugin else None,
        plugin.add_dependency_command if plugin else None,
    )

    state_view = StateService(workspace, spec.artifact, plugin).build_state_view()

    prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
    )
    if plugin:
        prompt += f"\n\n{plugin.prompt_supplement}"

    prompt += f"\n\nLanguage: {spec.language or 'not specified'}"
    if state_view.files:
        files_list = "\n".join(f"  - {f}" for f in state_view.files)
        prompt += f"\n\nExisting files in '{spec.artifact}':\n{files_list}"
    else:
        prompt += f"\n\nArtifact '{spec.artifact}' has no files yet — create all files from scratch."

    prompt += (
        "\n\nIMPORTANT: Your final response MUST list every file you created in new_files "
        "and every edit you made in edits. Do NOT return empty new_files and edits."
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
