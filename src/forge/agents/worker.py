"""Worker agent that executes a task using an adapter and tool registry."""

import logging

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.agents.critic import critic_agent
from forge.agents.referee import referee_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry
from forge.llm.providers import LLMProvider
from forge.tools.builtin import build_read_registry
from forge.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)

_FALLBACK_DELTA_EXAMPLE = (
    '{{\n  "new_files": [{{"path": "output/result.txt", "content": "..."}}],\n'
    '  "edits": [],\n  "dependencies": [],\n  "errors": [],\n'
    '  "base_version": {base_version}\n}}'
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

    if plugin:
        delta_example = plugin.delta_example.format(base_version=state_view.version)
    else:
        delta_example = _FALLBACK_DELTA_EXAMPLE.format(base_version=state_view.version)

    base_prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
        base_version=state_view.version,
        delta_example=delta_example,
    )
    if plugin:
        base_prompt += f"\n\n{plugin.prompt_supplement}"

    base_prompt += f"\n\nLanguage: {spec.language or 'not specified'}"
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
        "\n\nUse the available tools to understand the existing codebase and verify current state."
        "\nProduce ALL new files and edits in your final JSON response."
        "\nDo not attempt to write files via tools — workers are read-only."
    )

    base_prompt += (
        "\n\nIMPORTANT — your final JSON response MUST be a non-empty DeltaState:"
        "\n- new_files must contain the complete content of every file you create"
        "\n- edits must contain every change to existing files"
        "\n- An empty DeltaState is always wrong for a coding task"
        "\n- Do not summarise what you would do — write the actual file contents"
    )

    last_response: AgentResponse | None = None
    feedback: str | None = None

    for _attempt in range(max_attempts):
        prompt = base_prompt if feedback is None else f"{base_prompt}\n\n{feedback}"

        response = await run_agent(
            request,
            WorkSpec,
            provider,
            prompt,
            tools=tools,
            max_retries=max_retries,
            max_tool_iterations=max_tool_iterations,
        )
        last_response = response

        if critic_provider is None or referee_provider is None:
            return response

        if response.status != ResponseStatus.COMPLETED or response.delta is None:
            return response

        finding = await critic_agent(request, state_view, response.delta, critic_provider, registry)
        decision = await referee_agent(
            request, state_view, response.delta, finding, referee_provider, registry
        )

        if decision.disposition == CriticDisposition.ACCEPT:
            return response

        hints_text = (
            "\n".join(f"{i + 1}. {h}" for i, h in enumerate(finding.hints))
            if finding.hints
            else "(none)"
        )
        feedback = (
            f"Your previous attempt received feedback:\n"
            f"Disposition: {decision.disposition.value}\n"
            f"Rationale: {decision.rationale}\n"
            f"Hints:\n{hints_text}\n\n"
            f"Revise your implementation addressing the feedback above."
        )

    _logger.warning(
        "work_agent: max_attempts (%d) exhausted; returning best-effort delta", max_attempts
    )
    assert last_response is not None
    return last_response
