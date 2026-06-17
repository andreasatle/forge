"""Worker agent that executes a task using an adapter and tool registry."""

from pathlib import Path

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import AttemptLifecycle, RunAgentFailed, WorkOutputValidator
from forge.agents.base import run_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    FailureKind,
    ResponseStatus,
    StateView,
    WorkOutput,
    WorkSpec,
    render_agent_contract,
)
from forge.core.telemetry import TelemetrySink
from forge.core.workspace import Workspace, run_git
from forge.languages.registry import LanguagePlugin, LanguageRegistry
from forge.llm.providers import LLMProvider
from forge.tools.builtin import build_worktree_registry
from forge.tools.registry import ToolRegistry

_FALLBACK_WORK_OUTPUT_EXAMPLE = (
    "No language-specific file layout conventions are configured for this artifact."
)
_LANGUAGE_GUIDANCE_CONSTRAINT_PREFIX = "Language plugin guidance:"


def _worktree_has_changes(worktree_path: Path) -> bool:
    result = run_git(
        ["status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _request_with_language_guidance(
    request: AgentRequest,
    spec: WorkSpec,
    plugin: LanguagePlugin | None,
) -> AgentRequest:
    """Promote plugin-owned guidance into the canonical node contract."""
    if plugin is None or not plugin.prompt_supplement.strip():
        return request
    guidance = f"{_LANGUAGE_GUIDANCE_CONSTRAINT_PREFIX}\n{plugin.prompt_supplement.strip()}"
    if guidance in spec.contract.constraints:
        return request
    contract = spec.contract.model_copy(
        update={"constraints": [*spec.contract.constraints, guidance]}
    )
    updated_spec = spec.model_copy(update={"contract": contract})
    return request.model_copy(update={"spec": updated_spec})


class WorkTaskExecutor:
    """Run one work task from AgentRequest to AgentResponse."""

    def __init__(
        self,
        *,
        registry: AdapterRegistry,
        workspace: Workspace,
        language_registry: LanguageRegistry,
        provider: LLMProvider,
        max_retries: int = 3,
        max_tool_iterations: int = 25,
        critic_provider: LLMProvider | None = None,
        referee_provider: LLMProvider | None = None,
        max_attempts: int = 3,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.language_registry = language_registry
        self.provider = provider
        self.max_retries = max_retries
        self.max_tool_iterations = max_tool_iterations
        self.critic_provider = critic_provider
        self.referee_provider = referee_provider
        self.max_attempts = max_attempts
        self.telemetry_sink = telemetry_sink

    async def run(
        self,
        request: AgentRequest,
        state_view: StateView,
    ) -> AgentResponse:
        """Execute a single work task request and return the agent response."""
        spec = request.spec
        if not isinstance(spec, WorkSpec):
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"expected WorkSpec, got {type(spec).__name__}",
            )

        adapter = self.registry.get(spec.adapter)
        plugin = self.language_registry.get(spec.language) if spec.language else None
        contract_request = _request_with_language_guidance(request, spec, plugin)
        contract_spec = contract_request.spec
        if not isinstance(contract_spec, WorkSpec):
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"expected WorkSpec, got {type(contract_spec).__name__}",
            )
        node_id = str(request.id)
        worktree_path = self.workspace.create_worktree(spec.artifact, node_id)
        preserve_worktree_for_integration = False
        full_registry = build_worktree_registry(
            str(worktree_path),
            plugin.test_command if plugin else None,
        )

        try:
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
            registered = {t.name for t in tools}
            for vt_name in adapter.verification_tools:
                if vt_name not in registered:
                    try:
                        tools.register(full_registry.get(vt_name))
                    except KeyError:
                        pass

            if "{work_output_example}" in adapter.prompt_template:
                if plugin:
                    work_output_example = plugin.work_output_example.format()
                else:
                    work_output_example = _FALLBACK_WORK_OUTPUT_EXAMPLE
            else:
                work_output_example = ""

            base_prompt = adapter.prompt_template.format(
                objective=spec.objective,
                success_condition=spec.success_condition,
                work_output_example=work_output_example,
            )
            contract_block = render_agent_contract(contract_request)
            base_prompt += f"\n\n{contract_block}\n\nYour output must satisfy every requirement in this contract."
            if plugin:
                base_prompt += f"\n\nLanguage: {spec.language}"

            base_prompt += f"\n\nState version: {state_view.version}"
            if state_view.version_sha:
                base_prompt += f"\nBase commit: {state_view.version_sha}"
            if state_view.files:
                sections = "\n\n".join(
                    f"File: {fv.path}\n```\n{fv.content}\n```" for fv in state_view.files
                )
                base_prompt += f"\n\nExisting files in '{spec.artifact}':\n\n{sections}"
            else:
                base_prompt += (
                    f"\n\nArtifact '{spec.artifact}' has no files yet"
                    " — create all files from scratch."
                )

            base_prompt += (
                "\n\nModify files directly in the assigned worktree using the available write/edit tools."
                "\nThe framework will use git status and git diff as the source of truth."
                "\nDo not include complete file contents in your final response."
                "\nAfter all required edits are made and tests pass, stop calling tools and return final JSON with kind and summary. Do not return final JSON while tests are failing."
            )

            provider = self.provider
            max_retries = self.max_retries
            max_tool_iterations = self.max_tool_iterations

            async def _run_fn(prompt: str) -> AgentResponse:
                return await run_agent(
                    contract_request,
                    WorkSpec,
                    provider,
                    prompt,
                    tools=tools,
                    final_response_type=WorkOutput,
                    adapter_spec=adapter,
                    max_retries=max_retries,
                    max_tool_iterations=max_tool_iterations,
                )

            validator = WorkOutputValidator(adapter, state_view, worktree_path)
            lifecycle = AttemptLifecycle(
                request=contract_request,
                state_view=state_view,
                validator=validator,
                run_fn=_run_fn,
                registry=self.registry,
                critic_provider=self.critic_provider,
                referee_provider=self.referee_provider,
                max_attempts=self.max_attempts,
                telemetry_sink=self.telemetry_sink,
                run_id=getattr(self.telemetry_sink, "run_id", None),
                initial_revision=contract_request.initial_revision,
            )

            try:
                response = await lifecycle.run(base_prompt)
            except RunAgentFailed as e:
                response = e.response

            if response.status == ResponseStatus.COMPLETED:
                if _worktree_has_changes(worktree_path):
                    preserve_worktree_for_integration = True
                    if state_view.version_sha and not response.dispatch_sha:
                        return response.model_copy(update={"dispatch_sha": state_view.version_sha})
                    return response
                if adapter.requires_nonempty_output:
                    return AgentResponse(
                        request_id=request.id,
                        status=ResponseStatus.FAILED,
                        failure_kind=FailureKind.VALIDATION_REJECTED,
                        error="worker completed without producing worktree changes",
                        output=response.output,
                        ran_tests_and_passed=response.ran_tests_and_passed,
                    )
                return AgentResponse(
                    request_id=request.id,
                    status=ResponseStatus.ALREADY_DONE,
                    output=response.output,
                    ran_tests_and_passed=response.ran_tests_and_passed,
                )

            return response

        finally:
            if not preserve_worktree_for_integration:
                self.workspace.remove_worktree(spec.artifact, node_id)


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
    telemetry_sink: TelemetrySink | None = None,
) -> AgentResponse:
    """Run the agentic tool loop for a work request using the specified adapter and artifact."""
    return await WorkTaskExecutor(
        registry=registry,
        workspace=workspace,
        language_registry=language_registry,
        provider=provider,
        max_retries=max_retries,
        max_tool_iterations=max_tool_iterations,
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        max_attempts=max_attempts,
        telemetry_sink=telemetry_sink,
    ).run(request, state_view)
