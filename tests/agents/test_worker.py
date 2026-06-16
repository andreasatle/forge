"""Tests for worker agent prompt assembly and response wrapping."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.base import PromptBuilder
from forge.agents.planner import PlannerTaskExecutor
from forge.agents.worker import WorkTaskExecutor, work_agent
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentMessageKind,
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    FailureKind,
    FileView,
    GraphSplitDecision,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    StateView,
    WorkOutput,
    WorkSpec,
    render_agent_contract,
)
from forge.core.plan_expansion import PlanExpansionBuilder
from forge.core.state_service import StateService
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin, LanguageRegistry
from forge.tools.registry import ToolRegistry

MUTATING_TOOL_NAMES = {"write_file", "replace_in_file"}
PYTHON_PROMPT_WORDS = (
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "pythonpath",
    "pytest",
    "Use uv",
    "Always place source code under src/.",
    "Imports behave as if src/",
)


def _tool_names(registry: ToolRegistry) -> set[str]:
    return {tool.name for tool in registry}


def _available_tool_names(system_prompt: str) -> set[str]:
    return set(re.findall(r"^  ([a-z_]+): ", system_prompt, flags=re.MULTILINE))


def _registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(
        AdapterSpec(
            name="coding",
            description="test",
            tools=["write_file", "replace_in_file"],
            prompt_template="do: {objective}\nsuccess: {success_condition}",
        )
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
    lr.register(
        LanguagePlugin(
            name=name,
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            prompt_supplement="",
            work_output_example="",
        )
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


def _assert_no_python_prompt_words(prompt: str) -> None:
    for word in PYTHON_PROMPT_WORDS:
        assert word not in prompt


async def test_work_task_executor_runs_simple_work_task_successfully(tmp_path: Path) -> None:
    """WorkTaskExecutor runs a work task and returns the engine response."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    work_output = WorkOutput(summary="Completed worktree changes.")
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
        )
        response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.COMPLETED
    assert response.output == work_output


async def test_work_task_executor_accepts_write_file_then_metadata_only_work_output(
    tmp_path: Path,
) -> None:
    """WorkTaskExecutor accepts a mutating write_file call followed by metadata only."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(
        side_effect=[
            (
                '{"kind": "tool", "name": "write_file", '
                '"arguments": {"path": "src/main.py", "content": "print(42)\\n"}}'
            ),
            (
                '{"kind":"final","output":{"kind":"work_output",'
                '"summary":"Wrote src/main.py in the worktree.","base_version":"0"}}'
            ),
        ]
    )
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock()
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
        state_service=ss,
    )

    response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.COMPLETED
    assert response.output == WorkOutput(
        kind=AgentMessageKind.WORK_OUTPUT,
        summary="Wrote src/main.py in the worktree.",
    )
    ss.apply_work_output.assert_called_once()


async def test_work_task_executor_enforces_adapter_tools(tmp_path: Path) -> None:
    """WorkTaskExecutor passes adapter-declared tools plus available verification tools."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    adapter = adapter_registry.get("coding")
    # With a language plugin, verification tools (run_tests) are also injected
    expected = set(adapter.tools) | set(adapter.verification_tools)
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    executor = WorkTaskExecutor(
        registry=adapter_registry,
        workspace=workspace,
        language_registry=_language_registry_with_tests(),
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await executor.run(request, _state_view(language="python"))

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_work_task_executor_unknown_adapter_tool_returns_failed(tmp_path: Path) -> None:
    """WorkTaskExecutor returns FAILED when an adapter declares an unknown tool."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = AdapterRegistry()
    adapter_registry.register(
        AdapterSpec(
            name="broken",
            description="test",
            tools=["nonexistent_tool"],
            prompt_template="do: {objective}\nsuccess: {success_condition}",
        )
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
    executor = WorkTaskExecutor(
        registry=adapter_registry,
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=MagicMock(),
    )

    response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.FAILED
    assert "nonexistent_tool" in (response.error or "")


async def test_work_task_executor_python_language_supplement_appears_in_prompt(
    tmp_path: Path,
) -> None:
    """WorkTaskExecutor includes the language plugin prompt supplement."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    language_registry = LanguageRegistry()
    language_registry.register(
        LanguagePlugin(
            name="python",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            prompt_supplement="UNIQUE_EXECUTOR_SUPPLEMENT",
            work_output_example="",
        )
    )
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=language_registry,
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await executor.run(request, _state_view(language="python"))

    user_prompt = mock_run_agent.call_args.args[3]
    assert "UNIQUE_EXECUTOR_SUPPLEMENT" in user_prompt


async def test_work_producer_prompt_includes_canonical_contract_block(
    tmp_path: Path,
) -> None:
    """Worker producer prompt includes the same canonical AgentRequest contract block."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="inspect code",
            success_condition="report changes",
            contract=AgentContract(
                objective="inspect code",
                success_condition="report changes",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="names files")],
                constraints=["avoid rewrites"],
                non_goals=["format unrelated files"],
            ),
            adapter="coding",
            artifact="codebase",
            language="python",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=_language_registry_with_tests(),
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await executor.run(request, _state_view(language="python"))

    user_prompt = mock_run_agent.call_args.args[3]
    assert render_agent_contract(request) in user_prompt
    assert "Produce output satisfying this contract." in user_prompt


async def test_work_task_executor_prompts_for_worktree_mutation(tmp_path: Path) -> None:
    """WorkTaskExecutor tells workers to mutate the assigned worktree."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await executor.run(request, _state_view())

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Modify files directly in the assigned worktree" in user_prompt
    assert "git status and git diff" in user_prompt
    assert "Workers are read-only" not in user_prompt


async def test_worker_prompt_tool_mentions_match_registry(tmp_path: Path) -> None:
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = PromptBuilder(tools, WorkOutput).build()
    tool_names = _tool_names(tools)

    assert _available_tool_names(system_prompt) == tool_names
    for unavailable in MUTATING_TOOL_NAMES - tool_names:
        assert unavailable not in system_prompt
        assert unavailable not in user_prompt


async def test_worker_prompt_leaves_generic_mechanics_to_base(tmp_path: Path) -> None:
    """worker.py does not duplicate generic tool, JSON, or schema mechanics."""
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = PromptBuilder(tools, WorkOutput).build()
    schema_system_prompt = PromptBuilder(tools, WorkOutput, always_show_final=True).build()

    assert '"kind":"tool"' in system_prompt
    assert "Generated JSON schema" in schema_system_prompt
    assert "JSON only" in system_prompt
    assert "tool_call" not in user_prompt
    assert "Generated JSON schema" not in user_prompt
    assert "JSON only" not in user_prompt
    assert "final JSON response" not in user_prompt
    assert "Produce ALL" not in user_prompt


async def test_worker_prompt_uses_git_native_mutation_policy(tmp_path: Path) -> None:
    """worker.py tells workers to modify the worktree instead of proposing file payloads."""
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Modify files directly in the assigned worktree" in user_prompt
    assert "complete file contents in your final response" in user_prompt
    assert "Workers are read-only" not in user_prompt


async def test_worker_prompt_includes_state_version_and_file_context(tmp_path: Path) -> None:
    """worker.py still includes StateView version and existing file context."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="codebase",
        language=None,
        files=[FileView(path="src/app.py", content="print('hi')")],
        dependencies=[],
        version=7,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider, state_view)

    user_prompt = mock_run_agent.call_args.args[3]
    assert "State version: 7" in user_prompt
    assert "Existing files in 'codebase'" in user_prompt
    assert "File: src/app.py" in user_prompt
    assert "print('hi')" in user_prompt


async def test_coding_adapter_receives_exactly_declared_tools(tmp_path: Path) -> None:
    """work_agent passes adapter tools plus available verification tools for coding.yaml."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    adapter = adapter_registry.get("coding")
    # With a language plugin, run_tests is injected from verification_tools
    expected = set(adapter.tools) | set(adapter.verification_tools)
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
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


async def test_document_adapter_receives_exactly_declared_tools(tmp_path: Path) -> None:
    """work_agent passes exactly the tools declared in document.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("document").tools)
    request = _work_request("document")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_audit_adapter_receives_exactly_declared_tools(tmp_path: Path) -> None:
    """work_agent passes exactly the tools declared in audit.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("audit").tools)
    request = _work_request("audit")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools) == expected


async def test_audit_adapter_does_not_receive_run_tests(tmp_path: Path) -> None:
    """audit adapter receives declared file tools but not run_tests."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    request = _work_request("audit")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, adapter_registry, workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert "list_files" in _tool_names(tools)
    assert "write_file" in _tool_names(tools)
    assert "run_tests" not in _tool_names(tools)


async def test_worker_prompt_describes_git_native_work_output(tmp_path: Path) -> None:
    """coding.yaml tells workers that WorkOutput is completion metadata."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            _language_registry_with_tests(),
            provider,
            _state_view(language="python"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Modify files directly in the assigned worktree" in user_prompt
    assert "final WorkOutput is completion metadata only" in user_prompt


async def test_language_not_appended_when_no_plugin(tmp_path: Path) -> None:
    """work_agent omits plugin language context when no language plugin is configured."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Language: not specified" in user_prompt
    assert "Language: python" not in user_prompt


async def test_worker_prompt_uses_existing_files_not_codebase(tmp_path: Path) -> None:
    """work_agent prompt refers to 'existing files', not 'existing codebase'."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="codebase",
        language=None,
        files=[FileView(path="README.md", content="hello")],
        dependencies=[],
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider, state_view)

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Existing files" in user_prompt
    assert "existing codebase" not in user_prompt


async def test_language_supplement_appears_in_worker_prompt(tmp_path: Path) -> None:
    """work_agent injects language plugin guidance into the canonical contract."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    lr = LanguageRegistry()
    lr.register(
        LanguagePlugin(
            name="python",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            prompt_supplement="UNIQUE_SUPPLEMENT_MARKER",
            work_output_example="",
        )
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request, _registry(), workspace, lr, provider, _state_view(language="python")
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "UNIQUE_SUPPLEMENT_MARKER" in user_prompt
    assert "Language plugin guidance:" in user_prompt
    producer_request = mock_run_agent.call_args.args[0]
    assert isinstance(producer_request.spec, WorkSpec)
    assert any(
        "UNIQUE_SUPPLEMENT_MARKER" in constraint
        for constraint in producer_request.spec.contract.constraints
    )


async def test_producer_critic_and_referee_receive_same_plugin_guidance(
    tmp_path: Path,
) -> None:
    """The worker promotes plugin guidance into the request shared by producer and reviewers."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="toy")
    provider = MagicMock()
    provider.max_tokens = 8192
    language_registry = LanguageRegistry()
    language_registry.register(
        LanguagePlugin(
            name="toy",
            init_command="toy init",
            test_command="toy test",
            sync_command="toy sync",
            prompt_supplement="TOY_REVIEWER_CONTRACT_GUIDANCE",
            work_output_example='{{"files": [{{"path": "module.toy", "content": "ok"}}]}}',
        )
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.attempt.critic_agent", new_callable=AsyncMock) as mock_critic,
        patch("forge.agents.attempt.referee_agent", new_callable=AsyncMock) as mock_referee,
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        mock_critic.return_value = CriticFinding(
            disposition=CriticDisposition.ACCEPT,
            rationale="meets contract",
            hints=[],
        )
        mock_referee.return_value = RefereeDecision(
            disposition=CriticDisposition.ACCEPT,
            rationale="meets contract",
            override=False,
        )

        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            language_registry,
            provider,
            _state_view(language="toy"),
            critic_provider=MagicMock(),
            referee_provider=MagicMock(),
        )

    producer_request = mock_run_agent.call_args.args[0]
    critic_request = mock_critic.call_args.args[0]
    referee_request = mock_referee.call_args.args[0]
    assert producer_request == critic_request == referee_request
    assert isinstance(producer_request.spec, WorkSpec)
    assert any(
        "TOY_REVIEWER_CONTRACT_GUIDANCE" in constraint
        for constraint in producer_request.spec.contract.constraints
    )


async def test_python_worker_prompt_includes_packaging_guidance(tmp_path: Path) -> None:
    """Python coding workers receive packaging guidance from the language supplement."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    language_registry = LanguageRegistry()
    language_registry.load(Path(__file__).parents[2] / "languages")

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            language_registry,
            provider,
            _state_view(language="python"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "pyproject.toml" in user_prompt
    assert "uv" in user_prompt
    assert "requirements.txt" in user_prompt
    assert "setup.py" in user_prompt


async def test_coding_worker_prompt_without_language_plugin_is_language_agnostic(
    tmp_path: Path,
) -> None:
    """Core worker prompt assembly does not inject Python-specific wording without a plugin."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            request,
            _registry(),
            workspace,
            LanguageRegistry(),
            provider,
            _state_view(),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    _assert_no_python_prompt_words(user_prompt)


async def test_python_specific_instructions_come_from_python_plugin_only(tmp_path: Path) -> None:
    """Python wording appears when, and because, the Python language plugin is configured."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    language_registry = LanguageRegistry()
    language_registry.load(Path(__file__).parents[2] / "languages")

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            language_registry,
            provider,
            _state_view(language="python"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    python_supplement = language_registry.get("python").prompt_supplement
    assert python_supplement in user_prompt
    assert "pyproject.toml" in user_prompt
    assert "requirements.txt" in user_prompt


async def test_non_python_language_plugin_does_not_receive_python_wording(
    tmp_path: Path,
) -> None:
    """A non-Python language plugin can render its own text without Python conventions leaking in."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="toy")
    provider = MagicMock()
    provider.max_tokens = 8192
    language_registry = LanguageRegistry()
    language_registry.register(
        LanguagePlugin(
            name="toy",
            init_command="toy init",
            test_command="toy test",
            sync_command="toy sync",
            prompt_supplement="TOY_LANGUAGE_SUPPLEMENT",
            work_output_example='{{"files": [{{"path": "module.toy", "content": "ok"}}]}}',
        )
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            language_registry,
            provider,
            _state_view(language="toy"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "TOY_LANGUAGE_SUPPLEMENT" in user_prompt
    assert "module.toy" in user_prompt
    _assert_no_python_prompt_words(user_prompt)


async def test_language_work_output_example_appears_in_worker_prompt(tmp_path: Path) -> None:
    """work_agent renders language-specific conventions into the prompt."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192
    lr = LanguageRegistry()
    lr.register(
        LanguagePlugin(
            name="python",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            prompt_supplement="",
            work_output_example='{{"path": "WORK_OUTPUT_EXAMPLE_MARKER"}}',
        )
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
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
    assert "Language-specific worktree conventions" in user_prompt
    assert "Example of a valid WorkOutput response" not in user_prompt
    assert "WORK_OUTPUT_EXAMPLE_MARKER" in user_prompt


async def test_unknown_tool_in_adapter_returns_failed_response(tmp_path: Path) -> None:
    """work_agent returns FAILED with a clear message when adapter declares an unknown tool."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = AdapterRegistry()
    adapter_registry.register(
        AdapterSpec(
            name="broken",
            description="test",
            tools=["nonexistent_tool"],
            prompt_template="do: {objective}\nsuccess: {success_condition}",
        )
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


async def test_run_agent_failure_propagates_as_failed_response(tmp_path: Path) -> None:
    """work_agent returns the failed AgentResponse when AttemptLifecycle raises RunAgentFailed."""
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

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run.return_value = failed_response
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.PROVIDER_ERROR
    assert response.error == "provider error"


async def test_successful_engine_result_wrapped_in_completed_response(tmp_path: Path) -> None:
    """work_agent returns AgentResponse(COMPLETED, output=...) for a successful engine run."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    work_output = WorkOutput(summary="Completed worktree changes.")

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, output=work_output
        )
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.output == work_output


async def test_worker_prompt_shows_state_version_without_base_version_instruction(
    tmp_path: Path,
) -> None:
    """worker.py shows State version for context but does NOT instruct model to set base_version."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="codebase",
        language=None,
        files=[],
        dependencies=[],
        version=42,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider, state_view)

    user_prompt = mock_run_agent.call_args.args[3]
    assert "State version: 42" in user_prompt
    assert "base_version" not in user_prompt


async def test_worker_prompt_includes_base_commit_sha_for_context(tmp_path: Path) -> None:
    """worker.py shows Base commit SHA for context but does NOT ask model to echo it back."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="codebase",
        language=None,
        files=[],
        dependencies=[],
        version=3,
        version_sha="deadbeef1234abcd",
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider, state_view)

    user_prompt = mock_run_agent.call_args.args[3]
    assert "State version: 3" in user_prompt
    assert "Base commit: deadbeef1234abcd" in user_prompt
    assert "base_version" not in user_prompt


async def test_document_adapter_writes_file_then_returns_metadata_only_work_output(
    tmp_path: Path,
) -> None:
    """Document adapter worker writes docs to worktree then returns metadata-only WorkOutput."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("docs")
    adapter_registry = _yaml_adapter_registry()
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write API documentation",
            success_condition="API docs are complete",
            adapter="document",
            artifact="docs",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(
        side_effect=[
            (
                '{"kind": "tool", "name": "write_file", '
                '"arguments": {"path": "README.md", "content": "# API Docs\\n\\nThis is the API documentation.\\n"}}'
            ),
            (
                '{"kind":"final","output":{"kind":"work_output",'
                '"summary":"Created README.md with API documentation.","base_version":"0"}}'
            ),
        ]
    )
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock()
    executor = WorkTaskExecutor(
        registry=adapter_registry,
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
        state_service=ss,
    )

    response = await executor.run(request, _state_view("docs"))

    assert response.status == ResponseStatus.COMPLETED
    output = response.output
    assert isinstance(output, WorkOutput)
    assert output.summary == "Created README.md with API documentation."
    ss.apply_work_output.assert_called_once()
    assert "# API Docs" not in output.summary


async def test_document_adapter_prompt_instructs_write_file_and_json_only_response(
    tmp_path: Path,
) -> None:
    """document.yaml prompt tells workers to use write_file and return JSON-only final response."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("docs")
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write API documentation",
            success_condition="API docs are complete",
            adapter="document",
            artifact="docs",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Created README.md with API documentation."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            LanguageRegistry(),
            provider,
            _state_view("docs"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "write_file" in user_prompt
    assert "final WorkOutput is completion metadata only" in user_prompt
    assert "do not include markdown documentation in the final response" in user_prompt
    assert "JSON only" in user_prompt
    assert "summary should briefly describe" in user_prompt


async def test_document_adapter_prompt_includes_concrete_work_output_example(
    tmp_path: Path,
) -> None:
    """document.yaml prompt includes a concrete WorkOutput example without base_version."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("docs")
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write API documentation",
            success_condition="API docs are complete",
            adapter="document",
            artifact="docs",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="docs", language=None, files=[], dependencies=[], version=0
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Created README.md."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            LanguageRegistry(),
            provider,
            state_view,
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert '"kind":"work_output"' in user_prompt
    assert "base_version" not in user_prompt
    assert '"..."' not in user_prompt


async def test_document_adapter_prompt_prohibits_document_content_in_final_response(
    tmp_path: Path,
) -> None:
    """document.yaml explicitly tells the model not to put document/README content in the final response."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("docs")
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write a README",
            success_condition="README is written",
            adapter="document",
            artifact="docs",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Created README.md."),
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            LanguageRegistry(),
            provider,
            _state_view("docs"),
        )

    user_prompt = mock_run_agent.call_args.args[3]
    assert "Never put README contents" in user_prompt
    assert "do not include markdown documentation in the final response" in user_prompt
    assert "metadata only" in user_prompt


async def test_version_zero_prompt_has_no_sha_and_no_base_version_instruction(
    tmp_path: Path,
) -> None:
    """When version_sha is absent, neither prompt mentions a commit SHA or base_version."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    state_view = StateView(
        artifact_name="codebase",
        language=None,
        files=[],
        dependencies=[],
        version=0,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(request, _registry(), workspace, LanguageRegistry(), provider, state_view)

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = PromptBuilder(tools, WorkOutput, always_show_final=True).build()
    assert "Base commit:" not in user_prompt
    assert "base_version" not in user_prompt
    assert "base_version" not in system_prompt


async def test_worker_uses_work_output_as_final_response_type(tmp_path: Path) -> None:
    """WorkTaskExecutor passes final_response_type=WorkOutput to run_agent."""
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs.get("final_response_type") is WorkOutput


# --- Worktree ownership tests ---


async def test_worktree_removed_after_successful_integration(tmp_path: Path) -> None:
    """WorkTaskExecutor removes the worktree after successful integration."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    removed: list[tuple[str, str]] = []
    real_remove = workspace.remove_worktree

    def tracking_remove(artifact: str, node_id: str) -> None:
        removed.append((artifact, node_id))
        real_remove(artifact, node_id)

    workspace.remove_worktree = tracking_remove  # type: ignore[method-assign]
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock()
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
        state_service=ss,
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="done"),
        )
        await executor.run(request, _state_view())

    assert len(removed) == 1
    assert removed[0][0] == "codebase"


async def test_worktree_removed_after_integration_failure(tmp_path: Path) -> None:
    """WorkTaskExecutor removes the worktree even when integration raises."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    removed: list[tuple[str, str]] = []
    real_remove = workspace.remove_worktree

    def tracking_remove(artifact: str, node_id: str) -> None:
        removed.append((artifact, node_id))
        real_remove(artifact, node_id)

    workspace.remove_worktree = tracking_remove  # type: ignore[method-assign]
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("tests failed"))
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
        state_service=ss,
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="done"),
        )
        response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED
    assert len(removed) == 1


async def test_worktree_removed_after_engine_exception(tmp_path: Path) -> None:
    """WorkTaskExecutor removes the worktree even when the engine raises unexpectedly."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    removed: list[tuple[str, str]] = []
    real_remove = workspace.remove_worktree

    def tracking_remove(artifact: str, node_id: str) -> None:
        removed.append((artifact, node_id))
        real_remove(artifact, node_id)

    workspace.remove_worktree = tracking_remove  # type: ignore[method-assign]
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.side_effect = Exception("unexpected crash")
        try:
            await executor.run(request, _state_view())
        except Exception:
            pass

    assert len(removed) == 1


async def test_worktree_removed_when_no_changes_produced(tmp_path: Path) -> None:
    """WorkTaskExecutor removes the worktree on the ALREADY_DONE path (no changes)."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="inspect code",
            success_condition="report changes",
            adapter="readonly",
            artifact="codebase",
        ),
    )
    provider = MagicMock()
    provider.max_tokens = 8192
    removed: list[tuple[str, str]] = []
    real_remove = workspace.remove_worktree

    def tracking_remove(artifact: str, node_id: str) -> None:
        removed.append((artifact, node_id))
        real_remove(artifact, node_id)

    workspace.remove_worktree = tracking_remove  # type: ignore[method-assign]
    registry = AdapterRegistry()
    registry.register(
        AdapterSpec(
            name="readonly",
            description="read-only adapter",
            tools=["write_file", "replace_in_file"],
            prompt_template="do: {objective}\nsuccess: {success_condition}",
            requires_nonempty_output=False,
        )
    )
    executor = WorkTaskExecutor(
        registry=registry,
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=False),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.ALREADY_DONE
    assert len(removed) == 1


async def test_state_service_never_removes_worktree(tmp_path: Path) -> None:
    """StateService.apply_work_output does not call workspace.remove_worktree."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    removed: list[tuple[str, str]] = []
    real_remove = workspace.remove_worktree

    def tracking_remove(artifact: str, node_id: str) -> None:
        removed.append((artifact, node_id))
        real_remove(artifact, node_id)

    workspace.remove_worktree = tracking_remove  # type: ignore[method-assign]
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock()
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
        state_service=ss,
    )

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="done"),
        )
        await executor.run(request, _state_view())

    ss.apply_work_output.assert_called_once()
    assert len(removed) == 1


# --- Adapter/tool semantic mismatch tests ---


async def test_coding_adapter_starts_without_failing_when_no_language_plugin(
    tmp_path: Path,
) -> None:
    """coding adapter can start a work node when no language plugin is set (run_tests not available)."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        response = await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            LanguageRegistry(),
            provider,
            _state_view(),
        )

    assert response.status != ResponseStatus.FAILED or "unknown tool" not in (response.error or "")
    assert mock_run_agent.called
    tools = mock_run_agent.call_args.kwargs["tools"]
    assert "run_tests" not in _tool_names(tools)


async def test_coding_adapter_with_language_injects_run_tests(tmp_path: Path) -> None:
    """coding adapter injects run_tests from verification_tools when a language plugin provides it."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _work_request("coding", language="python")
    provider = MagicMock()
    provider.max_tokens = 8192

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
        )
        await work_agent(
            request,
            _yaml_adapter_registry(),
            workspace,
            _language_registry_with_tests(),
            provider,
            _state_view(language="python"),
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert "run_tests" in _tool_names(tools)


async def test_worker_receives_planner_normalized_language_guidance_and_run_tests(
    tmp_path: Path,
) -> None:
    """Planner-normalized coding tasks reach workers with language-specific guidance/tools."""
    plan_request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build scraper"),
    )
    planner_provider = MagicMock()
    planner_provider.max_tokens = 8192
    planner_provider.chat = AsyncMock(
        return_value=(
            '{"kind":"final","output":{"kind":"split_graph","nodes":['
            '{"id":"scraper","task":{"objective":"Implement scraper",'
            '"success_condition":"pytest passes","adapter":"coding",'
            '"artifact":"codebase"},"depends_on":[]}'
            "]}}"
        )
    )
    planner = PlannerTaskExecutor(
        provider=planner_provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
        artifact_types={"codebase": "coding"},
    )

    plan_response = await planner.run(plan_request)

    assert isinstance(plan_response.output, GraphSplitDecision)
    work_request = PlanExpansionBuilder(plan_request).build_from_decision(plan_response.output)[0]
    assert isinstance(work_request.spec, WorkSpec)
    assert work_request.spec.language == "python"

    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    language_registry = LanguageRegistry()
    language_registry.register(
        LanguagePlugin(
            name="python",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            prompt_supplement="PYTHON_NORMALIZED_GUIDANCE",
            work_output_example="PYTHON_NORMALIZED_EXAMPLE",
        )
    )
    provider = MagicMock()
    provider.max_tokens = 8192

    with (
        patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent,
        patch("forge.agents.worker._worktree_has_changes", return_value=True),
    ):
        mock_run_agent.return_value = AgentResponse(
            request_id=work_request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )
        await work_agent(
            work_request,
            _yaml_adapter_registry(),
            workspace,
            language_registry,
            provider,
            _state_view(language="python"),
        )

    producer_request = mock_run_agent.call_args.args[0]
    assert isinstance(producer_request.spec, WorkSpec)
    assert producer_request.spec.language == "python"
    user_prompt = mock_run_agent.call_args.args[3]
    assert "PYTHON_NORMALIZED_GUIDANCE" in user_prompt
    assert "PYTHON_NORMALIZED_EXAMPLE" in user_prompt
    tools = mock_run_agent.call_args.kwargs["tools"]
    assert "run_tests" in _tool_names(tools)
