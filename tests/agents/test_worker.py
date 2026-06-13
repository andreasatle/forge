"""Tests for worker agent prompt assembly and response wrapping."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.base import PromptBuilder
from forge.agents.worker import WorkTaskExecutor, work_agent
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FailureKind,
    FileView,
    FileWrite,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    StateView,
    WorkSpec,
    render_agent_contract,
)
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin, LanguageRegistry
from forge.tools.registry import ToolRegistry

BLACKBOARD_TOOL_NAMES = {"read_blackboard", "write_blackboard"}
MUTATING_TOOL_NAMES = {"write_file", "replace_in_file", "add_dependency", "write_blackboard"}
PYTHON_PROMPT_WORDS = (
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "pythonpath",
    "BeautifulSoup",
    "from scraper import",
    "pytest",
    "requests",
    "coverage",
    "__init__.py",
    "Use uv",
    "Always place source code under src/.",
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
            tools=[],
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
            package_manager="uv",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            add_dependency_command="uv add {package}",
            project_structure=[],
            prompt_supplement="",
            delta_example="",
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
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])
    executor = WorkTaskExecutor(
        registry=_registry(),
        workspace=workspace,
        language_registry=LanguageRegistry(),
        provider=provider,
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        response = await executor.run(request, _state_view())

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == delta


async def test_work_task_executor_enforces_adapter_tools(tmp_path: Path) -> None:
    """WorkTaskExecutor passes only the tools declared by the adapter."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("coding").tools)
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
            package_manager="uv",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            add_dependency_command="uv add {package}",
            project_structure=[],
            prompt_supplement="UNIQUE_EXECUTOR_SUPPLEMENT",
            delta_example="",
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


async def test_work_task_executor_keeps_read_only_policy(tmp_path: Path) -> None:
    """WorkTaskExecutor keeps the worker read-only policy in the prompt."""
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
    assert "Workers are read-only" in user_prompt
    assert "do not attempt to write files via tools" in user_prompt


async def test_worker_tools_do_not_include_blackboard_tools(tmp_path: Path) -> None:
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    assert _tool_names(tools).isdisjoint(BLACKBOARD_TOOL_NAMES)


async def test_worker_prompt_does_not_expose_blackboard_tools(tmp_path: Path) -> None:
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
        await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    tools = mock_run_agent.call_args.kwargs["tools"]
    user_prompt = mock_run_agent.call_args.args[3]
    system_prompt = PromptBuilder(tools, DeltaState).build()

    for tool_name in BLACKBOARD_TOOL_NAMES:
        assert tool_name not in system_prompt
        assert tool_name not in user_prompt


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
    system_prompt = PromptBuilder(tools, DeltaState).build()
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
    system_prompt = PromptBuilder(tools, DeltaState).build()
    schema_system_prompt = PromptBuilder(tools, DeltaState).build(
        DeltaState(new_files=[FileWrite(path="x.py", content="x")])
    )

    assert "tool_call" in system_prompt
    assert "Generated JSON schema" in schema_system_prompt
    assert "JSON only" in system_prompt
    assert "tool_call" not in user_prompt
    assert "Generated JSON schema" not in user_prompt
    assert "JSON only" not in user_prompt
    assert "final JSON response" not in user_prompt
    assert "Produce ALL" not in user_prompt


async def test_worker_prompt_keeps_read_only_policy(tmp_path: Path) -> None:
    """worker.py still tells workers to propose changes without mutating through tools."""
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
    assert "Workers are read-only" in user_prompt
    assert "do not attempt to write files via tools" in user_prompt
    assert "task result" in user_prompt


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


def test_production_adapter_yamls_expose_no_blackboard_tools() -> None:
    """No adapter YAML in the production adapters directory advertises blackboard tools."""
    adapters_dir = Path(__file__).parents[2] / "adapters"
    registry = AdapterRegistry()
    registry.load(adapters_dir)
    for name in registry.names():
        spec = registry.get(name)
        exposed = BLACKBOARD_TOOL_NAMES.intersection(spec.tools)
        assert not exposed, f"adapter '{name}' exposes blackboard tools: {exposed}"


async def test_coding_adapter_receives_exactly_declared_tools(tmp_path: Path) -> None:
    """work_agent passes exactly the tools declared in coding.yaml — no more, no less."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    adapter_registry = _yaml_adapter_registry()
    expected = set(adapter_registry.get("coding").tools)
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


async def test_audit_adapter_does_not_receive_list_files_or_run_tests(tmp_path: Path) -> None:
    """audit adapter declares only read_file — list_files and run_tests must not be exposed."""
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
    assert "list_files" not in _tool_names(tools)
    assert "run_tests" not in _tool_names(tools)


async def test_worker_prompt_warns_against_empty_delta(tmp_path: Path) -> None:
    """coding.yaml warns that empty DeltaState is always wrong."""
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
            delta=DeltaState(new_files=[FileWrite(path="x.py", content="x")]),
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
    assert "non-empty DeltaState" in user_prompt
    assert "empty DeltaState is always wrong" in user_prompt


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
            package_manager="uv",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            add_dependency_command="uv add {package}",
            project_structure=[],
            prompt_supplement="UNIQUE_SUPPLEMENT_MARKER",
            delta_example="",
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
            package_manager="toy-packages",
            init_command="toy init",
            test_command="toy test",
            sync_command="toy sync",
            add_dependency_command="toy add {package}",
            project_structure=["module.toy"],
            prompt_supplement="TOY_REVIEWER_CONTRACT_GUIDANCE",
            delta_example='{{"new_files": [{{"path": "module.toy", "content": "ok"}}], "base_version": {base_version}}}',
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
            delta=DeltaState(new_files=[FileWrite(path="module.toy", content="ok")]),
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
            delta=DeltaState(new_files=[FileWrite(path="x.py", content="x")]),
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
            delta=DeltaState(new_files=[FileWrite(path="main.txt", content="content")]),
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
            delta=DeltaState(new_files=[FileWrite(path="x.py", content="x")]),
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
            package_manager="toy-packages",
            init_command="toy init",
            test_command="toy test",
            sync_command="toy sync",
            add_dependency_command="toy add {package}",
            project_structure=["module.toy"],
            prompt_supplement="TOY_LANGUAGE_SUPPLEMENT",
            delta_example='{{"new_files": [{{"path": "module.toy", "content": "ok"}}], "base_version": {base_version}}}',
        )
    )

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta=DeltaState(new_files=[FileWrite(path="module.toy", content="ok")]),
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


async def test_language_delta_example_appears_in_worker_prompt(tmp_path: Path) -> None:
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
            package_manager="uv",
            init_command="uv init",
            test_command="pytest",
            sync_command="uv sync",
            add_dependency_command="uv add {package}",
            project_structure=[],
            prompt_supplement="",
            delta_example='{{"path": "DELTA_EXAMPLE_MARKER"}}',
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
    assert "Language-specific output conventions" in user_prompt
    assert "Example of a valid DeltaState response" not in user_prompt
    assert "DELTA_EXAMPLE_MARKER" in user_prompt


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

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = failed_response
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.PROVIDER_ERROR
    assert response.error == "provider error"


async def test_successful_engine_result_wrapped_in_completed_response(tmp_path: Path) -> None:
    """work_agent returns AgentResponse(COMPLETED, delta=...) for a successful engine run."""
    workspace = Workspace(tmp_path / "ws")
    workspace.init()
    workspace.init_artifact("codebase")
    request = _request()
    provider = MagicMock()
    provider.max_tokens = 8192
    delta = DeltaState(new_files=[FileWrite(path="main.py", content="code")])

    with patch("forge.agents.worker.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, delta=delta
        )
        response = await work_agent(
            request, _registry(), workspace, LanguageRegistry(), provider, _state_view()
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == delta


async def test_worker_prompt_includes_explicit_base_version_instruction(tmp_path: Path) -> None:
    """worker.py emits an explicit base_version instruction on the line after 'State version: N'."""
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
    assert "You MUST set base_version to 42 in your response." in user_prompt
    version_line_idx = user_prompt.index("State version: 42")
    must_line_idx = user_prompt.index("You MUST set base_version to 42 in your response.")
    assert must_line_idx > version_line_idx


async def test_worker_prompt_includes_base_commit_sha_instruction(tmp_path: Path) -> None:
    """worker.py emits 'Base commit: <sha>' and 'You MUST set base_version to <sha>' when version_sha is set."""
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
    assert "You MUST set base_version to deadbeef1234abcd in your response." in user_prompt
    assert "You MUST set base_version to 3 in your response." not in user_prompt
