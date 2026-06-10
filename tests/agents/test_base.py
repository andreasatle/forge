"""Tests for the run_agent base engine — plain chat loop with structured JSON parsing."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

from forge.agents.base import (
    PromptBuilder,
    ResponseParser,
    ToolError,
    ToolLoop,
    TrackedToolExecutor,
    _classify_failure,
    _merge_delta,
    run_agent,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    Edit,
    FailureKind,
    FileWrite,
    PlanResponse,
    RequestSource,
    ResponseStatus,
    ToolCallRequest,
    WorkSpec,
)
from forge.llm.providers import ProviderEmptyOutputError
from forge.tools.registry import Tool, ToolRegistry
from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    WriteFileRequest,
    WriteFileResponse,
)

_NONEMPTY_DELTA = (
    '{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}'
)


class _DoThingRequest(BaseModel):
    """Minimal no-field request used in agent base unit tests."""


class _DoThingResponse(BaseModel):
    """Minimal response carrying a result string used in agent base unit tests."""

    result: str


class _ExtendedResponse(BaseModel):
    """Synthetic response model used to prove prompt schema rendering is model-derived."""

    alpha: str
    beta: int


class _RequiredResponse(BaseModel):
    """Synthetic response model used to prove parser schema rejection."""

    required_value: str


def _work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write code",
            success_condition="tests pass",
            adapter="coding",
            artifact="main",
        ),
    )


def _mock_provider(chat_return: str = "{}") -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value=chat_return)
    return provider


def _make_registry() -> tuple[ToolRegistry, AsyncMock]:
    registry = ToolRegistry()
    mock_fn: AsyncMock = AsyncMock(return_value=_DoThingResponse(result="done"))
    registry.register(
        Tool(
            name="do_thing",
            description="does a thing",
            request_type=_DoThingRequest,
            response_type=_DoThingResponse,
            fn=mock_fn,
        )
    )
    return registry, mock_fn


def _make_write_file_tool() -> Tool:
    async def fn(req: WriteFileRequest) -> WriteFileResponse:
        return WriteFileResponse(path=req.path)

    return Tool(
        name="write_file",
        description="write a file",
        request_type=WriteFileRequest,
        response_type=WriteFileResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


def _make_replace_in_file_tool() -> Tool:
    async def fn(req: ReplaceInFileRequest) -> ReplaceInFileResponse:
        return ReplaceInFileResponse(path=req.path)

    return Tool(
        name="replace_in_file",
        description="replace in a file",
        request_type=ReplaceInFileRequest,
        response_type=ReplaceInFileResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


def _make_add_dependency_tool() -> Tool:
    async def fn(req: AddDependencyRequest) -> AddDependencyResponse:
        return AddDependencyResponse(package=req.package, success=True, output="")

    return Tool(
        name="add_dependency",
        description="add a dependency",
        request_type=AddDependencyRequest,
        response_type=AddDependencyResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


# --- ResponseParser ---


def test_response_parser_parses_tool_call_request():
    """ResponseParser returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = ResponseParser(DeltaState).parse(raw)
    assert isinstance(result, ToolCallRequest)
    assert result.name == "my_tool"


def test_response_parser_parses_valid_final_response():
    """ResponseParser returns the configured final response model for valid JSON."""
    raw = '{"required_value": "ok"}'
    result = ResponseParser(_RequiredResponse).parse(raw)
    assert isinstance(result, _RequiredResponse)
    assert result.required_value == "ok"


def test_response_parser_rejects_invalid_json():
    """ResponseParser raises ValueError with the existing invalid JSON message."""
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(DeltaState).parse("not json at all")


def test_response_parser_rejects_schema_invalid_final_response():
    """ResponseParser raises ValueError with the existing schema mismatch message."""
    with pytest.raises(ValueError, match="response does not match _RequiredResponse"):
        ResponseParser(_RequiredResponse).parse("{}")


def test_response_parser_correctly_parses_tool_call_request():
    """ResponseParser returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = ResponseParser(DeltaState).parse(raw)
    assert isinstance(result, ToolCallRequest)
    assert result.name == "my_tool"


def test_response_parser_correctly_parses_delta_state_as_final_response():
    """ResponseParser returns DeltaState when JSON matches DeltaState schema."""
    raw = '{"edits": [], "new_files": [], "dependencies": []}'
    result = ResponseParser(DeltaState).parse(raw)
    assert isinstance(result, DeltaState)


def test_response_parser_correctly_parses_plan_response_as_final_response():
    """ResponseParser returns PlanResponse when final_response_type is PlanResponse."""
    raw = '{"kind": "plan", "tasks": []}'
    result = ResponseParser(PlanResponse).parse(raw)
    assert isinstance(result, PlanResponse)
    assert result.tasks == []


def test_response_parser_raises_value_error_on_unknown_format():
    """ResponseParser raises ValueError when the response is not valid JSON."""
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(DeltaState).parse("not json at all")


# --- TrackedToolExecutor ---


async def test_tracked_tool_executor_executes_valid_tool():
    """TrackedToolExecutor returns a successful tool response for a valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})
    response, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1
    assert delta == DeltaState()


async def test_tracked_tool_executor_rejects_unknown_tool():
    """TrackedToolExecutor returns the existing failed response for an unknown tool."""
    registry = ToolRegistry()
    request = ToolCallRequest(kind="tool_call", name="nonexistent", arguments={})
    response, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert response.success is False
    assert response.error is not None
    assert delta == DeltaState()


async def test_tracked_tool_executor_validates_tool_arguments():
    """TrackedToolExecutor returns a failed response and unchanged delta for invalid arguments."""
    original_delta = DeltaState(dependencies=["existing"])
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="write_file",
        arguments={"path": "src/hello.py"},
    )
    response, delta = await TrackedToolExecutor(registry).execute(request, original_delta)
    assert response.success is False
    assert "content" in (response.error or "")
    assert delta == original_delta


async def test_tracked_tool_executor_tracks_write_file_delta():
    """TrackedToolExecutor tracks write_file calls in DeltaState."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="write_file",
        arguments={"path": "src/hello.py", "content": "print(1)\n"},
    )
    _, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert delta.new_files == [FileWrite(path="src/hello.py", content="print(1)\n")]
    assert delta.edits == []
    assert delta.dependencies == []


async def test_tracked_tool_executor_leaves_read_only_tool_delta_unchanged():
    """TrackedToolExecutor does not modify tracked_delta for non-mutating tools."""
    registry, _ = _make_registry()
    original_delta = DeltaState(dependencies=["existing"])
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})
    response, delta = await TrackedToolExecutor(registry).execute(request, original_delta)
    assert response.success is True
    assert delta == original_delta


async def test_tracked_tool_executor_returns_correct_tool_call_response():
    """TrackedToolExecutor returns (ToolCallResponse, delta) on valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})
    response, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1
    assert delta == DeltaState()


async def test_tracked_tool_executor_returns_error_tool_call_response_on_unknown_tool():
    """TrackedToolExecutor returns (error response, unchanged delta) for an unregistered tool."""
    registry = ToolRegistry()
    request = ToolCallRequest(kind="tool_call", name="nonexistent", arguments={})
    response, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert response.success is False
    assert response.error is not None
    assert delta == DeltaState()


async def test_tracked_tool_executor_tracks_write_file_in_new_files():
    """TrackedToolExecutor adds a FileWrite entry to tracked_delta when write_file succeeds."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="write_file",
        arguments={"path": "src/hello.py", "content": "print(1)\n"},
    )
    _, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert len(delta.new_files) == 1
    assert delta.new_files[0] == FileWrite(path="src/hello.py", content="print(1)\n")


async def test_tracked_tool_executor_write_file_overwrites_same_path():
    """Calling write_file twice for the same path keeps only the latest content."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    req1 = ToolCallRequest(
        kind="tool_call", name="write_file", arguments={"path": "a.py", "content": "v1"}
    )
    req2 = ToolCallRequest(
        kind="tool_call", name="write_file", arguments={"path": "a.py", "content": "v2"}
    )
    executor = TrackedToolExecutor(registry)
    _, after_first = await executor.execute(req1, DeltaState())
    _, after_second = await executor.execute(req2, after_first)
    assert len(after_second.new_files) == 1
    assert after_second.new_files[0].content == "v2"


async def test_tracked_tool_executor_tracks_replace_in_file_in_edits():
    """TrackedToolExecutor adds an Edit entry to tracked_delta when replace_in_file succeeds."""
    registry = ToolRegistry()
    registry.register(_make_replace_in_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="replace_in_file",
        arguments={"path": "src/main.py", "old": "x = 1", "new": "x = 2"},
    )
    _, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert len(delta.edits) == 1
    assert delta.edits[0] == Edit(path="src/main.py", old="x = 1", new="x = 2")


async def test_tracked_tool_executor_tracks_add_dependency_in_dependencies():
    """TrackedToolExecutor adds a package name to tracked_delta when add_dependency succeeds."""
    registry = ToolRegistry()
    registry.register(_make_add_dependency_tool())
    request = ToolCallRequest(
        kind="tool_call", name="add_dependency", arguments={"package": "requests"}
    )
    _, delta = await TrackedToolExecutor(registry).execute(request, DeltaState())
    assert delta.dependencies == ["requests"]


async def test_tracked_tool_executor_add_dependency_skips_duplicate():
    """add_dependency does not add the same package twice."""
    registry = ToolRegistry()
    registry.register(_make_add_dependency_tool())
    req = ToolCallRequest(
        kind="tool_call", name="add_dependency", arguments={"package": "requests"}
    )
    executor = TrackedToolExecutor(registry)
    _, after_first = await executor.execute(req, DeltaState())
    _, after_second = await executor.execute(req, after_first)
    assert after_second.dependencies == ["requests"]


async def test_tracked_tool_executor_returns_failed_response_when_replace_in_file_raises():
    """TrackedToolExecutor returns a failed ToolCallResponse and the original delta."""
    original_delta = DeltaState(dependencies=["existing"])
    registry = ToolRegistry()

    async def failing_fn(req: ReplaceInFileRequest) -> ReplaceInFileResponse:
        raise ValueError("old string not found in file")

    registry.register(
        Tool(
            name="replace_in_file",
            description="replace in a file",
            request_type=ReplaceInFileRequest,
            response_type=ReplaceInFileResponse,
            fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], failing_fn),
        )
    )
    request = ToolCallRequest(
        kind="tool_call",
        name="replace_in_file",
        arguments={"path": "src/main.py", "old": "missing", "new": "x = 2"},
    )
    response, delta = await TrackedToolExecutor(registry).execute(request, original_delta)

    assert response.success is False
    assert "not found" in (response.error or "")
    assert delta == original_delta


# --- _merge_delta ---


def test_merge_delta_combines_non_overlapping_entries():
    """_merge_delta unions new_files, edits, and dependencies when there is no overlap."""
    tracked = DeltaState(
        new_files=[FileWrite(path="a.py", content="a")],
        edits=[Edit(path="b.py", old="x", new="y")],
        dependencies=["requests"],
    )
    reported = DeltaState(
        new_files=[FileWrite(path="c.py", content="c")],
        edits=[Edit(path="d.py", old="p", new="q")],
        dependencies=["flask"],
    )
    merged = _merge_delta(tracked, reported)
    assert {fw.path for fw in merged.new_files} == {"a.py", "c.py"}
    assert {e.path for e in merged.edits} == {"b.py", "d.py"}
    assert set(merged.dependencies) == {"requests", "flask"}


def test_merge_delta_tracked_wins_on_path_conflict_in_new_files():
    """When tracked and reported both have new_files for the same path, tracked content wins."""
    tracked = DeltaState(new_files=[FileWrite(path="a.py", content="tracked")])
    reported = DeltaState(new_files=[FileWrite(path="a.py", content="reported")])
    merged = _merge_delta(tracked, reported)
    assert len(merged.new_files) == 1
    assert merged.new_files[0].content == "tracked"


def test_merge_delta_tracked_wins_on_path_conflict_in_edits():
    """When tracked and reported both have edits for the same path, tracked edit wins."""
    tracked = DeltaState(edits=[Edit(path="a.py", old="x", new="tracked")])
    reported = DeltaState(edits=[Edit(path="a.py", old="x", new="reported")])
    merged = _merge_delta(tracked, reported)
    assert len(merged.edits) == 1
    assert merged.edits[0].new == "tracked"


def test_merge_delta_deduplicates_dependencies():
    """_merge_delta does not duplicate a package present in both tracked and reported."""
    tracked = DeltaState(dependencies=["requests"])
    reported = DeltaState(dependencies=["requests", "flask"])
    merged = _merge_delta(tracked, reported)
    assert merged.dependencies.count("requests") == 1
    assert "flask" in merged.dependencies


# --- run_agent ---


async def test_tool_loop_returns_completed_final_response_without_tools():
    """ToolLoop returns COMPLETED for a valid final response without tool calls."""
    request = _work_request()
    provider = _mock_provider(_NONEMPTY_DELTA)

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=DeltaState,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 1


async def test_tool_loop_runs_tool_call_then_final_response():
    """ToolLoop executes one tool call and feeds the response back before final JSON."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_DELTA,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=DeltaState,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_tool_loop_retries_invalid_json():
    """ToolLoop retries invalid JSON and succeeds on the next response."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=["not valid json", _NONEMPTY_DELTA])

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=DeltaState,
        max_retries=3,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_tool_loop_rejects_empty_delta_for_work_agent():
    """ToolLoop preserves the empty DeltaState validation failure for WORK agents."""
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=DeltaState,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "empty delta" in (response.error or "")


async def test_run_agent_routes_tool_calls_correctly():
    """run_agent executes tool calls and feeds results back before the final response."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_DELTA,
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_returns_agent_response_on_final_response():
    """run_agent returns COMPLETED AgentResponse when the LLM returns valid non-empty final JSON."""
    request = _work_request()
    provider = _mock_provider(_NONEMPTY_DELTA)

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert isinstance(response, AgentResponse)
    assert response.status == ResponseStatus.COMPLETED


async def test_run_agent_retries_on_invalid_format():
    """run_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            "not valid json",
            _NONEMPTY_DELTA,
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=3)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_run_agent_returns_failed_after_max_iterations():
    """run_agent returns FAILED when the tool iteration limit is exhausted."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool_call", "name": "do_thing", "arguments": {}}')

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        max_tool_iterations=2,
        max_retries=0,
    )

    assert response.status == ResponseStatus.FAILED
    assert "exceeded" in (response.error or "")


async def test_run_agent_merges_tracked_write_file_into_delta():
    """write_file tool calls are automatically reflected in the final AgentResponse delta."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "write_file", "arguments": {"path": "src/app.py", "content": "x = 1"}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert len(response.delta.new_files) == 1
    assert response.delta.new_files[0] == FileWrite(path="src/app.py", content="x = 1")


async def test_run_agent_tracked_write_wins_over_llm_reported():
    """Framework-tracked content overrides LLM-reported content for the same path."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "write_file", "arguments": {"path": "a.py", "content": "tracked"}}',
            '{"edits": [], "new_files": [{"path": "a.py", "content": "reported"}], "dependencies": []}',
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.delta is not None
    assert len(response.delta.new_files) == 1
    assert response.delta.new_files[0].content == "tracked"


async def test_run_agent_merges_tracked_replace_in_file_into_delta():
    """replace_in_file tool calls are automatically reflected in the final AgentResponse delta."""
    registry = ToolRegistry()
    registry.register(_make_replace_in_file_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "replace_in_file", "arguments": {"path": "a.py", "old": "x=1", "new": "x=2"}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.delta is not None
    assert len(response.delta.edits) == 1
    assert response.delta.edits[0] == Edit(path="a.py", old="x=1", new="x=2")


async def test_run_agent_merges_tracked_add_dependency_into_delta():
    """add_dependency tool calls are automatically reflected in the final AgentResponse delta."""
    registry = ToolRegistry()
    registry.register(_make_add_dependency_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "add_dependency", "arguments": {"package": "httpx"}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.delta is not None
    assert response.delta.dependencies == ["httpx"]


async def test_run_agent_returns_completed_with_llm_reported_delta_after_read_tool_calls():
    """Workers using read-only tools: tracked_delta stays empty; LLM-reported DeltaState is returned."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_DELTA,
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert len(response.delta.new_files) == 1
    assert response.delta.new_files[0].path == "src/main.py"


# --- _classify_failure ---


def test_classify_failure_maps_value_error_to_invalid_json():
    """_classify_failure maps ValueError to FailureKind.INVALID_JSON."""
    assert _classify_failure(ValueError("bad json")) == FailureKind.INVALID_JSON


def test_classify_failure_maps_runtime_error_to_max_iterations():
    """_classify_failure maps RuntimeError to FailureKind.MAX_ITERATIONS."""
    assert _classify_failure(RuntimeError("exceeded")) == FailureKind.MAX_ITERATIONS


def test_classify_failure_maps_http_status_error_to_provider_error():
    """_classify_failure maps httpx.HTTPStatusError to FailureKind.PROVIDER_ERROR."""
    exc = httpx.HTTPStatusError(
        "error",
        request=httpx.Request("POST", "http://example.com"),
        response=httpx.Response(500),
    )
    assert _classify_failure(exc) == FailureKind.PROVIDER_ERROR


def test_classify_failure_maps_provider_empty_output_to_provider_error():
    """_classify_failure maps empty provider output to PROVIDER_ERROR, not INVALID_JSON."""
    failure_kind = _classify_failure(ProviderEmptyOutputError("empty content"))
    assert failure_kind == FailureKind.PROVIDER_ERROR


def test_classify_failure_maps_unknown_exception_to_unknown():
    """_classify_failure maps an unrecognized exception to FailureKind.UNKNOWN."""
    assert _classify_failure(KeyError("x")) == FailureKind.UNKNOWN


def test_classify_failure_maps_tool_error_to_tool_error():
    """_classify_failure maps ToolError to FailureKind.TOOL_ERROR, not INVALID_JSON."""
    assert _classify_failure(ToolError("replace_in_file failed")) == FailureKind.TOOL_ERROR


async def test_run_agent_sets_failure_kind_invalid_json_on_retry_exhaustion():
    """run_agent sets failure_kind=INVALID_JSON when JSON parse retries are exhausted."""
    request = _work_request()
    provider = _mock_provider("not valid json")

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=0)

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INVALID_JSON


async def test_run_agent_sets_failure_kind_max_iterations_on_loop_exhaustion():
    """run_agent sets failure_kind=MAX_ITERATIONS when the tool loop is exhausted."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool_call", "name": "do_thing", "arguments": {}}')

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        max_tool_iterations=2,
        max_retries=0,
    )

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.MAX_ITERATIONS


async def test_run_agent_sets_failure_kind_provider_error_on_http_error():
    """run_agent sets failure_kind=PROVIDER_ERROR when the provider raises HTTPStatusError."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("POST", "http://example.com"),
            response=httpx.Response(500),
        )
    )

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.PROVIDER_ERROR


# --- run_agent only uses provider.chat ---


async def test_run_agent_never_calls_chat_with_tools_no_tools_path():
    """run_agent uses provider.chat only, never chat_with_tools, when no tools are registered."""
    request = _work_request()
    provider = _mock_provider(_NONEMPTY_DELTA)
    provider.chat_with_tools = AsyncMock(
        side_effect=AssertionError("chat_with_tools must not be called")
    )

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert response.status == ResponseStatus.COMPLETED


async def test_run_agent_never_calls_chat_with_tools_tool_loop_path():
    """run_agent uses provider.chat only, never chat_with_tools, on the tool loop path."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_DELTA,
        ]
    )
    provider.chat_with_tools = AsyncMock(
        side_effect=AssertionError("chat_with_tools must not be called")
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED


# --- PromptBuilder ---


def test_prompt_builder_builds_prompt_with_tools():
    """PromptBuilder includes tool-call instructions when tools are configured."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, DeltaState).build()
    assert "You have two valid response formats" in prompt
    assert "tool_call" in prompt
    assert "do_thing" in prompt


def test_prompt_builder_builds_prompt_without_tools():
    """PromptBuilder omits tool-call instructions when tools are not configured."""
    prompt = PromptBuilder(None, DeltaState).build()
    assert "JSON only" in prompt
    assert "tool_call" not in prompt
    assert "Generated JSON schema" in prompt


def test_prompt_builder_includes_generated_pydantic_schema():
    """PromptBuilder renders schema from the configured final response model."""
    prompt = PromptBuilder(None, _ExtendedResponse).build()
    assert "Final response model: _ExtendedResponse" in prompt
    assert "alpha" in prompt
    assert "beta" in prompt
    assert "Generated JSON schema" in prompt


def test_prompt_builder_uses_tool_registry_descriptions():
    """PromptBuilder uses tool descriptions from the provided ToolRegistry."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, DeltaState).build()
    assert "do_thing: does a thing" in prompt


def test_prompt_builder_preserves_tracked_delta_schema_visibility():
    """PromptBuilder hides final schema before tool work and shows it after tracked delta exists."""
    registry, _ = _make_registry()
    first_turn = PromptBuilder(registry, DeltaState).build()
    after_tool_work = PromptBuilder(registry, DeltaState).build(
        DeltaState(new_files=[FileWrite(path="src/x.py", content="x")])
    )
    assert "new_files" not in first_turn
    assert "new_files" in after_tool_work


def test_prompt_builder_hides_delta_schema_on_first_turn_with_tools():
    """First-turn system prompt does not show DeltaState schema when tools are present."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, DeltaState).build()
    assert "new_files" not in prompt
    assert "edits" not in prompt


def test_prompt_builder_owns_json_only_instruction_once():
    """run_agent owns the JSON-only semantic instruction and emits it once."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, DeltaState).build()
    assert prompt.count("JSON only") == 1


def test_prompt_builder_shows_delta_schema_after_tool_work():
    """After tool work (non-empty tracked_delta), system prompt shows DeltaState schema."""
    registry, _ = _make_registry()
    tracked = DeltaState(new_files=[FileWrite(path="src/x.py", content="x")])
    prompt = PromptBuilder(registry, DeltaState).build(tracked)
    assert "new_files" in prompt


def test_generated_delta_schema_includes_all_top_level_model_fields():
    """DeltaState schema prompt is generated from all actual top-level Pydantic fields."""
    prompt = PromptBuilder.render_response_schema(DeltaState)
    for field_name in DeltaState.model_fields:
        assert field_name in prompt


def test_generated_plan_schema_includes_all_top_level_model_fields():
    """PlanResponse schema prompt is generated from all actual top-level Pydantic fields."""
    prompt = PromptBuilder.render_response_schema(PlanResponse)
    for field_name in PlanResponse.model_fields:
        assert field_name in prompt


def test_generated_schema_reflects_new_response_model_fields_automatically():
    """Adding fields to a response model is reflected by schema rendering without prompt edits."""
    prompt = PromptBuilder.render_response_schema(_ExtendedResponse)
    for field_name in _ExtendedResponse.model_fields:
        assert field_name in prompt


def test_prompt_builder_tool_guidance_uses_registered_tools_only():
    """Tool guidance names tools from the provided registry, not hardcoded worker tools."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, DeltaState).build()

    assert "do_thing" in prompt
    for unavailable in (
        "list_files",
        "read_file",
        "run_tests",
        "add_dependency",
        "write_blackboard",
    ):
        assert unavailable not in prompt


def test_prompt_builder_includes_new_files_vs_edits_format_clarification():
    """PromptBuilder includes new_files/edits format rules in the DeltaState system prompt."""
    prompt = PromptBuilder(None, DeltaState).build()
    assert "Format rules:" in prompt
    assert "new_files: create files that do not exist yet" in prompt
    assert "edits: replace existing text in existing files" in prompt
    assert "Never put file content in edits." in prompt
    assert "Never put old/new strings in new_files." in prompt


async def test_run_agent_rejects_premature_delta_state_when_no_tool_calls_made():
    """With tools and no prior tool calls, an empty DeltaState is rejected with tool-call format in the error."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        max_retries=0,
    )

    assert response.status == ResponseStatus.FAILED
    assert "tool_call" in (response.error or "")
    assert "do_thing" in (response.error or "")
    assert "list_files" not in (response.error or "")


async def test_run_agent_rejects_empty_delta_for_work_agent_with_no_tools():
    """run_agent rejects empty DeltaState for WORK agents even when no tools are configured."""
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=0)

    assert response.status == ResponseStatus.FAILED
    assert "empty delta" in (response.error or "")


async def test_run_agent_rejects_empty_delta_after_read_only_tool_call():
    """run_agent rejects empty DeltaState for WORK agents even after read-only tool calls."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        max_retries=0,
    )

    assert response.status == ResponseStatus.FAILED
    assert "empty delta" in (response.error or "")


async def test_run_agent_accepts_delta_state_after_tool_work():
    """With tools and non-empty tracked_delta, a DeltaState is accepted without correction."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "write_file", "arguments": {"path": "src/x.py", "content": "x = 1"}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


# --- adapter_spec / requires_nonempty_output ---


def _audit_adapter_spec() -> "AdapterSpec":
    from forge.adapters.registry import AdapterSpec

    return AdapterSpec(
        name="audit",
        description="audit",
        tools=[],
        prompt_template="",
        requires_nonempty_output=False,
        work_noun="findings",
    )


async def test_run_agent_allows_empty_delta_after_tool_call_when_adapter_allows_it():
    """run_agent accepts empty DeltaState after tool use when adapter allows empty output."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            '{"edits": [], "new_files": [], "dependencies": []}',
        ]
    )

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        adapter_spec=_audit_adapter_spec(),
        max_retries=0,
    )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == DeltaState()


async def test_tool_loop_correction_message_includes_format_reminder_on_delta_state_parse_failure():
    """Correction message after a DeltaState parse failure includes the field format reminder."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=["not valid json", _NONEMPTY_DELTA])

    await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=DeltaState,
        max_retries=3,
    ).run()

    second_call_messages = provider.chat.call_args_list[1][0][0]
    correction = second_call_messages[-1]["content"]
    assert "new_files must be a list of objects" in correction
    assert "edits must be a list of objects" in correction
    assert "Do not use dicts or nested objects" in correction


async def test_run_agent_allows_empty_delta_without_tool_calls_when_adapter_allows_it():
    """run_agent accepts empty DeltaState with no tool calls when adapter allows empty output."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await run_agent(
        request,
        WorkSpec,
        provider,
        "prompt",
        tools=registry,
        adapter_spec=_audit_adapter_spec(),
        max_retries=0,
    )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == DeltaState()
