"""Tests for the run_agent base engine — plain chat loop with structured JSON parsing."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

from forge.agents.base import (
    _classify_failure,
    _execute_tool,
    _merge_delta,
    _parse_response,
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
from forge.tools.registry import Tool, ToolRegistry
from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    WriteFileRequest,
    WriteFileResponse,
)


class _DoThingRequest(BaseModel):
    pass


class _DoThingResponse(BaseModel):
    result: str


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
    registry.register(Tool(
        name="do_thing",
        description="does a thing",
        request_type=_DoThingRequest,
        response_type=_DoThingResponse,
        fn=mock_fn,
    ))
    return registry, mock_fn


def _make_write_file_tool() -> Tool:
    async def fn(req: WriteFileRequest) -> WriteFileResponse:
        return WriteFileResponse(path=req.path)
    return Tool(
        name="write_file",
        description="write a file",
        request_type=WriteFileRequest,
        response_type=WriteFileResponse,
        fn=fn,
    )


def _make_replace_in_file_tool() -> Tool:
    async def fn(req: ReplaceInFileRequest) -> ReplaceInFileResponse:
        return ReplaceInFileResponse(path=req.path)
    return Tool(
        name="replace_in_file",
        description="replace in a file",
        request_type=ReplaceInFileRequest,
        response_type=ReplaceInFileResponse,
        fn=fn,
    )


def _make_add_dependency_tool() -> Tool:
    async def fn(req: AddDependencyRequest) -> AddDependencyResponse:
        return AddDependencyResponse(package=req.package, success=True, output="")
    return Tool(
        name="add_dependency",
        description="add a dependency",
        request_type=AddDependencyRequest,
        response_type=AddDependencyResponse,
        fn=fn,
    )


# --- _parse_response ---


def test_parse_response_correctly_parses_tool_call_request():
    """_parse_response returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = _parse_response(raw, None, DeltaState)
    assert isinstance(result, ToolCallRequest)
    assert result.name == "my_tool"


def test_parse_response_correctly_parses_delta_state_as_final_response():
    """_parse_response returns DeltaState when JSON matches DeltaState schema."""
    raw = '{"edits": [], "new_files": [], "dependencies": []}'
    result = _parse_response(raw, None, DeltaState)
    assert isinstance(result, DeltaState)


def test_parse_response_correctly_parses_plan_response_as_final_response():
    """_parse_response returns PlanResponse when final_response_type is PlanResponse."""
    raw = '{"kind": "plan", "tasks": []}'
    result = _parse_response(raw, None, PlanResponse)
    assert isinstance(result, PlanResponse)
    assert result.tasks == []


def test_parse_response_raises_value_error_on_unknown_format():
    """_parse_response raises ValueError when the response is not valid JSON."""
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_response("not json at all", None, DeltaState)


# --- _execute_tool ---


async def test_execute_tool_returns_correct_tool_call_response():
    """_execute_tool returns (ToolCallResponse, delta) with success=True on valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})
    response, delta = await _execute_tool(request, registry, DeltaState())
    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1
    assert delta == DeltaState()


async def test_execute_tool_returns_error_tool_call_response_on_unknown_tool():
    """_execute_tool returns (error response, unchanged delta) for an unregistered tool."""
    registry = ToolRegistry()
    request = ToolCallRequest(kind="tool_call", name="nonexistent", arguments={})
    response, delta = await _execute_tool(request, registry, DeltaState())
    assert response.success is False
    assert response.error is not None
    assert delta == DeltaState()


async def test_execute_tool_tracks_write_file_in_new_files():
    """_execute_tool adds a FileWrite entry to tracked_delta when write_file succeeds."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="write_file",
        arguments={"path": "src/hello.py", "content": "print(1)\n"},
    )
    _, delta = await _execute_tool(request, registry, DeltaState())
    assert len(delta.new_files) == 1
    assert delta.new_files[0] == FileWrite(path="src/hello.py", content="print(1)\n")


async def test_execute_tool_write_file_overwrites_same_path():
    """Calling write_file twice for the same path keeps only the latest content."""
    registry = ToolRegistry()
    registry.register(_make_write_file_tool())
    req1 = ToolCallRequest(
        kind="tool_call", name="write_file", arguments={"path": "a.py", "content": "v1"}
    )
    req2 = ToolCallRequest(
        kind="tool_call", name="write_file", arguments={"path": "a.py", "content": "v2"}
    )
    _, after_first = await _execute_tool(req1, registry, DeltaState())
    _, after_second = await _execute_tool(req2, registry, after_first)
    assert len(after_second.new_files) == 1
    assert after_second.new_files[0].content == "v2"


async def test_execute_tool_tracks_replace_in_file_in_edits():
    """_execute_tool adds an Edit entry to tracked_delta when replace_in_file succeeds."""
    registry = ToolRegistry()
    registry.register(_make_replace_in_file_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="replace_in_file",
        arguments={"path": "src/main.py", "old": "x = 1", "new": "x = 2"},
    )
    _, delta = await _execute_tool(request, registry, DeltaState())
    assert len(delta.edits) == 1
    assert delta.edits[0] == Edit(path="src/main.py", old="x = 1", new="x = 2")


async def test_execute_tool_tracks_add_dependency_in_dependencies():
    """_execute_tool adds a package name to tracked_delta when add_dependency succeeds."""
    registry = ToolRegistry()
    registry.register(_make_add_dependency_tool())
    request = ToolCallRequest(
        kind="tool_call", name="add_dependency", arguments={"package": "requests"}
    )
    _, delta = await _execute_tool(request, registry, DeltaState())
    assert delta.dependencies == ["requests"]


async def test_execute_tool_add_dependency_skips_duplicate():
    """add_dependency does not add the same package twice."""
    registry = ToolRegistry()
    registry.register(_make_add_dependency_tool())
    req = ToolCallRequest(
        kind="tool_call", name="add_dependency", arguments={"package": "requests"}
    )
    _, after_first = await _execute_tool(req, registry, DeltaState())
    _, after_second = await _execute_tool(req, registry, after_first)
    assert after_second.dependencies == ["requests"]


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


async def test_run_agent_routes_tool_calls_correctly():
    """run_agent executes tool calls and feeds results back before the final response."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_returns_agent_response_on_final_response():
    """run_agent returns COMPLETED AgentResponse when the LLM returns valid final JSON."""
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert isinstance(response, AgentResponse)
    assert response.status == ResponseStatus.COMPLETED


async def test_run_agent_retries_on_invalid_format():
    """run_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        "not valid json",
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=3)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_run_agent_returns_failed_after_max_iterations():
    """run_agent returns FAILED when the tool iteration limit is exhausted."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool_call", "name": "do_thing", "arguments": {}}')

    response = await run_agent(
        request, WorkSpec, provider, "prompt",
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
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "write_file", "arguments": {"path": "src/app.py", "content": "x = 1"}}',
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

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
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "write_file", "arguments": {"path": "a.py", "content": "tracked"}}',
        '{"edits": [], "new_files": [{"path": "a.py", "content": "reported"}], "dependencies": []}',
    ])

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
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "replace_in_file", "arguments": {"path": "a.py", "old": "x=1", "new": "x=2"}}',
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

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
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "add_dependency", "arguments": {"package": "httpx"}}',
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.delta is not None
    assert response.delta.dependencies == ["httpx"]


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


def test_classify_failure_maps_unknown_exception_to_unknown():
    """_classify_failure maps an unrecognized exception to FailureKind.UNKNOWN."""
    assert _classify_failure(KeyError("x")) == FailureKind.UNKNOWN


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
        request, WorkSpec, provider, "prompt",
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
    provider.chat = AsyncMock(side_effect=httpx.HTTPStatusError(
        "server error",
        request=httpx.Request("POST", "http://example.com"),
        response=httpx.Response(500),
    ))

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.PROVIDER_ERROR
