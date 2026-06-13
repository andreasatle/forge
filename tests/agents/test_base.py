"""Tests for the run_agent base engine — plain chat loop with structured JSON parsing."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

from forge.adapters.registry import AdapterSpec
from forge.agents.base import (
    PromptBuilder,
    ResponseParser,
    ToolError,
    ToolLoop,
    TrackedToolExecutor,
    _classify_failure,
    run_agent,
)
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    FailureKind,
    FileContent,
    PlanResponse,
    RequestSource,
    ResponseStatus,
    ToolCallRequest,
    WorkOutput,
    WorkSpec,
)
from forge.llm.providers import ProviderEmptyOutputError
from forge.tools.registry import Tool, ToolRegistry

_NONEMPTY_WORK_OUTPUT = (
    '{"files": [{"path": "src/main.py", "content": "x = 1"}], "dependencies": []}'
)
_MALFORMED_WORK_OUTPUT_WITH_STRING_FILE_ENTRIES = (
    '{"files": ['
    '{"path": "src/main.py", "content": "x = 1"}, '
    '"path:tests/test_scraper.py", '
    '"content:import pytest\\n"'
    '], "dependencies": [], "base_version": "abc"}'
)


class _DoThingRequest(BaseModel):
    """Minimal no-field request used in agent base unit tests."""


class _DoThingResponse(BaseModel):
    """Minimal response carrying a result string used in agent base unit tests."""

    result: str


class _NeedsContentRequest(BaseModel):
    """Synthetic request with one required field for validation tests."""

    content: str


class _NeedsContentResponse(BaseModel):
    """Synthetic response for validation tests."""

    accepted: bool


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


def _make_needs_content_tool() -> Tool:
    async def fn(req: _NeedsContentRequest) -> _NeedsContentResponse:
        return _NeedsContentResponse(accepted=bool(req.content))

    return Tool(
        name="needs_content",
        description="requires content",
        request_type=_NeedsContentRequest,
        response_type=_NeedsContentResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


# --- ResponseParser ---


def test_response_parser_parses_tool_call_request():
    """ResponseParser returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = ResponseParser(WorkOutput).parse(raw)
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
        ResponseParser(WorkOutput).parse("not json at all")


def test_response_parser_rejects_schema_invalid_final_response():
    """ResponseParser raises ValueError with the existing schema mismatch message."""
    with pytest.raises(ValueError, match="response does not match _RequiredResponse"):
        ResponseParser(_RequiredResponse).parse("{}")


def test_response_parser_correctly_parses_tool_call_request():
    """ResponseParser returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, ToolCallRequest)
    assert result.name == "my_tool"


def test_response_parser_correctly_parses_work_output_as_final_response():
    """ResponseParser returns WorkOutput when JSON matches WorkOutput schema."""
    raw = '{"files": [], "dependencies": []}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, WorkOutput)


def test_response_parser_correctly_parses_plan_response_as_final_response():
    """ResponseParser returns PlanResponse when final_response_type is PlanResponse."""
    raw = '{"kind": "plan", "tasks": []}'
    result = ResponseParser(PlanResponse).parse(raw)
    assert isinstance(result, PlanResponse)
    assert result.tasks == []


def test_response_parser_raises_value_error_on_unknown_format():
    """ResponseParser raises ValueError when the response is not valid JSON."""
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkOutput).parse("not json at all")


# --- TrackedToolExecutor ---


async def test_tracked_tool_executor_executes_valid_tool():
    """TrackedToolExecutor returns a successful tool response for a valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1


async def test_tracked_tool_executor_rejects_unknown_tool():
    """TrackedToolExecutor returns the existing failed response for an unknown tool."""
    registry = ToolRegistry()
    request = ToolCallRequest(kind="tool_call", name="nonexistent", arguments={})

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is False
    assert response.error is not None


async def test_tracked_tool_executor_validates_tool_arguments():
    """TrackedToolExecutor returns a failed response for invalid arguments."""
    registry = ToolRegistry()
    registry.register(_make_needs_content_tool())
    request = ToolCallRequest(
        kind="tool_call",
        name="needs_content",
        arguments={},
    )

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is False
    assert "content" in (response.error or "")


async def test_tracked_tool_executor_returns_failed_response_when_tool_raises():
    """TrackedToolExecutor returns a failed ToolCallResponse when a tool raises."""
    registry = ToolRegistry()

    async def failing_fn(req: _DoThingRequest) -> _DoThingResponse:
        raise ValueError("tool failed")

    registry.register(
        Tool(
            name="failing_tool",
            description="fails on purpose",
            request_type=_DoThingRequest,
            response_type=_DoThingResponse,
            fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], failing_fn),
        )
    )
    request = ToolCallRequest(
        kind="tool_call",
        name="failing_tool",
        arguments={},
    )

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is False
    assert "tool failed" in (response.error or "")


# --- run_agent ---


async def test_tool_loop_returns_completed_final_response_without_tools():
    """ToolLoop returns COMPLETED for a valid final response without tool calls."""
    request = _work_request()
    provider = _mock_provider(_NONEMPTY_WORK_OUTPUT)

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.output == WorkOutput(files=[FileContent(path="src/main.py", content="x = 1")])
    assert provider.chat.call_count == 1


async def test_tool_loop_runs_tool_call_then_final_response():
    """ToolLoop executes one tool call and feeds the response back before final JSON."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_tool_loop_retries_invalid_json():
    """ToolLoop retries invalid JSON and succeeds on the next response."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=["not valid json", _NONEMPTY_WORK_OUTPUT])

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
        max_retries=3,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_tool_loop_rejects_empty_work_output_for_work_agent():
    """ToolLoop rejects empty WorkOutput for WORK agents when nonempty output is required."""
    request = _work_request()
    provider = _mock_provider('{"files": [], "dependencies": []}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "empty work output" in (response.error or "")


async def test_run_agent_routes_tool_calls_correctly():
    """run_agent executes tool calls and feeds results back before the final response."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert isinstance(response.output, WorkOutput)
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_returns_agent_response_on_final_response():
    """run_agent returns COMPLETED AgentResponse when the LLM returns valid non-empty final JSON."""
    request = _work_request()
    provider = _mock_provider(_NONEMPTY_WORK_OUTPUT)

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert isinstance(response, AgentResponse)
    assert response.status == ResponseStatus.COMPLETED
    assert isinstance(response.output, WorkOutput)


async def test_run_agent_retries_on_invalid_format():
    """run_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=["not valid json", _NONEMPTY_WORK_OUTPUT])

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
    """_classify_failure maps ToolError to TOOL_ERROR, not INVALID_JSON."""
    assert _classify_failure(ToolError("tool failed")) == FailureKind.TOOL_ERROR


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
    provider = _mock_provider(_NONEMPTY_WORK_OUTPUT)
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
            _NONEMPTY_WORK_OUTPUT,
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
    prompt = PromptBuilder(registry, WorkOutput).build()
    assert "You have two valid response formats" in prompt
    assert "tool_call" in prompt
    assert "do_thing" in prompt


def test_prompt_builder_builds_prompt_without_tools():
    """PromptBuilder omits tool-call instructions when tools are not configured."""
    prompt = PromptBuilder(None, WorkOutput).build()
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
    prompt = PromptBuilder(registry, WorkOutput).build()
    assert "do_thing: does a thing" in prompt


def test_prompt_builder_owns_json_only_instruction_once():
    """run_agent owns the JSON-only semantic instruction and emits it once."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, WorkOutput).build()
    assert prompt.count("JSON only") == 1


def test_generated_work_output_schema_includes_all_top_level_model_fields():
    """WorkOutput schema prompt is generated from all actual top-level Pydantic fields."""
    prompt = PromptBuilder.render_response_schema(WorkOutput)
    for field_name in WorkOutput.model_fields:
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
    prompt = PromptBuilder(registry, WorkOutput).build()

    assert "do_thing" in prompt
    for unavailable in (
        "list_files",
        "read_file",
        "run_tests",
        "add_dependency",
    ):
        assert unavailable not in prompt


def test_prompt_builder_includes_work_output_format_clarification():
    """PromptBuilder includes WorkOutput format rules in the system prompt."""
    prompt = PromptBuilder(None, WorkOutput).build()
    assert "Format rules:" in prompt
    assert "files: provide complete file content" in prompt
    assert "dependencies: list any new runtime packages required" in prompt
    assert "base_version set to the current commit SHA" in prompt


def test_prompt_builder_always_shows_work_output_schema_for_work_agents():
    """PromptBuilder includes WorkOutput schema on first turn when always_show_final is True."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, WorkOutput, always_show_final=True).build()
    assert "files" in prompt


async def test_run_agent_rejects_premature_empty_work_output_when_no_tool_calls_made():
    """With tools and no prior tool calls, an empty WorkOutput asks for tool use."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"files": [], "dependencies": []}')

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


async def test_run_agent_rejects_empty_work_output_for_work_agent_with_no_tools():
    """run_agent rejects empty WorkOutput for WORK agents even when no tools are configured."""
    request = _work_request()
    provider = _mock_provider('{"files": [], "dependencies": []}')

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=0)

    assert response.status == ResponseStatus.FAILED
    assert "empty work output" in (response.error or "")


async def test_run_agent_rejects_empty_work_output_after_read_only_tool_call():
    """run_agent rejects empty WorkOutput for WORK agents even after read-only tool calls."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            '{"files": [], "dependencies": []}',
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
    assert "empty work output" in (response.error or "")


# --- adapter_spec / requires_nonempty_output ---


def _audit_adapter_spec() -> AdapterSpec:
    return AdapterSpec(
        name="audit",
        description="audit",
        tools=[],
        prompt_template="",
        requires_nonempty_output=False,
        work_noun="findings",
    )


async def test_run_agent_allows_empty_work_output_after_tool_call_when_adapter_allows_it():
    """run_agent accepts empty WorkOutput after tool use when adapter allows empty output."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
            '{"files": [], "dependencies": []}',
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
    assert response.output == WorkOutput()


async def test_tool_loop_preserves_raw_invalid_response_diagnostics_on_retry_exhaustion():
    """Parse-exhausted AgentResponse carries bounded invalid response diagnostics."""
    request = _work_request()
    provider = _mock_provider(_MALFORMED_WORK_OUTPUT_WITH_STRING_FILE_ENTRIES)

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INVALID_JSON
    assert response.diagnostics
    diagnostic = response.diagnostics[0]
    assert diagnostic.kind == "invalid_structured_output"
    assert diagnostic.validation_path == "files.1"
    assert diagnostic.bad_value_excerpt == '"path:tests/test_scraper.py"'
    assert diagnostic.raw_response_excerpt is not None
    assert "path:tests/test_scraper.py" in diagnostic.raw_response_excerpt
    assert len(diagnostic.raw_response_excerpt) <= 4000


async def test_tool_loop_retry_preserves_original_prompt_plus_json_repair_block():
    """Invalid structured output retry keeps the original prompt and appends repair guidance."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[_MALFORMED_WORK_OUTPUT_WITH_STRING_FILE_ENTRIES, _NONEMPTY_WORK_OUTPUT]
    )
    original_prompt = "AgentRequest contract: satisfy AC1"

    await ToolLoop(
        request=request,
        provider=provider,
        prompt=original_prompt,
        tools=None,
        final_response_type=WorkOutput,
        max_retries=3,
    ).run()

    second_call_messages = provider.chat.call_args_list[1][0][0]
    assert second_call_messages[1]["role"] == "user"
    assert second_call_messages[1]["content"] == original_prompt
    assert second_call_messages[-1]["role"] == "user"
    assert "Invalid response:" in second_call_messages[-1]["content"]


async def test_run_agent_allows_empty_work_output_without_tool_calls_when_adapter_allows_it():
    """run_agent accepts empty WorkOutput with no tool calls when adapter allows empty output."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"files": [], "dependencies": []}')

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
    assert response.output == WorkOutput()
