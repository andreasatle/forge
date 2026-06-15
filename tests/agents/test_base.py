"""Tests for the run_agent base engine — plain chat loop with structured JSON parsing."""

# pyright: reportPrivateUsage=false

from collections.abc import Awaitable, Callable
from pathlib import Path
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
    ToolLoopState,
    TrackedToolExecutor,
    _classify_failure,
    run_agent,
)
from forge.core.models import (
    AgentMessageKind,
    AgentRequest,
    AgentResponse,
    AgentType,
    FailureKind,
    RequestSource,
    ResponseStatus,
    ToolTurn,
    WorkDecision,
    WorkOutput,
    WorkSpec,
)
from forge.llm.providers import ProviderEmptyOutputError
from forge.tools.file_tools import make_write_file_tool_for_root
from forge.tools.registry import Tool, ToolRegistry
from forge.tools.schemas import RunTestsRequest, RunTestsResponse

_NONEMPTY_WORK_OUTPUT = '{"kind":"final","output":{"kind":"work_output","summary":"Changed files in the worktree.","base_version":""}}'
_MALFORMED_WORK_OUTPUT_WITH_BAD_SUMMARY = '{"kind":"final","output":{"kind":"work_output","summary":["not","a","string"],"base_version":"abc"}}'
_VALID_WORK_DECISION = '{"kind":"final","output":{"kind":"work","task":{"objective":"build it","success_condition":"tests pass","adapter":"coding","artifact":"codebase"}}}'


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


def test_response_parser_parses_valid_final_response():
    """ResponseParser unwraps kind='final' envelope and returns the configured final response."""
    raw = '{"kind":"final","output":{"kind":"work_output","summary":"done","base_version":"v1"}}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, WorkOutput)
    assert result.summary == "done"


def test_response_parser_rejects_invalid_json():
    """ResponseParser raises ValueError with the existing invalid JSON message."""
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkOutput).parse("not json at all")


def test_response_parser_rejects_schema_invalid_final_response():
    """ResponseParser raises ValueError for invalid FinalTurn output schema."""
    raw = '{"kind":"final","output":{"kind":"work_output","summary":["bad"],"base_version":""}}'
    with pytest.raises(ValueError, match="invalid final turn"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_rejects_old_tool_call_kind():
    """ResponseParser rejects legacy kind='tool_call' in the strict new protocol."""
    raw = '{"kind":"tool_call","name":"my_tool","arguments":{}}'
    with pytest.raises(ValueError, match="unknown protocol kind"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_rejects_plain_work_output():
    """ResponseParser rejects plain WorkOutput without a final envelope."""
    raw = '{"kind":"work_output","summary":"done","base_version":""}'
    with pytest.raises(ValueError, match="unknown protocol kind"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_rejects_missing_kind():
    """ResponseParser rejects JSON with no kind field."""
    raw = '{"summary":"done","base_version":""}'
    with pytest.raises(ValueError, match="unknown protocol kind"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_raises_value_error_on_unknown_format():
    """ResponseParser raises ValueError when the response is not valid JSON."""
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkOutput).parse("not json at all")


# --- ResponseParser — new two-shape protocol ---


def test_response_parser_parses_new_tool_turn():
    """ResponseParser accepts kind='tool' and returns ToolTurn."""
    raw = '{"kind":"tool","name":"write_file","arguments":{"path":"src/main.py","content":"x=1"}}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, ToolTurn)
    assert result.name == "write_file"
    assert result.arguments == {"path": "src/main.py", "content": "x=1"}


def test_response_parser_parses_new_tool_turn_empty_arguments():
    """ResponseParser accepts kind='tool' with no arguments field and defaults to {}."""
    raw = '{"kind":"tool","name":"run_tests"}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, ToolTurn)
    assert result.name == "run_tests"
    assert result.arguments == {}


def test_response_parser_parses_new_final_work_turn():
    """ResponseParser unwraps kind='final' envelope and ignores model-supplied base_version."""
    raw = '{"kind":"final","output":{"kind":"work_output","summary":"Wrote main.py","base_version":"abc123"}}'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, WorkOutput)
    assert result.summary == "Wrote main.py"
    assert not hasattr(result, "base_version")


def test_response_parser_parses_new_final_work_decision_turn():
    """ResponseParser unwraps kind='final' envelope and returns WorkDecision directly."""
    raw = '{"kind":"final","output":{"kind":"work","task":{"objective":"build it","success_condition":"tests pass","adapter":"coding","artifact":"codebase"}}}'
    result = ResponseParser(WorkDecision).parse(raw)
    assert isinstance(result, WorkDecision)
    assert result.task.objective == "build it"


def test_response_parser_new_tool_turn_missing_name_raises():
    """ResponseParser raises ValueError with 'invalid tool turn' when kind='tool' but name is absent."""
    raw = '{"kind":"tool","arguments":{}}'
    with pytest.raises(ValueError, match="invalid tool turn"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_new_final_turn_missing_output_raises():
    """ResponseParser raises ValueError with 'invalid final turn' when kind='final' but output is absent."""
    raw = '{"kind":"final"}'
    with pytest.raises(ValueError, match="invalid final turn"):
        ResponseParser(WorkOutput).parse(raw)


def test_response_parser_new_final_turn_unknown_output_kind_raises():
    """ResponseParser raises ValueError with 'invalid final turn' for unrecognized nested kind."""
    raw = '{"kind":"final","output":{"kind":"unknown_type"}}'
    with pytest.raises(ValueError, match="invalid final turn"):
        ResponseParser(WorkOutput).parse(raw)


# --- ResponseParser — JSON fence normalization ---


def test_response_parser_parses_fenced_json_final_turn():
    """ResponseParser accepts a final turn wrapped in a JSON code fence."""
    raw = f"```json\n{_VALID_WORK_DECISION}\n```"
    result = ResponseParser(WorkDecision).parse(raw)
    assert isinstance(result, WorkDecision)
    assert result.task.objective == "build it"


def test_response_parser_parses_fenced_json_tool_turn():
    """ResponseParser accepts a tool turn wrapped in a JSON code fence."""
    raw = '```json\n{"kind":"tool","name":"run_tests","arguments":{}}\n```'
    result = ResponseParser(WorkOutput).parse(raw)
    assert isinstance(result, ToolTurn)
    assert result.name == "run_tests"


def test_response_parser_rejects_fenced_json_with_prose_before():
    """ResponseParser rejects a fenced block preceded by prose text."""
    raw = f"Here is the result:\n```json\n{_VALID_WORK_DECISION}\n```"
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkDecision).parse(raw)


def test_response_parser_rejects_fenced_json_with_prose_after():
    """ResponseParser rejects a fenced block followed by prose text."""
    raw = f"```json\n{_VALID_WORK_DECISION}\n```\nDone!"
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkDecision).parse(raw)


def test_response_parser_rejects_multiple_fenced_blocks():
    """ResponseParser rejects a response containing multiple JSON fenced blocks."""
    raw = (
        '```json\n{"kind":"tool","name":"read_file","arguments":{}}\n```'
        "\n\n"
        f"```json\n{_VALID_WORK_DECISION}\n```"
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        ResponseParser(WorkDecision).parse(raw)


def test_response_parser_parses_unfenced_valid_json_unchanged():
    """ResponseParser continues to accept valid unfenced JSON without modification."""
    result = ResponseParser(WorkDecision).parse(_VALID_WORK_DECISION)
    assert isinstance(result, WorkDecision)
    assert result.task.objective == "build it"


# --- TrackedToolExecutor ---


async def test_tracked_tool_executor_executes_valid_tool():
    """TrackedToolExecutor returns a successful tool response for a valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolTurn(name="do_thing", arguments={})

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1


async def test_tracked_tool_executor_rejects_unknown_tool():
    """TrackedToolExecutor returns the existing failed response for an unknown tool."""
    registry = ToolRegistry()
    request = ToolTurn(name="nonexistent", arguments={})

    response = await TrackedToolExecutor(registry).execute(request)

    assert response.success is False
    assert response.error is not None


async def test_tracked_tool_executor_validates_tool_arguments():
    """TrackedToolExecutor returns a failed response for invalid arguments."""
    registry = ToolRegistry()
    registry.register(_make_needs_content_tool())
    request = ToolTurn(name="needs_content", arguments={})

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
    request = ToolTurn(name="failing_tool", arguments={})

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
    assert response.output == WorkOutput(summary="Changed files in the worktree.")
    assert provider.chat.call_count == 1


async def test_tool_loop_runs_tool_call_then_final_response():
    """ToolLoop executes one tool call and feeds the response back before final JSON."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
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


async def test_tool_loop_accepts_metadata_only_work_output_after_write_file(
    tmp_path: Path,
):
    """ToolLoop accepts metadata-only WorkOutput after a mutating write_file call."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            (
                '{"kind": "tool", "name": "write_file", '
                '"arguments": {"path": "src/main.py", "content": "print(42)\\n"}}'
            ),
            (
                '{"kind":"final","output":{"kind":"work_output",'
                '"summary":"Wrote src/main.py in the worktree.","base_version":"abc123"}}'
            ),
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
    assert response.output == WorkOutput(
        kind=AgentMessageKind.WORK_OUTPUT,
        summary="Wrote src/main.py in the worktree.",
    )
    assert (tmp_path / "src/main.py").read_text() == "print(42)\n"
    assert provider.chat.call_count == 2


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
    provider = _mock_provider(
        '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}'
    )

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
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
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
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

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
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

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
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
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
    assert "two JSON shapes" in prompt
    assert '"kind":"tool"' in prompt
    assert "do_thing" in prompt


def test_prompt_builder_makes_tool_call_protocol_explicit():
    """PromptBuilder explains the tool-call envelope and shows each tool invocation shape."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, WorkOutput).build()

    assert 'kind="tool"' in prompt
    assert "put the tool name in name" in prompt
    assert '{"kind":"tool","name":"do_thing","arguments":{...}}' in prompt
    assert "Never put a tool name in kind" in prompt
    assert '{"kind":"do_thing"' not in prompt
    assert '"kind": "do_thing"' not in prompt


def test_prompt_builder_builds_prompt_without_tools():
    """PromptBuilder omits tool-call instructions when tools are not configured."""
    prompt = PromptBuilder(None, WorkOutput).build()
    assert "JSON only" in prompt
    assert "tool_call" not in prompt
    assert "Generated JSON schema" in prompt


def test_prompt_builder_includes_generated_pydantic_schema():
    """PromptBuilder renders schema from the configured output object model."""
    prompt = PromptBuilder(None, _ExtendedResponse).build()
    assert "Output object model: _ExtendedResponse" in prompt
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


def test_generated_work_decision_schema_includes_all_top_level_model_fields():
    """WorkDecision schema prompt is generated from all actual top-level Pydantic fields."""
    prompt = PromptBuilder.render_response_schema(WorkDecision)
    for field_name in WorkDecision.model_fields:
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
    assert "Format rules (for the output object):" in prompt
    assert 'kind: must be "work_output"' in prompt
    assert "summary: briefly describe the worktree changes" in prompt
    assert "Dependency changes must be made in package manager files" in prompt
    assert "Do not include complete file contents" in prompt
    assert 'return final JSON with kind="final" and output containing kind and summary' in prompt
    assert "base_version" not in prompt


def test_prompt_builder_always_shows_work_output_schema_for_work_agents():
    """PromptBuilder includes WorkOutput schema on first turn when always_show_final is True."""
    registry, _ = _make_registry()
    prompt = PromptBuilder(registry, WorkOutput, always_show_final=True).build()
    assert "Output object fields: kind, summary" in prompt
    assert "base_version" not in prompt
    assert '"files"' not in prompt
    assert '"dependencies"' not in prompt


async def test_run_agent_rejects_premature_empty_work_output_when_no_tool_calls_made():
    """With tools and no prior tool calls, an empty WorkOutput asks for tool use."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider(
        '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}'
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
    assert '"kind":"tool"' in (response.error or "")
    assert "do_thing" in (response.error or "")
    assert "list_files" not in (response.error or "")


async def test_run_agent_rejects_empty_work_output_for_work_agent_with_no_tools():
    """run_agent rejects empty WorkOutput for WORK agents even when no tools are configured."""
    request = _work_request()
    provider = _mock_provider(
        '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}'
    )

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
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
            '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}',
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
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
            '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}',
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


async def test_tool_loop_preserves_raw_response_excerpt_for_json_parse_failure():
    """ToolLoop captures raw_response_excerpt when model returns unparseable text."""
    raw = "this is definitely not json and will cause a parse error"
    provider = _mock_provider(raw)

    response = await ToolLoop(
        request=_work_request(),
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INVALID_JSON
    assert response.diagnostics
    excerpt = response.diagnostics[0].raw_response_excerpt
    assert excerpt is not None
    assert raw in excerpt


async def test_tool_loop_preserves_raw_invalid_response_diagnostics_on_retry_exhaustion():
    """Parse-exhausted AgentResponse carries bounded invalid response diagnostics."""
    request = _work_request()
    provider = _mock_provider(_MALFORMED_WORK_OUTPUT_WITH_BAD_SUMMARY)

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
    assert diagnostic.validation_path == "summary"
    assert diagnostic.bad_value_excerpt == '["not", "a", "string"]'
    assert diagnostic.raw_response_excerpt is not None
    assert "not" in diagnostic.raw_response_excerpt
    assert len(diagnostic.raw_response_excerpt) <= 4000


async def test_tool_loop_retry_preserves_original_prompt_plus_json_repair_block():
    """Invalid structured output retry keeps the original prompt and appends repair guidance."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[_MALFORMED_WORK_OUTPUT_WITH_BAD_SUMMARY, _NONEMPTY_WORK_OUTPUT]
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
    repair_content = second_call_messages[-1]["content"]
    assert "Invalid response:" in repair_content
    assert '"kind":"tool"' in repair_content
    assert '"kind":"final"' in repair_content


async def test_tool_loop_repair_prompt_explains_two_shape_protocol():
    """Repair prompt for unknown kind explains the two-shape protocol."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "run_tests"}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=None,
        final_response_type=WorkOutput,
        max_retries=3,
    ).run()

    second_call_messages = provider.chat.call_args_list[1][0][0]
    repair_prompt = second_call_messages[-1]["content"]
    assert '"kind":"tool"' in repair_prompt
    assert '"kind":"final"' in repair_prompt


async def test_run_agent_allows_empty_work_output_without_tool_calls_when_adapter_allows_it():
    """run_agent accepts empty WorkOutput with no tool calls when adapter allows empty output."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider(
        '{"kind":"final","output":{"kind":"work_output","summary":"","base_version":""}}'
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


# --- final_response_only after run_tests passes ---


def _make_passing_run_tests_tool() -> Tool:
    async def fn(req: RunTestsRequest) -> RunTestsResponse:
        return RunTestsResponse(passed=True, failures=[], summary="1 passed", output="")

    return Tool(
        name="run_tests",
        description="run tests",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


async def test_tool_loop_disables_tools_after_successful_tests(tmp_path: Path) -> None:
    """After run_tests passes, the next chat turn receives no-tools system prompt and returns COMPLETED."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    registry.register(_make_passing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "f.py", "content": "x=1"}}',
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
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
    assert response.ran_tests_and_passed is True
    assert provider.chat.call_count == 3

    third_call_messages = provider.chat.call_args_list[2][0][0]
    system_content = third_call_messages[0]["content"]
    assert "write_file" not in system_content
    assert "run_tests" not in system_content
    assert "WorkOutput" in system_content

    last_user_content = third_call_messages[-1]["content"]
    assert "Tests passed" in last_user_content


async def test_tool_loop_rejects_tool_call_after_successful_tests() -> None:
    """After run_tests passes, a subsequent tool call is rejected without execution and the loop corrects to final JSON."""
    registry, mock_fn = _make_registry()
    registry.register(_make_passing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
            '{"kind": "tool", "name": "do_thing", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_retries=3,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.ran_tests_and_passed is True
    assert mock_fn.call_count == 0


# --- final_response_only without run_tests (no-test workers) ---


async def test_tool_loop_no_test_write_file_does_not_trigger_completion_pressure(
    tmp_path: Path,
) -> None:
    """Non-verifying adapter: successful write_file does not inject completion pressure."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "f.txt", "content": "hello"}}',
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

    second_call_messages = provider.chat.call_args_list[1][0][0]
    system_content = second_call_messages[0]["content"]
    assert "write_file" in system_content
    assert "WorkOutput" in system_content

    last_user_content = second_call_messages[-1]["content"]
    assert "Write complete" not in last_user_content


async def test_tool_loop_no_test_allows_multiple_writes(
    tmp_path: Path,
) -> None:
    """Non-verifying adapter: multiple successful write_file calls all execute before final output."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "a.txt", "content": "first"}}',
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "b.txt", "content": "second"}}',
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
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()


async def test_tool_loop_with_tests_write_file_alone_does_not_finalize(
    tmp_path: Path,
) -> None:
    """Test-enabled worker: successful write_file alone does not finalize until run_tests passes."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    registry.register(_make_passing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "f.txt", "content": "hello"}}',
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
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
    assert response.ran_tests_and_passed is True
    # 3 calls needed (write_file → run_tests → JSON) proves write_file alone did not finalize
    assert provider.chat.call_count == 3


async def test_tool_loop_max_iterations_preserves_bounded_tool_call_history():
    """ToolLoop max-iteration failure records bounded recent tool call names in diagnostics."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=3,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    diag = response.diagnostics[0]
    assert diag.kind == "max_iterations"
    assert "do_thing" in diag.message
    assert "last_tool_calls=" in diag.message


async def test_tool_loop_max_iterations_preserves_last_raw_response_excerpt():
    """ToolLoop max-iteration failure captures the last raw assistant response in the diagnostic."""
    registry, _ = _make_registry()
    request = _work_request()
    tool_call_raw = '{"kind": "tool", "name": "do_thing", "arguments": {}}'
    provider = _mock_provider(tool_call_raw)

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.diagnostics
    diag = response.diagnostics[0]
    assert diag.raw_response_excerpt is not None
    assert "do_thing" in diag.raw_response_excerpt


async def test_tool_loop_max_iterations_records_loop_state_flags():
    """Max-iteration diagnostic includes ran_tests_and_passed, final_response_only, has_run_tests, mutating_tool_succeeded."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.diagnostics
    diag = response.diagnostics[0]
    assert "ran_tests_and_passed=False" in diag.message
    assert "final_response_only=False" in diag.message
    assert "has_run_tests=False" in diag.message
    assert "mutating_tool_succeeded=False" in diag.message


async def test_tool_loop_max_iterations_bounded_to_five_most_recent():
    """Max-iteration diagnostic tool call list is bounded to the 5 most recent calls."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=7,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    diag = response.diagnostics[0]
    # 7 calls were made but only the last 5 should appear
    assert diag.message.count("do_thing") == 5


async def test_tool_loop_successful_run_has_no_diagnostics():
    """Normal successful ToolLoop run produces no diagnostics."""
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
    assert not response.diagnostics


async def test_tool_loop_with_tests_failed_tests_do_not_finalize(
    tmp_path: Path,
) -> None:
    """Test-enabled worker: failed run_tests does not finalize."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))

    async def _failing_run_tests(req: RunTestsRequest) -> RunTestsResponse:
        return RunTestsResponse(
            passed=False, failures=["AssertionError"], summary="1 failed", output=""
        )

    registry.register(
        Tool(
            name="run_tests",
            description="run tests",
            request_type=RunTestsRequest,
            response_type=RunTestsResponse,
            fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], _failing_run_tests),
        )
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "fix.txt", "content": "fix"}}',
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
    assert response.ran_tests_and_passed is False
    # write_file executed after failed tests proves failed run_tests did not finalize
    assert (tmp_path / "fix.txt").exists()


# --- ToolLoopState ---


def test_tool_loop_state_defaults() -> None:
    """ToolLoopState initialises all counters and flags to zero/False."""
    state = ToolLoopState()
    assert state.mutation_count == 0
    assert state.verification_count == 0
    assert state.mutating_tool_succeeded is False
    assert state.verification_passed is False
    assert state.final_response_only is False
    assert state.iteration_at_last_mutation == 0
    assert state.iteration_at_completion_pressure == 0
    assert state.iteration_at_verification_stability is None


# --- adapter-declared mutating tool names ---


class _CustomWriteRequest(BaseModel):
    """Request for the custom_write test tool."""

    path: str
    content: str


class _CustomWriteResponse(BaseModel):
    """Response from the custom_write test tool."""

    ok: bool


def _make_custom_write_tool(tmp_path: Path) -> Tool:
    """Tool named 'custom_write' that writes a file, for testing custom mutating_tools."""

    async def fn(req: _CustomWriteRequest) -> _CustomWriteResponse:
        (tmp_path / req.path).write_text(req.content)
        return _CustomWriteResponse(ok=True)

    return Tool(
        name="custom_write",
        description="custom write",
        request_type=_CustomWriteRequest,
        response_type=_CustomWriteResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


async def test_tool_loop_adapter_declared_mutating_tool_does_not_trigger_finalization(
    tmp_path: Path,
) -> None:
    """Non-verifying adapter: adapter-declared mutating tool does not trigger completion pressure."""
    registry = ToolRegistry()
    registry.register(_make_custom_write_tool(tmp_path))
    spec = AdapterSpec(
        name="custom",
        description="custom adapter",
        tools=["custom_write"],
        prompt_template="",
        mutating_tools=["custom_write"],
        verification_tools=[],
        verification_required=False,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "custom_write", "arguments": {"path": "out.txt", "content": "hi"}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    second_call_messages = provider.chat.call_args_list[1][0][0]
    system_content = second_call_messages[0]["content"]
    assert "custom_write" in system_content
    last_user_content = second_call_messages[-1]["content"]
    assert "Write complete" not in last_user_content


async def test_tool_loop_write_file_does_not_finalize_when_not_in_mutating_tools(
    tmp_path: Path,
) -> None:
    """ToolLoop does not finalize when write_file is not in adapter-declared mutating_tools."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    spec = AdapterSpec(
        name="custom",
        description="custom adapter",
        tools=["write_file"],
        prompt_template="",
        mutating_tools=[],
        verification_tools=[],
        verification_required=False,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "f.txt", "content": "x"}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    second_call_messages = provider.chat.call_args_list[1][0][0]
    last_user_content = second_call_messages[-1]["content"]
    assert "Write complete" not in last_user_content


# --- adapter-declared verification tool names ---


def _make_custom_check_tool(*, passed: bool) -> Tool:
    """Tool named 'custom_check' that returns a passed result, for testing verification_tools."""
    from forge.tools.schemas import RunTestsRequest, RunTestsResponse

    async def fn(req: RunTestsRequest) -> RunTestsResponse:
        return RunTestsResponse(passed=passed, failures=[], summary="ok", output="")

    return Tool(
        name="custom_check",
        description="custom check",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


async def test_tool_loop_adapter_declared_verification_tool_sets_verification_passed() -> None:
    """ToolLoop uses adapter-declared verification_tools to set verification_passed."""
    registry = ToolRegistry()
    registry.register(_make_custom_check_tool(passed=True))
    spec = AdapterSpec(
        name="custom",
        description="custom adapter",
        tools=["custom_check"],
        prompt_template="",
        mutating_tools=[],
        verification_tools=["custom_check"],
        verification_required=True,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "custom_check", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.ran_tests_and_passed is True
    second_call_messages = provider.chat.call_args_list[1][0][0]
    last_user_content = second_call_messages[-1]["content"]
    assert "Tests passed" in last_user_content


async def test_tool_loop_run_tests_does_not_finalize_when_not_in_verification_tools() -> None:
    """ToolLoop does not treat run_tests as verification when not in adapter-declared verification_tools."""
    registry = ToolRegistry()
    registry.register(_make_passing_run_tests_tool())
    spec = AdapterSpec(
        name="custom",
        description="custom adapter",
        tools=["run_tests"],
        prompt_template="",
        mutating_tools=[],
        verification_tools=[],
        verification_required=False,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.ran_tests_and_passed is False


# --- ToolLoopState mutation and verification count tracking ---


async def test_tool_loop_state_counts_mutations_in_diagnostic(tmp_path: Path) -> None:
    """mutation_count increments per successful mutating tool call; reflected via mutating_tool_succeeded."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    registry.register(_make_passing_run_tests_tool())
    # verification_required=True so write_file does not trigger finalization alone
    spec = AdapterSpec(
        name="test",
        description="test",
        tools=["write_file", "run_tests"],
        prompt_template="",
        mutating_tools=["write_file"],
        verification_tools=["run_tests"],
        verification_required=True,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "a.txt", "content": "1"}}',
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "b.txt", "content": "2"}}',
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "c.txt", "content": "3"}}',
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
        max_tool_iterations=3,
        max_retries=0,
    ).run()

    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    assert "mutating_tool_succeeded=True" in response.diagnostics[0].message


async def test_tool_loop_state_counts_verifications_in_diagnostic() -> None:
    """verification_count increments when a verification tool passes; reflected via ran_tests_and_passed."""
    registry = ToolRegistry()
    registry.register(_make_passing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=['{"kind": "tool", "name": "run_tests", "arguments": {}}']
    )

    # max_tool_iterations=1 so the loop exits after the single run_tests call (which passes),
    # without the model getting another turn to produce final JSON.
    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=1,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.ran_tests_and_passed is True
    assert response.diagnostics
    assert "ran_tests_and_passed=True" in response.diagnostics[0].message


# --- Phase 2: non-verifying adapter completion pressure ---


async def test_non_verifying_adapter_allows_multiple_mutations_before_final_output(
    tmp_path: Path,
) -> None:
    """Non-verifying adapter: three successful mutations all execute before the model returns final output."""
    registry = ToolRegistry()
    registry.register(_make_custom_write_tool(tmp_path))
    spec = AdapterSpec(
        name="document",
        description="document adapter",
        tools=["custom_write"],
        prompt_template="",
        mutating_tools=["custom_write"],
        verification_tools=[],
        verification_required=False,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "custom_write", "arguments": {"path": "a.md", "content": "# A"}}',
            '{"kind": "tool", "name": "custom_write", "arguments": {"path": "b.md", "content": "# B"}}',
            '{"kind": "tool", "name": "custom_write", "arguments": {"path": "c.md", "content": "# C"}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert (tmp_path / "a.md").exists()
    assert (tmp_path / "b.md").exists()
    assert (tmp_path / "c.md").exists()
    assert provider.chat.call_count == 4


async def test_verifying_adapter_gets_completion_pressure_after_verification_passes(
    tmp_path: Path,
) -> None:
    """Verifying adapter: verification pass still triggers final_response_only regardless of mutation count."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    registry.register(_make_custom_check_tool(passed=True))
    spec = AdapterSpec(
        name="coding",
        description="coding adapter",
        tools=["write_file", "custom_check"],
        prompt_template="",
        mutating_tools=["write_file"],
        verification_tools=["custom_check"],
        verification_required=True,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "x.py", "content": "x=1"}}',
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "y.py", "content": "y=2"}}',
            '{"kind": "tool", "name": "custom_check", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.ran_tests_and_passed is True
    assert provider.chat.call_count == 4
    fourth_call_messages = provider.chat.call_args_list[3][0][0]
    system_content = fourth_call_messages[0]["content"]
    assert "write_file" not in system_content
    last_user_content = fourth_call_messages[-1]["content"]
    assert "Tests passed" in last_user_content


# --- Phase 3: convergence telemetry ---


def _make_fixed_failing_run_tests_tool() -> Tool:
    """Tool named 'run_tests' that always fails with the same result."""

    async def fn(req: RunTestsRequest) -> RunTestsResponse:
        return RunTestsResponse(
            passed=False,
            failures=["AssertionError: expected 1 got 2"],
            summary="1 failed",
            output="FAILED test_x.py",
        )

    return Tool(
        name="run_tests",
        description="run tests",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


def _make_varying_run_tests_tool() -> Tool:
    """Tool named 'run_tests' that returns unique failures on each call."""
    call_count = 0

    async def fn(req: RunTestsRequest) -> RunTestsResponse:
        nonlocal call_count
        call_count += 1
        return RunTestsResponse(
            passed=False,
            failures=[f"error-call-{call_count}"],
            summary=f"failed call {call_count}",
            output="",
        )

    return Tool(
        name="run_tests",
        description="run tests",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


def _make_noop_write_file_tool() -> Tool:
    """Tool named 'write_file' that succeeds without filesystem access."""

    async def fn(req: _DoThingRequest) -> _DoThingResponse:
        return _DoThingResponse(result="ok")

    return Tool(
        name="write_file",
        description="write file",
        request_type=_DoThingRequest,
        response_type=_DoThingResponse,
        fn=cast(Callable[[BaseModel], Awaitable[BaseModel]], fn),
    )


async def test_tool_loop_identical_failing_verification_sets_verification_stable() -> None:
    """Two identical failing run_tests results set verification_stable=True in diagnostics."""
    registry = ToolRegistry()
    registry.register(_make_fixed_failing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    assert "verification_stable=True" in response.diagnostics[0].message


async def test_tool_loop_changed_verification_result_resets_verification_stable_count() -> None:
    """Different failing run_tests results reset verification_stable_count to 0."""
    registry = ToolRegistry()
    registry.register(_make_varying_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=3,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    msg = response.diagnostics[0].message
    assert "verification_stable=False" in msg
    assert "verification_stable_count=0" in msg


async def test_tool_loop_successful_mutation_updates_last_progress_iteration() -> None:
    """Successful mutating tool call sets last_progress_iteration=0 in diagnostics."""
    registry = ToolRegistry()
    registry.register(_make_noop_write_file_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "write_file", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=1,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    assert "last_progress_iteration=0" in response.diagnostics[0].message


async def test_tool_loop_changed_verification_fingerprint_updates_last_progress_iteration() -> None:
    """A changed failing verification fingerprint sets last_progress_iteration in diagnostics."""
    registry = ToolRegistry()
    registry.register(_make_varying_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    assert "last_progress_iteration=1" in response.diagnostics[0].message


async def test_tool_loop_max_iterations_diagnostic_includes_convergence_telemetry() -> None:
    """Max-iteration diagnostic message includes all Phase 3 convergence telemetry fields."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "do_thing", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    msg = response.diagnostics[0].message
    assert "verification_stable=" in msg
    assert "verification_stable_count=" in msg
    assert "last_progress_iteration=" in msg
    assert "iterations_since_progress=" in msg


# --- Phase 3.5: iteration_at_verification_stability telemetry ---


async def test_tool_loop_iteration_at_verification_stability_set_when_first_stable() -> None:
    """iteration_at_verification_stability is set to the iteration when stability first occurs."""
    registry = ToolRegistry()
    registry.register(_make_fixed_failing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    # iteration 0: first run_tests sets fingerprint; iteration 1: match → stable=True at iteration 1
    assert "iteration_at_verification_stability=1" in response.diagnostics[0].message


async def test_tool_loop_iteration_at_verification_stability_not_overwritten() -> None:
    """Subsequent identical verification results do not overwrite the first stability iteration."""
    registry = ToolRegistry()
    registry.register(_make_fixed_failing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=3,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    # stability first occurred at iteration 1; iteration 2 repeats but must not overwrite
    assert "iteration_at_verification_stability=1" in response.diagnostics[0].message


async def test_tool_loop_max_iterations_diagnostic_includes_verification_stability_iteration() -> (
    None
):
    """Max-iteration diagnostic includes the iteration_at_verification_stability field."""
    registry = ToolRegistry()
    registry.register(_make_fixed_failing_run_tests_tool())
    request = _work_request()
    provider = _mock_provider('{"kind": "tool", "name": "run_tests", "arguments": {}}')

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        max_tool_iterations=2,
        max_retries=0,
    ).run()

    assert response.failure_kind == FailureKind.MAX_ITERATIONS
    assert response.diagnostics
    assert "iteration_at_verification_stability=" in response.diagnostics[0].message


async def test_coding_adapter_write_alone_does_not_finalize_unchanged(
    tmp_path: Path,
) -> None:
    """Coding adapter (verification_required=True): write_file alone still does not trigger finalization."""
    registry = ToolRegistry()
    registry.register(make_write_file_tool_for_root(tmp_path))
    registry.register(_make_passing_run_tests_tool())
    spec = AdapterSpec(
        name="coding",
        description="coding adapter",
        tools=["write_file", "run_tests"],
        prompt_template="",
        mutating_tools=["write_file"],
        verification_tools=["run_tests"],
        verification_required=True,
    )
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            '{"kind": "tool", "name": "write_file", "arguments": {"path": "f.py", "content": "x=1"}}',
            '{"kind": "tool", "name": "run_tests", "arguments": {}}',
            _NONEMPTY_WORK_OUTPUT,
        ]
    )

    response = await ToolLoop(
        request=request,
        provider=provider,
        prompt="prompt",
        tools=registry,
        final_response_type=WorkOutput,
        adapter_spec=spec,
    ).run()

    assert response.status == ResponseStatus.COMPLETED
    assert response.ran_tests_and_passed is True
    # 3 calls proves write_file alone did not finalize; run_tests was still needed
    assert provider.chat.call_count == 3
