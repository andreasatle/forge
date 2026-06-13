"""Base agent runner — universal engine with plain chat loop and structured JSON parsing."""

import json
import re
from collections.abc import Callable
from typing import cast

import httpx
from pydantic import BaseModel, ValidationError

from forge.adapters.registry import AdapterSpec
from forge.core.models import (
    AgentDiagnostic,
    AgentRequest,
    AgentResponse,
    AgentType,
    FailureKind,
    PlanResponse,
    ProducerOutput,
    ResponseStatus,
    ToolCallRequest,
    ToolCallResponse,
    WorkOutput,
)
from forge.llm.providers import ChatMessage, LLMProvider, ProviderError
from forge.tools.registry import ToolRegistry


class ToolError(Exception):
    """Raised when a tool call fails during execution (distinct from JSON parse errors)."""


_MAX_RAW_RESPONSE_DIAGNOSTIC_CHARS = 4000
_MAX_BAD_VALUE_CHARS = 300


def _classify_failure(exc: Exception) -> FailureKind:
    """Map an exception to a FailureKind."""
    if isinstance(exc, httpx.TimeoutException):
        return FailureKind.TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        return FailureKind.PROVIDER_ERROR
    if isinstance(exc, ProviderError):
        return FailureKind.PROVIDER_ERROR
    if isinstance(exc, RuntimeError):
        return FailureKind.MAX_ITERATIONS
    if isinstance(exc, ToolError):
        return FailureKind.TOOL_ERROR
    if isinstance(exc, ValueError):
        return FailureKind.INVALID_JSON
    return FailureKind.UNKNOWN


def _truncate_text(text: str, limit: int) -> str:
    """Return a bounded diagnostic excerpt."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 14]}...[truncated]"


def _validation_path(loc: object) -> str | None:
    """Render a Pydantic error location as a compact dotted path."""
    if not isinstance(loc, tuple) or not loc:
        return None
    parts = cast(tuple[object, ...], loc)
    return ".".join(str(part) for part in parts)


def _compact_bad_value(value: object) -> str:
    """Render a compact, bounded representation of a validation input value."""
    try:
        rendered = json.dumps(value, ensure_ascii=False)
    except TypeError:
        rendered = repr(value)
    return _truncate_text(rendered, _MAX_BAD_VALUE_CHARS)


def _validation_error_items(error: Exception) -> list[tuple[str | None, str | None]]:
    """Extract validation paths and bad-value excerpts from wrapped Pydantic errors."""
    cause = error.__cause__
    if not isinstance(cause, ValidationError):
        return []
    items: list[tuple[str | None, str | None]] = []
    for entry in cause.errors():
        path = _validation_path(entry.get("loc"))
        bad_value = entry.get("input")
        bad_excerpt = _compact_bad_value(bad_value) if "input" in entry else None
        items.append((path, bad_excerpt))
    return items


def _invalid_response_diagnostic(error: Exception, raw: str) -> AgentDiagnostic:
    """Build a bounded diagnostic for an invalid structured response."""
    items = _validation_error_items(error)
    first_path = items[0][0] if items else None
    first_bad_value = items[0][1] if items else None
    return AgentDiagnostic(
        kind="invalid_structured_output",
        message=str(error),
        validation_path=first_path,
        bad_value_excerpt=first_bad_value,
        raw_response_excerpt=_truncate_text(raw, _MAX_RAW_RESPONSE_DIAGNOSTIC_CHARS),
    )


class PromptBuilder:
    """Build the generic system prompt for tool use and final response schemas."""

    def __init__(
        self,
        tools: ToolRegistry | None,
        final_response_type: type[BaseModel],
        *,
        always_show_final: bool = False,
    ) -> None:
        self.tools = tools
        self.final_response_type = final_response_type
        self.always_show_final = always_show_final

    @staticmethod
    def compact_response_schema(response_type: type[BaseModel]) -> dict[str, object]:
        """Return a compact JSON schema view derived from a Pydantic response model."""
        schema = response_type.model_json_schema()
        compact: dict[str, object] = {
            "title": schema.get("title", response_type.__name__),
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
        }
        if "required" in schema:
            compact["required"] = schema["required"]
        if "$defs" in schema:
            compact["$defs"] = schema["$defs"]
        return compact

    @classmethod
    def render_response_schema(cls, response_type: type[BaseModel]) -> str:
        """Render final-response schema instructions from the actual Pydantic model."""
        schema = cls.compact_response_schema(response_type)
        fields = (
            ", ".join(cast(dict[str, object], schema["properties"]).keys())
            if isinstance(schema["properties"], dict)
            else ""
        )
        lines = [
            f"Final response model: {response_type.__name__}",
            f"Top-level fields: {fields}",
            "Generated JSON schema:",
            json.dumps(schema, indent=2),
        ]
        return "\n".join(lines)

    def build(self) -> str:
        """Build the system prompt string, showing tool and schema sections as appropriate."""
        has_tools = self.tools is not None and bool(self.tools)
        show_final = self.tools is None or self.always_show_final
        step2 = "2. " if has_tools and show_final else ""
        lines: list[str] = [
            "You must respond with JSON only — no markdown, no explanation.",
            "",
        ]
        if self.tools is not None:
            lines += [
                "You have two valid response formats:",
                "",
                "1. To call a tool — use this exact format:",
                '{"kind": "tool_call", "name": "<tool_name>", "arguments": {"key": "value"}}',
                "",
                "Available tools:",
            ]
            for tool in self.tools:
                lines.append(f"  {tool.name}: {tool.description}")
                lines.append(
                    f"    input schema: {json.dumps(tool.request_type.model_json_schema())}"
                )
                lines.append(
                    f"    response schema: {json.dumps(tool.response_type.model_json_schema())}"
                )
                lines.append("")
            lines += [
                "Tool-use rules:",
                "  - Only call tools listed above, using exactly those tool names.",
                "  - Do not invent or reference tools that are not listed above.",
                "  - If a needed capability is not listed, include the requested result in your final JSON response instead.",
                "",
            ]
        if show_final:
            lines += [
                f"{step2}When you have completed your task, respond with JSON matching this generated schema:",
                self.render_response_schema(self.final_response_type),
            ]
            if self.final_response_type is WorkOutput:
                lines += [
                    "",
                    "Rules for your final response:",
                    "  - Provide complete file content for every file you create or modify.",
                    "  - The framework computes the diff via git.",
                    "  - Do not return an empty change set if you created or modified files.",
                    "",
                    "Format rules:",
                    "- files: provide complete file content for every file you create or modify.",
                    '  each entry: {"path": "...", "content": "full file content"}',
                    "- The framework computes the diff via git — do not compute diffs yourself.",
                    "- dependencies: list any new runtime packages required.",
                    "IMPORTANT: your response must include base_version set to the current commit SHA shown above.",
                ]
        return "\n".join(lines)


def build_system_prompt(
    tools: ToolRegistry | None,
    final_response_type: type[BaseModel],
) -> str:
    """Build a system prompt for the given tools and response type."""
    return PromptBuilder(tools, final_response_type).build()


class ResponseParser:
    """Parse raw model text into either a tool call or the final response model."""

    def __init__(self, final_response_type: type[BaseModel]) -> None:
        self.final_response_type = final_response_type

    def parse(self, raw: str) -> ToolCallRequest | BaseModel:
        """Parse raw LLM text into a ToolCallRequest or the final response model."""
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            data: object = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"response is not valid JSON: {e}") from e
        data_dict = cast(dict[str, object], data) if isinstance(data, dict) else None
        if data_dict is not None and data_dict.get("kind") == "tool_call":
            try:
                return ToolCallRequest.model_validate(data_dict)
            except Exception as e:
                raise ValueError(f"invalid tool_call format: {e}") from e
        try:
            return self.final_response_type.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"response does not match {self.final_response_type.__name__}: {e}"
            ) from e


def parse_response(
    raw: str, tools: ToolRegistry | None, final_response_type: type[BaseModel]
) -> ToolCallRequest | BaseModel:
    """Parse a raw LLM response into a tool call or final response model."""
    return ResponseParser(final_response_type).parse(raw)


class TrackedToolExecutor:
    """Execute tool calls and return framework responses."""

    def __init__(self, tools: ToolRegistry | None) -> None:
        self.tools = tools

    async def execute(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        """Execute a tool call and return the response."""
        if self.tools is None:
            return ToolCallResponse(
                kind="tool_response",
                name=request.name,
                success=False,
                result=None,
                error="no tools registered",
            )
        try:
            tool = self.tools.get(request.name)
        except KeyError as e:
            return ToolCallResponse(
                kind="tool_response",
                name=request.name,
                success=False,
                result=None,
                error=str(e),
            )
        try:
            request_obj = tool.request_type.model_validate(request.arguments)
            result = await tool.fn(request_obj)
            if not isinstance(result, tool.response_type):
                raise ValueError(
                    f"tool returned {type(result).__name__}, expected {tool.response_type.__name__}"
                )
            return ToolCallResponse(
                kind="tool_response",
                name=request.name,
                success=True,
                result=result.model_dump(),
            )
        except Exception as e:
            return ToolCallResponse(
                kind="tool_response",
                name=request.name,
                success=False,
                result=None,
                error=str(e),
            )


def _is_empty_work_output(output: WorkOutput) -> bool:
    return not output.files and not output.dependencies


class ToolLoop:
    """Run the mutable provider/tool/retry loop for one agent request."""

    def __init__(
        self,
        request: AgentRequest,
        provider: LLMProvider,
        prompt: str,
        tools: ToolRegistry | None,
        final_response_type: type[BaseModel],
        *,
        max_retries: int = 3,
        max_tool_iterations: int = 25,
        correction_prompt_fn: Callable[[Exception, str], str] | None = None,
        adapter_spec: AdapterSpec | None = None,
        follow_up_builder: Callable[[BaseModel], list[AgentRequest]] | None = None,
    ) -> None:
        self.request = request
        self.provider = provider
        self.prompt = prompt
        self.tools = tools
        self.final_response_type = final_response_type
        self.max_retries = max_retries
        self.max_tool_iterations = max_tool_iterations
        self.correction_prompt_fn = correction_prompt_fn
        self.adapter_spec = adapter_spec
        self.follow_up_builder = follow_up_builder
        self.prompt_builder = PromptBuilder(
            tools,
            final_response_type,
            always_show_final=request.agent_type == AgentType.WORK,
        )
        self.response_parser = ResponseParser(final_response_type)
        self.tool_executor = TrackedToolExecutor(tools)

    async def run(self) -> AgentResponse:
        """Run the tool loop until a final response is produced or limits are exceeded."""
        print(f"[debug] system prompt (initial):\n{self.prompt_builder.build()}")
        print(f"[debug] user prompt:\n{self.prompt}")
        messages: list[ChatMessage] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": self.prompt},
        ]

        requires_nonempty = (
            self.adapter_spec.requires_nonempty_output if self.adapter_spec is not None else True
        )
        any_tool_called = False
        ran_tests_and_passed = False
        retry_count = 0
        invalid_response_diagnostics: list[AgentDiagnostic] = []

        for _ in range(self.max_tool_iterations):
            messages[0] = {
                "role": "system",
                "content": self.prompt_builder.build(),
            }
            raw = await self.provider.chat(messages)

            try:
                parsed = self.response_parser.parse(raw)
                if (
                    self.tools is not None
                    and not any_tool_called
                    and isinstance(parsed, WorkOutput)
                    and _is_empty_work_output(parsed)
                    and requires_nonempty
                ):
                    available_tools = ", ".join(tool.name for tool in self.tools) or "(none)"
                    raise ValueError(
                        "You must call tools before returning a final response. "
                        f"Available tools: {available_tools}. "
                        "Call one of the available tools using this format: "
                        '{"kind": "tool_call", "name": "<tool_name>", "arguments": {}}'
                    )
                if (
                    isinstance(parsed, WorkOutput)
                    and _is_empty_work_output(parsed)
                    and self.request.agent_type == AgentType.WORK
                    and requires_nonempty
                ):
                    return AgentResponse(
                        request_id=self.request.id,
                        status=ResponseStatus.FAILED,
                        error="empty work output: no files or dependencies produced",
                        failure_kind=FailureKind.VALIDATION_REJECTED,
                        ran_tests_and_passed=ran_tests_and_passed,
                    )
            except ValueError as e:
                invalid_response_diagnostics.append(_invalid_response_diagnostic(e, raw))
                if retry_count >= self.max_retries:
                    return AgentResponse(
                        request_id=self.request.id,
                        status=ResponseStatus.FAILED,
                        error=f"agent failed after {self.max_retries} retries: {e}",
                        failure_kind=FailureKind.INVALID_JSON,
                        diagnostics=invalid_response_diagnostics,
                    )
                retry_count += 1
                print(f"  agent retry {retry_count}/{self.max_retries}: {e}")
                correction = (
                    self.correction_prompt_fn(e, raw)
                    if self.correction_prompt_fn
                    else f"Invalid response: {e}. Respond with valid JSON."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                continue

            if isinstance(parsed, ToolCallRequest):
                tool_response = await self.tool_executor.execute(parsed)
                any_tool_called = True
                if (
                    tool_response.name == "run_tests"
                    and tool_response.success
                    and isinstance(tool_response.result, dict)
                    and cast(dict[str, object], tool_response.result).get("passed") is True
                ):
                    ran_tests_and_passed = True
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": tool_response.model_dump_json()})
                continue

            output: ProducerOutput | None = None
            if isinstance(parsed, PlanResponse):
                output = parsed
            elif isinstance(parsed, WorkOutput):
                output = parsed
            return AgentResponse(
                request_id=self.request.id,
                status=ResponseStatus.COMPLETED,
                output=output,
                follow_up=self.follow_up_builder(parsed)
                if self.follow_up_builder is not None
                else [],
                ran_tests_and_passed=ran_tests_and_passed,
            )

        raise RuntimeError(f"agent loop exceeded {self.max_tool_iterations} iterations")


async def run_agent[S: BaseModel](
    request: AgentRequest,
    spec_type: type[S],
    provider: LLMProvider,
    prompt: str,
    tools: ToolRegistry | None = None,
    final_response_type: type[BaseModel] = WorkOutput,
    max_retries: int = 3,
    max_tool_iterations: int = 25,
    correction_prompt_fn: Callable[[Exception, str], str] | None = None,
    adapter_spec: AdapterSpec | None = None,
    follow_up_builder: Callable[[BaseModel], list[AgentRequest]] | None = None,
) -> AgentResponse:
    """Universal agent engine — plain chat loop with structured JSON parsing."""
    try:
        if not isinstance(request.spec, spec_type):
            raise TypeError(f"expected {spec_type.__name__}, got {type(request.spec).__name__}")

        return await ToolLoop(
            request=request,
            provider=provider,
            prompt=prompt,
            tools=tools,
            final_response_type=final_response_type,
            max_retries=max_retries,
            max_tool_iterations=max_tool_iterations,
            correction_prompt_fn=correction_prompt_fn,
            adapter_spec=adapter_spec,
            follow_up_builder=follow_up_builder,
        ).run()

    except Exception as e:
        print(f"agent error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
            failure_kind=_classify_failure(e),
        )
