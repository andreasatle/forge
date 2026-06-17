"""Base agent runner — universal engine with plain chat loop and structured JSON parsing."""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import httpx
from pydantic import BaseModel, ValidationError

from forge.adapters.registry import AdapterSpec
from forge.core.models import (
    AgentDiagnostic,
    AgentMessageKind,
    AgentRequest,
    AgentResponse,
    AgentType,
    FailureKind,
    PlannerOutputModel,
    ProducerOutput,
    ResponseStatus,
    ToolCallResponse,
    ToolTurn,
    WorkOutput,
)
from forge.llm.providers import ChatMessage, LLMProvider, ProviderError
from forge.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Raised when a tool call fails during execution (distinct from JSON parse errors)."""


_MAX_RAW_RESPONSE_DIAGNOSTIC_CHARS = 4000
_MAX_BAD_VALUE_CHARS = 300
_MAX_RECENT_TOOL_CALLS = 5


@dataclass
class ToolLoopState:
    """Iteration-level telemetry accumulated during one ToolLoop run."""

    mutation_count: int = 0
    verification_count: int = 0
    mutating_tool_succeeded: bool = False
    verification_passed: bool = False
    final_response_only: bool = False
    iteration_at_last_mutation: int = 0
    iteration_at_completion_pressure: int = 0
    last_verification_fingerprint: str | None = None
    previous_verification_fingerprint: str | None = None
    verification_stable: bool = False
    verification_stable_count: int = 0
    last_progress_iteration: int | None = None
    iterations_since_progress: int | None = None
    iteration_at_verification_stability: int | None = None


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


def _max_iterations_diagnostic(
    recent_tool_calls: list[str],
    last_raw: str,
    *,
    state: ToolLoopState,
    verification_required: bool,
) -> AgentDiagnostic:
    """Build a bounded diagnostic for a max-iterations failure."""
    tool_summary = ", ".join(recent_tool_calls) if recent_tool_calls else "(none)"
    message = (
        f"last_tool_calls=[{tool_summary}] "
        f"ran_tests_and_passed={state.verification_passed} "
        f"final_response_only={state.final_response_only} "
        f"has_run_tests={verification_required} "
        f"mutating_tool_succeeded={state.mutating_tool_succeeded} "
        f"verification_stable={state.verification_stable} "
        f"verification_stable_count={state.verification_stable_count} "
        f"iteration_at_verification_stability={state.iteration_at_verification_stability} "
        f"last_progress_iteration={state.last_progress_iteration} "
        f"iterations_since_progress={state.iterations_since_progress}"
    )
    return AgentDiagnostic(
        kind="max_iterations",
        message=message,
        raw_response_excerpt=_truncate_text(last_raw, _MAX_RAW_RESPONSE_DIAGNOSTIC_CHARS)
        if last_raw
        else None,
    )


def _default_correction_prompt(error: Exception) -> str:
    """Return generic structured-output repair guidance for the next model turn."""
    return "\n".join(
        [
            f"Invalid response: {error}.",
            "Respond with valid JSON only.",
            'Tool-call shape: {"kind":"tool","name":"<tool_name>","arguments":{}}',
            'Final-answer shape: {"kind":"final","output":{<output object>}}',
        ]
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
        for key in ("oneOf", "anyOf", "discriminator"):
            if key in schema:
                compact[key] = schema[key]
        return compact

    @classmethod
    def render_response_schema(cls, response_type: type[BaseModel]) -> str:
        """Render output object schema instructions from the actual Pydantic model."""
        schema = cls.compact_response_schema(response_type)
        fields = (
            ", ".join(cast(dict[str, object], schema["properties"]).keys())
            if isinstance(schema["properties"], dict)
            else ""
        )
        lines = [
            f"Output object model: {response_type.__name__}",
            f"Output object fields: {fields}",
            "Generated JSON schema:",
            json.dumps(schema, indent=2),
        ]
        return "\n".join(lines)

    def build(self) -> str:
        """Build the system prompt string, showing tool and schema sections as appropriate."""
        has_tools = self.tools is not None and bool(self.tools)
        show_final = self.tools is None or self.always_show_final

        lines: list[str] = [
            "You must respond with JSON only — no markdown, no explanation.",
            "",
        ]

        if self.tools is not None and has_tools:
            lines += [
                "Every response must be exactly one of two JSON shapes:",
                "",
                "Shape 1 — tool call:",
                '{"kind":"tool","name":"<tool_name>","arguments":{"key":"value"}}',
                "",
                "Tool-call rules:",
                '  - Use kind="tool" for every tool call; put the tool name in name.',
                "  - Never put a tool name in kind.",
                "  - arguments must match the tool's input schema; use {} when empty.",
                "  - Only call tools listed below.",
                "  - If a needed capability is not listed, include the result in your final response instead.",
                "",
                "Available tools:",
            ]
            for tool in self.tools:
                lines.append(f"  {tool.name}: {tool.description}")
                lines.append(
                    f'    invocation shape: {{"kind":"tool","name":"{tool.name}","arguments":{{...}}}}'
                )
                lines.append(
                    f"    input schema: {json.dumps(tool.request_type.model_json_schema())}"
                )
                lines.append(
                    f"    response schema: {json.dumps(tool.response_type.model_json_schema())}"
                )
                lines.append("")

        if show_final:
            shape_label = "Shape 2" if has_tools else "Final response"
            lines += [
                f"{shape_label} — task complete:",
                '{"kind":"final","output":{<output object — schema below>}}',
                "",
                self.render_response_schema(self.final_response_type),
            ]
            if self.final_response_type is WorkOutput:
                lines += [
                    "",
                    "Rules for your final response:",
                    "  - Modify files directly in the assigned worktree before responding.",
                    '  - After all required edits are made and tests pass, stop calling tools and return final JSON with kind="final" and output containing kind and summary. Do not return final JSON while tests are failing.',
                    "  - The framework uses git status and git diff as the source of truth.",
                    "  - Do not include complete file contents in your final response.",
                    "",
                    "Format rules (for the output object):",
                    '- kind: must be "work_output".',
                    "- summary: briefly describe the worktree changes you made.",
                    "- Dependency changes must be made in package manager files in the worktree.",
                ]
        return "\n".join(lines)


def build_system_prompt(
    tools: ToolRegistry | None,
    final_response_type: type[BaseModel],
) -> str:
    """Build a system prompt for the given tools and response type."""
    return PromptBuilder(tools, final_response_type).build()


def _strip_json_fence(text: str) -> str:
    """Return inner JSON if the entire response is one fenced block; otherwise return stripped text."""
    stripped = text.strip()
    if stripped.startswith("```json\n"):
        inner_start = stripped.index("\n") + 1
    elif stripped.startswith("```\n"):
        inner_start = 4
    else:
        return stripped
    if not stripped.endswith("\n```"):
        return stripped
    inner = stripped[inner_start:-4]
    if "```" in inner:
        return stripped
    return inner


class ResponseParser:
    """Parse raw model text into either a tool call or the final response model."""

    def __init__(
        self, final_response_type: type[BaseModel], tools: ToolRegistry | None = None
    ) -> None:
        self.final_response_type = final_response_type
        self.tools = tools

    def parse(self, raw: str) -> ToolTurn | BaseModel:
        """Parse raw LLM text into a ToolTurn or the final response model."""
        try:
            data: object = json.loads(_strip_json_fence(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"response is not valid JSON: {e}") from e
        data_dict = cast(dict[str, object], data) if isinstance(data, dict) else None
        kind = data_dict.get("kind") if data_dict is not None else None

        if kind == "tool":
            try:
                return ToolTurn.model_validate(data_dict)
            except Exception as e:
                raise ValueError(f"invalid tool turn: {e}") from e

        if kind == "final":
            try:
                if data_dict is None or "output" not in data_dict:
                    raise ValueError("missing output field")
                return self.final_response_type.model_validate(data_dict["output"])
            except Exception as e:
                raise ValueError(f"invalid final turn: {e}") from e

        raise ValueError(f"unknown protocol kind: {kind!r}; expected 'tool' or 'final'")


def parse_response(
    raw: str, tools: ToolRegistry | None, final_response_type: type[BaseModel]
) -> ToolTurn | BaseModel:
    """Parse a raw LLM response into a tool call or final response model."""
    return ResponseParser(final_response_type, tools).parse(raw)


class TrackedToolExecutor:
    """Execute tool calls and return framework responses."""

    def __init__(self, tools: ToolRegistry | None) -> None:
        self.tools = tools

    async def execute(
        self,
        request: ToolTurn,
    ) -> ToolCallResponse:
        """Execute a tool call and return the response."""
        if self.tools is None:
            return ToolCallResponse(
                kind=AgentMessageKind.TOOL_RESPONSE,
                name=request.name,
                success=False,
                result=None,
                error="no tools registered",
            )
        try:
            tool = self.tools.get(request.name)
        except KeyError as e:
            return ToolCallResponse(
                kind=AgentMessageKind.TOOL_RESPONSE,
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
                kind=AgentMessageKind.TOOL_RESPONSE,
                name=request.name,
                success=True,
                result=result.model_dump(),
            )
        except Exception as e:
            return ToolCallResponse(
                kind=AgentMessageKind.TOOL_RESPONSE,
                name=request.name,
                success=False,
                result=None,
                error=str(e),
            )


def _is_empty_work_output(output: WorkOutput) -> bool:
    return not output.summary.strip()


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

        if adapter_spec is not None:
            self._mutating_tool_names = frozenset(adapter_spec.mutating_tools)
            self._verification_tool_names = frozenset(adapter_spec.verification_tools)
            self._verification_required = (
                adapter_spec.verification_required
                if adapter_spec.verification_required is not None
                else (
                    tools is not None
                    and any(t.name in self._verification_tool_names for t in tools)
                )
            )
        else:
            self._mutating_tool_names = frozenset({"write_file", "replace_in_file"})
            self._verification_tool_names = frozenset({"run_tests"})
            self._verification_required = tools is not None and any(
                t.name in self._verification_tool_names for t in tools
            )

        self.prompt_builder = PromptBuilder(
            tools,
            final_response_type,
            always_show_final=request.agent_type == AgentType.WORK,
        )
        self.response_parser = ResponseParser(final_response_type, tools)
        self.tool_executor = TrackedToolExecutor(tools)

    async def run(self) -> AgentResponse:
        """Run the tool loop until a final response is produced or limits are exceeded."""
        _logger.debug("system prompt (initial):\n%s", self.prompt_builder.build())
        _logger.debug("user prompt:\n%s", self.prompt)
        messages: list[ChatMessage] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": self.prompt},
        ]

        requires_nonempty = (
            self.adapter_spec.requires_nonempty_output if self.adapter_spec is not None else True
        )
        any_tool_called = False
        state = ToolLoopState()
        retry_count = 0
        invalid_response_diagnostics: list[AgentDiagnostic] = []
        recent_tool_calls: list[str] = []
        last_raw: str = ""

        for iteration in range(self.max_tool_iterations):
            active_builder = (
                PromptBuilder(None, self.final_response_type, always_show_final=True)
                if state.final_response_only
                else self.prompt_builder
            )
            messages[0] = {
                "role": "system",
                "content": active_builder.build(),
            }
            raw = await self.provider.chat(messages)
            last_raw = raw

            try:
                parsed = self.response_parser.parse(raw)
                if state.final_response_only and isinstance(parsed, ToolTurn):
                    if state.verification_passed:
                        raise ValueError(
                            "File changes are complete. "
                            "Return final WorkOutput JSON now instead of calling tools."
                        )
                    raise ValueError(
                        "Verification results have stabilized with the same failing result. "
                        "Stop calling tools and return final WorkOutput JSON now."
                    )
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
                        '{"kind":"tool","name":"<tool_name>","arguments":{}}'
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
                        error="empty work output: no completion summary produced",
                        failure_kind=FailureKind.VALIDATION_REJECTED,
                        ran_tests_and_passed=state.verification_passed,
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
                _logger.debug("agent retry %d/%d: %s", retry_count, self.max_retries, e)
                correction = (
                    self.correction_prompt_fn(e, raw)
                    if self.correction_prompt_fn
                    else _default_correction_prompt(e)
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                continue

            if isinstance(parsed, ToolTurn):
                tool_response = await self.tool_executor.execute(parsed)
                any_tool_called = True
                recent_tool_calls.append(parsed.name)
                recent_tool_calls = recent_tool_calls[-_MAX_RECENT_TOOL_CALLS:]
                if parsed.name in self._mutating_tool_names and tool_response.success:
                    state.mutation_count += 1
                    state.mutating_tool_succeeded = True
                    state.iteration_at_last_mutation = iteration
                    state.last_progress_iteration = iteration
                coercion: str | None = None
                if parsed.name in self._verification_tool_names:
                    raw_result = tool_response.result
                    result_for_fp: dict[str, object] = (
                        cast(dict[str, object], raw_result) if isinstance(raw_result, dict) else {}
                    )
                    fingerprint = hashlib.sha256(
                        json.dumps(result_for_fp, sort_keys=True).encode()
                    ).hexdigest()
                    prev_fp = state.last_verification_fingerprint
                    state.previous_verification_fingerprint = prev_fp
                    state.last_verification_fingerprint = fingerprint
                    if prev_fp is not None:
                        if fingerprint == prev_fp:
                            state.verification_stable_count += 1
                        else:
                            state.verification_stable_count = 0
                            state.last_progress_iteration = iteration
                    result_passed = (
                        tool_response.success
                        and isinstance(raw_result, dict)
                        and cast(dict[str, object], raw_result).get("passed") is True
                    )
                    newly_stable = not result_passed and state.verification_stable_count >= 1
                    if newly_stable and not state.verification_stable:
                        state.iteration_at_verification_stability = iteration
                    state.verification_stable = newly_stable
                    if result_passed:
                        state.verification_count += 1
                        state.verification_passed = True
                        state.final_response_only = True
                        state.iteration_at_completion_pressure = iteration
                        coercion = (
                            "Tests passed. Stop calling tools and return final WorkOutput JSON now."
                        )
                    elif state.verification_stable:
                        state.final_response_only = True
                        if state.iteration_at_completion_pressure == 0:
                            state.iteration_at_completion_pressure = iteration
                        coercion = (
                            "Verification results have stabilized with the same failing result. "
                            "Stop calling tools and return final WorkOutput JSON now."
                        )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": tool_response.model_dump_json()})
                if coercion is not None:
                    messages.append({"role": "user", "content": coercion})
                continue

            output: ProducerOutput | None = None
            if isinstance(parsed, WorkOutput):
                output = parsed
            elif isinstance(parsed, PlannerOutputModel):
                output = parsed.root
            return AgentResponse(
                request_id=self.request.id,
                status=ResponseStatus.COMPLETED,
                output=output,
                ran_tests_and_passed=state.verification_passed,
            )

        if state.last_progress_iteration is not None:
            state.iterations_since_progress = (
                self.max_tool_iterations - 1
            ) - state.last_progress_iteration
        diagnostic = _max_iterations_diagnostic(
            recent_tool_calls,
            last_raw,
            state=state,
            verification_required=self._verification_required,
        )
        _logger.debug(
            "tool loop exhausted after %d iterations: %s",
            self.max_tool_iterations,
            diagnostic.message,
        )
        return AgentResponse(
            request_id=self.request.id,
            status=ResponseStatus.FAILED,
            error=f"agent loop exceeded {self.max_tool_iterations} iterations",
            failure_kind=FailureKind.MAX_ITERATIONS,
            ran_tests_and_passed=state.verification_passed,
            diagnostics=[diagnostic],
        )


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
        ).run()

    except Exception as e:
        _logger.exception("agent error: %s", e)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
            failure_kind=_classify_failure(e),
        )
