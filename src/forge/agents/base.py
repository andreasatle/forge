"""Base agent runner — universal engine with plain chat loop and structured JSON parsing."""

import json
import re
from collections.abc import Callable
from typing import cast

import httpx
from pydantic import BaseModel

from forge.adapters.registry import AdapterSpec
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    Edit,
    FailureKind,
    FileWrite,
    ResponseStatus,
    StateView,
    ToolCallRequest,
    ToolCallResponse,
)
from forge.llm.providers import ChatMessage, LLMProvider, ProviderError
from forge.tools.registry import ToolRegistry


class ToolError(Exception):
    """Raised when a tool call fails during execution (distinct from JSON parse errors)."""


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

    def build(self, tracked_delta: DeltaState | None = None) -> str:
        """Build the system prompt string, showing tool and schema sections as appropriate."""
        tracked_delta = tracked_delta or DeltaState()
        has_tools = self.tools is not None and bool(self.tools)
        show_final = (
            self.tools is None or not _is_empty_delta(tracked_delta) or self.always_show_final
        )
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
            if self.final_response_type is DeltaState:
                lines += [
                    "",
                    "Rules for your final response:",
                    "  - Include complete file contents for newly created files.",
                    "  - Existing-file edits must identify exact unique text to replace.",
                    "  - Do not return an empty change set if you created or modified files.",
                    "",
                    "Format rules:",
                    "- new_files: create files that do not exist yet",
                    '  each entry: {"path": "...", "content": "full file content"}',
                    "- edits: replace existing text in existing files",
                    '  each entry: {"path": "...", "old": "exact text to replace", "new": "replacement"}',
                    "- Never put file content in edits.",
                    "- Never put old/new strings in new_files.",
                    "IMPORTANT: your response must include base_version set to the current state version shown above.",
                ]
        return "\n".join(lines)


def build_system_prompt(
    tools: ToolRegistry | None,
    final_response_type: type[BaseModel],
    tracked_delta: DeltaState = DeltaState(),
) -> str:
    """Build a system prompt for the given tools and response type."""
    return PromptBuilder(tools, final_response_type).build(tracked_delta)


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
    """Execute tool calls and track framework-observed DeltaState changes."""

    def __init__(self, tools: ToolRegistry | None) -> None:
        self.tools = tools

    async def execute(
        self,
        request: ToolCallRequest,
        tracked_delta: DeltaState,
    ) -> tuple[ToolCallResponse, DeltaState]:
        """Execute a tool call and return the response plus updated tracked delta."""
        if self.tools is None:
            return (
                ToolCallResponse(
                    kind="tool_response",
                    name=request.name,
                    success=False,
                    result=None,
                    error="no tools registered",
                ),
                tracked_delta,
            )
        try:
            tool = self.tools.get(request.name)
        except KeyError as e:
            return (
                ToolCallResponse(
                    kind="tool_response",
                    name=request.name,
                    success=False,
                    result=None,
                    error=str(e),
                ),
                tracked_delta,
            )
        try:
            request_obj = tool.request_type.model_validate(request.arguments)
            result = await tool.fn(request_obj)
            if not isinstance(result, tool.response_type):
                raise ValueError(
                    f"tool returned {type(result).__name__}, expected {tool.response_type.__name__}"
                )
            updated = tracked_delta
            if request.name == "write_file":
                fw = FileWrite(path=request.arguments["path"], content=request.arguments["content"])
                new_files = [f for f in updated.new_files if f.path != fw.path] + [fw]
                updated = updated.model_copy(update={"new_files": new_files})
            elif request.name == "replace_in_file":
                edit = Edit(
                    path=request.arguments["path"],
                    old=request.arguments["old"],
                    new=request.arguments["new"],
                )
                updated = updated.model_copy(update={"edits": list(updated.edits) + [edit]})
            elif request.name == "add_dependency":
                pkg = request.arguments["package"]
                if pkg not in updated.dependencies:
                    updated = updated.model_copy(
                        update={"dependencies": list(updated.dependencies) + [pkg]}
                    )
            print(
                f"[debug] tracked: tool={request.name}"
                f" delta_files={len(updated.new_files)}"
                f" delta_edits={len(updated.edits)}"
                f" delta_deps={len(updated.dependencies)}"
            )
            return (
                ToolCallResponse(
                    kind="tool_response",
                    name=request.name,
                    success=True,
                    result=result.model_dump(),
                ),
                updated,
            )
        except Exception as e:
            return (
                ToolCallResponse(
                    kind="tool_response",
                    name=request.name,
                    success=False,
                    result=None,
                    error=str(e),
                ),
                tracked_delta,
            )


def _merge_delta(tracked: DeltaState, reported: DeltaState) -> DeltaState:
    """Merge tracked (framework-observed) and reported (LLM-declared) deltas; tracked wins on conflict."""
    files: dict[str, FileWrite] = {fw.path: fw for fw in reported.new_files}
    files.update({fw.path: fw for fw in tracked.new_files})
    edits: dict[str, Edit] = {e.path: e for e in reported.edits}
    edits.update({e.path: e for e in tracked.edits})
    seen: set[str] = set()
    deps: list[str] = []
    for d in [*tracked.dependencies, *reported.dependencies]:
        if d not in seen:
            seen.add(d)
            deps.append(d)
    return DeltaState(new_files=list(files.values()), edits=list(edits.values()), dependencies=deps)


def _is_empty_delta(delta: DeltaState) -> bool:
    return not delta.new_files and not delta.edits and not delta.dependencies


def render_files(delta: DeltaState, state_view: StateView) -> str:
    """Render produced files, applied edits, and existing artifact state as a readable block."""
    lines: list[str] = []
    if delta.new_files:
        lines.append("Files produced:")
        for fw in delta.new_files:
            lines += [f"\nFile: {fw.path}", "```", fw.content, "```"]
    if delta.edits:
        if lines:
            lines.append("")
        lines.append("Edits applied:")
        for edit in delta.edits:
            lines += [f"\nFile: {edit.path}", "  old:", edit.old, "  new:", edit.new]
    if not delta.new_files and not delta.edits:
        lines.append("No files or edits were produced.")
    if state_view.files:
        if lines:
            lines.append("")
        lines.append("Existing artifact files:")
        for fv in state_view.files:
            lines += [f"\nFile: {fv.path}", "```", fv.content, "```"]
    return "\n".join(lines)


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
        print(f"[debug] system prompt (initial):\n{self.prompt_builder.build(DeltaState())}")
        print(f"[debug] user prompt:\n{self.prompt}")
        messages: list[ChatMessage] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": self.prompt},
        ]

        requires_nonempty = (
            self.adapter_spec.requires_nonempty_output if self.adapter_spec is not None else True
        )
        tracked_delta = DeltaState()
        any_tool_called = False
        ran_tests_and_passed = False
        retry_count = 0

        for _ in range(self.max_tool_iterations):
            messages[0] = {
                "role": "system",
                "content": self.prompt_builder.build(tracked_delta),
            }
            raw = await self.provider.chat(messages)

            try:
                parsed = self.response_parser.parse(raw)
                if (
                    self.tools is not None
                    and not any_tool_called
                    and isinstance(parsed, DeltaState)
                    and _is_empty_delta(parsed)
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
                    isinstance(parsed, DeltaState)
                    and _is_empty_delta(_merge_delta(tracked_delta, parsed))
                    and self.request.agent_type == AgentType.WORK
                    and requires_nonempty
                ):
                    return AgentResponse(
                        request_id=self.request.id,
                        status=ResponseStatus.FAILED,
                        error="empty delta: no new_files, edits, or dependencies produced",
                        failure_kind=FailureKind.VALIDATION_REJECTED,
                        delta=DeltaState(),
                        ran_tests_and_passed=ran_tests_and_passed,
                    )
            except ValueError as e:
                if retry_count >= self.max_retries:
                    return AgentResponse(
                        request_id=self.request.id,
                        status=ResponseStatus.FAILED,
                        error=f"agent failed after {self.max_retries} retries: {e}",
                        failure_kind=FailureKind.INVALID_JSON,
                    )
                retry_count += 1
                print(f"  agent retry {retry_count}/{self.max_retries}: {e}")
                correction = (
                    self.correction_prompt_fn(e, raw)
                    if self.correction_prompt_fn
                    else (
                        f"Invalid response: {e}. "
                        'new_files must be a list of objects: [{"path": "...", "content": "..."}]\n'
                        'edits must be a list of objects: [{"path": "...", "old": "...", "new": "..."}]\n'
                        "Do not use dicts or nested objects. Respond with valid JSON."
                        if self.final_response_type is DeltaState
                        else f"Invalid response: {e}. Respond with valid JSON."
                    )
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                continue

            if isinstance(parsed, ToolCallRequest):
                tool_response, tracked_delta = await self.tool_executor.execute(
                    parsed, tracked_delta
                )
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

            return AgentResponse(
                request_id=self.request.id,
                status=ResponseStatus.COMPLETED,
                delta=_merge_delta(tracked_delta, parsed)
                if isinstance(parsed, DeltaState)
                else None,
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
    final_response_type: type[BaseModel] = DeltaState,
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
