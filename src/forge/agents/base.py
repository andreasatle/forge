"""Base agent runner — universal engine with plain chat loop and structured JSON parsing."""

import json
import re
from collections.abc import Callable
from typing import TypeVar, cast

import httpx
from pydantic import BaseModel

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
    StateView,
    ToolCallRequest,
    ToolCallResponse,
    WorkSpec,
)
from forge.llm.providers import ChatMessage, LLMProvider, ProviderError
from forge.tools.registry import ToolRegistry

S = TypeVar("S", bound=BaseModel)


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


def _compact_response_schema(response_type: type[BaseModel]) -> dict[str, object]:
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


def _render_response_schema(response_type: type[BaseModel]) -> str:
    """Render final-response schema instructions from the actual Pydantic model."""
    schema = _compact_response_schema(response_type)
    fields = (
        ", ".join(schema["properties"].keys()) if isinstance(schema["properties"], dict) else ""
    )
    lines = [
        f"Final response model: {response_type.__name__}",
        f"Top-level fields: {fields}",
        "Generated JSON schema:",
        json.dumps(schema, indent=2),
    ]
    return "\n".join(lines)


def _build_system_prompt(
    tools: ToolRegistry | None,
    final_response_type: type[BaseModel],
    tracked_delta: DeltaState = DeltaState(),
) -> str:
    has_tools = tools is not None and bool(tools)
    show_final = tools is None or not _is_empty_delta(tracked_delta)
    step2 = "2. " if has_tools and show_final else ""
    lines: list[str] = [
        "You must respond with JSON only — no markdown, no explanation.",
        "",
    ]
    if tools is not None:
        lines += [
            "You have two valid response formats:",
            "",
            "1. To call a tool — use this exact format:",
            '{"kind": "tool_call", "name": "<tool_name>", "arguments": {"key": "value"}}',
            "",
            "Available tools:",
        ]
        for tool in tools:
            lines.append(f"  {tool.name}: {tool.description}")
            lines.append(f"    input schema: {json.dumps(tool.request_type.model_json_schema())}")
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
            _render_response_schema(final_response_type),
        ]
        if final_response_type is DeltaState:
            lines += [
                "",
                "Rules for your final response:",
                "  - Include complete file contents for newly created files.",
                "  - Existing-file edits must identify exact unique text to replace.",
                "  - Do not return an empty change set if you created or modified files.",
            ]
    return "\n".join(lines)


def _parse_response(
    raw: str, tools: ToolRegistry | None, final_response_type: type[BaseModel]
) -> ToolCallRequest | BaseModel:
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
        return final_response_type.model_validate(data)
    except Exception as e:
        raise ValueError(f"response does not match {final_response_type.__name__}: {e}") from e


async def _execute_tool(
    request: ToolCallRequest,
    tools: ToolRegistry | None,
    tracked_delta: DeltaState,
) -> tuple[ToolCallResponse, DeltaState]:
    if tools is None:
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
        tool = tools.get(request.name)
    except KeyError as e:
        return (
            ToolCallResponse(
                kind="tool_response", name=request.name, success=False, result=None, error=str(e)
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
                kind="tool_response", name=request.name, success=True, result=result.model_dump()
            ),
            updated,
        )
    except Exception as e:
        return (
            ToolCallResponse(
                kind="tool_response", name=request.name, success=False, result=None, error=str(e)
            ),
            tracked_delta,
        )


def _to_follow_up(plan: PlanResponse, request: AgentRequest) -> list[AgentRequest]:
    """Convert a PlanResponse into work follow-up nodes with remapped dependencies."""
    if not plan.tasks:
        return []

    # Step 1: bare work nodes — stable IDs, no deps yet
    work_nodes = [
        AgentRequest(
            agent_type=AgentType.WORK,
            source=RequestSource.PLANNER,
            spec=WorkSpec(
                objective=task.objective,
                success_condition=task.success_condition,
                adapter=task.adapter,
                artifact=task.artifact,
                language=task.language,
            ),
        )
        for task in plan.tasks
    ]

    # Step 2: remap work deps — depends_on=[i] means depend on work_nodes[i]
    return [
        work.model_copy(
            update={
                "dependencies": frozenset(
                    work_nodes[j].id for j in task.depends_on if 0 <= j < len(work_nodes)
                )
            }
        )
        for work, task in zip(work_nodes, plan.tasks)
    ]


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


def _render_files(delta: DeltaState, state_view: StateView) -> str:
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


async def run_agent(
    request: AgentRequest,
    spec_type: type[S],
    provider: LLMProvider,
    prompt: str,
    tools: ToolRegistry | None = None,
    final_response_type: type[BaseModel] = DeltaState,
    max_retries: int = 3,
    max_tool_iterations: int = 25,
    correction_prompt_fn: Callable[[Exception, str], str] | None = None,
) -> AgentResponse:
    """Universal agent engine — plain chat loop with structured JSON parsing."""
    try:
        if not isinstance(request.spec, spec_type):
            raise TypeError(f"expected {spec_type.__name__}, got {type(request.spec).__name__}")

        print(
            f"[debug] system prompt (initial):\n{_build_system_prompt(tools, final_response_type, DeltaState())}"
        )
        print(f"[debug] user prompt:\n{prompt}")
        messages: list[ChatMessage] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        tracked_delta = DeltaState()
        any_tool_called = False
        retry_count = 0

        for _ in range(max_tool_iterations):
            messages[0] = {
                "role": "system",
                "content": _build_system_prompt(tools, final_response_type, tracked_delta),
            }
            raw = await provider.chat(messages)

            try:
                parsed = _parse_response(raw, tools, final_response_type)
                if (
                    tools is not None
                    and not any_tool_called
                    and isinstance(parsed, DeltaState)
                    and _is_empty_delta(parsed)
                ):
                    available_tools = ", ".join(tool.name for tool in tools) or "(none)"
                    raise ValueError(
                        "You must call tools before returning a final response. "
                        f"Available tools: {available_tools}. "
                        "Call one of the available tools using this format: "
                        '{"kind": "tool_call", "name": "<tool_name>", "arguments": {}}'
                    )
                if (
                    isinstance(parsed, DeltaState)
                    and _is_empty_delta(_merge_delta(tracked_delta, parsed))
                    and request.agent_type == AgentType.WORK
                ):
                    raise ValueError(
                        "Your response contained an empty DeltaState with no new_files, edits, "
                        "or dependencies. You MUST produce actual file content. Return a "
                        "DeltaState with at least one entry in new_files or edits."
                    )
            except ValueError as e:
                if retry_count >= max_retries:
                    return AgentResponse(
                        request_id=request.id,
                        status=ResponseStatus.FAILED,
                        error=f"agent failed after {max_retries} retries: {e}",
                        failure_kind=FailureKind.INVALID_JSON,
                    )
                retry_count += 1
                print(f"  agent retry {retry_count}/{max_retries}: {e}")
                correction = (
                    correction_prompt_fn(e, raw)
                    if correction_prompt_fn
                    else (
                        f"Invalid response: {e}. new_files and edits must be lists of objects, not dicts. Respond with valid JSON."
                        if final_response_type is DeltaState
                        else f"Invalid response: {e}. Respond with valid JSON."
                    )
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                continue

            if isinstance(parsed, ToolCallRequest):
                tool_response, tracked_delta = await _execute_tool(parsed, tools, tracked_delta)
                any_tool_called = True
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": tool_response.model_dump_json()})
                continue

            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                delta=_merge_delta(tracked_delta, parsed)
                if isinstance(parsed, DeltaState)
                else None,
                follow_up=_to_follow_up(parsed, request)
                if isinstance(parsed, PlanResponse)
                else [],
            )

        raise RuntimeError(f"agent loop exceeded {max_tool_iterations} iterations")

    except Exception as e:
        print(f"agent error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
            failure_kind=_classify_failure(e),
        )
