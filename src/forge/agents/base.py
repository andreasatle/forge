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
    IntegrateSpec,
    PlanResponse,
    RequestSource,
    ResponseStatus,
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


def _build_system_prompt(tools: ToolRegistry | None, final_response_type: type[BaseModel], tracked_delta: DeltaState = DeltaState()) -> str:
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
            lines.append(f"    response schema: {json.dumps(tool.response_type.model_json_schema())}")
            lines.append("")
        lines += [
            "IMPORTANT: Use tools only to READ and understand the existing codebase.",
            "  - Use list_files to discover what exists",
            "  - Use read_file to read existing files",
            "  - Use run_tests to check current test status",
            "  - All file creation and editing goes in your final JSON response — not via tools",
            "",
        ]
    if show_final:
        if final_response_type is DeltaState:
            lines += [
                f"{step2}When ALL your work is done, respond with this exact JSON structure:",
                "{",
                '  "new_files": [',
                '    {"path": "src/example.py", "content": "# complete file content here\\n"}',
                '  ],',
                '  "edits": [',
                '    {"path": "src/existing.py", "old": "exact_string_to_replace", "new": "replacement_string"}',
                '  ],',
                '  "dependencies": ["<package-name>"]',
                "}",
                "",
                "Rules for your final response:",
                "  - new_files: every new file you created, with its FULL content as a string (not a summary)",
                "  - edits: every change to an existing file — old must be the exact unique string you replaced",
                "  - dependencies: every package you installed",
                "  - NEVER return empty new_files and edits if you created or modified any files",
                "  - Include the COMPLETE content of every file — do not truncate or abbreviate",
            ]
        elif final_response_type is PlanResponse:
            lines += [
                f"{step2}When you have completed your task, respond with this exact JSON structure:",
                json.dumps(
                    {
                        "kind": "plan",
                        "tasks": [
                            {
                                "objective": "<task description>",
                                "success_condition": "<how to verify it is done>",
                                "adapter": "coding",
                                "artifact": "<artifact-name>",
                                "language": "python",
                                "depends_on": [],
                            }
                        ],
                    },
                    indent=2,
                ),
            ]
        else:
            lines += [
                f"{step2}When you have completed your task:",
                json.dumps(final_response_type.model_json_schema(), indent=2),
            ]
    lines += ["", "Respond with JSON only."]
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
                kind="tool_response", name=request.name, success=False, result=None, error="no tools registered"
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
            raise ValueError(f"tool returned {type(result).__name__}, expected {tool.response_type.__name__}")
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
                updated = updated.model_copy(update={"dependencies": list(updated.dependencies) + [pkg]})
        print(
            f"[debug] tracked: tool={request.name}"
            f" delta_files={len(updated.new_files)}"
            f" delta_edits={len(updated.edits)}"
            f" delta_deps={len(updated.dependencies)}"
        )
        return (
            ToolCallResponse(kind="tool_response", name=request.name, success=True, result=result.model_dump()),
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
    if not plan.tasks:
        return []
    work_requests = [
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
    id_map = {i: req.id for i, req in enumerate(work_requests)}
    work_with_deps = [
        req.model_copy(update={
            "dependencies": frozenset(id_map[j] for j in task.depends_on if 0 <= j < len(work_requests))
        })
        for req, task in zip(work_requests, plan.tasks)
    ]
    first_task = plan.tasks[0]
    integrate_node = AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.PLANNER,
        spec=IntegrateSpec(
            objective="integrate completed work",
            artifact=first_task.artifact,
            language=first_task.language,
            work_request_ids=[req.id for req in work_with_deps],
        ),
        dependencies=frozenset(req.id for req in work_with_deps),
    )
    return [*work_with_deps, integrate_node]


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

        print(f"[debug] system prompt (initial):\n{_build_system_prompt(tools, final_response_type, DeltaState())}")
        print(f"[debug] user prompt:\n{prompt}")
        messages: list[ChatMessage] = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        tracked_delta = DeltaState()
        any_tool_called = False
        retry_count = 0

        for _ in range(max_tool_iterations):
            messages[0] = {"role": "system", "content": _build_system_prompt(tools, final_response_type, tracked_delta)}
            raw = await provider.chat(messages)

            try:
                parsed = _parse_response(raw, tools, final_response_type)
                if (
                    tools is not None
                    and not any_tool_called
                    and isinstance(parsed, DeltaState)
                    and _is_empty_delta(parsed)
                ):
                    raise ValueError(
                        'You must call tools before returning a final response. '
                        'Call a tool using this exact format:\n'
                        '{"kind": "tool_call", "name": "list_files", "arguments": {}}'
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
                    else f"Invalid response: {e}. Respond with valid JSON."
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
                delta=_merge_delta(tracked_delta, parsed) if isinstance(parsed, DeltaState) else None,
                follow_up=_to_follow_up(parsed, request) if isinstance(parsed, PlanResponse) else [],
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
