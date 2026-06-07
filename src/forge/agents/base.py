"""Base agent runner — universal engine with plain chat loop and structured JSON parsing."""

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    PlanResponse,
    RequestSource,
    ResponseStatus,
    ToolCallRequest,
    ToolCallResponse,
    WorkSpec,
)
from forge.llm.providers import LLMProvider
from forge.tools.registry import ToolRegistry

S = TypeVar("S", bound=BaseModel)


def _build_system_prompt(tools: ToolRegistry | None, final_response_type: type[BaseModel]) -> str:
    lines = [
        "You must respond with JSON only — no markdown, no explanation.",
        "",
        "You have two valid response formats:",
        "",
        "1. To call a tool:",
        '{"kind": "tool_call", "name": "<tool_name>", "arguments": {<arguments>}}',
        "",
        "Available tools:",
    ]
    if tools is not None:
        for tool in tools._tools.values():
            lines.append(f"  {tool.name}: {tool.description}")
            lines.append(f"    input: {json.dumps(tool.request_type.model_json_schema())}")
            lines.append(f"    response you will receive: {json.dumps(tool.response_type.model_json_schema())}")
            lines.append("")
    lines += [
        "2. When you have completed your task:",
        json.dumps(final_response_type.model_json_schema()),
        "",
        "Respond with JSON only.",
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
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not valid JSON: {e}") from e
    if isinstance(data, dict) and data.get("kind") == "tool_call":
        try:
            return ToolCallRequest.model_validate(data)
        except Exception as e:
            raise ValueError(f"invalid tool_call format: {e}") from e
    try:
        return final_response_type.model_validate(data)
    except Exception as e:
        raise ValueError(f"response does not match {final_response_type.__name__}: {e}") from e


async def _execute_tool(request: ToolCallRequest, tools: ToolRegistry | None) -> ToolCallResponse:
    if tools is None:
        return ToolCallResponse(
            kind="tool_response", name=request.name, success=False, result=None, error="no tools registered"
        )
    try:
        tool = tools.get(request.name)
    except KeyError as e:
        return ToolCallResponse(
            kind="tool_response", name=request.name, success=False, result=None, error=str(e)
        )
    try:
        request_obj = tool.request_type.model_validate(request.arguments)
        result = await tool.fn(request_obj)
        if not isinstance(result, tool.response_type):
            raise ValueError(f"tool returned {type(result).__name__}, expected {tool.response_type.__name__}")
        return ToolCallResponse(
            kind="tool_response", name=request.name, success=True, result=result.model_dump()
        )
    except Exception as e:
        return ToolCallResponse(
            kind="tool_response", name=request.name, success=False, result=None, error=str(e)
        )


def _to_follow_up(plan: PlanResponse, request: AgentRequest) -> list[AgentRequest]:
    requests = [
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
    id_map = {i: req.id for i, req in enumerate(requests)}
    return [
        req.model_copy(update={
            "dependencies": frozenset(id_map[j] for j in task.depends_on if 0 <= j < len(requests))
        })
        for req, task in zip(requests, plan.tasks)
    ]


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

        system_prompt = _build_system_prompt(tools, final_response_type)
        messages: list[dict] = [  # type: ignore[type-arg]
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        retry_count = 0

        for _ in range(max_tool_iterations):
            raw = await provider.chat(messages)

            try:
                parsed = _parse_response(raw, tools, final_response_type)
            except ValueError as e:
                if retry_count >= max_retries:
                    return AgentResponse(
                        request_id=request.id,
                        status=ResponseStatus.FAILED,
                        error=f"agent failed after {max_retries} retries: {e}",
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
                tool_response = await _execute_tool(parsed, tools)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": tool_response.model_dump_json()})
                continue

            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                delta=parsed if isinstance(parsed, DeltaState) else None,
                follow_up=_to_follow_up(parsed, request) if isinstance(parsed, PlanResponse) else [],
            )

        raise RuntimeError(f"agent loop exceeded {max_tool_iterations} iterations")

    except Exception as e:
        print(f"agent error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
        )
