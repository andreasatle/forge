"""Base agent runner — universal engine with retry, tool loop, and error handling."""

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from forge.core.models import AgentRequest, AgentResponse, ResponseStatus
from forge.llm.providers import LLMProvider
from forge.tools.registry import ToolRegistry

S = TypeVar("S", bound=BaseModel)

_MAX_TOOL_ITERATIONS = 25


async def run_agent(
    request: AgentRequest,
    spec_type: type[S],
    provider: LLMProvider,
    prompt: str,
    tools: ToolRegistry | None = None,
    tool_schema: list[dict] | None = None,  # type: ignore[type-arg]
    max_retries: int = 3,
    max_tool_iterations: int = _MAX_TOOL_ITERATIONS,
    correction_prompt_fn: Callable[[Exception, str], str] | None = None,
    response_fn: Callable[[str], AgentResponse] | None = None,
) -> AgentResponse:
    """Universal agent engine — runs tool loop, retries on ValueError, returns AgentResponse."""
    try:
        if not isinstance(request.spec, spec_type):
            raise TypeError(f"expected {spec_type.__name__}, got {type(request.spec).__name__}")
        if tools is not None:
            return await _run_tool_loop(request, provider, prompt, tools, tool_schema or [], max_tool_iterations)
        return await _run_with_retries(
            request, provider, prompt, max_retries, correction_prompt_fn, response_fn
        )
    except Exception as e:
        print(f"agent error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
        )


async def _run_with_retries(
    request: AgentRequest,
    provider: LLMProvider,
    prompt: str,
    max_retries: int,
    correction_prompt_fn: Callable[[Exception, str], str] | None,
    response_fn: Callable[[str], AgentResponse] | None,
) -> AgentResponse:
    current_prompt = prompt
    last_text = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"  agent retry {attempt - 1}/{max_retries - 1}: {last_error}")
            if correction_prompt_fn is not None:
                current_prompt = correction_prompt_fn(last_error, last_text)  # type: ignore[arg-type]
        try:
            last_text = await provider.chat(current_prompt, provider.max_tokens)
            if response_fn is not None:
                return response_fn(last_text)
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
            )
        except ValueError as e:
            last_error = e
        except Exception as e:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"{type(e).__name__}: {e}",
            )

    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error=f"agent failed after {max_retries} attempts. Last error: {last_error}",
    )


async def _run_tool_loop(
    request: AgentRequest,
    provider: LLMProvider,
    prompt: str,
    tools: ToolRegistry,
    tool_schema: list[dict],  # type: ignore[type-arg]
    max_tool_iterations: int,
) -> AgentResponse:
    messages: list[dict] = [{"role": "user", "content": prompt}]  # type: ignore[type-arg]
    tool_names = ", ".join(t["function"]["name"] for t in tool_schema)
    print(f"[debug] tools available: {tool_names}")

    for _ in range(max_tool_iterations):
        text, tool_calls = await provider.chat_with_tools(messages, tool_schema, provider.max_tokens)
        if text is not None:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
            )
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            print(f"[debug] tool call: {name!r} args={args!r}")
            tool = tools.get(name)
            try:
                request_obj = tool.request_type.model_validate(args)
            except Exception as e:
                raise ValueError(f"tool {name!r} request validation failed: {e}") from e
            result = await tool.fn(request_obj)  # type: ignore[arg-type]
            if not isinstance(result, tool.response_type):
                raise ValueError(
                    f"tool {name!r} returned {type(result).__name__}, expected {tool.response_type.__name__}"
                )
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.model_dump_json()})

    raise RuntimeError(
        f"agentic loop exceeded {max_tool_iterations} iterations without a final response"
    )
