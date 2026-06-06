"""Base agent runner — universal engine with retry, tool loop, and error handling."""

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from forge.core.models import AgentRequest, AgentResponse, ResponseStatus
from forge.llm.providers import LLMProvider
from forge.tools.registry import ToolRegistry

S = TypeVar("S", bound=BaseModel)

_MAX_TOOL_ITERATIONS = 10


async def run_agent(
    request: AgentRequest,
    spec_type: type[S],
    provider: LLMProvider,
    prompt: str,
    tools: ToolRegistry | None = None,
    tool_schema: list[dict] | None = None,  # type: ignore[type-arg]
    max_retries: int = 3,
    correction_prompt_fn: Callable[[Exception, str], str] | None = None,
    response_fn: Callable[[str], AgentResponse] | None = None,
) -> AgentResponse:
    """Universal agent engine — runs tool loop, retries on ValueError, returns AgentResponse."""
    try:
        if not isinstance(request.spec, spec_type):
            raise TypeError(f"expected {spec_type.__name__}, got {type(request.spec).__name__}")
        if tools is not None:
            return await _run_tool_loop(request, provider, prompt, tools, tool_schema or [])
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
                delta={"result": last_text},
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
) -> AgentResponse:
    messages: list[dict] = [{"role": "user", "content": prompt}]  # type: ignore[type-arg]

    for _ in range(_MAX_TOOL_ITERATIONS):
        text, tool_calls = await provider.chat_with_tools(messages, tool_schema, provider.max_tokens)
        if text is not None:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                delta={"result": text},
            )
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            print(f"[debug] tool call: {name!r} args={args!r}")
            tool = tools.get(name)
            result = await tool.fn(**args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    raise RuntimeError(
        f"agentic loop exceeded {_MAX_TOOL_ITERATIONS} iterations without a final response"
    )
