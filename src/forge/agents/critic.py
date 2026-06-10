"""Critic agent — reviews worker output against the success condition."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import _build_system_prompt, _parse_response, _render_files
from forge.core.models import AgentRequest, CriticFinding, DeltaState, StateView, WorkSpec
from forge.llm.providers import ChatMessage, LLMProvider


async def critic_agent(
    request: AgentRequest,
    state_view: StateView,
    delta: DeltaState,
    provider: LLMProvider,
    registry: AdapterRegistry,
    max_retries: int = 3,
) -> CriticFinding:
    """Review a worker's DeltaState against the request's success condition."""
    spec = request.spec
    if not isinstance(spec, WorkSpec):
        raise TypeError(f"expected WorkSpec, got {type(spec).__name__}")

    adapter = registry.get("critic")
    user_prompt = adapter.prompt_template.format(
        objective=spec.objective,
        success_condition=spec.success_condition,
        files=_render_files(delta, state_view),
        language=spec.language or "not specified",
    )
    system_prompt = _build_system_prompt(None, CriticFinding)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw = await provider.chat(messages)
        try:
            parsed = _parse_response(raw, None, CriticFinding)
            return parsed  # type: ignore[return-value]
        except ValueError as e:
            if attempt >= max_retries:
                raise ValueError(f"critic_agent failed after {max_retries} retries: {e}") from e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Invalid response: {e}. Respond with valid JSON."}
            )
    raise ValueError("critic_agent: loop exhausted")  # unreachable
