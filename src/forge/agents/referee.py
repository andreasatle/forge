"""Referee agent — reviews critic finding and makes the final disposition."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import _build_system_prompt, _parse_response
from forge.core.models import (
    AgentRequest,
    CriticFinding,
    PlanSpec,
    RefereeDecision,
    StateView,
    WorkSpec,
)
from forge.llm.providers import ChatMessage, LLMProvider


def _render_hints(hints: list[str]) -> str:
    return ", ".join(hints) if hints else "(none)"


async def referee_agent(
    request: AgentRequest,
    state_view: StateView,
    output_text: str,
    critic_finding: CriticFinding,
    provider: LLMProvider,
    registry: AdapterRegistry,
    max_retries: int = 3,
) -> RefereeDecision:
    """Review the critic's finding and agent output; return the final RefereeDecision."""
    spec = request.spec
    if isinstance(spec, WorkSpec):
        objective = spec.objective
        success_condition = spec.success_condition
        language = spec.language or "not specified"
    elif isinstance(spec, PlanSpec):
        objective = spec.northstar
        success_condition = "Plan comprehensively addresses the northstar goal"
        language = "n/a"
    else:
        raise TypeError(f"unsupported spec type: {type(spec).__name__}")

    adapter = registry.get("referee")
    user_prompt = adapter.prompt_template.format(
        objective=objective,
        success_condition=success_condition,
        output_text=output_text,
        language=language,
        critic_disposition=critic_finding.disposition.value,
        critic_rationale=critic_finding.rationale,
        critic_hints=_render_hints(critic_finding.hints),
    )
    system_prompt = _build_system_prompt(None, RefereeDecision)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw = await provider.chat(messages)
        try:
            parsed = _parse_response(raw, None, RefereeDecision)
            return parsed  # type: ignore[return-value]
        except ValueError as e:
            if attempt >= max_retries:
                raise ValueError(f"referee_agent failed after {max_retries} retries: {e}") from e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Invalid response: {e}. Respond with valid JSON."}
            )
    raise ValueError("referee_agent: loop exhausted")  # unreachable
