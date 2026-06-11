"""Referee agent — reviews critic finding and makes the final disposition."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import build_system_prompt, parse_response
from forge.core.models import (
    AgentRequest,
    CriticFinding,
    RefereeDecision,
    ReviewContext,
    StateView,
    WorkSpec,
)
from forge.llm.providers import ChatMessage, LLMProvider

_DEFAULT_REVIEW_CONTEXT = ReviewContext(
    output_noun="work",
    review_focus="whether the output genuinely meets the success condition",
    empty_output_guidance="If no files were produced, reject it.",
)


def _render_hints(hints: list[str]) -> str:
    return ", ".join(hints) if hints else "(none)"


async def referee_agent(
    request: AgentRequest,
    state_view: StateView,
    output_text: str,
    critic_finding: CriticFinding,
    provider: LLMProvider,
    registry: AdapterRegistry,
    review_context: ReviewContext = _DEFAULT_REVIEW_CONTEXT,
    max_retries: int = 3,
) -> RefereeDecision:
    """Review the critic's finding and agent output; return the final RefereeDecision."""
    spec = request.spec
    if isinstance(spec, WorkSpec):
        objective = spec.objective
        success_condition = spec.success_condition
        language = spec.language or "not specified"
    else:
        objective = spec.northstar
        success_condition = "Plan comprehensively addresses the northstar goal"
        language = "n/a"

    adapter = registry.get("referee")
    user_prompt = adapter.prompt_template.format(
        objective=objective,
        success_condition=success_condition,
        output_text=output_text,
        language=language,
        output_noun=review_context.output_noun,
        review_focus=review_context.review_focus,
        empty_output_guidance=review_context.empty_output_guidance,
        critic_disposition=critic_finding.disposition.value,
        critic_rationale=critic_finding.rationale,
        critic_hints=_render_hints(critic_finding.hints),
    )
    system_prompt = build_system_prompt(None, RefereeDecision)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw = await provider.chat(messages)
        try:
            parsed = parse_response(raw, None, RefereeDecision)
            return parsed  # type: ignore[return-value]
        except ValueError as e:
            if attempt >= max_retries:
                raise ValueError(f"referee_agent failed after {max_retries} retries: {e}") from e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Invalid response: {e}. Respond with valid JSON."}
            )
    raise ValueError("referee_agent: loop exhausted")  # unreachable
