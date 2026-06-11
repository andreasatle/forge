"""Critic agent — reviews agent output against the success condition."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import build_system_prompt, parse_response
from forge.core.models import AgentRequest, CriticFinding, ReviewContext, StateView, WorkSpec
from forge.llm.providers import ChatMessage, LLMProvider

_DEFAULT_REVIEW_CONTEXT = ReviewContext(
    output_noun="work",
    review_focus="whether the output genuinely meets the success condition",
    empty_output_guidance="If no files were produced, reject it.",
)


async def critic_agent(
    request: AgentRequest,
    state_view: StateView,
    output_text: str,
    provider: LLMProvider,
    registry: AdapterRegistry,
    review_context: ReviewContext = _DEFAULT_REVIEW_CONTEXT,
    max_retries: int = 3,
) -> CriticFinding:
    """Review agent output against the request's success condition."""
    spec = request.spec
    if isinstance(spec, WorkSpec):
        objective = spec.objective
        success_condition = spec.success_condition
        language = spec.language or "not specified"
    else:
        objective = spec.northstar
        success_condition = "Plan comprehensively addresses the northstar goal"
        language = "n/a"

    adapter = registry.get("critic")
    user_prompt = adapter.prompt_template.format(
        objective=objective,
        success_condition=success_condition,
        output_text=output_text,
        language=language,
        output_noun=review_context.output_noun,
        review_focus=review_context.review_focus,
        empty_output_guidance=review_context.empty_output_guidance,
    )
    system_prompt = build_system_prompt(None, CriticFinding)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw = await provider.chat(messages)
        try:
            parsed = parse_response(raw, None, CriticFinding)
            return parsed  # type: ignore[return-value]
        except ValueError as e:
            if attempt >= max_retries:
                raise ValueError(f"critic_agent failed after {max_retries} retries: {e}") from e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Invalid response: {e}. Respond with valid JSON."}
            )
    raise ValueError("critic_agent: loop exhausted")  # unreachable
