"""Critic agent — reviews agent output against the AgentRequest contract."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import build_system_prompt, parse_response
from forge.core.models import (
    AgentRequest,
    CriticFinding,
    ReviewContext,
    StateView,
    WorkSpec,
    render_agent_contract,
)
from forge.llm.providers import ChatMessage, LLMProvider

_DEFAULT_REVIEW_CONTEXT = ReviewContext(
    output_noun="work",
    review_focus="whether the output genuinely satisfies the AgentRequest contract",
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
    """Review agent output against the request's AgentRequest contract."""
    spec = request.spec
    objective = spec.contract.objective
    success_condition = spec.contract.success_condition
    if isinstance(spec, WorkSpec):
        language = spec.language or "not specified"
    else:
        language = "n/a"

    adapter = registry.get("critic")
    user_prompt = adapter.prompt_template.format(
        objective=objective,
        success_condition=success_condition,
        contract_block=render_agent_contract(request),
        output_text=output_text,
        language=language,
        output_noun=review_context.output_noun,
        review_focus=review_context.review_focus,
        empty_output_guidance=review_context.empty_output_guidance,
        topology_rules=review_context.topology_rules,
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
