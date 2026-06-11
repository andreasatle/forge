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
    render_agent_contract,
)
from forge.llm.providers import ChatMessage, LLMProvider

_DEFAULT_REVIEW_CONTEXT = ReviewContext(
    output_noun="work",
    review_focus="whether the output genuinely satisfies the AgentRequest contract",
    empty_output_guidance="If no files were produced, reject it.",
)


def _render_hints(hints: list[str]) -> str:
    return ", ".join(hints) if hints else "(none)"


def _render_revision_items(finding: CriticFinding) -> str:
    """Render structured critic revision items for referee review."""
    if not finding.revision_items:
        return "(none)"
    lines: list[str] = []
    for index, item in enumerate(finding.revision_items, start=1):
        criterion = f" [{item.criterion_id}]" if item.criterion_id else ""
        lines.append(f"{index}. Required change{criterion}: {item.required_change}")
        if item.rationale:
            lines.append(f"   Rationale: {item.rationale}")
    return "\n".join(lines)


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
    objective = spec.contract.objective
    success_condition = spec.contract.success_condition
    if isinstance(spec, WorkSpec):
        language = spec.language or "not specified"
    else:
        language = "n/a"

    adapter = registry.get("referee")
    user_prompt = adapter.prompt_template.format(
        objective=objective,
        success_condition=success_condition,
        contract_block=render_agent_contract(request),
        output_text=output_text,
        language=language,
        output_noun=review_context.output_noun,
        review_focus=review_context.review_focus,
        empty_output_guidance=review_context.empty_output_guidance,
        critic_disposition=critic_finding.disposition.value,
        critic_rationale=critic_finding.rationale,
        critic_hints=_render_hints(critic_finding.hints),
        critic_revision_items=_render_revision_items(critic_finding),
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
