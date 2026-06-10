"""Referee agent — reviews critic finding and makes the final disposition."""

from forge.agents.base import _build_system_prompt, _parse_response
from forge.core.models import (
    AgentRequest,
    CriticFinding,
    DeltaState,
    RefereeDecision,
    StateView,
    WorkSpec,
)
from forge.llm.providers import ChatMessage, LLMProvider


def _build_referee_prompt(
    spec: WorkSpec,
    state_view: StateView,
    delta: DeltaState,
    finding: CriticFinding,
) -> str:
    lines = [
        f"Objective: {spec.objective}",
        f"Success condition: {spec.success_condition}",
        "",
        f"Critic disposition: {finding.disposition.value}",
        f"Critic rationale: {finding.rationale}",
    ]
    if finding.hints:
        lines.append("Critic hints:")
        for hint in finding.hints:
            lines.append(f"  - {hint}")
    lines.append("")
    if delta.new_files:
        lines.append("Files produced:")
        for fw in delta.new_files:
            lines += [f"\nFile: {fw.path}", "```", fw.content, "```"]
    if delta.edits:
        lines.append("\nEdits applied:")
        for edit in delta.edits:
            lines += [f"\nFile: {edit.path}", "  old:", edit.old, "  new:", edit.new]
    if not delta.new_files and not delta.edits:
        lines.append("No files or edits were produced.")
    if state_view.files:
        lines.append("\nExisting artifact files:")
        for fv in state_view.files:
            lines += [f"\nFile: {fv.path}", "```", fv.content, "```"]
    lines += [
        "",
        "Make the final accept/revise/reject decision.",
        "You may agree with the critic or override. Set override=true if your disposition "
        "differs from the critic's.",
    ]
    return "\n".join(lines)


async def referee_agent(
    request: AgentRequest,
    state_view: StateView,
    delta: DeltaState,
    critic_finding: CriticFinding,
    provider: LLMProvider,
    max_retries: int = 3,
) -> RefereeDecision:
    """Review the critic's finding and worker's delta; return the final RefereeDecision."""
    spec = request.spec
    if not isinstance(spec, WorkSpec):
        raise TypeError(f"expected WorkSpec, got {type(spec).__name__}")

    system_prompt = _build_system_prompt(None, RefereeDecision)
    user_prompt = _build_referee_prompt(spec, state_view, delta, critic_finding)
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
