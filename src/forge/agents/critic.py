"""Critic agent — reviews worker output against the success condition."""

from forge.agents.base import _build_system_prompt, _parse_response
from forge.core.models import AgentRequest, CriticFinding, DeltaState, StateView, WorkSpec
from forge.llm.providers import ChatMessage, LLMProvider


def _build_critic_prompt(spec: WorkSpec, state_view: StateView, delta: DeltaState) -> str:
    lines = [
        f"Objective: {spec.objective}",
        f"Success condition: {spec.success_condition}",
        "",
    ]
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
        "Assess whether the work above meets the success condition.",
        "Return ACCEPT if it does, REVISE if it is on the right track but incomplete or incorrect, "
        "REJECT if it does not meet the success condition at all.",
    ]
    return "\n".join(lines)


async def critic_agent(
    request: AgentRequest,
    state_view: StateView,
    delta: DeltaState,
    provider: LLMProvider,
    max_retries: int = 3,
) -> CriticFinding:
    """Review a worker's DeltaState against the request's success condition."""
    spec = request.spec
    if not isinstance(spec, WorkSpec):
        raise TypeError(f"expected WorkSpec, got {type(spec).__name__}")

    system_prompt = _build_system_prompt(None, CriticFinding)
    user_prompt = _build_critic_prompt(spec, state_view, delta)
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
