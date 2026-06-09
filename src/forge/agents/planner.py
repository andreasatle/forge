"""Planning agent that decomposes a northstar goal into concrete work tasks."""

from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, PlanResponse, PlanSpec, ResponseStatus
from forge.llm.providers import LLMProvider

CORRECTION_PROMPT = """
Your previous response could not be parsed. Error: {error}

Your previous response was:
{bad_response}

Original instructions:
{original_prompt}

Fix the error and return corrected JSON only — no explanation, no markdown.
"""

PLAN_PROMPT = """
You are a planning agent. Given a goal, decompose it into at most 5 concrete tasks.

Available artifacts and their languages:
{artifact_language_list}

Each coding task must declare the correct language for its artifact.

Rules:
- EVERY task MUST include the "artifact" field — omitting it is an error
- artifact must be one of: {artifact_names}
- depends_on contains indices (0-based) of tasks this task depends on
- adapter must be one of: coding, document, audit
- No more than 5 tasks

Goal: {northstar}
"""


async def plan_agent(
    request: AgentRequest,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
    provider: LLMProvider,
    max_retries: int = 3,
) -> AgentResponse:
    """Send the northstar goal to the planner LLM and return follow-up work requests."""
    spec = request.spec
    if not isinstance(spec, PlanSpec):
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"expected PlanSpec, got {type(spec).__name__}",
        )

    artifact_language_list = "\n".join(
        f"  {name}: {lang}" for name, lang in artifact_languages.items()
    ) or "  (no languages declared)"
    prompt = PLAN_PROMPT.format(
        northstar=spec.northstar,
        artifact_names=", ".join(artifact_names),
        artifact_language_list=artifact_language_list,
    )

    def correction_fn(error: Exception, bad_response: str) -> str:
        return CORRECTION_PROMPT.format(
            original_prompt=prompt,
            bad_response=bad_response,
            error=error,
        )

    return await run_agent(
        request,
        PlanSpec,
        provider,
        prompt,
        max_retries=max_retries,
        correction_prompt_fn=correction_fn,
        final_response_type=PlanResponse,
    )
