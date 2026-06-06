"""Planning agent that decomposes a northstar goal into concrete work tasks."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, PlanSpec, ResponseStatus
from forge.parsers.plan import parse_plan

PLANNER_MODEL = "gemma4:e4b"

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

Respond with ONLY a JSON object in this exact format:
{{
  "tasks": [
    {{
      "objective": "specific task description",
      "success_condition": "how to know this task is done",
      "adapter": "coding|document|audit",
      "artifact": "name of the artifact this task writes to",
      "language": "language name for coding tasks, or null for non-coding tasks",
      "depends_on": []
    }}
  ]
}}

Example task:
{{
  "objective": "example task",
  "success_condition": "example done",
  "adapter": "coding",
  "artifact": "{first_artifact}",
  "language": "python",
  "depends_on": []
}}

Available artifacts and their languages:
{artifact_language_list}

Each coding task must declare the correct language for its artifact.

Rules:
- EVERY task MUST include the "artifact" field — omitting it is an error
- artifact must be one of: {artifact_names}
- depends_on contains indices (0-based) of tasks this task depends on
- adapter must be one of: coding, document, audit
- No more than 5 tasks
- Respond with JSON only — no explanation, no markdown

Goal: {northstar}
"""


async def plan_agent(
    request: AgentRequest,
    registry: AdapterRegistry,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
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
        first_artifact=artifact_names[0],
        artifact_language_list=artifact_language_list,
    )

    def correction_fn(error: Exception, bad_response: str) -> str:
        return CORRECTION_PROMPT.format(
            original_prompt=prompt,
            bad_response=bad_response,
            error=error,
        )

    def process_response(raw_text: str) -> AgentResponse:
        try:
            follow_up = parse_plan(raw_text, registry)
        except Exception as e:
            raise ValueError(f"parse failed: {type(e).__name__}: {e}") from e
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            follow_up=follow_up,
        )

    return await run_agent(
        request,
        PlanSpec,
        PLANNER_MODEL,
        prompt,
        max_retries=max_retries,
        correction_prompt_fn=correction_fn,
        response_fn=process_response,
    )
