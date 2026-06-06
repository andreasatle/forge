"""Planning agent that decomposes a northstar goal into concrete work tasks."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, PlanSpec, ResponseStatus
from forge.llm import client as llm
from forge.parsers.plan import parse_plan

PLANNER_MODEL = "gemma4:e4b"

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
) -> AgentResponse:
    """Send the northstar goal to the planner LLM and return follow-up work requests."""
    async def build(spec: PlanSpec) -> AgentResponse:
        artifact_language_list = "\n".join(
            f"  {name}: {lang}" for name, lang in artifact_languages.items()
        ) or "  (no languages declared)"
        prompt = PLAN_PROMPT.format(
            northstar=spec.northstar,
            artifact_names=", ".join(artifact_names),
            first_artifact=artifact_names[0],
            artifact_language_list=artifact_language_list,
        )
        raw = await llm.chat(PLANNER_MODEL, prompt)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            follow_up=parse_plan(raw, registry),
        )

    return await run_agent(request, PlanSpec, build)
