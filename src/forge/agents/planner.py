from forge.adapters.registry import AdapterRegistry
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
      "depends_on": []
    }}
  ]
}}

Rules:
- depends_on contains indices (0-based) of tasks this task depends on
- adapter must be one of: coding, document, audit
- No more than 5 tasks
- Respond with JSON only — no explanation, no markdown

Goal: {northstar}
"""


async def plan_agent(request: AgentRequest, registry: AdapterRegistry) -> AgentResponse:
    try:
        if not isinstance(request.spec, PlanSpec):
            raise TypeError(f"expected PlanSpec, got {type(request.spec).__name__}")
        prompt = PLAN_PROMPT.format(northstar=request.spec.northstar)
        raw = await llm.chat(PLANNER_MODEL, prompt)
        follow_up = parse_plan(raw, registry)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            follow_up=follow_up,
        )
    except Exception as e:
        print(f"planner error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
        )
