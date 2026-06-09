"""Parser that converts a planner LLM response into a list of AgentRequests."""

import json
import re

from pydantic import ValidationError

from forge.adapters.registry import AdapterRegistry
from forge.core.models import AgentRequest, AgentType, PlanResponse, RequestSource, WorkSpec


def parse_plan(response: str, registry: AdapterRegistry) -> list[AgentRequest]:
    """Parse a JSON plan into ordered AgentRequests with dependencies."""
    text = response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        plan = PlanResponse.model_validate(json.loads(text))
    except ValidationError as e:
        raise ValueError(f"invalid plan response: {e}") from e
    tasks = plan.tasks

    if not tasks:
        return []

    known_adapters = set(registry.names())

    requests: list[AgentRequest] = []
    for i, task in enumerate(tasks):
        adapter = task.adapter
        if adapter not in known_adapters:
            adapter = "coding"
        artifact = task.artifact
        if not artifact:
            raise ValueError(f"task {i} missing required 'artifact' field")
        requests.append(
            AgentRequest(
                agent_type=AgentType.WORK,
                source=RequestSource.PLANNER,
                spec=WorkSpec(
                    objective=task.objective,
                    success_condition=task.success_condition,
                    adapter=adapter,
                    artifact=artifact,
                    language=task.language,
                ),
            )
        )

    id_map = {i: req.id for i, req in enumerate(requests)}

    final: list[AgentRequest] = []
    for req, task in zip(requests, tasks):
        dep_indices = task.depends_on
        dep_ids = frozenset(id_map[j] for j in dep_indices if 0 <= j < len(requests))
        final.append(req.model_copy(update={"dependencies": dep_ids}))

    return final
