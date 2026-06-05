import json
import re

from forge.adapters.registry import AdapterRegistry
from forge.core.models import AgentRequest, AgentType, RequestSource, WorkSpec


def parse_plan(response: str, registry: AdapterRegistry) -> list[AgentRequest]:
    text = response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    data = json.loads(text)
    tasks: list[dict] = data.get("tasks", [])  # type: ignore[assignment]

    if not tasks:
        return []

    known_adapters = set(registry.names())

    requests: list[AgentRequest] = []
    for task in tasks:
        adapter = task.get("adapter", "coding")
        if adapter not in known_adapters:
            adapter = "coding"
        requests.append(
            AgentRequest(
                agent_type=AgentType.WORK,
                source=RequestSource.PLANNER,
                spec=WorkSpec(
                    objective=task["objective"],
                    success_condition=task["success_condition"],
                    adapter=adapter,
                ),
            )
        )

    id_map = {i: req.id for i, req in enumerate(requests)}

    final: list[AgentRequest] = []
    for req, task in zip(requests, tasks):
        dep_indices: list[int] = task.get("depends_on", [])
        dep_ids = frozenset(id_map[j] for j in dep_indices if 0 <= j < len(requests))
        final.append(req.model_copy(update={"dependencies": dep_ids}))

    return final
