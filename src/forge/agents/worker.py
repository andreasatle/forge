from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, WorkSpec
from forge.llm import client as llm

WORK_MODEL = "gemma4:e4b"


async def work_agent(request: AgentRequest, registry: AdapterRegistry) -> AgentResponse:
    async def build(spec: WorkSpec) -> AgentResponse:
        adapter = registry.get(spec.adapter)
        prompt = adapter.prompt_template.format(
            objective=spec.objective,
            success_condition=spec.success_condition,
        )
        raw = await llm.chat(WORK_MODEL, prompt)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta={"result": raw, "adapter": spec.adapter},
        )

    return await run_agent(request, WorkSpec, build)
