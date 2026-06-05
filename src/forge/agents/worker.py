from forge.adapters.registry import AdapterRegistry
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, WorkSpec
from forge.llm import client as llm

WORK_MODEL = "gemma4:e4b"


async def work_agent(request: AgentRequest, registry: AdapterRegistry) -> AgentResponse:
    try:
        if not isinstance(request.spec, WorkSpec):
            raise TypeError(f"expected WorkSpec, got {type(request.spec).__name__}")
        adapter = registry.get(request.spec.adapter)
        prompt = adapter.prompt_template.format(
            objective=request.spec.objective,
            success_condition=request.spec.success_condition,
        )
        raw = await llm.chat(WORK_MODEL, prompt)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta={"result": raw},
        )
    except Exception as e:
        print(f"work error: {type(e).__name__}: {e}")
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error=f"{type(e).__name__}: {e}",
        )
