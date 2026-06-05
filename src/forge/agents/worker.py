"""Worker agent that executes a task using an adapter and tool registry."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.core.models import AgentRequest, AgentResponse, ResponseStatus, WorkSpec
from forge.llm import client as llm
from forge.tools.registry import ToolRegistry

WORK_MODEL = "gemma4:e4b"


async def work_agent(request: AgentRequest, registry: AdapterRegistry, tools: ToolRegistry) -> AgentResponse:
    """Run the agentic tool loop for a work request using the specified adapter."""
    async def build(spec: WorkSpec) -> AgentResponse:
        adapter = registry.get(spec.adapter)
        tool_names = adapter.tools
        tool_schema = tools.to_ollama_schema(tool_names)
        prompt = adapter.prompt_template.format(
            objective=spec.objective,
            success_condition=spec.success_condition,
        )

        messages: list[dict] = [{"role": "user", "content": prompt}]  # type: ignore[type-arg]
        files: dict[str, str] = {}
        blackboard: dict[str, str] = {}

        for _ in range(10):
            text, tool_calls = await llm.chat_with_tools(WORK_MODEL, messages, tool_schema)
            if text is not None:
                return AgentResponse(
                    request_id=request.id,
                    status=ResponseStatus.COMPLETED,
                    delta={"result": text, "files": files, "blackboard": blackboard},
                )
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in tool_calls
                ],
            })
            for call in tool_calls:
                tool = tools.get(call["name"])
                result = await tool.fn(**call["arguments"])
                messages.append({"role": "tool", "content": result, "name": call["name"]})

        raise RuntimeError("agentic loop exceeded 10 iterations without a final response")

    return await run_agent(request, WorkSpec, build)
