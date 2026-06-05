import asyncio

from forge.core.models import (
    AgentRequest,
    AgentType,
    PlanSpec,
    Priority,
    RequestSource,
    SchedulerState,
)
from forge.core.runner import (
    Runner,
    scripted_plan_handler,
    stub_integrate_handler,
    stub_work_handler,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    runner = Runner()
    runner.register(AgentType.PLAN, scripted_plan_handler)
    runner.register(AgentType.WORK, stub_work_handler)
    runner.register(AgentType.INTEGRATE, stub_integrate_handler)

    state = SchedulerState(northstar="hello world")

    global_planner = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="hello world"),
        priority=Priority.HIGH,
    )

    callbacks = SchedulerCallbacks(
        on_node_dispatched=lambda node: print(
            f"→ dispatched: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_completed=lambda node: print(
            f"✓ completed: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_node_failed=lambda node: print(
            f"✗ failed: {node.request.agent_type.value} ({node.request.id})"
        ),
        on_idle=lambda state: print(f"~ idle: {len(state.dag)} nodes in DAG"),
    )

    final = await Scheduler(runner=runner, callbacks=callbacks).run(state, global_planner)

    completed = sum(1 for n in final.dag.values() if n.node_state.value == "completed")
    failed = sum(1 for n in final.dag.values() if n.node_state.value == "failed")
    cancelled = sum(1 for n in final.dag.values() if n.node_state.value == "cancelled")
    print(f"\nDone — completed: {completed}, failed: {failed}, cancelled: {cancelled}")

    print("\nDAG summary:")
    for node in sorted(final.dag.values(), key=lambda n: len(n.request.dependencies)):
        short_id = str(node.request.id)[:8]
        agent_type = node.request.agent_type.value
        node_state = node.node_state.value
        n_deps = len(node.request.dependencies)
        delta = node.response.delta if node.response else None
        print(f"  {short_id}  {agent_type:<10}  {node_state:<10}  deps={n_deps}  delta={delta}")
