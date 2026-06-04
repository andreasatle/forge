import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    DAGNode,
    NodeState,
    RequestId,
    SchedulerState,
)

AgentRunner = Callable[[AgentRequest], Awaitable[AgentResponse]]

logger = logging.getLogger(__name__)


@dataclass
class SchedulerCallbacks:
    on_node_dispatched: Callable[[DAGNode], None] | None = None
    on_node_completed: Callable[[DAGNode], None] | None = None
    on_node_failed: Callable[[DAGNode], None] | None = None
    on_idle: Callable[[SchedulerState], None] | None = None


class Scheduler:
    def __init__(
        self,
        runner: AgentRunner,
        callbacks: SchedulerCallbacks | None = None,
    ) -> None:
        self._runner = runner
        self._callbacks = callbacks or SchedulerCallbacks()

    async def run(
        self,
        state: SchedulerState,
        global_planner: AgentRequest,
    ) -> SchedulerState:
        if not state.dag:
            state = state.add_nodes([DAGNode(request=global_planner)])

        pending_termination_id: RequestId | None = None

        while True:
            ready = state.ready_nodes()

            if not ready:
                self._fire_state(self._callbacks.on_idle, state)
                new_planner = global_planner.model_copy(update={"id": uuid4()})
                state = state.add_nodes([DAGNode(request=new_planner)])
                pending_termination_id = new_planner.id
                continue

            to_dispatch = ready[: state.max_concurrency]

            for node in to_dispatch:
                running = node.with_state(NodeState.RUNNING)
                state = state.update_node(running)
                self._fire_node(self._callbacks.on_node_dispatched, running)

            coros = [self._runner(node.request) for node in to_dispatch]
            raw = await asyncio.gather(*coros, return_exceptions=True)

            should_terminate = False

            for node, result in zip(to_dispatch, raw):
                current = state.dag[node.request.id]

                if isinstance(result, BaseException):
                    failed = current.with_state(NodeState.FAILED)
                    state = state.update_node(failed)
                    self._fire_node(self._callbacks.on_node_failed, failed)
                    state = self._cancel_dependents(state, node.request.id)
                else:
                    response: AgentResponse = result
                    updated = current.with_response(response)
                    state = state.update_node(updated)

                    if updated.node_state == NodeState.COMPLETED:
                        follow_ups = [DAGNode(request=r) for r in response.follow_up]
                        if follow_ups:
                            state = state.add_nodes(follow_ups)
                        self._fire_node(self._callbacks.on_node_completed, updated)

                        if pending_termination_id == node.request.id and not response.follow_up:
                            should_terminate = True
                    else:
                        self._fire_node(self._callbacks.on_node_failed, updated)
                        state = self._cancel_dependents(state, node.request.id)

            if should_terminate:
                break

        return state

    def _cancel_dependents(self, state: SchedulerState, failed_id: RequestId) -> SchedulerState:
        failed_ids: set[RequestId] = {failed_id}
        while True:
            to_cancel = [
                node
                for nid, node in state.dag.items()
                if node.node_state == NodeState.PENDING
                and node.request.dependencies & failed_ids
                and nid not in failed_ids
            ]
            if not to_cancel:
                break
            for node in to_cancel:
                state = state.update_node(node.with_state(NodeState.CANCELLED))
                failed_ids.add(node.request.id)
        return state

    def _fire_node(self, callback: Callable[[DAGNode], None] | None, node: DAGNode) -> None:
        if callback is None:
            return
        try:
            callback(node)
        except Exception:
            logger.exception("Scheduler callback raised an exception")

    def _fire_state(
        self, callback: Callable[[SchedulerState], None] | None, state: SchedulerState
    ) -> None:
        if callback is None:
            return
        try:
            callback(state)
        except Exception:
            logger.exception("Scheduler callback raised an exception")
