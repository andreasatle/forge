"""Async DAG scheduler that dispatches agent requests up to max_concurrency."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from forge.agents.integrator import integrate
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    DeltaState,
    FailureKind,
    NodeState,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkSpec,
)
from forge.core.state_service import StateService

AgentRunner = Callable[[AgentRequest], Awaitable[AgentResponse]]

logger = logging.getLogger(__name__)


@dataclass
class SchedulerCallbacks:
    """Optional lifecycle callbacks fired by the scheduler at key state transitions."""

    on_node_dispatched: Callable[[DAGNode], None] | None = None
    on_node_completed: Callable[[DAGNode], None] | None = None
    on_node_failed: Callable[[DAGNode], None] | None = None
    on_idle: Callable[[SchedulerState], None] | None = None


class Scheduler:
    """Async scheduler that drives a DAG of agent requests to completion."""

    def __init__(
        self,
        runner: AgentRunner,
        state_services: dict[str, StateService] | None = None,
        callbacks: SchedulerCallbacks | None = None,
    ) -> None:
        self._runner = runner
        self._state_services = state_services
        self._callbacks = callbacks or SchedulerCallbacks()
        self._stale_retry_counts: dict[RequestId, int] = {}

    async def run(
        self,
        state: SchedulerState,
        global_planner: AgentRequest,
    ) -> SchedulerState:
        """Drive state forward until no PENDING or RUNNING nodes remain."""
        if not state.dag:
            state = state.add_nodes([DAGNode(request=global_planner)])

        while True:
            ready = state.ready_nodes()

            if not ready:
                self._fire_state(self._callbacks.on_idle, state)
                if not any(
                    n.node_state in (NodeState.PENDING, NodeState.RUNNING)
                    for n in state.dag.values()
                ):
                    break
                new_planner = global_planner.model_copy(
                    update={"id": uuid4(), "source": RequestSource.PLANNER}
                )
                state = state.add_nodes([DAGNode(request=new_planner)])
                continue

            to_dispatch = ready[: state.max_concurrency]

            for node in to_dispatch:
                running = node.with_state(NodeState.RUNNING)
                state = state.update_node(running)
                self._fire_node(self._callbacks.on_node_dispatched, running)

            coros = [self._runner(node.request) for node in to_dispatch]
            raw = await asyncio.gather(*coros, return_exceptions=True)

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

                    if updated.node_state == NodeState.INTEGRATED:
                        integration_failed = False
                        stale_retry = False
                        if (
                            node.request.agent_type == AgentType.WORK
                            and self._state_services is not None
                        ):
                            spec = node.request.spec
                            if isinstance(spec, WorkSpec):
                                ss = self._state_services.get(spec.artifact)
                                if ss is not None:
                                    integration_response = await integrate(
                                        request_id=node.request.id,
                                        state_service=ss,
                                        delta=response.delta or DeltaState(),
                                    )
                                    if integration_response.status == ResponseStatus.FAILED:
                                        if (
                                            integration_response.failure_kind
                                            == FailureKind.STALE_DELTA
                                        ):
                                            retry_count = self._stale_retry_counts.get(
                                                node.request.id, 0
                                            )
                                            if retry_count < 3:
                                                self._stale_retry_counts[node.request.id] = (
                                                    retry_count + 1
                                                )
                                                pending_node = current.with_state(NodeState.PENDING)
                                                state = state.update_node(pending_node)
                                                stale_retry = True
                                            else:
                                                integration_failed = True
                                        else:
                                            integration_failed = True

                        if integration_failed:
                            failed_node = updated.with_state(NodeState.FAILED)
                            state = state.update_node(failed_node)
                            self._fire_node(self._callbacks.on_node_failed, failed_node)
                            state = self._cancel_dependents(state, node.request.id)
                        elif not stale_retry:
                            follow_ups = [DAGNode(request=r) for r in response.follow_up]
                            if follow_ups:
                                state = state.add_nodes(follow_ups)
                            self._fire_node(self._callbacks.on_node_completed, updated)
                    else:
                        self._fire_node(self._callbacks.on_node_failed, updated)
                        state = self._cancel_dependents(state, node.request.id)

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
