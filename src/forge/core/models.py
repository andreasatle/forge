"""Core Pydantic models and enums shared across all forge components."""

from enum import Enum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

RequestId = UUID


class AgentType(Enum):
    """Discriminator enum for the three agent roles in the system."""

    PLAN = "plan"
    WORK = "work"
    INTEGRATE = "integrate"


class RequestSource(Enum):
    """Identifies who originated an agent request."""

    USER = "user"
    PLANNER = "planner"
    WORKER = "worker"


class Priority(Enum):
    """Scheduling priority for an agent request."""

    HIGH = "high"
    NORMAL = "normal"


class NodeState(Enum):
    """Lifecycle states a DAG node moves through during a scheduler run."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResponseStatus(Enum):
    """Terminal outcome reported by an agent in its response."""

    COMPLETED = "completed"
    FAILED = "failed"


class PlanSpec(BaseModel):
    """Spec for a planning agent request carrying the northstar goal."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["plan"] = "plan"
    northstar: str
    goal: str | None = None


class WorkSpec(BaseModel):
    """Spec for a work agent request describing a single concrete task."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["work"] = "work"
    objective: str
    success_condition: str
    target_entity: str | None = None
    adapter: str


class IntegrateSpec(BaseModel):
    """Spec for an integrate agent request that consolidates a completed work result."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["integrate"] = "integrate"
    source_request_id: RequestId


AgentSpec = Annotated[
    PlanSpec | WorkSpec | IntegrateSpec,
    Field(discriminator="kind"),
]


class AgentRequest(BaseModel):
    """Immutable description of a unit of work dispatched to an agent."""

    model_config = ConfigDict(frozen=True)

    id: RequestId = Field(default_factory=uuid4)
    agent_type: AgentType
    source: RequestSource
    spec: AgentSpec
    context_chain: tuple[str, ...] = ()
    dependencies: frozenset[RequestId] = frozenset()
    priority: Priority = Priority.NORMAL


class AgentResponse(BaseModel):
    """Immutable result returned by an agent after processing a request."""

    model_config = ConfigDict(frozen=True)

    request_id: RequestId
    status: ResponseStatus
    delta: dict | None = None  # type: ignore[type-arg]
    follow_up: list[AgentRequest] = Field(default_factory=list)
    error: str | None = None


class DAGNode(BaseModel):
    """A single node in the scheduler DAG wrapping a request and its current state."""

    model_config = ConfigDict(frozen=True)

    request: AgentRequest
    node_state: NodeState = NodeState.PENDING
    response: AgentResponse | None = None

    def with_state(self, node_state: NodeState) -> "DAGNode":
        """Return a copy of this node with the given node_state."""
        return self.model_copy(update={"node_state": node_state})

    def with_response(self, response: AgentResponse) -> "DAGNode":
        """Return a copy of this node with the response set and state derived from its status."""
        node_state = (
            NodeState.COMPLETED if response.status == ResponseStatus.COMPLETED else NodeState.FAILED
        )
        return self.model_copy(update={"node_state": node_state, "response": response})


class SchedulerState(BaseModel):
    """Immutable snapshot of the full DAG and scheduler configuration at a point in time."""

    model_config = ConfigDict(frozen=True)

    dag: dict[RequestId, DAGNode] = Field(default_factory=dict)
    northstar: str
    max_concurrency: int = 1

    def add_nodes(self, nodes: list[DAGNode]) -> "SchedulerState":
        """Return a new state with the given nodes merged into the DAG."""
        return self.model_copy(update={"dag": {**self.dag, **{n.request.id: n for n in nodes}}})

    def update_node(self, node: DAGNode) -> "SchedulerState":
        """Return a new state with the given node replacing its previous entry in the DAG."""
        return self.model_copy(update={"dag": {**self.dag, node.request.id: node}})

    def ready_nodes(self) -> list[DAGNode]:
        """Return all PENDING nodes whose dependencies are all COMPLETED."""
        completed = {nid for nid, n in self.dag.items() if n.node_state == NodeState.COMPLETED}
        return [
            n
            for n in self.dag.values()
            if n.node_state == NodeState.PENDING and n.request.dependencies <= completed
        ]
