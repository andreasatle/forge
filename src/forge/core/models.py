from enum import Enum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

RequestId = UUID


class AgentType(Enum):
    PLAN = "plan"
    WORK = "work"
    INTEGRATE = "integrate"


class RequestSource(Enum):
    USER = "user"
    PLANNER = "planner"
    WORKER = "worker"


class Priority(Enum):
    HIGH = "high"
    NORMAL = "normal"


class NodeState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResponseStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class AdapterType(Enum):
    CODING = "coding"
    DOCUMENT = "document"
    AUDIT = "audit"


class PlanSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["plan"] = "plan"
    northstar: str
    goal: str | None = None


class WorkSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["work"] = "work"
    objective: str
    success_condition: str
    target_entity: str | None = None
    adapter_type: AdapterType


class IntegrateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["integrate"] = "integrate"
    source_request_id: RequestId


AgentSpec = Annotated[
    PlanSpec | WorkSpec | IntegrateSpec,
    Field(discriminator="kind"),
]


class AgentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: RequestId = Field(default_factory=uuid4)
    agent_type: AgentType
    source: RequestSource
    spec: AgentSpec
    context_chain: tuple[str, ...] = ()
    dependencies: frozenset[RequestId] = frozenset()
    priority: Priority = Priority.NORMAL


class AgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: RequestId
    status: ResponseStatus
    delta: dict | None = None  # type: ignore[type-arg]
    follow_up: list[AgentRequest] = Field(default_factory=list)
    error: str | None = None


class DAGNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: AgentRequest
    node_state: NodeState = NodeState.PENDING
    response: AgentResponse | None = None

    def with_state(self, node_state: NodeState) -> "DAGNode":
        return self.model_copy(update={"node_state": node_state})

    def with_response(self, response: AgentResponse) -> "DAGNode":
        node_state = (
            NodeState.COMPLETED if response.status == ResponseStatus.COMPLETED else NodeState.FAILED
        )
        return self.model_copy(update={"node_state": node_state, "response": response})


class SchedulerState(BaseModel):
    model_config = ConfigDict(frozen=True)

    dag: dict[RequestId, DAGNode] = Field(default_factory=dict)
    northstar: str
    max_concurrency: int = 1

    def add_nodes(self, nodes: list[DAGNode]) -> "SchedulerState":
        return self.model_copy(update={"dag": {**self.dag, **{n.request.id: n for n in nodes}}})

    def update_node(self, node: DAGNode) -> "SchedulerState":
        return self.model_copy(update={"dag": {**self.dag, node.request.id: node}})

    def ready_nodes(self) -> list[DAGNode]:
        completed = {nid for nid, n in self.dag.items() if n.node_state == NodeState.COMPLETED}
        return [
            n
            for n in self.dag.values()
            if n.node_state == NodeState.PENDING and n.request.dependencies <= completed
        ]
