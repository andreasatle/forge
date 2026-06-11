"""Core Pydantic models and enums shared across all forge components."""

from enum import Enum
from typing import Annotated, Any, Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

RequestId = UUID


def _empty_request_ids() -> frozenset[RequestId]:
    return frozenset()


def _empty_strings() -> list[str]:
    return []


class AgentType(Enum):
    """Discriminator enum for the two agent roles in the system."""

    PLAN = "plan"
    WORK = "work"


class RequestSource(Enum):
    """Identifies who originated an agent request."""

    USER = "user"
    PLANNER = "planner"


class NodeState(Enum):
    """Lifecycle states a DAG node moves through during a scheduler run."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    INTEGRATED = "integrated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResponseStatus(Enum):
    """Terminal outcome reported by an agent in its response."""

    COMPLETED = "completed"
    FAILED = "failed"
    ALREADY_DONE = "already_done"


class FailureKind(Enum):
    """Classification of why an agent failed."""

    INVALID_JSON = "invalid_json"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    TOOL_ERROR = "tool_error"
    STALE_DELTA = "stale_delta"
    INTEGRATION_FAILED = "integration_failed"
    TEST_FAILED = "test_failed"
    VALIDATION_REJECTED = "validation_rejected"
    UNKNOWN = "unknown"


class CriticDisposition(Enum):
    """Verdict returned by a critic agent."""

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    ALREADY_DONE = "already_done"


class RevisionItem(BaseModel, frozen=True):
    """One required change requested by a critic/referee revision."""

    criterion_id: str | None = Field(
        default=None,
        description="Acceptance criterion id this change addresses, when applicable.",
    )
    required_change: str
    rationale: str | None = None


def _empty_revision_items() -> list[RevisionItem]:
    return []


class RevisionRequest(BaseModel, frozen=True):
    """Typed request for a producer to revise output against the same AgentRequest contract."""

    disposition: Literal["revise"] = "revise"
    rationale: str
    items: list[RevisionItem]
    prior_attempts: int


class CriticFinding(BaseModel, frozen=True):
    """A critic's assessment of a piece of work."""

    disposition: CriticDisposition
    rationale: str
    hints: list[str] = Field(default_factory=list)
    revision_items: list[RevisionItem] = Field(default_factory=_empty_revision_items)


class RefereeDecision(BaseModel, frozen=True):
    """Final adjudication that may override the critic's finding."""

    disposition: CriticDisposition
    rationale: str
    override: bool
    revision_items: list[RevisionItem] = Field(default_factory=_empty_revision_items)


class ReviewContext(BaseModel, frozen=True):
    """Language used to frame critic/referee review for a typed producer output."""

    output_noun: str
    review_focus: str
    empty_output_guidance: str


class AcceptanceCriterion(BaseModel, frozen=True):
    """One explicit contract criterion for accepting an agent output."""

    id: str
    text: str


def _empty_acceptance_criteria() -> list[AcceptanceCriterion]:
    return []


class AgentContract(BaseModel, frozen=True):
    """Authoritative contract for producer/critic/referee judgment."""

    objective: str
    success_condition: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=_empty_acceptance_criteria
    )
    constraints: list[str] = Field(default_factory=_empty_strings)
    non_goals: list[str] = Field(default_factory=_empty_strings)


def _contract_value(data: dict[str, Any], key: str) -> object:
    contract = data.get("contract")
    if isinstance(contract, AgentContract):
        return getattr(contract, key)
    if isinstance(contract, dict):
        contract_dict = cast(dict[str, object], contract)
        return contract_dict.get(key)
    return None


class PlanSpec(BaseModel):
    """Spec for a planning agent request carrying the northstar goal."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["plan"] = "plan"
    northstar: str = ""
    contract: AgentContract = Field(
        default_factory=lambda: AgentContract(
            objective="",
            success_condition="A bounded plan is produced for this objective.",
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _backfill_contract(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(cast(dict[str, Any], data))
        objective = data.get("northstar") or _contract_value(data, "objective")
        if objective is not None and "northstar" not in data:
            data["northstar"] = objective
        if "contract" not in data and objective is not None:
            data["contract"] = {
                "objective": objective,
                "success_condition": "A bounded plan is produced for this objective.",
            }
        return data

    @model_validator(mode="after")
    def _require_contract_fields(self) -> "PlanSpec":
        if not self.northstar:
            raise ValueError("PlanSpec requires northstar or contract.objective")
        if not self.contract.objective:
            raise ValueError("PlanSpec contract requires objective")
        if not self.contract.success_condition:
            raise ValueError("PlanSpec contract requires success_condition")
        return self


class WorkSpec(BaseModel):
    """Spec for a work agent request describing a single concrete task."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["work"] = "work"
    objective: str = ""
    success_condition: str = ""
    contract: AgentContract = Field(
        default_factory=lambda: AgentContract(objective="", success_condition="")
    )
    adapter: str
    artifact: str
    language: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _backfill_contract(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(cast(dict[str, Any], data))
        objective = data.get("objective") or _contract_value(data, "objective")
        success_condition = data.get("success_condition") or _contract_value(
            data, "success_condition"
        )
        if objective is not None and "objective" not in data:
            data["objective"] = objective
        if success_condition is not None and "success_condition" not in data:
            data["success_condition"] = success_condition
        if "contract" not in data and objective is not None and success_condition is not None:
            data["contract"] = {
                "objective": objective,
                "success_condition": success_condition,
            }
        return data

    @model_validator(mode="after")
    def _require_contract_fields(self) -> "WorkSpec":
        if not self.objective:
            raise ValueError("WorkSpec requires objective or contract.objective")
        if not self.success_condition:
            raise ValueError("WorkSpec requires success_condition or contract.success_condition")
        if not self.contract.objective:
            raise ValueError("WorkSpec contract requires objective")
        if not self.contract.success_condition:
            raise ValueError("WorkSpec contract requires success_condition")
        return self


AgentSpec = Annotated[
    PlanSpec | WorkSpec,
    Field(discriminator="kind"),
]


class AgentRequest(BaseModel):
    """Immutable description of a unit of work dispatched to an agent."""

    model_config = ConfigDict(frozen=True)

    id: RequestId = Field(default_factory=uuid4)
    agent_type: AgentType
    source: RequestSource
    spec: AgentSpec
    dependencies: frozenset[RequestId] = Field(default_factory=_empty_request_ids)


def _render_list_section(title: str, values: list[str]) -> list[str]:
    lines = [f"{title}:"]
    if not values:
        lines.append("- (none)")
    else:
        lines.extend(f"- {value}" for value in values)
    return lines


def render_agent_contract(request: AgentRequest) -> str:
    """Render the canonical AgentRequest contract block for prompts."""
    spec = request.spec
    contract = spec.contract
    lines = [
        "AgentRequest contract:",
        f"Objective: {contract.objective}",
        f"Success condition: {contract.success_condition}",
        "Acceptance criteria:",
    ]
    if contract.acceptance_criteria:
        lines.extend(
            f"- {criterion.id}: {criterion.text}" for criterion in contract.acceptance_criteria
        )
    else:
        lines.append("- (none)")
    lines.extend(_render_list_section("Constraints", contract.constraints))
    lines.extend(_render_list_section("Non-goals", contract.non_goals))
    if isinstance(spec, WorkSpec):
        lines.extend(
            [
                f"Artifact: {spec.artifact}",
                f"Adapter: {spec.adapter}",
                f"Language: {spec.language or 'not specified'}",
            ]
        )
    else:
        lines.extend(
            [
                "Artifact: n/a",
                "Adapter: n/a",
                "Language: n/a",
            ]
        )
    return "\n".join(lines)


class Edit(BaseModel, frozen=True):
    """A surgical edit to an existing file — old must be unique in the file."""

    path: str
    old: str
    new: str


class FileWrite(BaseModel, frozen=True):
    """A new file to be written to the artifact directory."""

    path: str
    content: str


def _empty_edits() -> list[Edit]:
    return []


def _empty_file_writes() -> list[FileWrite]:
    return []


def _empty_worker_ids() -> list[RequestId]:
    return []


class IntegrationError(BaseModel, frozen=True):
    """An error encountered during integration — conflict, apply failure, test failure, etc."""

    kind: str
    description: str
    path: str | None = None
    worker_ids: list[RequestId] = Field(default_factory=_empty_worker_ids)


def _empty_integration_errors() -> list[IntegrationError]:
    return []


class DeltaState(BaseModel, frozen=True):
    """The state change produced by a worker or integrator agent."""

    new_files: list[FileWrite] = Field(default_factory=_empty_file_writes)
    edits: list[Edit] = Field(default_factory=_empty_edits)
    dependencies: list[str] = Field(default_factory=_empty_strings)
    errors: list[IntegrationError] = Field(default_factory=_empty_integration_errors)
    base_version: int = 0


class RunResult(BaseModel, frozen=True):
    """Result of running tests against the current artifact state."""

    passed: bool
    failures: list[str] = Field(default_factory=_empty_strings)
    summary: str = ""
    output: str = ""


class FileView(BaseModel, frozen=True):
    """A file in the artifact directory with its path and full content."""

    path: str
    content: str


class StateView(BaseModel, frozen=True):
    """A high-level projection of the current artifact state for LLM context."""

    artifact_name: str
    language: str | None
    files: list[FileView]
    dependencies: list[str]
    test_summary: str | None = None
    version: int = 0


def _empty_ints() -> list[int]:
    return []


class TaskSpec(BaseModel, frozen=True):
    """A single task emitted by the planner."""

    objective: str
    success_condition: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=_empty_acceptance_criteria
    )
    constraints: list[str] = Field(default_factory=_empty_strings)
    non_goals: list[str] = Field(default_factory=_empty_strings)
    adapter: str
    artifact: str
    language: str | None = None
    depends_on: list[int] = Field(default_factory=_empty_ints)


class PlanResponse(BaseModel, frozen=True):
    """The planner's final output — a list of tasks to execute."""

    kind: Literal["plan"] = "plan"
    tasks: list[TaskSpec]


ProducerOutput = PlanResponse | DeltaState


def _empty_agent_requests() -> list[AgentRequest]:
    return []


class AgentResponse(BaseModel):
    """Immutable result returned by an agent after processing a request."""

    model_config = ConfigDict(frozen=True)

    request_id: RequestId
    status: ResponseStatus
    output: ProducerOutput | None = None
    delta: DeltaState | None = None
    follow_up: list[AgentRequest] = Field(default_factory=_empty_agent_requests)
    error: str | None = None
    failure_kind: FailureKind | None = None
    ran_tests_and_passed: bool = False

    @model_validator(mode="before")
    @classmethod
    def _sync_legacy_delta(cls, data: object) -> object:
        """Keep legacy delta construction compatible while output becomes canonical."""
        if not isinstance(data, dict):
            return data
        values: dict[str, object] = dict(cast(dict[str, object], data))
        output = values.get("output")
        delta = values.get("delta")
        if output is None and isinstance(delta, DeltaState):
            values["output"] = delta
        elif delta is None and isinstance(output, DeltaState):
            values["delta"] = output
        return values

    @model_validator(mode="after")
    def _sync_parsed_legacy_delta(self) -> "AgentResponse":
        """Sync parsed legacy delta payloads after Pydantic has built nested models."""
        if self.output is None and self.delta is not None:
            object.__setattr__(self, "output", self.delta)
        elif self.delta is None and isinstance(self.output, DeltaState):
            object.__setattr__(self, "delta", self.output)
        return self


class ToolCallRequest(BaseModel, frozen=True):
    """LLM requests a tool to be executed."""

    kind: Literal["tool_call"]
    name: str
    arguments: dict[str, Any]


class ToolCallResponse(BaseModel, frozen=True):
    """Framework response to a tool call — fed back to the LLM."""

    kind: Literal["tool_response"]
    name: str
    success: bool
    result: Any
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
            NodeState.INTEGRATED
            if response.status in (ResponseStatus.COMPLETED, ResponseStatus.ALREADY_DONE)
            else NodeState.FAILED
        )
        return self.model_copy(update={"node_state": node_state, "response": response})


def _empty_dag() -> dict[RequestId, DAGNode]:
    return {}


class SchedulerState(BaseModel):
    """Immutable snapshot of the full DAG and scheduler configuration at a point in time."""

    model_config = ConfigDict(frozen=True)

    dag: dict[RequestId, DAGNode] = Field(default_factory=_empty_dag)
    northstar: str
    max_concurrency: int = 1

    def add_nodes(self, nodes: list[DAGNode]) -> "SchedulerState":
        """Return a new state with the given nodes merged into the DAG."""
        return self.model_copy(update={"dag": {**self.dag, **{n.request.id: n for n in nodes}}})

    def update_node(self, node: DAGNode) -> "SchedulerState":
        """Return a new state with the given node replacing its previous entry in the DAG."""
        return self.model_copy(update={"dag": {**self.dag, node.request.id: node}})

    def ready_nodes(self) -> list[DAGNode]:
        """Return all PENDING nodes whose dependencies are all INTEGRATED."""
        integrated = {nid for nid, n in self.dag.items() if n.node_state == NodeState.INTEGRATED}
        return [
            n
            for n in self.dag.values()
            if n.node_state == NodeState.PENDING and n.request.dependencies <= integrated
        ]
