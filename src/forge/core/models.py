"""Core Pydantic models and enums shared across all forge components."""

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, assert_never, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

RequestId = UUID


def _empty_request_ids() -> frozenset[RequestId]:
    return frozenset()


def _empty_strings() -> list[str]:
    return []


class AgentType(Enum):
    """Discriminator enum for the two agent roles in the system."""

    PLAN = "plan"
    WORK = "work"


class AgentMessageKind(StrEnum):
    """Protocol-level discriminator values for agent messages."""

    TOOL_RESPONSE = "tool_response"
    WORK_OUTPUT = "work_output"


class RequestSource(Enum):
    """Identifies who originated an agent request."""

    USER = "user"
    PLANNER = "planner"


class NodeState(Enum):
    """Lifecycle states a DAG node moves through during a scheduler run."""

    PENDING = "pending"
    RUNNING = "running"
    INTEGRATED = "integrated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResponseStatus(Enum):
    """Terminal outcome reported by an agent in its response."""

    COMPLETED = "completed"
    FAILED = "failed"
    ALREADY_DONE = "already_done"
    DECOMPOSE = "decompose"


class FailureKind(Enum):
    """Classification of why an agent failed."""

    INVALID_JSON = "invalid_json"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    TOOL_ERROR = "tool_error"
    STALE_WORK_OUTPUT = "stale_work_output"
    INTEGRATION_FAILED = "integration_failed"
    TEST_FAILED = "test_failed"
    VALIDATION_REJECTED = "validation_rejected"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


class CriticDisposition(Enum):
    """Verdict returned by a critic agent."""

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    ALREADY_DONE = "already_done"
    DECOMPOSE = "decompose"


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
    topology_rules: str = ""


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
    def _derive_contract(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        d: dict[str, object] = dict(data)  # type: ignore[arg-type]
        if "contract" not in d and d.get("northstar"):
            d["contract"] = {
                "objective": d["northstar"],
                "success_condition": "A bounded plan is produced for this objective.",
            }
        return d

    @model_validator(mode="after")
    def _require_contract_fields(self) -> "PlanSpec":
        if not self.northstar:
            raise ValueError("PlanSpec requires northstar")
        if not self.contract.objective:
            raise ValueError("PlanSpec contract requires objective")
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
    def _derive_contract(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        d: dict[str, object] = dict(data)  # type: ignore[arg-type]
        if "contract" not in d and d.get("objective") and d.get("success_condition"):
            d["contract"] = {
                "objective": d["objective"],
                "success_condition": d["success_condition"],
            }
        return d

    @model_validator(mode="after")
    def _require_contract_fields(self) -> "WorkSpec":
        if not self.objective:
            raise ValueError("WorkSpec requires objective")
        if not self.success_condition:
            raise ValueError("WorkSpec requires success_condition")
        if not self.contract.objective:
            raise ValueError("WorkSpec contract requires objective")
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
    model_profile: str = "default"
    initial_revision: RevisionRequest | None = None


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


class WorkOutput(BaseModel, frozen=True):
    """Completion metadata for work already written to the assigned git worktree."""

    kind: Literal[AgentMessageKind.WORK_OUTPUT] = AgentMessageKind.WORK_OUTPUT
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        d = cast("dict[str, object]", data)
        if "files" in d or "dependencies" in d:
            raise ValueError(
                "WorkOutput no longer accepts 'files' or 'dependencies' fields. "
                "Write changes to the worktree directly and return a summary only."
            )
        return d


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
    version: int = 0
    version_sha: str = ""


class TaskSpec(BaseModel, frozen=True):
    """A single task emitted by the planner — always expands to a work node."""

    kind: Literal["work_task"] = "work_task"
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


class DecompositionTask(BaseModel, frozen=True):
    """A child task that expands to a plan node rather than a work node."""

    kind: Literal["decomposition_task"] = "decomposition_task"
    objective: str
    success_condition: str


ChildTask = Annotated[
    TaskSpec | DecompositionTask,
    Field(discriminator="kind"),
]


class WorkDecision(BaseModel, frozen=True):
    """Decomposition decision to execute a task directly without further splitting."""

    kind: Literal["work"] = "work"
    task: WorkSpec


def _default_child_task_kinds(tasks: object) -> object:
    """Insert kind='work_task' into task dicts that omit it — LLMs don't emit the discriminator."""
    if not isinstance(tasks, list):
        return tasks
    result: list[Any] = []
    for t in cast("list[Any]", tasks):
        result.append({**t, "kind": "work_task"} if isinstance(t, dict) and "kind" not in t else t)
    return result


class DecompositionNodeSpec(BaseModel, frozen=True):
    """One node in a graph decomposition — carries a child task and its intra-graph dependencies."""

    id: str
    task: ChildTask
    depends_on: list[str] = Field(default_factory=_empty_strings)

    @field_validator("task", mode="before")
    @classmethod
    def _default_task_kind(cls, v: object) -> object:
        return cast("list[object]", _default_child_task_kinds([v]))[0]


class GraphSplitDecision(BaseModel, frozen=True):
    """Decomposition decision for a mixed-topology DAG with explicit per-node depends_on."""

    kind: Literal["split_graph"] = "split_graph"
    nodes: list[DecompositionNodeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_depends_on_refs(self) -> "GraphSplitDecision":
        node_ids = {node.id for node in self.nodes}
        for node in self.nodes:
            if node.id in node.depends_on:
                raise ValueError(f"Node '{node.id}' depends on itself.")
            for ref in node.depends_on:
                if ref not in node_ids:
                    raise ValueError(
                        f"Node '{node.id}' depends_on '{ref}' which is not a known node id. "
                        f"Known ids: {sorted(node_ids)}"
                    )
        return self


DecompositionDecision = Annotated[
    WorkDecision | GraphSplitDecision,
    Field(discriminator="kind"),
]


class PlannerOutputModel(
    RootModel[
        Annotated[
            WorkDecision | GraphSplitDecision,
            Field(discriminator="kind"),
        ]
    ],
    frozen=True,
):
    """Planner final response type — the canonical decomposition decision."""


ProducerOutput = WorkDecision | GraphSplitDecision | WorkOutput


class ToolTurn(BaseModel, frozen=True):
    """Strict protocol envelope for one LLM tool call turn."""

    kind: Literal["tool"] = "tool"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalTurn(BaseModel, frozen=True):
    """Strict protocol envelope for one LLM final-answer turn."""

    kind: Literal["final"] = "final"
    output: Annotated[
        WorkOutput | WorkDecision | GraphSplitDecision,
        Field(discriminator="kind"),
    ]


VALIDATION_EXHAUSTED_DIAGNOSTIC = "validation_exhausted"


class AgentDiagnostic(BaseModel, frozen=True):
    """Bounded diagnostic context captured from a failed agent attempt."""

    kind: str
    message: str
    validation_path: str | None = None
    bad_value_excerpt: str | None = None
    raw_response_excerpt: str | None = None


def _empty_agent_diagnostics() -> list[AgentDiagnostic]:
    return []


class AgentResponse(BaseModel):
    """Immutable result returned by an agent after processing a request."""

    model_config = ConfigDict(frozen=True)

    request_id: RequestId
    status: ResponseStatus
    output: ProducerOutput | None = None
    error: str | None = None
    failure_kind: FailureKind | None = None
    ran_tests_and_passed: bool = False
    diagnostics: list[AgentDiagnostic] = Field(default_factory=_empty_agent_diagnostics)
    revision: RevisionRequest | None = None
    dispatch_sha: str = ""


class ToolCallResponse(BaseModel, frozen=True):
    """Framework response to a tool call — fed back to the LLM."""

    kind: Literal[AgentMessageKind.TOOL_RESPONSE]
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
    decomposition_depth: int = 0
    retry_of: RequestId | None = None
    profile_escalation_attempt: int = 0
    post_merge_retry_attempt: int = 0
    prior_profiles: tuple[str, ...] = ()

    def with_state(self, node_state: NodeState) -> "DAGNode":
        """Return a copy of this node with the given node_state."""
        return self.model_copy(update={"node_state": node_state})

    def with_response(self, response: AgentResponse) -> "DAGNode":
        """Return a copy of this node with the response set and state derived from its status."""
        match response.status:
            case ResponseStatus.DECOMPOSE:
                node_state = NodeState.CANCELLED
            case ResponseStatus.COMPLETED | ResponseStatus.ALREADY_DONE:
                node_state = NodeState.INTEGRATED
            case ResponseStatus.FAILED:
                node_state = NodeState.FAILED
            case _ as unreachable:
                assert_never(unreachable)
        return self.model_copy(update={"node_state": node_state, "response": response})


def _empty_dag() -> dict[RequestId, DAGNode]:
    return {}


class SchedulerState(BaseModel):
    """Immutable snapshot of the full DAG and scheduler configuration at a point in time."""

    model_config = ConfigDict(frozen=True)

    dag: dict[RequestId, DAGNode] = Field(default_factory=_empty_dag)
    northstar: str
    max_concurrency: int = 1
    max_plan_depth: int = 8
    max_dag_nodes: int = 1000

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
