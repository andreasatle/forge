"""AttemptEngine — generic attempt/validation/retry loop for work and plan tasks."""

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar, cast, runtime_checkable
from uuid import UUID

from pydantic import BaseModel

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.critic import critic_agent
from forge.agents.referee import referee_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    CriticFinding,
    FailureKind,
    PlanResponse,
    ResponseStatus,
    ReviewContext,
    RevisionItem,
    RevisionRequest,
    StateView,
    WorkOutput,
)
from forge.core.telemetry import TelemetryEvent, TelemetrySink, safe_append_telemetry
from forge.llm.providers import LLMProvider

_logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_REVISION_RATIONALE_CHARS = 1200
_MAX_REVISION_CHANGE_CHARS = 1600
_MAX_REVISION_ITEM_RATIONALE_CHARS = 900
_REPEATED_CONTRACT_MARKER = (
    "[omitted repeated AgentRequest contract; apply the contract block above]"
)
_REPEATED_PLUGIN_GUIDANCE_MARKER = (
    "[omitted repeated language plugin guidance; apply the binding language constraints above]"
)


@runtime_checkable
class OutputValidator(Protocol[T]):
    """Protocol for validating and rendering output in the producer/review retry loop."""

    def extract_from_response(self, response: AgentResponse) -> T | None:
        """Extract typed output from an AgentResponse, or None if unavailable."""
        ...

    def is_empty(self, output: T) -> bool:
        """Return True when the output contains no meaningful work."""
        ...

    def render_for_critic(self, output: T) -> str:
        """Render output as a human-readable string for the critic/referee."""
        ...

    def work_noun(self) -> str:
        """Return the singular noun describing this kind of output (e.g. 'implementation')."""
        ...

    def requires_nonempty(self) -> bool:
        """Return True when empty output should trigger a retry rather than ALREADY_DONE."""
        ...

    def review_context(self) -> ReviewContext:
        """Return language for critic/referee prompts for this output type."""
        ...

    def final_output_reminder(self) -> str:
        """Return a compact output-format reminder for retry prompts."""
        ...


class WorkOutputValidator:
    """OutputValidator for WorkOutput — validates full-file-content output from work agents."""

    def __init__(self, adapter_spec: AdapterSpec, state_view: StateView) -> None:
        self._adapter = adapter_spec
        self._state_view = state_view

    def extract_from_response(self, response: AgentResponse) -> WorkOutput | None:
        """Return typed WorkOutput output from the response."""
        return response.output if isinstance(response.output, WorkOutput) else None

    def is_empty(self, output: WorkOutput) -> bool:
        """Return True when the WorkOutput has no files or dependencies."""
        return not output.files and not output.dependencies

    def render_for_critic(self, output: WorkOutput) -> str:
        """Render WorkOutput files and existing artifact state for the critic."""
        lines: list[str] = []
        if output.files:
            lines.append("Files proposed:")
            for fc in output.files:
                lines += [f"\nFile: {fc.path}", "```", fc.content, "```"]
        else:
            lines.append("No files were proposed.")
        if self._state_view.files:
            if lines:
                lines.append("")
            lines.append("Existing artifact files:")
            for fv in self._state_view.files:
                lines += [f"\nFile: {fv.path}", "```", fv.content, "```"]
        return "\n".join(lines)

    def work_noun(self) -> str:
        """Return the adapter's work noun."""
        return self._adapter.work_noun

    def requires_nonempty(self) -> bool:
        """Return the adapter's requires_nonempty_output flag."""
        return self._adapter.requires_nonempty_output

    def review_context(self) -> ReviewContext:
        """Return worker-output review language."""
        return ReviewContext(
            output_noun=self._adapter.work_noun,
            review_focus="whether the proposed files satisfy the task",
            empty_output_guidance=(
                "If no files were proposed, reject unless the "
                "success condition is already demonstrably met."
            ),
        )

    def final_output_reminder(self) -> str:
        """Return a compact WorkOutput format reminder."""
        return "\n".join(
            [
                "FINAL OUTPUT FORMAT REMINDER",
                "Return valid JSON only matching WorkOutput.",
                "- files must be an array of JSON objects.",
                '  Each files item must be {"path": "...", "content": "..."}',
                '- Do not use string entries like "path:...".',
            ]
        )


class PlanResponseValidator:
    """OutputValidator for PlanResponse — validates task decomposition from plan agents."""

    def extract_from_response(self, response: AgentResponse) -> PlanResponse | None:
        """Return typed PlanResponse output from the response."""
        return response.output if isinstance(response.output, PlanResponse) else None

    def is_empty(self, output: PlanResponse) -> bool:
        """Always returns False — planners never trigger ALREADY_DONE."""
        return False

    def render_for_critic(self, output: PlanResponse) -> str:
        """Render plan tasks as a numbered list for the critic."""
        if not output.tasks:
            return "(no tasks)"
        lines: list[str] = []
        for i, task in enumerate(output.tasks):
            lines.append(f"Task {i}: {task.objective}")
            lines.append(f"  Success condition: {task.success_condition}")
            if task.acceptance_criteria:
                lines.append("  Acceptance criteria:")
                lines.extend(
                    f"    - {criterion.id}: {criterion.text}"
                    for criterion in task.acceptance_criteria
                )
            if task.constraints:
                lines.append("  Constraints:")
                lines.extend(f"    - {constraint}" for constraint in task.constraints)
            if task.non_goals:
                lines.append("  Non-goals:")
                lines.extend(f"    - {non_goal}" for non_goal in task.non_goals)
            if task.artifact:
                lines.append(f"  Artifact: {task.artifact}")
            if task.language:
                lines.append(f"  Language: {task.language}")
        return "\n".join(lines)

    def work_noun(self) -> str:
        """Return 'plan'."""
        return "plan"

    def requires_nonempty(self) -> bool:
        """Return True — plan agents must always produce tasks."""
        return True

    def review_context(self) -> ReviewContext:
        """Return planner-output review language."""
        return ReviewContext(
            output_noun="plan",
            review_focus="whether the task decomposition satisfies the planning contract",
            empty_output_guidance="If the plan contains no tasks, reject it.",
        )

    def final_output_reminder(self) -> str:
        """Return a compact PlanResponse output-format reminder."""
        return "\n".join(
            [
                "FINAL OUTPUT FORMAT REMINDER",
                "Return valid JSON only matching PlanResponse.",
                '- Top-level kind must be "plan".',
                "- tasks must be an array of task objects satisfying the AgentRequest contract.",
            ]
        )


class RunAgentFailed(Exception):
    """Raised when run_agent returns a non-COMPLETED response."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        super().__init__(response.error or "run_agent failed")


def _validation_rejected_response(
    request: AgentRequest,
    disposition: CriticDisposition,
    rationale: str,
    output_noun: str = "output",
) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error=(
            f"validation rejected {output_noun} with disposition '{disposition.value}': {rationale}"
        ),
        failure_kind=FailureKind.VALIDATION_REJECTED,
    )


def _validation_parse_failed_response(request: AgentRequest, error: ValueError) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error=f"validation response could not be parsed: {error}",
        failure_kind=FailureKind.INVALID_JSON,
    )


def _revision_items_from_hints(hints: list[str], rationale: str) -> list[RevisionItem]:
    """Convert free-form hints into structured revision items."""
    return [
        RevisionItem(required_change=hint, rationale=rationale) for hint in hints if hint.strip()
    ]


def _revision_items_from_finding(finding: CriticFinding) -> list[RevisionItem]:
    """Return structured critic revision items, falling back to free-form hints."""
    if finding.revision_items:
        return finding.revision_items
    return _revision_items_from_hints(finding.hints, finding.rationale)


def _build_revision_request(
    *,
    rationale: str,
    prior_attempts: int,
    items: list[RevisionItem],
) -> RevisionRequest:
    """Build a non-empty typed RevisionRequest from structured items or rationale."""
    if not items:
        items = [RevisionItem(required_change=rationale, rationale=rationale)]
    return RevisionRequest(
        disposition="revise",
        rationale=rationale,
        items=items,
        prior_attempts=prior_attempts,
    )


def _strip_repeated_block(text: str, heading: str, marker: str) -> str:
    """Remove an embedded invariant prompt block from reviewer-supplied retry text."""
    start = text.find(heading)
    if start == -1:
        return text
    before = text[:start].rstrip()
    rest = text[start + len(heading) :]
    match = re.search(r"\n\s*\n", rest)
    after = rest[match.end() :].lstrip() if match else ""
    parts = [part for part in (before, marker, after) if part]
    return "\n".join(parts)


def _truncate_revision_text(text: str, limit: int) -> str:
    """Bound a retry-history field while preserving enough text to act on it."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 15].rstrip()} ...[truncated]"


def _compact_revision_text(text: str | None, limit: int) -> str:
    """Compact reviewer text before placing it in accumulated producer retry history."""
    if not text:
        return ""
    compact = _strip_repeated_block(
        text,
        "AgentRequest contract:",
        _REPEATED_CONTRACT_MARKER,
    )
    compact = _strip_repeated_block(
        compact,
        "Language plugin guidance:",
        _REPEATED_PLUGIN_GUIDANCE_MARKER,
    )
    return _truncate_revision_text(compact, limit)


def _render_revision_requests(
    revision_requests: list[RevisionRequest],
    output_noun: str,
    final_output_reminder: str,
) -> str:
    """Render accumulated RevisionRequests as a prominent producer retry block."""
    lines = [
        "REQUIRED REVISION",
        "You must revise your next output against the same AgentRequest contract above.",
        "The next output must address every required change listed below.",
    ]
    for request_index, revision_request in enumerate(revision_requests, start=1):
        lines.extend(
            [
                "",
                f"Revision request {request_index} "
                f"(after {revision_request.prior_attempts} prior attempt(s)):",
                f"Previous disposition: {revision_request.disposition}",
                "Rationale: "
                f"{_compact_revision_text(revision_request.rationale, _MAX_REVISION_RATIONALE_CHARS)}",
                "Required changes:",
            ]
        )
        for item_index, item in enumerate(revision_request.items, start=1):
            criterion = f" [{item.criterion_id}]" if item.criterion_id else ""
            lines.append(
                f"{item_index}. Required change{criterion}: "
                f"{_compact_revision_text(item.required_change, _MAX_REVISION_CHANGE_CHARS)}"
            )
            if item.rationale:
                lines.append(
                    "   Rationale: "
                    f"{_compact_revision_text(item.rationale, _MAX_REVISION_ITEM_RATIONALE_CHARS)}"
                )
    lines.extend(
        [
            "",
            f"Revise your {output_noun} now.",
            "Do not repeat the previous output unless it has been changed to address every required change.",
        ]
    )
    if final_output_reminder:
        lines.extend(["", final_output_reminder])
    return "\n".join(lines)


def _preview(text: str | None, limit: int = 500) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 15].rstrip()} ...[truncated]"


def _work_output_summary(output: WorkOutput) -> dict[str, object]:
    return {
        "file_count": len(output.files),
        "dependency_count": len(output.dependencies),
        "file_paths": [file.path for file in output.files],
        "dependencies": list(output.dependencies),
        "base_version": output.base_version,
    }


def _plan_summary(plan: PlanResponse) -> dict[str, object]:
    return {
        "task_count": len(plan.tasks),
        "tasks": [
            {
                "objective": task.objective,
                "adapter": task.adapter,
                "artifact": task.artifact,
                "language": task.language,
                "depends_on": list(task.depends_on),
            }
            for task in plan.tasks
        ],
    }


def _producer_response_summary(response: AgentResponse) -> dict[str, object]:
    output_type = type(response.output).__name__ if response.output is not None else None
    data: dict[str, object] = {
        "status": response.status.value,
        "failure_kind": response.failure_kind.value if response.failure_kind else None,
        "error": _preview(response.error),
        "output_type": output_type,
        "ran_tests_and_passed": response.ran_tests_and_passed,
    }
    if isinstance(response.output, WorkOutput):
        data["work_output"] = _work_output_summary(response.output)
    elif isinstance(response.output, PlanResponse):
        data["plan"] = _plan_summary(response.output)
    return data


def _model_data(model: BaseModel) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="json"))


class AttemptEngine[T]:
    """Generic producer/critic/referee retry loop for work and plan agents."""

    def __init__(
        self,
        request: AgentRequest,
        state_view: StateView,
        validator: OutputValidator[T],
        run_fn: Callable[[str], Awaitable[AgentResponse]],
        registry: AdapterRegistry | None = None,
        critic_provider: LLMProvider | None = None,
        referee_provider: LLMProvider | None = None,
        max_attempts: int = 3,
        telemetry_sink: TelemetrySink | None = None,
        run_id: UUID | None = None,
        initial_revision: RevisionRequest | None = None,
    ) -> None:
        self._request = request
        self._state_view = state_view
        self._validator = validator
        self._run_fn = run_fn
        self._registry = registry
        self._critic_provider = critic_provider
        self._referee_provider = referee_provider
        self._max_attempts = max_attempts
        self._telemetry_sink = telemetry_sink
        self._run_id = run_id
        self._initial_revision = initial_revision

    def _emit(
        self,
        *,
        attempt_number: int | None,
        role: str,
        phase: str,
        event_type: str,
        status: str | None = None,
        summary: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        if self._run_id is None:
            return
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=self._request.id,
                request_id=self._request.id,
                agent_type=self._request.agent_type.value,
                attempt_number=attempt_number,
                role=role,
                phase=phase,
                event_type=event_type,
                status=status,
                summary=summary,
                data=data or {},
            ),
        )

    async def run(self, prompt: str) -> AgentResponse:
        """Run attempts with validation/retry; return AgentResponse."""
        revision_requests: list[RevisionRequest] = (
            [self._initial_revision] if self._initial_revision is not None else []
        )

        for attempt in range(self._max_attempts):
            attempt_number = attempt + 1
            self._emit(
                attempt_number=attempt_number,
                role="producer",
                phase="pwc",
                event_type="pwc.attempt.started",
                status="started",
                summary=f"attempt {attempt_number}/{self._max_attempts} started",
                data={"max_attempts": self._max_attempts},
            )
            current_prompt = (
                prompt
                if not revision_requests
                else (
                    f"{prompt}\n\n"
                    f"{
                        _render_revision_requests(
                            revision_requests,
                            self._validator.work_noun(),
                            self._validator.final_output_reminder(),
                        )
                    }"
                )
            )
            response = await self._run_fn(current_prompt)
            output = self._validator.extract_from_response(response)
            self._emit(
                attempt_number=attempt_number,
                role="producer",
                phase="producer",
                event_type="producer.response.parsed",
                status=response.status.value,
                summary=f"producer returned {response.status.value}",
                data=_producer_response_summary(response),
            )

            if (
                response.status == ResponseStatus.FAILED
                and response.failure_kind == FailureKind.VALIDATION_REJECTED
                and output is not None
                and self._validator.is_empty(output)
            ):
                if response.ran_tests_and_passed:
                    _logger.info(
                        "attempt %d/%d: empty output but ran_tests_and_passed — ALREADY_DONE",
                        attempt_number,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.ALREADY_DONE,
                        output=response.output,
                    )
                if self._critic_provider is None:
                    is_last = attempt == self._max_attempts - 1
                    if not is_last:
                        _logger.info(
                            "attempt %d/%d: empty output, no critic — injecting correction, retrying",
                            attempt_number,
                            self._max_attempts,
                        )
                        revision_request = _build_revision_request(
                            rationale=(
                                f"Your previous attempt produced no {self._validator.work_noun()}."
                            ),
                            prior_attempts=attempt_number,
                            items=[
                                RevisionItem(
                                    required_change=(
                                        "Produce concrete output satisfying the "
                                        "AgentRequest contract."
                                    ),
                                    rationale="The previous attempt produced no output.",
                                )
                            ],
                        )
                        revision_requests.append(revision_request)
                        self._emit(
                            attempt_number=attempt_number,
                            role="revision_loop",
                            phase="pwc",
                            event_type="pwc.revision.appended",
                            status="revise",
                            summary="revision request appended",
                            data={"revision_request": _model_data(revision_request)},
                        )
                        continue
                    if not self._validator.requires_nonempty():
                        _logger.info(
                            "attempt %d/%d: empty output, no critic, last attempt — ALREADY_DONE",
                            attempt_number,
                            self._max_attempts,
                        )
                        return AgentResponse(
                            request_id=self._request.id,
                            status=ResponseStatus.ALREADY_DONE,
                            output=response.output,
                        )
                    _logger.info(
                        "attempt %d/%d: empty output, no critic, last attempt, requires nonempty — FAILED",
                        attempt_number,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.FAILED,
                        error=(
                            f"producer produced no {self._validator.work_noun()} "
                            f"after {self._max_attempts} attempts"
                        ),
                        failure_kind=FailureKind.VALIDATION_REJECTED,
                    )
                try:
                    finding = await critic_agent(
                        self._request,
                        self._state_view,
                        self._validator.render_for_critic(output),
                        self._critic_provider,
                        cast(AdapterRegistry, self._registry),
                        review_context=self._validator.review_context(),
                    )
                except ValueError as e:
                    _logger.warning(
                        "attempt %d/%d: critic failed on empty output: %s",
                        attempt_number,
                        self._max_attempts,
                        e,
                    )
                    return _validation_parse_failed_response(self._request, e)
                self._emit(
                    attempt_number=attempt_number,
                    role="critic",
                    phase="critic",
                    event_type="critic.finding.parsed",
                    status=finding.disposition.value,
                    summary=_preview(finding.rationale),
                    data={"critic_finding": _model_data(finding)},
                )
                if finding.disposition == CriticDisposition.ALREADY_DONE:
                    _logger.info(
                        "attempt %d/%d: critic confirmed ALREADY_DONE",
                        attempt_number,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.ALREADY_DONE,
                        output=response.output,
                    )
                if finding.disposition == CriticDisposition.REJECT:
                    _logger.info(
                        "attempt %d/%d: critic rejected empty output",
                        attempt_number,
                        self._max_attempts,
                    )
                    return _validation_rejected_response(
                        self._request,
                        finding.disposition,
                        finding.rationale,
                        self._validator.work_noun(),
                    )
                _logger.info(
                    "attempt %d/%d: critic=%s on empty output — retrying",
                    attempt_number,
                    self._max_attempts,
                    finding.disposition.value,
                )
                revision_request = _build_revision_request(
                    rationale=finding.rationale,
                    prior_attempts=attempt_number,
                    items=_revision_items_from_finding(finding),
                )
                revision_requests.append(revision_request)
                self._emit(
                    attempt_number=attempt_number,
                    role="revision_loop",
                    phase="pwc",
                    event_type="pwc.revision.appended",
                    status="revise",
                    summary="revision request appended",
                    data={"revision_request": _model_data(revision_request)},
                )
                continue

            if response.status != ResponseStatus.COMPLETED or output is None:
                raise RunAgentFailed(response)

            if self._critic_provider is None or self._referee_provider is None:
                return response

            try:
                output_text = self._validator.render_for_critic(output)
                finding = await critic_agent(
                    self._request,
                    self._state_view,
                    output_text,
                    self._critic_provider,
                    cast(AdapterRegistry, self._registry),
                    review_context=self._validator.review_context(),
                )
                decision = await referee_agent(
                    self._request,
                    self._state_view,
                    output_text,
                    finding,
                    self._referee_provider,
                    cast(AdapterRegistry, self._registry),
                    review_context=self._validator.review_context(),
                )
            except ValueError as e:
                _logger.warning(
                    "attempt %d/%d: validation parsing failed: %s",
                    attempt_number,
                    self._max_attempts,
                    e,
                )
                return _validation_parse_failed_response(self._request, e)

            _logger.info(
                "attempt %d/%d: critic=%s referee=%s — %s",
                attempt_number,
                self._max_attempts,
                finding.disposition.value,
                decision.disposition.value,
                "returning" if decision.disposition == CriticDisposition.ACCEPT else "retrying",
            )
            self._emit(
                attempt_number=attempt_number,
                role="critic",
                phase="critic",
                event_type="critic.finding.parsed",
                status=finding.disposition.value,
                summary=_preview(finding.rationale),
                data={"critic_finding": _model_data(finding)},
            )
            self._emit(
                attempt_number=attempt_number,
                role="referee",
                phase="referee",
                event_type="referee.decision.parsed",
                status=decision.disposition.value,
                summary=_preview(decision.rationale),
                data={"referee_decision": _model_data(decision)},
            )

            if decision.disposition == CriticDisposition.ACCEPT:
                return response
            if decision.disposition == CriticDisposition.REJECT:
                return _validation_rejected_response(
                    self._request,
                    decision.disposition,
                    decision.rationale,
                    self._validator.work_noun(),
                )
            if decision.disposition == CriticDisposition.DECOMPOSE:
                _logger.info(
                    "attempt %d/%d: referee requested decomposition",
                    attempt_number,
                    self._max_attempts,
                )
                self._emit(
                    attempt_number=attempt_number,
                    role="referee",
                    phase="referee",
                    event_type="pwc.decompose.requested",
                    status="decompose",
                    summary=_preview(decision.rationale),
                    data={"referee_decision": _model_data(decision)},
                )
                return AgentResponse(
                    request_id=self._request.id,
                    status=ResponseStatus.DECOMPOSE,
                )

            revision_items = (
                decision.revision_items
                if decision.revision_items
                else _revision_items_from_finding(finding)
            )
            revision_request = _build_revision_request(
                rationale=decision.rationale,
                prior_attempts=attempt_number,
                items=revision_items,
            )
            revision_requests.append(revision_request)
            self._emit(
                attempt_number=attempt_number,
                role="revision_loop",
                phase="pwc",
                event_type="pwc.revision.appended",
                status="revise",
                summary="revision request appended",
                data={"revision_request": _model_data(revision_request)},
            )

        _logger.warning(
            "max_attempts (%d) exhausted; validation did not accept",
            self._max_attempts,
        )
        self._emit(
            attempt_number=self._max_attempts,
            role="revision_loop",
            phase="pwc",
            event_type="pwc.exhausted",
            status="failed",
            summary="maximum validation attempts exhausted without an accept disposition",
            data={
                "attempt_count": self._max_attempts,
                "final_disposition": CriticDisposition.REVISE.value,
                "error": "maximum validation attempts exhausted without an accept disposition",
            },
        )
        return _validation_rejected_response(
            self._request,
            CriticDisposition.REVISE,
            "maximum validation attempts exhausted without an accept disposition",
            self._validator.work_noun(),
        )
