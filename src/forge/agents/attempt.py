"""AttemptLifecycle for work and plan PWC execution."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, TypeVar, cast, runtime_checkable
from uuid import UUID

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.attempt_telemetry import AttemptTelemetryReporter
from forge.agents.critic import critic_agent
from forge.agents.referee import referee_agent
from forge.agents.revisions import RevisionHistory
from forge.core.file_filters import (
    CRITIC_EVIDENCE_EXCLUDED_DIRS,
    CRITIC_EVIDENCE_EXCLUDED_SUFFIXES,
    EXCLUDED_FILE_NAMES,
)
from forge.core.models import (
    VALIDATION_EXHAUSTED_DIAGNOSTIC,
    AgentDiagnostic,
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    FailureKind,
    GraphSplitDecision,
    ResponseStatus,
    ReviewContext,
    RevisionItem,
    RevisionRequest,
    StateView,
    TaskSpec,
    WorkDecision,
    WorkOutput,
)
from forge.core.telemetry import TelemetrySink
from forge.core.workspace import run_git
from forge.llm.providers import LLMProvider

_logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_UNTRACKED_BYTES = 64 * 1024
_NO_WORKTREE_CHANGE_REQUIRED_CHANGE = (
    "Actually modify the assigned worktree to satisfy the task; the previous attempt "
    "returned WorkOutput metadata but produced no file changes."
)
_NO_WORKTREE_CHANGE_PHRASES = (
    "no actual file changes",
    "no file changes",
    "no files changed",
    "no files were changed",
    "no worktree changes",
    "no worktree change",
    "empty git diff",
    "git diff is empty",
    "git diff was empty",
    "worktree was empty",
    "worktree is empty",
    "no evidence of changes",
    "no evidence of file changes",
    "produced no file changes",
    "did not change any files",
    "didn't change any files",
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
    """OutputValidator for WorkOutput — validates git-native work completion metadata."""

    def __init__(
        self,
        adapter_spec: AdapterSpec,
        state_view: StateView,
        worktree_path: Path | None = None,
    ) -> None:
        self._adapter = adapter_spec
        self._state_view = state_view
        self._worktree_path = worktree_path

    def extract_from_response(self, response: AgentResponse) -> WorkOutput | None:
        """Return typed WorkOutput output from the response."""
        return response.output if isinstance(response.output, WorkOutput) else None

    def is_empty(self, output: WorkOutput) -> bool:
        """Return True when the WorkOutput has no completion summary."""
        return not output.summary.strip()

    def render_for_critic(self, output: WorkOutput) -> str:
        """Render worktree status/diff and existing artifact state for the critic."""
        lines: list[str] = [f"Worker summary: {output.summary or '(none)'}"]
        if self._worktree_path is not None:
            lines.extend(["", "Git status:", "```", self._git_output("status", "--short"), "```"])
            lines.extend(["", "Git diff:", "```", self._git_output("diff", "--", "."), "```"])
            for path_str in self._untracked_paths():
                content = self._read_untracked(path_str)
                if content is not None:
                    lines += [f"\nNew file: {path_str}", "```", content, "```"]
        if self._state_view.files:
            if lines:
                lines.append("")
            lines.append("Existing artifact files:")
            for fv in self._state_view.files:
                lines += [f"\nFile: {fv.path}", "```", fv.content, "```"]
        return "\n".join(lines)

    def has_worktree_changes(self) -> bool:
        """Return True when git evidence shows tracked or relevant untracked changes."""
        if self._worktree_path is None:
            return False
        status = self._git_output("status", "--short")
        diff = self._git_output("diff", "--", ".")
        return status != "(none)" or diff != "(none)" or bool(self._untracked_paths())

    def _git_output(self, *args: str) -> str:
        if self._worktree_path is None:
            return "(worktree unavailable)"
        result = run_git(
            [*args],
            cwd=self._worktree_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "(none)"

    def _untracked_paths(self) -> list[str]:
        """Return paths of untracked non-noise files via git ls-files."""
        if self._worktree_path is None:
            return []
        result = run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=self._worktree_path,
            capture_output=True,
            text=True,
        )
        paths: list[str] = []
        for raw in result.stdout.splitlines():
            path_str = raw.strip()
            if not path_str:
                continue
            p = Path(path_str)
            if (
                any(part in CRITIC_EVIDENCE_EXCLUDED_DIRS for part in p.parts)
                or p.name in EXCLUDED_FILE_NAMES
                or p.suffix in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
            ):
                continue
            paths.append(path_str)
        return paths

    def _read_untracked(self, path_str: str) -> str | None:
        """Return UTF-8 content of an untracked file if it is small and text-readable."""
        if self._worktree_path is None:
            return None
        full = self._worktree_path / path_str
        if not full.is_file() or full.stat().st_size > _MAX_UNTRACKED_BYTES:
            return None
        try:
            return full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None

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
            review_focus="whether the worktree changes satisfy the task",
            empty_output_guidance=(
                "If no worktree changes were made, reject unless the "
                "success condition is already demonstrably met."
            ),
        )

    def final_output_reminder(self) -> str:
        """Return a compact WorkOutput format reminder."""
        return "\n".join(
            [
                "FINAL OUTPUT FORMAT REMINDER",
                "Return valid JSON only matching WorkOutput.",
                "- summary must describe the worktree changes made.",
                "- Do not include full file contents in the final response.",
            ]
        )


_PlannerOutput = WorkDecision | GraphSplitDecision

_DECOMPOSITION_TOPOLOGY_RULES = """\
Decomposition topology rules (apply when reviewing split_graph or work decisions):

For each split_graph decision, validate:
- Are the depends_on edges justified by real artifact or information flow?
  A downstream node must genuinely consume output produced by an upstream node.
- For tasks that can proceed independently, depends_on must be empty.

Policy:
- Use split_graph for any multi-task decomposition — independent, dependent, or mixed topology.
- A genuine ordering constraint is required to justify each depends_on edge.
  Convention, symmetry, and aesthetic balance are NOT ordering constraints.

If a split_graph has depends_on edges with no genuine information flow justification, issue REVISE.\
"""


class PlannerOutputValidator:
    """OutputValidator for planner output — WorkDecision or GraphSplitDecision."""

    def extract_from_response(self, response: AgentResponse) -> _PlannerOutput | None:
        """Return typed planner output from the response."""
        if isinstance(response.output, (WorkDecision, GraphSplitDecision)):
            return response.output
        return None

    def is_empty(self, output: _PlannerOutput) -> bool:
        """Always returns False — planners never trigger ALREADY_DONE."""
        return False

    def render_for_critic(self, output: _PlannerOutput) -> str:
        """Render planner output as a human-readable string for the critic."""
        if isinstance(output, WorkDecision):
            return (
                f"Decision: work\n"
                f"Objective: {output.task.objective}\n"
                f"Success condition: {output.task.success_condition}\n"
                f"Artifact: {output.task.artifact}"
            )
        lines = ["Decision: split_graph (mixed topology)"]
        for node in output.nodes:
            dep_str = f" (depends_on: {', '.join(node.depends_on)})" if node.depends_on else ""
            lines.append(f"Node {node.id}{dep_str}: {node.task.objective}")
            lines.append(f"  Success condition: {node.task.success_condition}")
            if isinstance(node.task, TaskSpec) and node.task.artifact:
                lines.append(f"  Artifact: {node.task.artifact}")
        return "\n".join(lines)

    def work_noun(self) -> str:
        """Return 'plan'."""
        return "plan"

    def requires_nonempty(self) -> bool:
        """Return True — planners must always produce output."""
        return True

    def review_context(self) -> ReviewContext:
        """Return planner-output review language with decomposition topology enforcement."""
        return ReviewContext(
            output_noun="plan",
            review_focus="whether the decomposition decision satisfies the planning contract",
            empty_output_guidance="If the decision contains no tasks, reject it.",
            topology_rules=_DECOMPOSITION_TOPOLOGY_RULES,
        )

    def final_output_reminder(self) -> str:
        """Return a compact planner output-format reminder."""
        return "\n".join(
            [
                "FINAL OUTPUT FORMAT REMINDER",
                "Return one of these decision kinds:",
                '  {"kind":"work","task":{...WorkSpec...}}',
                '  {"kind":"split_graph","nodes":[{"id":"a","task":{...TaskSpec...},"depends_on":[]},{"id":"b","task":{...},"depends_on":["a"]}]}',
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


def _mentions_no_worktree_changes(*texts: str | None) -> bool:
    haystack = " ".join(text or "" for text in texts).lower()
    return any(phrase in haystack for phrase in _NO_WORKTREE_CHANGE_PHRASES)


def _no_worktree_change_revision(
    *,
    rationale: str,
    prior_attempts: int,
) -> RevisionRequest:
    return RevisionRequest(
        rationale=rationale,
        prior_attempts=prior_attempts,
        items=[
            RevisionItem(
                required_change=_NO_WORKTREE_CHANGE_REQUIRED_CHANGE,
                rationale=rationale,
            )
        ],
    )


def _claimed_work_has_no_worktree_changes(output: object, validator: object) -> bool:
    return (
        isinstance(output, WorkOutput)
        and bool(output.summary.strip())
        and isinstance(validator, WorkOutputValidator)
        and not validator.has_worktree_changes()
    )


class AttemptLifecycle[T]:
    """Owner of producer/critic/referee execution for work and plan agents."""

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
        self._initial_revision = initial_revision
        self._telemetry = AttemptTelemetryReporter(
            telemetry_sink, run_id, request.id, request.agent_type
        )

    async def run(self, prompt: str) -> AgentResponse:
        """Run attempts with validation/retry; return AgentResponse."""
        history = RevisionHistory(
            [self._initial_revision] if self._initial_revision is not None else []
        )

        for attempt in range(self._max_attempts):
            attempt_number = attempt + 1
            self._telemetry.attempt_started(attempt_number, self._max_attempts)
            current_prompt = (
                prompt
                if not history.requests
                else (
                    f"{prompt}\n\n"
                    f"{history.render(self._validator.work_noun(), self._validator.final_output_reminder())}"
                )
            )
            response = await self._run_fn(current_prompt)
            output = self._validator.extract_from_response(response)
            self._telemetry.producer_response_parsed(
                attempt_number, response, dispatch_sha=self._state_view.version_sha
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
                        revision_request = RevisionRequest(
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
                        history = history.append(revision_request)
                        self._telemetry.revision_appended(attempt_number, revision_request)
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
                self._telemetry.critic_finding_parsed(attempt_number, finding)
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
                history = history.append_from_review(
                    rationale=finding.rationale,
                    prior_attempts=attempt_number,
                    critic_finding=finding,
                )
                self._telemetry.revision_appended(attempt_number, history.requests[-1])
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
            self._telemetry.critic_finding_parsed(attempt_number, finding)
            self._telemetry.referee_decision_parsed(attempt_number, decision)

            if decision.disposition == CriticDisposition.ACCEPT:
                return response
            if decision.disposition == CriticDisposition.REJECT:
                if (
                    attempt < self._max_attempts - 1
                    and _claimed_work_has_no_worktree_changes(output, self._validator)
                    and _mentions_no_worktree_changes(finding.rationale, decision.rationale)
                ):
                    _logger.info(
                        "attempt %d/%d: no worktree changes despite WorkOutput — retrying",
                        attempt_number,
                        self._max_attempts,
                    )
                    revision_request = _no_worktree_change_revision(
                        rationale=decision.rationale or finding.rationale,
                        prior_attempts=attempt_number,
                    )
                    history = history.append(revision_request)
                    self._telemetry.revision_appended(attempt_number, revision_request)
                    continue
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
                self._telemetry.decompose_requested(attempt_number, decision)
                return AgentResponse(
                    request_id=self._request.id,
                    status=ResponseStatus.DECOMPOSE,
                )

            history = history.append_from_review(
                rationale=decision.rationale,
                prior_attempts=attempt_number,
                critic_finding=finding,
                referee_decision=decision,
            )
            self._telemetry.revision_appended(attempt_number, history.requests[-1])

        _logger.warning(
            "max_attempts (%d) exhausted; validation did not accept",
            self._max_attempts,
        )
        self._telemetry.exhausted(
            self._max_attempts,
            "maximum validation attempts exhausted without an accept disposition",
        )
        return _validation_rejected_response(
            self._request,
            CriticDisposition.REVISE,
            "maximum validation attempts exhausted without an accept disposition",
            self._validator.work_noun(),
        ).model_copy(
            update={
                "diagnostics": [
                    AgentDiagnostic(
                        kind=VALIDATION_EXHAUSTED_DIAGNOSTIC,
                        message="maximum validation attempts exhausted without an accept disposition",
                    )
                ]
            }
        )
