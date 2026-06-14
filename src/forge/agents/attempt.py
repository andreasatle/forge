"""AttemptEngine and AttemptLifecycle for work and plan PWC execution."""

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
from forge.core.models import (
    VALIDATION_EXHAUSTED_DIAGNOSTIC,
    AgentDiagnostic,
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    FailureKind,
    PlanResponse,
    ResponseStatus,
    ReviewContext,
    RevisionItem,
    RevisionRequest,
    StateView,
    WorkOutput,
)
from forge.core.telemetry import TelemetrySink
from forge.core.workspace import run_git
from forge.llm.providers import LLMProvider

_logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_UNTRACKED_BYTES = 64 * 1024
_UNTRACKED_NOISE_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".venv"}
)
_UNTRACKED_NOISE_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd", ".lock"})
_UNTRACKED_NOISE_NAMES = frozenset({"CACHEDIR.TAG", "pyvenv.cfg"})


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
                any(part in _UNTRACKED_NOISE_DIRS for part in p.parts)
                or p.name in _UNTRACKED_NOISE_NAMES
                or p.suffix in _UNTRACKED_NOISE_SUFFIXES
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
            self._telemetry.producer_response_parsed(attempt_number, response)

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


class AttemptEngine[T]:
    """Compatibility entry point that delegates PWC execution to AttemptLifecycle."""

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
        self._lifecycle = AttemptLifecycle(
            request=request,
            state_view=state_view,
            validator=validator,
            run_fn=run_fn,
            registry=registry,
            critic_provider=critic_provider,
            referee_provider=referee_provider,
            max_attempts=max_attempts,
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            initial_revision=initial_revision,
        )

    async def run(self, prompt: str) -> AgentResponse:
        """Run attempts with validation/retry; return AgentResponse."""
        return await self._lifecycle.run(prompt)
