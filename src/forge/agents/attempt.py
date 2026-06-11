"""AttemptEngine — generic attempt/validation/retry loop for work and plan tasks."""

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar, cast, runtime_checkable

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.base import render_files
from forge.agents.critic import critic_agent
from forge.agents.referee import referee_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    DeltaState,
    FailureKind,
    PlanResponse,
    ResponseStatus,
    ReviewContext,
    StateView,
)
from forge.llm.providers import LLMProvider

_logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class OutputValidator(Protocol[T]):
    """Protocol for validating and rendering agent output for the PWC loop."""

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


class DeltaStateValidator:
    """OutputValidator for DeltaState — validates file/edit output from work agents."""

    def __init__(self, adapter_spec: AdapterSpec, state_view: StateView) -> None:
        self._adapter = adapter_spec
        self._state_view = state_view

    def extract_from_response(self, response: AgentResponse) -> DeltaState | None:
        """Return typed DeltaState output from the response."""
        return response.output if isinstance(response.output, DeltaState) else None

    def is_empty(self, output: DeltaState) -> bool:
        """Return True when the delta has no files, edits, or dependencies."""
        return not output.new_files and not output.edits and not output.dependencies

    def render_for_critic(self, output: DeltaState) -> str:
        """Render delta files and existing artifact state for the critic."""
        return render_files(output, self._state_view)

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
            review_focus="whether the proposed file, edit, and dependency delta satisfies the task",
            empty_output_guidance=(
                "If no files, edits, or dependencies were produced, reject unless the "
                "success condition is already demonstrably met."
            ),
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


class AttemptEngine[T]:
    """Generic PWC (Plan-Work-Critique) retry loop for work and plan agents."""

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
    ) -> None:
        self._request = request
        self._state_view = state_view
        self._validator = validator
        self._run_fn = run_fn
        self._registry = registry
        self._critic_provider = critic_provider
        self._referee_provider = referee_provider
        self._max_attempts = max_attempts

    async def run(self, prompt: str) -> AgentResponse:
        """Run attempts with validation/retry; return AgentResponse."""
        feedback: str | None = None

        for attempt in range(self._max_attempts):
            current_prompt = prompt if feedback is None else f"{prompt}\n\n{feedback}"
            response = await self._run_fn(current_prompt)
            output = self._validator.extract_from_response(response)

            if (
                response.status == ResponseStatus.FAILED
                and response.failure_kind == FailureKind.VALIDATION_REJECTED
                and output is not None
                and self._validator.is_empty(output)
            ):
                if response.ran_tests_and_passed:
                    _logger.info(
                        "attempt %d/%d: empty output but ran_tests_and_passed — ALREADY_DONE",
                        attempt + 1,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.ALREADY_DONE,
                        output=response.output,
                        delta=response.delta,
                    )
                if self._critic_provider is None:
                    is_last = attempt == self._max_attempts - 1
                    if not is_last:
                        _logger.info(
                            "attempt %d/%d: empty output, no critic — injecting correction, retrying",
                            attempt + 1,
                            self._max_attempts,
                        )
                        feedback = (
                            f"Your previous attempt produced no {self._validator.work_noun()}. "
                            f"You must produce concrete output to satisfy the AgentRequest contract."
                        )
                        continue
                    if not self._validator.requires_nonempty():
                        _logger.info(
                            "attempt %d/%d: empty output, no critic, last attempt — ALREADY_DONE",
                            attempt + 1,
                            self._max_attempts,
                        )
                        return AgentResponse(
                            request_id=self._request.id,
                            status=ResponseStatus.ALREADY_DONE,
                            output=response.output,
                            delta=response.delta,
                        )
                    _logger.info(
                        "attempt %d/%d: empty output, no critic, last attempt, requires nonempty — FAILED",
                        attempt + 1,
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
                        attempt + 1,
                        self._max_attempts,
                        e,
                    )
                    return _validation_parse_failed_response(self._request, e)
                if finding.disposition == CriticDisposition.ALREADY_DONE:
                    _logger.info(
                        "attempt %d/%d: critic confirmed ALREADY_DONE",
                        attempt + 1,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.ALREADY_DONE,
                        output=response.output,
                        delta=response.delta,
                    )
                if finding.disposition == CriticDisposition.REJECT:
                    _logger.info(
                        "attempt %d/%d: critic rejected empty output",
                        attempt + 1,
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
                    attempt + 1,
                    self._max_attempts,
                    finding.disposition.value,
                )
                hints_text = (
                    "\n".join(f"{i + 1}. {h}" for i, h in enumerate(finding.hints))
                    if finding.hints
                    else "(none)"
                )
                feedback = (
                    f"Your previous attempt received feedback:\n"
                    f"Disposition: {finding.disposition.value}\n"
                    f"Rationale: {finding.rationale}\n"
                    f"Hints:\n{hints_text}\n\n"
                    f"Revise your {self._validator.work_noun()} addressing the feedback above."
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
                    attempt + 1,
                    self._max_attempts,
                    e,
                )
                return _validation_parse_failed_response(self._request, e)

            _logger.info(
                "attempt %d/%d: critic=%s referee=%s — %s",
                attempt + 1,
                self._max_attempts,
                finding.disposition.value,
                decision.disposition.value,
                "returning" if decision.disposition == CriticDisposition.ACCEPT else "retrying",
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

            hints_text = (
                "\n".join(f"{i + 1}. {h}" for i, h in enumerate(finding.hints))
                if finding.hints
                else "(none)"
            )
            feedback = (
                f"Your previous attempt received feedback:\n"
                f"Disposition: {decision.disposition.value}\n"
                f"Rationale: {decision.rationale}\n"
                f"Hints:\n{hints_text}\n\n"
                f"Revise your {self._validator.work_noun()} addressing the feedback above."
            )

        _logger.warning(
            "max_attempts (%d) exhausted; validation did not accept",
            self._max_attempts,
        )
        return _validation_rejected_response(
            self._request,
            CriticDisposition.REVISE,
            "maximum validation attempts exhausted without an accept disposition",
            self._validator.work_noun(),
        )


# Backwards-compatible alias kept so existing imports don't break immediately.
TaskAttemptEngine = AttemptEngine
