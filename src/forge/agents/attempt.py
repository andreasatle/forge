"""AttemptEngine — generic attempt/validation/retry loop for work and plan tasks."""

import logging
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.agents.base import _render_files, run_agent
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
    StateView,
    TaskSpec,
    WorkSpec,
)
from forge.llm.providers import LLMProvider
from forge.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class OutputValidator(Protocol[T]):
    def extract_from_response(self, response: AgentResponse) -> T | None: ...
    def is_empty(self, output: T) -> bool: ...
    def render_for_critic(self, output: T) -> str: ...
    def work_noun(self) -> str: ...
    def requires_nonempty(self) -> bool: ...


class DeltaStateValidator:
    def __init__(self, adapter_spec: AdapterSpec, state_view: StateView) -> None:
        self._adapter = adapter_spec
        self._state_view = state_view

    def extract_from_response(self, response: AgentResponse) -> DeltaState | None:
        return response.delta

    def is_empty(self, output: DeltaState) -> bool:
        return not output.new_files and not output.edits and not output.dependencies

    def render_for_critic(self, output: DeltaState) -> str:
        return _render_files(output, self._state_view)

    def work_noun(self) -> str:
        return self._adapter.work_noun

    def requires_nonempty(self) -> bool:
        return self._adapter.requires_nonempty_output


class PlanResponseValidator:
    def extract_from_response(self, response: AgentResponse) -> PlanResponse | None:
        if response.status != ResponseStatus.COMPLETED:
            return None
        tasks = [
            TaskSpec(
                objective=req.spec.objective,
                success_condition=req.spec.success_condition,
                adapter=req.spec.adapter,
                artifact=req.spec.artifact,
                language=req.spec.language,
            )
            for req in response.follow_up
            if isinstance(req.spec, WorkSpec)
        ]
        return PlanResponse(tasks=tasks)

    def is_empty(self, output: PlanResponse) -> bool:
        return False

    def render_for_critic(self, output: PlanResponse) -> str:
        if not output.tasks:
            return "(no tasks)"
        lines: list[str] = []
        for i, task in enumerate(output.tasks):
            lines.append(f"Task {i}: {task.objective}")
            lines.append(f"  Success condition: {task.success_condition}")
            if task.artifact:
                lines.append(f"  Artifact: {task.artifact}")
            if task.language:
                lines.append(f"  Language: {task.language}")
        return "\n".join(lines)

    def work_noun(self) -> str:
        return "plan"

    def requires_nonempty(self) -> bool:
        return True


class RunAgentFailed(Exception):
    """Raised when run_agent returns a non-COMPLETED response."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        super().__init__(response.error or "run_agent failed")


def _validation_rejected_response(
    request: AgentRequest,
    disposition: CriticDisposition,
    rationale: str,
) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error=(f"validation rejected work with disposition '{disposition.value}': {rationale}"),
        failure_kind=FailureKind.VALIDATION_REJECTED,
    )


def _validation_parse_failed_response(request: AgentRequest, error: ValueError) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error=f"validation response could not be parsed: {error}",
        failure_kind=FailureKind.INVALID_JSON,
    )


class AttemptEngine(Generic[T]):
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
                if self._critic_provider is None:
                    _logger.info(
                        "attempt %d/%d: empty output, no critic — ALREADY_DONE",
                        attempt + 1,
                        self._max_attempts,
                    )
                    return AgentResponse(
                        request_id=self._request.id,
                        status=ResponseStatus.ALREADY_DONE,
                        delta=response.delta,
                    )
                try:
                    finding = await critic_agent(
                        self._request,
                        self._state_view,
                        self._validator.render_for_critic(output),
                        self._critic_provider,
                        cast(AdapterRegistry, self._registry),
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
                        delta=response.delta,
                    )
                if finding.disposition == CriticDisposition.REJECT:
                    _logger.info(
                        "attempt %d/%d: critic rejected empty output",
                        attempt + 1,
                        self._max_attempts,
                    )
                    return _validation_rejected_response(
                        self._request, finding.disposition, finding.rationale
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
                )
                decision = await referee_agent(
                    self._request,
                    self._state_view,
                    output_text,
                    finding,
                    self._referee_provider,
                    cast(AdapterRegistry, self._registry),
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
                    self._request, decision.disposition, decision.rationale
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
        )


# Backwards-compatible alias kept so existing imports don't break immediately.
TaskAttemptEngine = AttemptEngine
