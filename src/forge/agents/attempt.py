"""TaskAttemptEngine — owns the attempt/validation/retry loop for worker tasks."""

import logging

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import run_agent
from forge.agents.critic import critic_agent
from forge.agents.referee import referee_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    CriticDisposition,
    DeltaState,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.llm.providers import LLMProvider
from forge.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)


class RunAgentFailed(Exception):
    """Raised when run_agent returns a non-COMPLETED response."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        super().__init__(response.error or "run_agent failed")


class TaskAttemptEngine:
    def __init__(
        self,
        request: AgentRequest,
        state_view: StateView,
        provider: LLMProvider,
        registry: AdapterRegistry,
        tools: ToolRegistry,
        critic_provider: LLMProvider | None = None,
        referee_provider: LLMProvider | None = None,
        max_attempts: int = 3,
        max_retries: int = 3,
        max_tool_iterations: int = 25,
    ) -> None:
        self._request = request
        self._state_view = state_view
        self._provider = provider
        self._registry = registry
        self._tools = tools
        self._critic_provider = critic_provider
        self._referee_provider = referee_provider
        self._max_attempts = max_attempts
        self._max_retries = max_retries
        self._max_tool_iterations = max_tool_iterations

    async def run(self, prompt: str) -> DeltaState:
        """Run attempts with validation/retry; return best DeltaState.

        Raises RunAgentFailed if run_agent returns a non-COMPLETED response.
        """
        last_delta: DeltaState | None = None
        feedback: str | None = None

        for attempt in range(self._max_attempts):
            current_prompt = prompt if feedback is None else f"{prompt}\n\n{feedback}"
            response = await run_agent(
                self._request,
                WorkSpec,
                self._provider,
                current_prompt,
                tools=self._tools,
                max_retries=self._max_retries,
                max_tool_iterations=self._max_tool_iterations,
            )

            if response.status != ResponseStatus.COMPLETED or response.delta is None:
                raise RunAgentFailed(response)

            last_delta = response.delta

            if self._critic_provider is None or self._referee_provider is None:
                return last_delta

            try:
                finding = await critic_agent(
                    self._request,
                    self._state_view,
                    last_delta,
                    self._critic_provider,
                    self._registry,
                )
                decision = await referee_agent(
                    self._request,
                    self._state_view,
                    last_delta,
                    finding,
                    self._referee_provider,
                    self._registry,
                )
            except ValueError as e:
                _logger.warning(
                    "attempt %d/%d: validation parsing failed: %s — returning last delta",
                    attempt + 1,
                    self._max_attempts,
                    e,
                )
                return last_delta

            _logger.info(
                "attempt %d/%d: critic=%s referee=%s — %s",
                attempt + 1,
                self._max_attempts,
                finding.disposition.value,
                decision.disposition.value,
                "returning" if decision.disposition == CriticDisposition.ACCEPT else "retrying",
            )

            if decision.disposition == CriticDisposition.ACCEPT:
                return last_delta

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
                f"Revise your implementation addressing the feedback above."
            )

        _logger.warning("max_attempts (%d) exhausted; returning last delta", self._max_attempts)
        assert last_delta is not None, "max_attempts must be >= 1"
        return last_delta
