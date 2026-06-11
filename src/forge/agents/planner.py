"""Planning agent that decomposes a northstar goal into concrete work tasks."""

from collections.abc import Callable
from typing import cast

from pydantic import BaseModel

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import AttemptEngine, PlanResponseValidator, RunAgentFailed
from forge.agents.base import run_agent
from forge.agents.plan_follow_up import PlanFollowUpBuilder
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    PlanResponse,
    PlanSpec,
    ResponseStatus,
    StateView,
    WorkSpec,
)
from forge.llm.providers import LLMProvider

CORRECTION_PROMPT = """
Your previous response could not be parsed. Error: {error}

Your previous response was:
{bad_response}

Original instructions:
{original_prompt}

Fix the error and return corrected JSON only — no explanation, no markdown.
"""

PLAN_PROMPT = """
You are a planning agent. Given a goal, decompose it into at most 5 concrete tasks.

Available artifacts and their languages:
{artifact_language_list}

Each coding task must declare the correct language for its artifact.

Rules:
- EVERY task MUST include the "artifact" field — omitting it is an error
- artifact must be one of: {artifact_names}
- depends_on contains indices (0-based) of tasks this task depends on
- adapter must be one of: coding, document, audit
- No more than 5 tasks
- Success conditions must describe observable outcomes
  (tests pass, output matches, endpoint returns X)
- Every coding task success condition must be verifiable by running tests
  — phrase it as an observable test outcome, not as a description of the implementation

Goal: {northstar}
"""

_DUMMY_STATE_VIEW = StateView(artifact_name="", language=None, files=[], dependencies=[])


class PlannerTaskExecutor:
    """Own planner prompt construction and PlanResponse execution."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        artifact_names: list[str],
        artifact_languages: dict[str, str],
        max_retries: int = 3,
        critic_provider: LLMProvider | None = None,
        referee_provider: LLMProvider | None = None,
        registry: AdapterRegistry | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.provider = provider
        self.artifact_names = artifact_names
        self.artifact_languages = artifact_languages
        self.max_retries = max_retries
        self.critic_provider = critic_provider
        self.referee_provider = referee_provider
        self.registry = registry
        self.max_attempts = max_attempts

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Send the northstar goal to the planner LLM and return follow-up work requests."""
        spec = request.spec
        if not isinstance(spec, PlanSpec):
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"expected PlanSpec, got {type(spec).__name__}",
            )

        artifact_language_list = (
            "\n".join(f"  {name}: {lang}" for name, lang in self.artifact_languages.items())
            or "  (no languages declared)"
        )
        prompt = PLAN_PROMPT.format(
            northstar=spec.northstar,
            artifact_names=", ".join(self.artifact_names),
            artifact_language_list=artifact_language_list,
        )

        def correction_fn(error: Exception, bad_response: str) -> str:
            return CORRECTION_PROMPT.format(
                original_prompt=prompt,
                bad_response=bad_response,
                error=error,
            )

        follow_up_builder = cast(
            Callable[[BaseModel], list[AgentRequest]],
            PlanFollowUpBuilder(request).build,
        )
        provider = self.provider
        max_retries = self.max_retries

        async def _run_fn(current_prompt: str) -> AgentResponse:
            return await run_agent(
                request,
                PlanSpec,
                provider,
                current_prompt,
                correction_prompt_fn=correction_fn,
                final_response_type=PlanResponse,
                follow_up_builder=follow_up_builder,
                max_retries=max_retries,
            )

        engine = AttemptEngine(
            request=request,
            state_view=_DUMMY_STATE_VIEW,
            validator=PlanResponseValidator(),
            run_fn=_run_fn,
            registry=self.registry,
            critic_provider=self.critic_provider,
            referee_provider=self.referee_provider,
            max_attempts=self.max_attempts,
        )

        try:
            return await engine.run(prompt)
        except RunAgentFailed as e:
            return e.response


async def plan_agent(
    request: AgentRequest,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
    provider: LLMProvider,
    max_retries: int = 3,
) -> AgentResponse:
    """Send the northstar goal to the planner LLM and return follow-up work requests."""
    return await PlannerTaskExecutor(
        provider=provider,
        artifact_names=artifact_names,
        artifact_languages=artifact_languages,
        max_retries=max_retries,
    ).run(request)
