"""Planning agent that decomposes a northstar goal into concrete work tasks."""

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import AttemptLifecycle, PlannerOutputValidator, RunAgentFailed
from forge.agents.base import run_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    PlannerOutputModel,
    PlanSpec,
    ResponseStatus,
    StateView,
    render_agent_contract,
)
from forge.core.telemetry import TelemetrySink
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
You are a planning agent. Given a goal, choose a decomposition decision.

Available artifacts:
{artifact_details}

Each coding task must declare the correct language for its artifact.

Rules:
- EVERY task MUST include the "artifact" field — omitting it is an error
- artifact must be one of: {artifact_names}
- adapter must be one of: coding, document, audit
- No more than 5 tasks per decision
- Success conditions must describe observable outcomes
  (tests pass, output matches, endpoint returns X)
- Every coding task success condition must be verifiable by running tests
  — phrase it as an observable test outcome, not as a description of the implementation
- Task objectives, success conditions, acceptance criteria, constraints, and non-goals
  must not contradict artifact-specific language guidance

Decomposition decision kinds (choose one):

  kind="work" — task is small enough to execute directly.
    Use when the goal fits in one focused implementation session.
    {{"kind":"work","task":{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}}}}

  kind="split_graph" — explicit DAG with per-node depends_on. PREFERRED for mixed topologies.
    Use when some tasks depend on others but not all tasks need to be ordered.
    Assign each node a short string id. List ids in depends_on that must complete first.
    {{"kind":"split_graph","nodes":[
      {{"id":"setup","task":{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},"depends_on":[]}},
      {{"id":"readme","task":{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},"depends_on":[]}},
      {{"id":"scraper","task":{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},"depends_on":["setup"]}},
      {{"id":"cli","task":{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},"depends_on":["scraper"]}}
    ]}}

  kind="split_orthogonal" — children are independent; Forge creates no sibling dependencies.
    Use when ALL tasks can run in parallel with no ordering requirement.
    {{"kind":"split_orthogonal","tasks":[{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},...] }}

  kind="split_dependent" — children must run in order; Forge creates a strict chain A→B→C→D.
    Use ONLY when EVERY task in the list must wait for the previous one to complete.
    {{"kind":"split_dependent","tasks":[{{"objective":"...","success_condition":"...","adapter":"...","artifact":"..."}},...] }}

Legacy (still accepted):
  kind="plan" — list tasks with explicit depends_on indices.
    {{"kind":"plan","tasks":[{{"objective":"...","depends_on":[0],...}},...] }}

Decomposition policy:
- Prefer split_graph for mixed topologies — when some tasks depend on others but not all.
- Use split_orthogonal when ALL child tasks can proceed independently.
- Use split_dependent ONLY when every task must wait for the previous (strict chain).
  split_dependent requires a genuine ordering constraint — not caution, symmetry, or convention.
  split_dependent is rarely correct — it forces unnecessary sequencing between unrelated tasks.
- Never create dependencies for symmetry, aesthetic balance, or convention.
- Only add a depends_on edge when one task genuinely produces output that another must consume.
- Balanced and symmetric trees are not a goal. An uneven tree that
  exposes more parallel work is better than a balanced tree that forces
  unnecessary sequencing.
- Maximize safe concurrency. When in doubt, remove edges.

Good decomposition:
  Goal: "build scraper with CLI"
  → split_graph:
      setup (no deps)
      readme (no deps)
      scraper (depends_on: setup)
      cli (depends_on: scraper)
  This exposes maximum concurrency: setup, readme run immediately;
  scraper starts after setup; cli starts after scraper.

  Goal: "all tasks independent"
  → split_orthogonal: readme, tests, config   (all independent)

Bad decomposition:
  Goal: "build scraper with CLI"
  → split_dependent: setup → readme → scraper → cli
  (readme does not depend on setup — this forces unnecessary sequencing)

  Goal: "build scraper with CLI"
  → split_orthogonal: setup, readme, scraper, cli
  (cli requires scraper output — missing a required dependency)
{decomposition_context}
Goal: {northstar}

{contract_block}

Produce output satisfying this contract.
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
        artifact_types: dict[str, str] | None = None,
        artifact_descriptions: dict[str, str] | None = None,
        artifact_language_guidance: dict[str, str] | None = None,
        max_retries: int = 3,
        critic_provider: LLMProvider | None = None,
        referee_provider: LLMProvider | None = None,
        registry: AdapterRegistry | None = None,
        max_attempts: int = 3,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self.provider = provider
        self.artifact_names = artifact_names
        self.artifact_languages = artifact_languages
        self.artifact_types = artifact_types or {}
        self.artifact_descriptions = artifact_descriptions or {}
        self.artifact_language_guidance = artifact_language_guidance or {}
        self.max_retries = max_retries
        self.critic_provider = critic_provider
        self.referee_provider = referee_provider
        self.registry = registry
        self.max_attempts = max_attempts
        self.telemetry_sink = telemetry_sink

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Send the northstar goal to the planner LLM and return a PlanResponse."""
        spec = request.spec
        if not isinstance(spec, PlanSpec):
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                error=f"expected PlanSpec, got {type(spec).__name__}",
            )

        artifact_details = self._render_artifact_details()
        decomposition_context = ""
        if spec.contract.constraints or spec.contract.non_goals:
            decomposition_context = (
                "\nThis task was too broad for a single implementation.\n"
                "Decompose it into focused, non-overlapping subtasks where\n"
                "each subtask has exactly one concern."
            )
        prompt = PLAN_PROMPT.format(
            northstar=spec.northstar,
            artifact_names=", ".join(self.artifact_names),
            artifact_details=artifact_details,
            contract_block=render_agent_contract(request),
            decomposition_context=decomposition_context,
        )

        def correction_fn(error: Exception, bad_response: str) -> str:
            return CORRECTION_PROMPT.format(
                original_prompt=prompt,
                bad_response=bad_response,
                error=error,
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
                final_response_type=PlannerOutputModel,
                max_retries=max_retries,
            )

        lifecycle = AttemptLifecycle(
            request=request,
            state_view=_DUMMY_STATE_VIEW,
            validator=PlannerOutputValidator(),
            run_fn=_run_fn,
            registry=self.registry,
            critic_provider=self.critic_provider,
            referee_provider=self.referee_provider,
            max_attempts=self.max_attempts,
            telemetry_sink=self.telemetry_sink,
            run_id=getattr(self.telemetry_sink, "run_id", None),
        )

        try:
            return await lifecycle.run(prompt)
        except RunAgentFailed as e:
            return e.response

    def _render_artifact_details(self) -> str:
        blocks: list[str] = []
        for name in self.artifact_names:
            lines = [f"  {name}:"]
            artifact_type = self.artifact_types.get(name)
            if artifact_type:
                lines.append(f"    type: {artifact_type}")
            language = self.artifact_languages.get(name)
            if language:
                lines.append(f"    language: {language}")
            description = self.artifact_descriptions.get(name)
            if description:
                lines.append(f"    description: {description}")
            guidance = self.artifact_language_guidance.get(name)
            if guidance:
                lines.append("    language guidance:")
                lines.extend(f"      {line}" for line in guidance.splitlines() if line.strip())
            blocks.append("\n".join(lines))
        return "\n".join(blocks)


async def plan_agent(
    request: AgentRequest,
    artifact_names: list[str],
    artifact_languages: dict[str, str],
    provider: LLMProvider,
    max_retries: int = 3,
    critic_provider: LLMProvider | None = None,
    referee_provider: LLMProvider | None = None,
    registry: AdapterRegistry | None = None,
    artifact_types: dict[str, str] | None = None,
    artifact_descriptions: dict[str, str] | None = None,
    artifact_language_guidance: dict[str, str] | None = None,
    telemetry_sink: TelemetrySink | None = None,
    max_attempts: int = 3,
) -> AgentResponse:
    """Send the northstar goal to the planner LLM and return a PlanResponse."""
    return await PlannerTaskExecutor(
        provider=provider,
        artifact_names=artifact_names,
        artifact_languages=artifact_languages,
        artifact_types=artifact_types,
        artifact_descriptions=artifact_descriptions,
        artifact_language_guidance=artifact_language_guidance,
        max_retries=max_retries,
        max_attempts=max_attempts,
        critic_provider=critic_provider,
        referee_provider=referee_provider,
        registry=registry,
        telemetry_sink=telemetry_sink,
    ).run(request)
