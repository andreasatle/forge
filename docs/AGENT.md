# Forge AI Bootstrap Brief

Read this before advising on or modifying Forge.

## Project Goal

Forge is a scheduler-centric agentic framework.

The core abstraction is:

Scheduler DAG node = bounded game.

Each node has:

1. AgentRequest
2. Producer
3. Critic
4. Referee
5. Typed output
6. Scheduler-owned consequence

The framework must preserve typed inputs and outputs across all boundaries.

## Core Invariants

* AgentRequest is law.
* Producer, critic, and referee must receive the same node contract.
* Critic/referee judge only against the AgentRequest contract, not against imagined ideal completeness.
* Producer output must remain typed through PWC.
* Planner output is `WorkDecision | GraphSplitDecision`, wrapped by `PlannerOutputModel` (the canonical final response type). The scheduler receives this via `AgentResponse.output` as a `PlannerOutputModel` whose `.root` is the actual decision.
* Worker output is WorkOutput.
* Scheduler derives work nodes from accepted planner output (a `PlannerOutputModel` containing a `WorkDecision` or `GraphSplitDecision`).
* Scheduler integrates accepted WorkOutput through StateService.apply_work_output.
* Integration is deterministic and framework-owned.
* Language-specific rules belong in language plugins, not Forge core.
* Adapter-specific behavior belongs behind adapters.
* Core scheduler/orchestration must remain project-type and language agnostic.

## Design Preferences

* Do not overfit to the current failing run.
* Do not patch symptoms before identifying the broken abstraction.
* Prefer correct architectural boundaries over smallest diff.
* Encapsulation is required.
* Prefer classes for major framework concepts.
* Avoid free-floating helper functions when behavior has an owner.
* Avoid `_private_helper` style as a substitute for real ownership.
* Use Pydantic request/response models at boundaries.
* Favor input -> typed output -> explicit side effect boundary.
* Side effects should happen only in named services.

## Anti-Patterns

Do not:

* Add planner-specific hacks.
* Add scraper-specific hacks.
* Add Python-specific behavior to Forge core.
* Let critic/referee invent requirements outside the contract.
* Let planner emit scheduler artifacts directly.
* Let workers mutate state directly.
* Hide scheduler consequences inside producer responses.
* Increase max_attempts just to make failures disappear.
* Disable critic/referee to make a run pass.
* Move code into classes if the class does not represent a real framework concept.
* Preserve a bad abstraction just because it is cheaper to patch.
* Use `if isinstance` fallthrough or bare `else` for enum dispatch — use `match` + `assert_never`.
* Compare enum values as strings — use direct enum comparisons.

## Taxonomy

| Category | Members | Directory |
|---|---|---|
| **role** | producer, critic, referee | `roles/` |
| **node type** | plan, work | (enum; no directory) |
| **adapter** | coding, document, audit | `adapters/` |
| **language** | python, rust, zig | `languages/` |
| **model profile** | fast, default, strong | (config; no directory) |

- **Roles** are filled by agents during PWC evaluation of a single DAG node. They are not node types.
- **Node types** (plan / work) are carried by `AgentType` — they describe *what a node produces*, not who evaluates it.
- **Adapters** are artifact-type plugins — they supply tool lists, output format rules, and prompts for a category of work (code, document, audit). Always language-agnostic.
- **Language plugins** own everything language-specific: package managers, test commands, project manifests.
- **Model profiles** are named capability tiers used by `ProfileAssigner`; they map to concrete LLM configurations at runtime.

`producer.yaml` is not yet implemented. `roles/` currently contains only `critic.yaml` and `referee.yaml`.

## Current Architecture Direction

Important concepts:

* Scheduler
* DAG node
* AgentRequest (carries `initial_revision` for scheduler-generated revision entry points)
* AgentContract
* Producer
* Critic
* Referee
* RevisionRequest
* AgentResponse.output
* WorkOutput
* StateService (raises `PostMergeTestFailure` on post-merge test failures)
* Language plugin
* Adapter registry
* TelemetryService / RunLedger (implemented — see `core/telemetry.py` and trace commands)
* ProfileAssigner / ComplexityProfileAssigner (complexity-to-profile routing for WORK nodes)
* TaskComplexityClassifier (Protocol; LLMTaskComplexityClassifier is the LLM-backed impl)
* ProfileEscalationPolicy / StaticProfileEscalationPolicy (retry on stronger profile after failure)
* `match` + `assert_never` — required for all enum and tagged-union dispatch

## Current Known Direction

Telemetry / RunLedger is implemented. Immutable run events are written under
`workspaces/<workspace>/telemetry/runs/<run_id>/` and surfaced via `uv run forge trace`.

Current areas of active development:

* Complexity routing — LLM-based task classification and profile mapping (wired; config-driven)
* Profile escalation — retry failed WORK nodes with a stronger model profile (implemented; not yet config-wired)
* Integration-test retries — scheduler-driven retry after post-merge test failures (implemented; not yet config-wired)
* Hierarchical planning — planning tasks that emit further planning tasks (not implemented)
* Versioned integration and conflict classification (partial)

## Required Development Discipline

Before implementing, answer:

1. What invariant is violated?
2. What abstraction should enforce it?
3. Is this a framework fix or a local patch?
4. What existing boundary owns this behavior?
5. What tests prove the invariant?
6. What tempting shortcut should be rejected?

Only then modify code.

## Validation Commands

Run:

```bash
uv run pytest
uv run pyright
uv run ruff check .
```

For runtime validation:

```bash
uv run forge reset forge.yaml && uv run forge start forge.yaml
```

## Current Runtime Goal

Get a complete Forge run where:

* planner integrates
* all work nodes integrate
* failed nodes have clear classified reasons
* no language-specific policy leaks into Forge core
* failures expose enough telemetry/diagnostics to repair
