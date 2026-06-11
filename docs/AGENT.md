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
* Planner output is PlanResponse.
* Worker output is DeltaState.
* Scheduler derives work nodes from accepted PlanResponse.
* Scheduler integrates accepted DeltaState.
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

## Current Architecture Direction

Important concepts:

* Scheduler
* DAG node
* AgentRequest
* AgentContract
* Producer
* Critic
* Referee
* RevisionRequest
* AgentResponse.output
* PlanResponse
* DeltaState
* StateService
* Integrator
* Language plugin
* Adapter registry
* Future: TelemetryService / RunLedger

## Current Known Direction

Next major future feature:

Telemetry / RunLedger.

This should store immutable run events for replay and a future web UI:

* run started/completed
* node dispatched/completed/failed
* producer prompt/raw response/parsed output
* critic finding
* referee decision
* revision request
* integration result
* test output
* state version changes

This is not a blackboard. Agents should not use it as shared mutable memory.

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
