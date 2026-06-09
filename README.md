# Forge

A scheduler-centric agentic framework. The scheduler owns the DAG; agents are opaque functions `AgentRequest → AgentResponse`.

## Status

The core loop runs end-to-end through provider-neutral chat adapters:

- **Core models** (`forge/core/models.py`) — `AgentRequest`, `AgentResponse`, `DAGNode`, `SchedulerState`, specs, `StateView`, and `DeltaState`. Pydantic models define the runtime contracts and LLM output schemas.
- **Scheduler** (`forge/core/scheduler.py`) — async DAG-driven dispatch loop. Owns node lifecycle, dependencies, failure propagation, idle planner reinjection, and termination.
- **Runner** (`forge/core/runner.py`) — routes `AgentRequest`s to registered handlers by `AgentType`.
- **Planner agent** (`forge/agents/planner.py`) — decomposes the northstar goal into `WorkSpec` follow-ups. Final response schema instructions are owned by `run_agent()`.
- **Worker agent** (`forge/agents/worker.py`) — read-only proposal generator. Workers inspect artifact state and return structured `DeltaState` proposals; they do not directly mutate artifacts or hidden shared state.
- **Integrator** (`forge/agents/integrator.py`) — merges worker proposals, applies non-conflicting changes, and runs tests.
- **StateService** (`forge/core/state_service.py`) — artifact mutation boundary. Builds `StateView`, applies `DeltaState`, and runs language test commands.
- **ToolRegistry** (`forge/tools/registry.py`) — source of truth for LLM-facing tools. Tool instructions are generated from the actual registry passed to `run_agent()`.
- **Providers** (`forge/llm/providers.py`) — transport/API adapters for Ollama, Claude, and OpenAI. Providers do not inject prompt semantics.
- **Workspace** (`forge/core/workspace.py`) — root directory for run state, artifacts, blackboard storage, and logs.
- **Persistence** (`forge/core/persistence.py`) — saves and loads `SchedulerState` to/from `workspace/state.json`. Running nodes are reset to pending on resume.
- **Config** (`forge/core/config.py`) — YAML-driven `ForgeConfig` with `northstar`, `workspace`, models, concurrency, and retry/token limits.
- **Entry point** (`forge/run.py`) — `start` and `reset` subcommands driven by a config file.

## Running

```bash
uv sync
```

Create a config file (or copy `forge.yaml`):

```yaml
northstar: "build a web scraper in Python"
workspace: ./workspaces/scraper
concurrency: 1
verbose: true
```

Start a run (resumes automatically if `state.json` exists in the workspace):

```bash
uv run forge start forge.yaml
```

Reset the workspace state and start fresh next time:

```bash
uv run forge reset forge.yaml
uv run forge start forge.yaml
```

Example output:

```
adapters: ['audit', 'coding', 'document']
→ dispatched: plan (...)
✓ completed: plan (...)
→ dispatched: work (...)
✓ completed: work (...)
~ idle: 6 nodes in DAG
→ dispatched: plan (...)
✓ completed: plan (...)
run saved: ./workspaces/scraper/state.json

Done — completed: 3, failed: 0, cancelled: 4

DAG summary:
  ...  plan        completed   deps=0  delta=None
  ...  work        completed   deps=0  delta=None
  ...  plan        completed   deps=0  delta=None
  ...  work        cancelled   deps=1  delta=None
```

## Tests

```bash
uv run pytest
```

The test suite covers scheduler flow, runner routing, agent behavior, provider payloads, prompt/schema ownership, tool registries, workspace lifecycle, persistence, and config loading. No live model calls required.

## Architecture

```
forge.yaml (ForgeConfig)
  ↓
Workspace.init()  ←→  state.json (save/load/resume)
  ↓
SchedulerState (DAG + northstar + concurrency)
  ↓
Scheduler emits READY nodes to Runner
  ↓
Runner routes by AgentType → Handler
  │
  ├── plan      → PlannerAgent → run_agent() → provider.chat() → WorkSpec follow-ups
  ├── work      → WorkAgent    → StateView + read-only tools → DeltaState proposal
  └── integrate → Integrator   → merge/apply proposals → StateService
  ↓
Scheduler records AgentResponse, grows DAG, re-checks for idle
  ↓
Global planner re-injected on idle → terminates when it returns empty follow_up
```

The scheduler has no knowledge of prompt construction, provider details, tools, or artifact mutation. Handlers have no knowledge of DAG scheduling beyond the request they receive.

## Ownership Boundaries

- **SchedulerState** owns DAG nodes, dependencies, lifecycle state, northstar, and concurrency.
- **StateView** is a projected read-only view of current artifact state for agents.
- **DeltaState** is a proposed state change, not direct mutation.
- **Integrator** owns merge/apply decisions for completed worker proposals.
- **StateService** is the only artifact mutation boundary.
- **ToolRegistry** is the source of truth for tools exposed to an LLM call.
- **Pydantic models** are the source of truth for final output schemas shown to models.
- **Providers** translate messages into backend-specific API payloads only.

## Worker Invariant

Workers are read-only proposal generators:

- They may inspect files, projected state, blackboard values, and test status when the corresponding read-only tools are available.
- They return structured `DeltaState` proposals.
- They do not directly mutate artifacts.
- They do not write blackboard/shared hidden state; `write_blackboard` is not exposed in the read-only worker registry.

All artifact writes flow through `DeltaState` proposals and the integrator/state-service path.

## Prompt Ownership

`run_agent()` owns LLM prompt semantics:

- Tool instructions are generated from the actual `ToolRegistry`.
- Final response schema instructions are generated from Pydantic response models.
- Providers must not append JSON-only text, schemas, role guidance, or other semantic instructions.
- Provider differences are limited to transport details such as payload shape, system-message placement, response-format arguments, token settings, and retry behavior.

## Concurrency

The scheduler can dispatch independent ready DAG nodes concurrently up to `max_concurrency`. Integration is the current safety boundary for applying worker proposals.

Known limitation: full async conflict safety is not complete yet. State versioning, stale-delta rejection, and read/write conflict detection across separately integrated concurrent workers are planned work, not current behavior.
