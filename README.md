# Forge

A scheduler-centric agentic framework. Early-stage — the core loop runs, real agent logic does not exist yet.

## Status

The following pieces are in place:

- **Core data structures** (`forge/core/models.py`) — `AgentRequest`, `AgentResponse`, `DAGNode`, `SchedulerState`, and the spec types (`PlanSpec`, `WorkSpec`, `IntegrateSpec`). All are frozen Pydantic models.
- **Scheduler** (`forge/core/scheduler.py`) — async DAG-driven dispatch loop. Respects `max_concurrency`, propagates failures to dependents, re-injects the global planner on idle, and terminates when the planner returns an empty follow-up list.
- **Runner** (`forge/core/runner.py`) — routes `AgentRequest`s to registered handlers by `AgentType`. Includes stub handlers and a `scripted_plan_handler` that emits a three-node dependency chain (A → B → C) for end-to-end testing.
- **Entry point** (`forge/run.py`) — wires everything together and prints a live event log plus a DAG summary. Run with `uv run forge`.

Nothing talks to a real LLM yet. There are no adapters, no state service, no prompt layer, and no parsers. Those are the next layers to build.

## Running

```bash
uv sync
uv run forge
```

Example output:

```
→ dispatched: plan (...)
✓ completed: plan (...)
→ dispatched: work (...)
✓ completed: work (...)
→ dispatched: work (...)
✓ completed: work (...)
→ dispatched: work (...)
✓ completed: work (...)
~ idle: 4 nodes in DAG
→ dispatched: plan (...)
✓ completed: plan (...)

Done — completed: 5, failed: 0, cancelled: 0

DAG summary:
  ...  plan        completed   deps=0  delta=None
  ...  work        completed   deps=0  delta={'result': 'stub result for adapter: coding'}
  ...  work        completed   deps=1  delta={'result': 'stub result for adapter: coding'}
  ...  work        completed   deps=1  delta={'result': 'stub result for adapter: coding'}
  ...  plan        completed   deps=0  delta=None
```

## Tests

```bash
uv run pytest
```

36 tests covering the scheduler loop, runner routing, and handler behaviour. No live model calls required.

## Architecture

```
SchedulerState (DAG + config)
  ↓
Scheduler emits READY nodes to Runner
  ↓
Runner routes by AgentType → Handler
  ↓
Handler returns AgentResponse (with optional follow_up nodes)
  ↓
Scheduler applies result, grows DAG, re-checks for idle
  ↓
Global planner re-injected on idle → terminates when it returns empty follow_up
```

The scheduler has no knowledge of what handlers do. Handlers have no knowledge of the DAG.
