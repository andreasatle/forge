# Proposed 8-task plan:
## Task 1 — Update NodeState and AgentType in models.py
Rename COMPLETED → INTEGRATED, add CANCELLED to NodeState. Remove INTEGRATE from AgentType. Remove IntegrateSpec and update AgentSpec union. This is pure model surgery — no logic changes yet. All tests that reference these will fail until the rest lands, so we fix references in the same prompt.
## Task 2 — Simplify integrate_agent signature
Change integrate_agent(request, workspace, language_registry, completed_deltas) to integrate_agent(delta, state_service). Single delta in, not a batch. Remove the internal merging logic that was handling multiple worker deltas. Update tests/agents/test_integrator.py accordingly.
## Task 3 — Update the scheduler
Three things in one file: (a) call state_service.apply_delta() inline after a work task completes instead of emitting an INTEGRATE node; (b) after any node goes to FAILED, mark all transitive dependents CANCELLED; (c) clean termination — stop when no PENDING or RUNNING nodes remain. Update tests/core/test_scheduler.py.
## Task 4 — Update the planner
Remove the logic that emits an INTEGRATE follow-up node after work nodes. The planner now only emits PLAN and WORK nodes. Update tests/agents/test_planner.py.
## Task 5 — Update the runner
Remove make_integrate_handler. Remove integrate registration from the runner assembly. Inject StateService into the scheduler instead. Update tests/core/test_runner.py.
## Task 6 — Update entry point
Update run.py (or wherever the runner/scheduler is assembled) to reflect the new wiring — no integrate handler, StateService passed to scheduler directly.
## Task 7 — Fix all remaining test breakage
Run the full test suite, collect failures, fix anything that slipped through the earlier tasks. Commit only when green.
## Task 8 — Commit checkpoint
git commit with a message summarising the refactor. Update memory/codebase index.


# AUDIT from chat after the 6 first tasks

**Architecture Audit**
- **Task Lifecycle: PARTIAL**. Runtime has `PENDING/RUNNING/INTEGRATED/FAILED/CANCELLED`, not explicit `EMITTED/DISPATCHED`; `READY` exists but is not stored as a transition. `INTEGRATED` is assigned from agent completion before integrator outcome is considered, so “only integrated tasks affect reality” is not fully true. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:38), [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:245), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:94).
- **Task Kinds: MOSTLY COMPLETE**. `PlanSpec` and `WorkSpec` are explicit, planner returns `PlanResponse`/`TaskSpec`, workers return `DeltaState`. Drift remains because the first-class unit is still `AgentRequest`, not `Task`, and planner output is immediately converted to work requests only. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:67), [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:148), [base.py](/Users/andreasatle/Projects/Forge/src/forge/agents/base.py:210).
- **Planner: PARTIAL**. Planner emits tasks and owns local `depends_on` semantics, and it does not mutate state. It does not observe a real `StateView`, only the northstar plus artifact metadata, and planner-source plan requests are explicitly no-ops. Planners are partially inside the DAG: the initial planner is a DAG node, but idle replanning is scheduler-injected and disabled by handler behavior. See [planner.py](/Users/andreasatle/Projects/Forge/src/forge/agents/planner.py:19), [runner.py](/Users/andreasatle/Projects/Forge/src/forge/core/runner.py:67), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:64).
- **Scheduler: MOSTLY COMPLETE**. It dispatches ready nodes, respects dependencies, cancels dependents on failure, and uses `asyncio.gather` up to `max_concurrency`. Missing: scheduler still understands work integration, ignores integration results, has unused priority, and has an odd idle planner injection path. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:274), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:75), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:98).
- **Integrator: PARTIAL**. Integrator applies deltas and runs tests, but failures are embedded in `delta.errors` while response status remains `COMPLETED`; scheduler ignores that returned response. There is no rollback/rejection path, and failed tests do not prevent the node being marked integrated. See [integrator.py](/Users/andreasatle/Projects/Forge/src/forge/agents/integrator.py:21), [integrator.py](/Users/andreasatle/Projects/Forge/src/forge/agents/integrator.py:37), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:104).
- **State Ownership: MOSTLY COMPLETE**. Active workers use `build_read_registry`, build a `StateView`, and are instructed to produce `DeltaState`; `StateService.apply_delta` is the main mutation boundary. Remaining drift: `build_write_registry`, write tools, blackboard tools, and `build_default_registry` still exist as stale surfaces, even if not active in workers. See [worker.py](/Users/andreasatle/Projects/Forge/src/forge/agents/worker.py:34), [worker.py](/Users/andreasatle/Projects/Forge/src/forge/agents/worker.py:68), [state_service.py](/Users/andreasatle/Projects/Forge/src/forge/core/state_service.py:97), [builtin.py](/Users/andreasatle/Projects/Forge/src/forge/tools/builtin.py:28).
- **Hierarchical Planning: PARTIAL**. A root planning task can emit additional work tasks, but emitted tasks cannot themselves be planning tasks, and tests assert planner follow-ups are only work nodes. Subplans cannot depend on other subplans because there is no planning-task `TaskSpec` representation. See [test_planner.py](/Users/andreasatle/Projects/Forge/tests/agents/test_planner.py:125), [test_base.py](/Users/andreasatle/Projects/Forge/tests/agents/test_base.py:729).
- **Concurrency Correctness: PARTIAL**. Basic concurrency and dependency cancellation exist. `StateView.state_version`, proposal/base versioning, stale delta detection, real conflict classification, and safe integration semantics are absent. Current conflict handling is limited to missing/non-unique edit strings during apply. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:182), [state_service.py](/Users/andreasatle/Projects/Forge/src/forge/core/state_service.py:106), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:119).
- **Dead Architecture: PARTIAL**. The codebase still carries old agent vocabulary (`AgentType`, `AgentRequest`, `AgentResponse`), `RequestSource.WORKER`, `Priority`, blackboard tools, write registries, and an “integrator agent” concept that is not a DAG task. These are stale relative to the task-first design. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:16), [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:23), [builtin.py](/Users/andreasatle/Projects/Forge/src/forge/tools/builtin.py:47).

**Top 5 Remaining Tasks**
1. Make integration outcome drive node state, including rejection/failure propagation.
2. Add state/proposal versioning to `StateView`, `DeltaState`, and integration.
3. Allow `TaskSpec` to represent planning tasks as well as work tasks.
4. Move planner-emitted tasks through integration, not direct `follow_up` insertion.
5. Remove or quarantine obsolete agent/write-registry/blackboard abstractions.

**Estimate**
- Current implementation is roughly **55% complete** toward `DESIGN-DOC.md`.
- Remaining work is **mostly new capability**, not just cleanup: hierarchical planning, versioned integration, stale delta detection, and integration-safe failure semantics are not implemented yet. Cleanup is still significant, but secondary.