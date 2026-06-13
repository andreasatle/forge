# Proposed 8-task plan:
## Task 1 — Update NodeState and AgentType in models.py
Rename COMPLETED → INTEGRATED, add CANCELLED to NodeState. Remove INTEGRATE from AgentType. Remove IntegrateSpec and update AgentSpec union. This is pure model surgery — no logic changes yet. All tests that reference these will fail until the rest lands, so we fix references in the same prompt.
## Task 2 — Remove legacy integration agent
The old integrate agent path has been removed. Accepted worker output now reaches `StateService.apply_work_output` through the scheduler.
## Task 3 — Update the scheduler
Three things in one file: (a) call `StateService.apply_work_output()` inline after a work task completes instead of emitting an INTEGRATE node; (b) after any node goes to FAILED, mark all transitive dependents CANCELLED; (c) clean termination — stop when no PENDING or RUNNING nodes remain. Update tests/core/test_scheduler.py.
## Task 4 — Update the planner
Remove the logic that emits an INTEGRATE scheduler node after work nodes. The planner now emits typed `PlanResponse` tasks. Update tests/agents/test_planner.py.
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
- **Task Lifecycle: PARTIAL**. Runtime has `PENDING/RUNNING/INTEGRATED/FAILED/CANCELLED`, not explicit `EMITTED/DISPATCHED`; `READY` exists but is not stored as a transition. Accepted worker output is now integrated through `StateService.apply_work_output`; remaining lifecycle drift is that integration outcome handling is still coupled to scheduler state transitions. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:38), [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:245), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:94).
- **Task Kinds: MOSTLY COMPLETE**. `PlanSpec` and `WorkSpec` are explicit, planner returns `PlanResponse`/`TaskSpec`, active workers return `WorkOutput`. Drift remains because the first-class unit is still `AgentRequest`, not `Task`, and planner output is immediately converted to work requests only.
- **Planner: PARTIAL**. Planner emits tasks and owns local `depends_on` semantics, and it does not mutate state. It does not observe a real `StateView`, only the northstar plus artifact metadata, and planner-source plan requests are explicitly no-ops. Planners are partially inside the DAG: the initial planner is a DAG node, but idle replanning is scheduler-injected and disabled by handler behavior. See [planner.py](/Users/andreasatle/Projects/Forge/src/forge/agents/planner.py:19), [runner.py](/Users/andreasatle/Projects/Forge/src/forge/core/runner.py:67), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:64).
- **Scheduler: MOSTLY COMPLETE**. It dispatches ready nodes, derives work nodes from accepted `PlanResponse`, integrates accepted `WorkOutput` through `StateService.apply_work_output`, cancels dependents on failure, and uses `asyncio.gather` up to `max_concurrency`. Remaining drift: integration outcome handling is still coupled to scheduler state transitions, and the idle planner injection path is still odd. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:274), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:75), [scheduler.py](/Users/andreasatle/Projects/Forge/src/forge/core/scheduler.py:98).
- **Integration: MOSTLY COMPLETE**. Accepted `WorkOutput` now flows through `StateService.apply_work_output`, which uses git worktrees, commits, merge, tests, and rollback on failure.
- **State Ownership: MOSTLY COMPLETE**. Active workers use `build_read_registry`, build a `StateView`, and are instructed to produce `WorkOutput`; `StateService.apply_work_output` is the main mutation boundary. Remaining drift: retained mutating tool APIs and blackboard-related workspace cleanup still exist as stale surfaces, even if not active in workers.
- **Hierarchical Planning: PARTIAL**. A root planning task can emit additional work tasks, but emitted tasks cannot themselves be planning tasks. Subplans cannot depend on other subplans because there is no planning-task `TaskSpec` representation. See [test_planner.py](/Users/andreasatle/Projects/Forge/tests/agents/test_planner.py:125), [test_base.py](/Users/andreasatle/Projects/Forge/tests/agents/test_base.py:729).
- **Concurrency Correctness: PARTIAL**. Basic concurrency and dependency cancellation exist. `WorkOutput.base_version` is checked against the artifact HEAD SHA, but real conflict classification and richer integration failure semantics remain incomplete.
- **Dead Architecture: PARTIAL**. The codebase still carries agent vocabulary (`AgentType`, `AgentRequest`, `AgentResponse`) that is live but increasingly misleading relative to the task-first design. Retained mutating tool APIs and blackboard-related workspace cleanup are the clearest stale surfaces. See [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:16), [models.py](/Users/andreasatle/Projects/Forge/src/forge/core/models.py:23), [builtin.py](/Users/andreasatle/Projects/Forge/src/forge/tools/builtin.py:47).

**Top 5 Remaining Tasks**
1. Make integration outcome drive node state, including rejection/failure propagation.
2. Improve state/proposal versioning and conflict classification around `WorkOutput` integration.
3. Allow `TaskSpec` to represent planning tasks as well as work tasks.
4. Keep scheduler-derived DAG expansion out of `AgentResponse`.
5. Remove or quarantine obsolete agent/write-registry/blackboard abstractions.

**Estimate**
- Current implementation is roughly **55% complete** toward `DESIGN-DOC.md`.
- Remaining work is **mostly new capability**, not just cleanup: hierarchical planning, versioned integration, stale work output detection, and integration-safe failure semantics are not implemented yet. Cleanup is still significant, but secondary.
