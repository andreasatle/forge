"""Integrator agent — merges worker DeltaStates, applies to disk, runs tests."""

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    DeltaState,
    Edit,
    FileWrite,
    IntegrateSpec,
    IntegrationError,
    ResponseStatus,
)
from forge.core.state_service import StateService
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry


def _edits_conflict(a: Edit, b: Edit) -> bool:
    """True if two edits target overlapping regions of the same file."""
    if a.path != b.path:
        return False
    if a.old == b.old:
        return a.new != b.new
    return a.old in b.old or b.old in a.old


async def integrate_agent(
    request: AgentRequest,
    workspace: Workspace,
    language_registry: LanguageRegistry,
    completed_deltas: list[DeltaState],
) -> AgentResponse:
    """Merge worker DeltaStates, apply to disk via StateService, run tests, return AgentResponse."""
    spec = request.spec
    if not isinstance(spec, IntegrateSpec):
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta=DeltaState(errors=[IntegrationError(
                kind="invalid_spec",
                description=f"expected IntegrateSpec, got {type(spec).__name__}",
            )]),
        )

    plugin = language_registry.get(spec.language) if spec.language else None
    state_service = StateService(workspace, spec.artifact, plugin)
    errors: list[IntegrationError] = []

    # Merge new_files: last writer wins per path; report one conflict per path
    file_map: dict[str, FileWrite] = {}
    conflicted_file_paths: set[str] = set()
    for delta in completed_deltas:
        for fw in delta.new_files:
            if fw.path in file_map and file_map[fw.path].content != fw.content:
                if fw.path not in conflicted_file_paths:
                    errors.append(IntegrationError(
                        kind="conflict",
                        description=f"two workers wrote different content to {fw.path}",
                        path=fw.path,
                    ))
                conflicted_file_paths.add(fw.path)
            file_map[fw.path] = fw

    # Collect all edits flat, detect conflicts pairwise
    all_edits: list[Edit] = [e for d in completed_deltas for e in d.edits]
    conflicted_edit_indices: set[int] = set()
    reported_edit_conflict_paths: set[str] = set()
    for i in range(len(all_edits)):
        for j in range(i):
            if _edits_conflict(all_edits[j], all_edits[i]):
                conflicted_edit_indices.add(i)
                conflicted_edit_indices.add(j)
                if all_edits[i].path not in reported_edit_conflict_paths:
                    reported_edit_conflict_paths.add(all_edits[i].path)
                    errors.append(IntegrationError(
                        kind="conflict",
                        description=f"two workers edited overlapping regions in {all_edits[i].path}",
                        path=all_edits[i].path,
                    ))

    # Merge dependencies: union, deduplicated
    seen_deps: set[str] = set()
    merged_deps: list[str] = []
    for delta in completed_deltas:
        for dep in delta.dependencies:
            if dep not in seen_deps:
                seen_deps.add(dep)
                merged_deps.append(dep)

    # Build clean (conflict-free) delta, deduplicating edits by (path, old)
    clean_files = [fw for fw in file_map.values() if fw.path not in conflicted_file_paths]
    seen_edit_keys: set[tuple[str, str]] = set()
    clean_edits: list[Edit] = []
    for i, e in enumerate(all_edits):
        if i not in conflicted_edit_indices:
            key = (e.path, e.old)
            if key not in seen_edit_keys:
                seen_edit_keys.add(key)
                clean_edits.append(e)

    clean_delta = DeltaState(new_files=clean_files, edits=clean_edits, dependencies=merged_deps)

    # Apply non-conflicting changes to disk
    try:
        state_service.apply_delta(clean_delta)
    except Exception as e:
        errors.append(IntegrationError(kind="apply_failed", description=str(e)))
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            delta=DeltaState(
                new_files=clean_files,
                edits=clean_edits,
                dependencies=merged_deps,
                errors=errors,
            ),
        )

    # Run tests
    test_result = state_service.run_tests()
    if not test_result.passed:
        lines = [test_result.summary, *test_result.failures]
        description = "\n".join(line for line in lines if line)
        errors.append(IntegrationError(kind="test_failed", description=description))

    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta=DeltaState(
            new_files=clean_files,
            edits=clean_edits,
            dependencies=merged_deps,
            errors=errors,
        ),
    )
