"""Tests for the integrator agent — merge, conflict detection, apply, and test reporting."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from forge.agents.integrator import integrate_agent
from forge.core.models import (
    AgentRequest,
    AgentType,
    DeltaState,
    Edit,
    FileWrite,
    IntegrateSpec,
    RequestSource,
    ResponseStatus,
    RunResult,
)
from forge.core.state_service import StateService
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry


def _integrate_request(artifact: str = "codebase") -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.WORKER,
        spec=IntegrateSpec(objective="integrate workers", artifact=artifact, work_request_id=uuid4()),
    )


def _mock_ss(passed: bool = True, failures: list[str] | None = None, summary: str = "") -> MagicMock:
    ss = MagicMock(spec=StateService)
    ss.run_tests.return_value = RunResult(passed=passed, failures=failures or [], summary=summary)
    return ss


def _patch_ss(mock: MagicMock):
    return patch("forge.agents.integrator.StateService", return_value=mock)


# --- merge ---


async def test_merges_two_non_conflicting_worker_deltas(tmp_path):
    """Two deltas with different paths produce a combined delta with no errors."""
    ss = _mock_ss()
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]),
                DeltaState(new_files=[FileWrite(path="b.py", content="y = 2")]),
            ],
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert response.delta.errors == []
    assert {fw.path for fw in response.delta.new_files} == {"a.py", "b.py"}


# --- conflict detection: new_files ---


async def test_detects_conflict_when_two_workers_write_different_content_to_same_path(tmp_path):
    """Two workers writing different content to the same path → one conflict error."""
    ss = _mock_ss()
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(new_files=[FileWrite(path="a.py", content="version 1")]),
                DeltaState(new_files=[FileWrite(path="a.py", content="version 2")]),
            ],
        )

    assert response.delta is not None
    conflicts = [e for e in response.delta.errors if e.kind == "conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].path == "a.py"


async def test_no_conflict_when_two_workers_write_identical_content_to_same_path(tmp_path):
    """Two workers writing the same content to the same path is not a conflict."""
    ss = _mock_ss()
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]),
                DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]),
            ],
        )

    assert response.delta is not None
    assert response.delta.errors == []


# --- conflict detection: edits ---


async def test_detects_conflict_when_two_workers_edit_overlapping_regions(tmp_path):
    """Two workers editing overlapping old strings in the same file → one conflict error."""
    ss = _mock_ss()
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(edits=[Edit(path="main.py", old="def foo():\n    return 1", new="def foo():\n    return 2")]),
                DeltaState(edits=[Edit(path="main.py", old="def foo():", new="def bar():")]),
            ],
        )

    assert response.delta is not None
    conflicts = [e for e in response.delta.errors if e.kind == "conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].path == "main.py"


# --- apply ---


async def test_applies_non_conflicting_changes_via_state_service(tmp_path):
    """apply_delta is called once with the merged clean delta."""
    ss = _mock_ss()
    with _patch_ss(ss):
        await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(
                    new_files=[FileWrite(path="src/main.py", content="x = 1")],
                    dependencies=["requests"],
                )
            ],
        )

    ss.apply_delta.assert_called_once()
    applied: DeltaState = ss.apply_delta.call_args[0][0]
    assert applied.new_files[0].path == "src/main.py"
    assert applied.dependencies == ["requests"]


# --- test failure ---


async def test_adds_test_failed_error_when_tests_fail(tmp_path):
    """A failing test run adds IntegrationError(kind='test_failed') to delta.errors."""
    ss = _mock_ss(passed=False, failures=["FAILED tests/test_foo.py::test_x"], summary="1 failed")
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[DeltaState()],
        )

    assert response.delta is not None
    test_errors = [e for e in response.delta.errors if e.kind == "test_failed"]
    assert len(test_errors) == 1
    assert "1 failed" in test_errors[0].description


# --- always COMPLETED ---


async def test_returns_completed_even_when_there_are_conflict_errors(tmp_path):
    """integrate_agent always returns COMPLETED — errors live in delta.errors."""
    ss = _mock_ss()
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[
                DeltaState(new_files=[FileWrite(path="a.py", content="v1")]),
                DeltaState(new_files=[FileWrite(path="a.py", content="v2")]),
            ],
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert len(response.delta.errors) > 0


# --- clean integration ---


async def test_returns_empty_errors_on_clean_integration(tmp_path):
    """No conflicts, apply succeeds, tests pass → errors is empty."""
    ss = _mock_ss(passed=True)
    with _patch_ss(ss):
        response = await integrate_agent(
            request=_integrate_request(),
            workspace=Workspace(tmp_path),
            language_registry=LanguageRegistry(),
            completed_deltas=[DeltaState(new_files=[FileWrite(path="x.py", content="x = 1")])],
        )

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta is not None
    assert response.delta.errors == []
