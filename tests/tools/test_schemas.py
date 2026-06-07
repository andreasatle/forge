"""Tests verifying all tool schemas are frozen and have correct fields."""

import pytest
from pydantic import ValidationError

from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
    ListFilesRequest,
    ListFilesResponse,
    ReadBlackboardRequest,
    ReadBlackboardResponse,
    ReadFileRequest,
    ReadFileResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    RunTestsRequest,
    RunTestsResponse,
    WriteBlackboardRequest,
    WriteBlackboardResponse,
    WriteFileRequest,
    WriteFileResponse,
)


def test_read_file_request_is_frozen() -> None:
    req = ReadFileRequest(path="foo.txt")
    with pytest.raises(ValidationError):
        req.path = "bar.txt"  # type: ignore[misc]


def test_read_file_response_is_frozen() -> None:
    resp = ReadFileResponse(content="hello")
    with pytest.raises(ValidationError):
        resp.content = "world"  # type: ignore[misc]


def test_write_file_request_is_frozen() -> None:
    req = WriteFileRequest(path="out.txt", content="data")
    with pytest.raises(ValidationError):
        req.path = "other.txt"  # type: ignore[misc]


def test_write_file_response_is_frozen() -> None:
    resp = WriteFileResponse(path="out.txt")
    with pytest.raises(ValidationError):
        resp.path = "other.txt"  # type: ignore[misc]


def test_replace_in_file_request_is_frozen() -> None:
    req = ReplaceInFileRequest(path="f.txt", old="a", new="b")
    with pytest.raises(ValidationError):
        req.old = "x"  # type: ignore[misc]


def test_replace_in_file_response_is_frozen() -> None:
    resp = ReplaceInFileResponse(path="f.txt")
    with pytest.raises(ValidationError):
        resp.path = "other.txt"  # type: ignore[misc]


def test_list_files_request_is_frozen() -> None:
    req = ListFilesRequest(directory="src")
    with pytest.raises(ValidationError):
        req.directory = "tests"  # type: ignore[misc]


def test_list_files_response_is_frozen() -> None:
    resp = ListFilesResponse(paths=["a.txt", "b.txt"])
    with pytest.raises(ValidationError):
        resp.paths = []  # type: ignore[misc]


def test_run_tests_request_is_frozen() -> None:
    assert RunTestsRequest.model_fields == {}


def test_run_tests_response_is_frozen() -> None:
    resp = RunTestsResponse(passed=True, failures=[], summary="1 passed")
    with pytest.raises(ValidationError):
        resp.passed = False  # type: ignore[misc]


def test_add_dependency_request_is_frozen() -> None:
    req = AddDependencyRequest(package="numpy")
    with pytest.raises(ValidationError):
        req.package = "pandas"  # type: ignore[misc]


def test_add_dependency_response_is_frozen() -> None:
    resp = AddDependencyResponse(package="numpy", success=True, output="installed")
    with pytest.raises(ValidationError):
        resp.success = False  # type: ignore[misc]


def test_read_blackboard_request_is_frozen() -> None:
    req = ReadBlackboardRequest(key="x")
    with pytest.raises(ValidationError):
        req.key = "y"  # type: ignore[misc]


def test_read_blackboard_response_value_can_be_none() -> None:
    resp = ReadBlackboardResponse(key="x", value=None)
    assert resp.value is None


def test_read_blackboard_response_is_frozen() -> None:
    resp = ReadBlackboardResponse(key="x", value="42")
    with pytest.raises(ValidationError):
        resp.value = "99"  # type: ignore[misc]


def test_write_blackboard_request_is_frozen() -> None:
    req = WriteBlackboardRequest(key="x", value="42")
    with pytest.raises(ValidationError):
        req.value = "99"  # type: ignore[misc]


def test_write_blackboard_response_is_frozen() -> None:
    resp = WriteBlackboardResponse(key="x")
    with pytest.raises(ValidationError):
        resp.key = "y"  # type: ignore[misc]


def test_read_file_request_has_correct_fields() -> None:
    req = ReadFileRequest(path="foo.txt")
    assert req.path == "foo.txt"


def test_write_file_request_has_correct_fields() -> None:
    req = WriteFileRequest(path="out.txt", content="data")
    assert req.path == "out.txt"
    assert req.content == "data"


def test_replace_in_file_request_has_correct_fields() -> None:
    req = ReplaceInFileRequest(path="f.txt", old="a", new="b")
    assert req.path == "f.txt"
    assert req.old == "a"
    assert req.new == "b"


def test_list_files_response_has_correct_fields() -> None:
    resp = ListFilesResponse(paths=["a.txt", "b.txt"])
    assert resp.paths == ["a.txt", "b.txt"]


def test_run_tests_response_has_correct_fields() -> None:
    resp = RunTestsResponse(passed=False, failures=["FAILED x::y"], summary="1 failed")
    assert resp.passed is False
    assert resp.failures == ["FAILED x::y"]
    assert resp.summary == "1 failed"


def test_add_dependency_response_has_correct_fields() -> None:
    resp = AddDependencyResponse(package="numpy", success=True, output="installed numpy")
    assert resp.package == "numpy"
    assert resp.success is True
    assert resp.output == "installed numpy"
