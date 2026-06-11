"""Tests verifying all tool schemas are frozen and have correct fields."""

import pytest
from pydantic import ValidationError

from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
    ListFilesRequest,
    ListFilesResponse,
    ReadFileRequest,
    ReadFileResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    RunTestsRequest,
    RunTestsResponse,
    WriteFileRequest,
    WriteFileResponse,
)


def test_read_file_request_is_frozen() -> None:
    """ReadFileRequest raises ValidationError when a field is mutated after construction."""
    req = ReadFileRequest(path="foo.txt")
    with pytest.raises(ValidationError):
        req.path = "bar.txt"  # type: ignore[misc]


def test_read_file_response_is_frozen() -> None:
    """ReadFileResponse raises ValidationError when a field is mutated after construction."""
    resp = ReadFileResponse(content="hello")
    with pytest.raises(ValidationError):
        resp.content = "world"  # type: ignore[misc]


def test_write_file_request_is_frozen() -> None:
    """WriteFileRequest raises ValidationError when a field is mutated after construction."""
    req = WriteFileRequest(path="out.txt", content="data")
    with pytest.raises(ValidationError):
        req.path = "other.txt"  # type: ignore[misc]


def test_write_file_response_is_frozen() -> None:
    """WriteFileResponse raises ValidationError when a field is mutated after construction."""
    resp = WriteFileResponse(path="out.txt")
    with pytest.raises(ValidationError):
        resp.path = "other.txt"  # type: ignore[misc]


def test_replace_in_file_request_is_frozen() -> None:
    """ReplaceInFileRequest raises ValidationError when a field is mutated after construction."""
    req = ReplaceInFileRequest(path="f.txt", old="a", new="b")
    with pytest.raises(ValidationError):
        req.old = "x"  # type: ignore[misc]


def test_replace_in_file_response_is_frozen() -> None:
    """ReplaceInFileResponse raises ValidationError when a field is mutated after construction."""
    resp = ReplaceInFileResponse(path="f.txt")
    with pytest.raises(ValidationError):
        resp.path = "other.txt"  # type: ignore[misc]


def test_list_files_request_is_frozen() -> None:
    """ListFilesRequest raises ValidationError when a field is mutated after construction."""
    req = ListFilesRequest(directory="src")
    with pytest.raises(ValidationError):
        req.directory = "tests"  # type: ignore[misc]


def test_list_files_response_is_frozen() -> None:
    """ListFilesResponse raises ValidationError when a field is mutated after construction."""
    resp = ListFilesResponse(paths=["a.txt", "b.txt"])
    with pytest.raises(ValidationError):
        resp.paths = []  # type: ignore[misc]


def test_run_tests_request_is_frozen() -> None:
    """RunTestsRequest has no fields — it is a zero-argument trigger."""
    assert RunTestsRequest.model_fields == {}


def test_run_tests_response_is_frozen() -> None:
    """RunTestsResponse raises ValidationError when a field is mutated after construction."""
    resp = RunTestsResponse(passed=True, failures=[], summary="1 passed")
    with pytest.raises(ValidationError):
        resp.passed = False  # type: ignore[misc]


def test_add_dependency_request_is_frozen() -> None:
    """AddDependencyRequest raises ValidationError when a field is mutated after construction."""
    req = AddDependencyRequest(package="numpy")
    with pytest.raises(ValidationError):
        req.package = "pandas"  # type: ignore[misc]


def test_add_dependency_response_is_frozen() -> None:
    """AddDependencyResponse raises ValidationError when a field is mutated after construction."""
    resp = AddDependencyResponse(package="numpy", success=True, output="installed")
    with pytest.raises(ValidationError):
        resp.success = False  # type: ignore[misc]


def test_read_file_request_has_correct_fields() -> None:
    """ReadFileRequest stores the path field correctly."""
    req = ReadFileRequest(path="foo.txt")
    assert req.path == "foo.txt"


def test_write_file_request_has_correct_fields() -> None:
    """WriteFileRequest stores path and content fields correctly."""
    req = WriteFileRequest(path="out.txt", content="data")
    assert req.path == "out.txt"
    assert req.content == "data"


def test_replace_in_file_request_has_correct_fields() -> None:
    """ReplaceInFileRequest stores path, old, and new fields correctly."""
    req = ReplaceInFileRequest(path="f.txt", old="a", new="b")
    assert req.path == "f.txt"
    assert req.old == "a"
    assert req.new == "b"


def test_list_files_response_has_correct_fields() -> None:
    """ListFilesResponse stores the paths list correctly."""
    resp = ListFilesResponse(paths=["a.txt", "b.txt"])
    assert resp.paths == ["a.txt", "b.txt"]


def test_run_tests_response_has_correct_fields() -> None:
    """RunTestsResponse stores passed, failures, summary, and output fields correctly."""
    resp = RunTestsResponse(
        passed=False,
        failures=["FAILED x::y"],
        summary="1 failed",
        output="full output",
    )
    assert resp.passed is False
    assert resp.failures == ["FAILED x::y"]
    assert resp.summary == "1 failed"
    assert resp.output == "full output"


def test_add_dependency_response_has_correct_fields() -> None:
    """AddDependencyResponse stores package, success, and output fields correctly."""
    resp = AddDependencyResponse(package="numpy", success=True, output="installed numpy")
    assert resp.package == "numpy"
    assert resp.success is True
    assert resp.output == "installed numpy"
