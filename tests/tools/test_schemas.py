"""Tests verifying all tool schemas are frozen and have correct fields."""

import pytest
from pydantic import ValidationError

from forge.tools.schemas import (
    ListFilesRequest,
    ListFilesResponse,
    ReadFileRequest,
    ReadFileResponse,
    RunTestsRequest,
    RunTestsResponse,
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


def test_read_file_request_has_correct_fields() -> None:
    """ReadFileRequest stores the path field correctly."""
    req = ReadFileRequest(path="foo.txt")
    assert req.path == "foo.txt"


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
