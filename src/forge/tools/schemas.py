"""Typed request/response pairs for all forge tools."""

from pydantic import BaseModel, ConfigDict


class ReadFileRequest(BaseModel):
    """Request to read a file at the given path."""

    model_config = ConfigDict(frozen=True)

    path: str


class ReadFileResponse(BaseModel):
    """Response containing the text content of a read file."""

    model_config = ConfigDict(frozen=True)

    content: str


class ListFilesRequest(BaseModel):
    """Request to list all files under a given directory."""

    model_config = ConfigDict(frozen=True)

    directory: str


class ListFilesResponse(BaseModel):
    """Response containing the relative paths of all listed files."""

    model_config = ConfigDict(frozen=True)

    paths: list[str]


class RunTestsRequest(BaseModel):
    """Request to execute the project test suite."""

    model_config = ConfigDict(frozen=True)


class RunTestsResponse(BaseModel):
    """Response with the outcome of running the test suite."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    failures: list[str]
    summary: str
    output: str = ""
