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


class WriteFileRequest(BaseModel):
    """Request to write content to a file, creating it if it does not exist."""

    model_config = ConfigDict(frozen=True)

    path: str
    content: str


class WriteFileResponse(BaseModel):
    """Response confirming the path that was written."""

    model_config = ConfigDict(frozen=True)

    path: str


class ReplaceInFileRequest(BaseModel):
    """Request to replace a unique substring in a file with new text."""

    model_config = ConfigDict(frozen=True)

    path: str
    old: str
    new: str


class ReplaceInFileResponse(BaseModel):
    """Response confirming the path where the replacement was applied."""

    model_config = ConfigDict(frozen=True)

    path: str


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


class AddDependencyRequest(BaseModel):
    """Request to add a package as a project dependency."""

    model_config = ConfigDict(frozen=True)

    package: str


class AddDependencyResponse(BaseModel):
    """Response confirming whether the dependency was successfully added."""

    model_config = ConfigDict(frozen=True)

    package: str
    success: bool
    output: str


class ReadBlackboardRequest(BaseModel):
    """Request to read a value from the shared blackboard by key."""

    model_config = ConfigDict(frozen=True)

    key: str


class ReadBlackboardResponse(BaseModel):
    """Response with the blackboard value for the requested key, or None if absent."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: str | None


class WriteBlackboardRequest(BaseModel):
    """Request to write a key-value pair to the shared blackboard."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: str


class WriteBlackboardResponse(BaseModel):
    """Response confirming the key that was written to the blackboard."""

    model_config = ConfigDict(frozen=True)

    key: str
