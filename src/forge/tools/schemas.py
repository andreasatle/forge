"""Typed request/response pairs for all forge tools."""

from pydantic import BaseModel, ConfigDict


class ReadFileRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str


class ReadFileResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str


class WriteFileRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    content: str


class WriteFileResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str


class ReplaceInFileRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    old: str
    new: str


class ReplaceInFileResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str


class ListFilesRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: str


class ListFilesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    paths: list[str]


class RunTestsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)


class RunTestsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    failures: list[str]
    summary: str


class AddDependencyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: str


class AddDependencyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: str
    success: bool
    output: str


class ReadBlackboardRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str


class ReadBlackboardResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: str | None


class WriteBlackboardRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: str


class WriteBlackboardResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
