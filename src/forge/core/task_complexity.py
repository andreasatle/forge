"""Task complexity classification abstractions."""

import json
import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from forge.core.models import AcceptanceCriterion, AgentRequest, AgentType, WorkSpec
from forge.llm.providers import ChatMessage, LLMProvider

_CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the worker task complexity from the compact metadata provided.\n"
    "Return JSON only. Do not return prose, markdown, code fences, bullets, or commentary.\n"
    'Return exactly two keys: "complexity" and "rationale". Do not include any other keys.\n'
    'The "complexity" value must be exactly one of: "easy", "medium", "hard".\n'
    'The "rationale" value must be a short string.\n'
    "Do not include provider model names, routing profile names, or profile-selection hints.\n"
    'Valid example: {"complexity":"medium","rationale":"requires coordinated but bounded changes"}'
)
_RAW_OUTPUT_EXCERPT_LIMIT = 240
_JSON_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*$", re.IGNORECASE)


class TaskComplexity(StrEnum):
    """Coarse task complexity labels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskComplexityInput(BaseModel):
    """Compact worker task metadata used for task complexity classification."""

    model_config = ConfigDict(frozen=True)

    objective: str
    success_condition: str
    acceptance_criteria: list[AcceptanceCriterion]
    constraints: list[str]
    non_goals: list[str]
    adapter: str
    artifact: str
    language: str | None = None


class TaskComplexityResponse(BaseModel):
    """Strict model response for task complexity classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    complexity: TaskComplexity
    rationale: str


class TaskComplexityClassifier(Protocol):
    """Classify an AgentRequest by task complexity."""

    async def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the task complexity for request."""
        ...


class DefaultTaskComplexityClassifier:
    """Default classifier that preserves behavior by returning a fixed complexity."""

    async def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the default task complexity."""
        return TaskComplexity.MEDIUM


def task_complexity_input_from_request(request: AgentRequest) -> TaskComplexityInput:
    """Extract compact worker task metadata from a WORK request."""
    if request.agent_type is not AgentType.WORK or not isinstance(request.spec, WorkSpec):
        raise ValueError("task complexity input requires a WORK request with WorkSpec")

    spec = request.spec
    contract = spec.contract
    return TaskComplexityInput(
        objective=spec.objective,
        success_condition=spec.success_condition,
        acceptance_criteria=contract.acceptance_criteria,
        constraints=contract.constraints,
        non_goals=contract.non_goals,
        adapter=spec.adapter,
        artifact=spec.artifact,
        language=spec.language,
    )


def parse_task_complexity_response(raw: str) -> TaskComplexityResponse:
    """Parse a strict task complexity classifier JSON response."""
    raw_stripped = raw.strip()
    if not raw_stripped:
        raise ValueError("empty classifier output")
    payload = _strip_single_json_fence(raw_stripped)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid task complexity JSON: {exc.msg}; "
            f"raw output excerpt: {_raw_output_excerpt(raw_stripped)}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("invalid task complexity response: expected a JSON object")

    try:
        return TaskComplexityResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid task complexity response schema: {exc}") from exc


def _raw_output_excerpt(raw: str, limit: int = _RAW_OUTPUT_EXCERPT_LIMIT) -> str:
    excerpt = " ".join(raw.split())
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3].rstrip() + "..."


def _strip_single_json_fence(raw: str) -> str:
    """Return inner JSON only when raw is exactly one json or bare fenced block."""
    if not raw.startswith("```"):
        return raw

    lines = raw.splitlines()
    if len(lines) < 3:
        return raw
    if _JSON_FENCE_OPEN_RE.fullmatch(lines[0]) is None:
        return raw
    if lines[-1].strip() != "```":
        return raw

    inner = "\n".join(lines[1:-1])
    if "```" in inner:
        return raw
    return inner.strip()


class LLMTaskComplexityClassifier:
    """LLM-backed classifier for compact worker task metadata."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return only the parsed complexity label for a WORK request."""
        return (await self.classify_with_response(request)).complexity

    async def classify_with_response(self, request: AgentRequest) -> TaskComplexityResponse:
        """Return the parsed complexity response, including rationale."""
        metadata = task_complexity_input_from_request(request)
        messages: list[ChatMessage] = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": metadata.model_dump_json()},
        ]
        raw = await self.provider.chat(messages)
        return parse_task_complexity_response(raw)
