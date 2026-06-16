"""Task complexity classification abstractions."""

from enum import StrEnum
from typing import Protocol

from forge.core.models import AgentRequest


class TaskComplexity(StrEnum):
    """Coarse task complexity labels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskComplexityClassifier(Protocol):
    """Classify an AgentRequest by task complexity."""

    def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the task complexity for request."""
        ...


class DefaultTaskComplexityClassifier:
    """Default classifier that preserves behavior by returning a fixed complexity."""

    def classify(self, request: AgentRequest) -> TaskComplexity:
        """Return the default task complexity."""
        return TaskComplexity.MEDIUM
