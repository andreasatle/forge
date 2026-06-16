"""Profile assignment for worker AgentRequests."""

from typing import Protocol

from forge.core.models import AgentRequest


class ProfileAssigner(Protocol):
    """Assign a model profile to an AgentRequest."""

    def assign(self, request: AgentRequest) -> str:
        """Return the model profile name for request."""
        ...


class DefaultProfileAssigner:
    """Default profile assigner that preserves existing routing behavior."""

    def assign(self, request: AgentRequest) -> str:
        """Return the default worker profile."""
        return "default"
