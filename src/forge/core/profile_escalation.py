"""Provider-agnostic worker profile escalation policies."""

from dataclasses import dataclass
from typing import Protocol

from forge.core.models import AgentResponse, AgentType, DAGNode, FailureKind


class ProfileEscalationPolicy(Protocol):
    """Return a stronger profile for a failed node, when escalation is allowed."""

    def next_profile(self, node: DAGNode, response: AgentResponse) -> str | None:
        """Return the next profile name for a retry, or None for terminal failure."""
        ...


class NoProfileEscalationPolicy:
    """Disabled policy preserving existing scheduler behavior."""

    def next_profile(self, node: DAGNode, response: AgentResponse) -> str | None:
        """Return no escalation target."""
        return None


@dataclass(frozen=True)
class StaticProfileEscalationPolicy:
    """Escalate through an explicit profile chain for selected failure kinds."""

    profile_chain: tuple[str, ...]
    escalatable_failures: frozenset[FailureKind]
    max_escalations: int

    def next_profile(self, node: DAGNode, response: AgentResponse) -> str | None:
        """Return the next stronger profile when the failed worker is eligible."""
        if node.request.agent_type is not AgentType.WORK:
            return None
        if response.failure_kind not in self.escalatable_failures:
            return None
        if node.profile_escalation_attempt >= self.max_escalations:
            return None

        current_profile = node.request.model_profile
        try:
            current_index = self.profile_chain.index(current_profile)
        except ValueError:
            return None

        next_index = current_index + 1
        if next_index >= len(self.profile_chain):
            return None

        next_profile = self.profile_chain[next_index]
        if next_profile in node.prior_profiles or next_profile == current_profile:
            return None
        return next_profile
