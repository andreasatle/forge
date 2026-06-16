"""Tests for provider-agnostic worker profile escalation policy."""

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    FailureKind,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    WorkSpec,
)
from forge.core.profile_escalation import (
    NoProfileEscalationPolicy,
    StaticProfileEscalationPolicy,
)


def _work_node(
    *,
    profile: str = "fast",
    attempt: int = 0,
    prior_profiles: tuple[str, ...] = (),
) -> DAGNode:
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do work",
            success_condition="work done",
            adapter="coding",
            artifact="codebase",
        ),
        model_profile=profile,
    )
    return DAGNode(
        request=request,
        profile_escalation_attempt=attempt,
        prior_profiles=prior_profiles,
    )


def _plan_node() -> DAGNode:
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test northstar"),
        model_profile="fast",
    )
    return DAGNode(request=request)


def _response(failure_kind: FailureKind = FailureKind.MAX_ITERATIONS) -> AgentResponse:
    return AgentResponse(
        request_id=_work_node().request.id,
        status=ResponseStatus.FAILED,
        failure_kind=failure_kind,
        error="agent loop exceeded 25 iterations",
    )


def _policy(max_escalations: int = 2) -> StaticProfileEscalationPolicy:
    return StaticProfileEscalationPolicy(
        profile_chain=("fast", "default", "strong"),
        escalatable_failures=frozenset({FailureKind.MAX_ITERATIONS}),
        max_escalations=max_escalations,
    )


def test_disabled_policy_returns_none() -> None:
    """Disabled escalation policy always returns no retry profile."""
    node = _work_node()

    assert NoProfileEscalationPolicy().next_profile(node, _response()) is None


def test_fast_max_iterations_escalates_to_default() -> None:
    """MAX_ITERATIONS on fast escalates to the next configured profile."""
    node = _work_node(profile="fast")

    assert _policy().next_profile(node, _response()) == "default"


def test_default_max_iterations_escalates_to_strong() -> None:
    """MAX_ITERATIONS on default escalates to strong when attempts remain."""
    node = _work_node(profile="default", attempt=1, prior_profiles=("fast",))

    assert _policy().next_profile(node, _response()) == "strong"


def test_strong_max_iterations_returns_none() -> None:
    """MAX_ITERATIONS on the final profile has no escalation target."""
    node = _work_node(profile="strong", attempt=2, prior_profiles=("fast", "default"))

    assert _policy().next_profile(node, _response()) is None


def test_non_work_node_returns_none() -> None:
    """Profile escalation does not apply to non-WORK nodes."""
    assert _policy().next_profile(_plan_node(), _response()) is None


def test_non_escalatable_failure_returns_none() -> None:
    """Failures outside the configured failure set are not escalated."""
    node = _work_node(profile="fast")

    response = _response(FailureKind.VALIDATION_REJECTED)

    assert _policy().next_profile(node, response) is None


def test_max_escalations_prevents_retry() -> None:
    """The max_escalations cap prevents additional retry profiles."""
    node = _work_node(profile="fast", attempt=1)

    assert _policy(max_escalations=1).next_profile(node, _response()) is None


def test_prior_profiles_prevents_cycling() -> None:
    """A next profile already present in prior_profiles is rejected."""
    node = _work_node(profile="fast", prior_profiles=("default",))

    assert _policy().next_profile(node, _response()) is None
