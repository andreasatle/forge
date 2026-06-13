"""Tests for RevisionHistory — accumulated producer revision requirements."""

from forge.agents.revisions import RevisionHistory
from forge.core.models import (
    CriticDisposition,
    CriticFinding,
    RefereeDecision,
    RevisionItem,
    RevisionRequest,
)


def _finding(
    rationale: str = "needs work",
    hints: list[str] | None = None,
    revision_items: list[RevisionItem] | None = None,
) -> CriticFinding:
    return CriticFinding(
        disposition=CriticDisposition.REVISE,
        rationale=rationale,
        hints=hints or [],
        revision_items=revision_items or [],
    )


def _decision(
    rationale: str = "agreed",
    revision_items: list[RevisionItem] | None = None,
) -> RefereeDecision:
    return RefereeDecision(
        disposition=CriticDisposition.REVISE,
        rationale=rationale,
        override=False,
        revision_items=revision_items or [],
    )


def test_empty_history_has_no_requests() -> None:
    """RevisionHistory starts with an empty requests tuple."""
    assert RevisionHistory().requests == ()


def test_append_from_review_uses_referee_revision_items_when_present() -> None:
    """Referee revision_items take precedence over critic items."""
    history = RevisionHistory()
    finding = _finding(
        revision_items=[
            RevisionItem(criterion_id="AC1", required_change="critic change", rationale="c")
        ]
    )
    decision = _decision(
        rationale="referee rationale",
        revision_items=[
            RevisionItem(criterion_id="AC2", required_change="referee change", rationale="r")
        ],
    )
    result = history.append_from_review(
        rationale="referee rationale",
        prior_attempts=1,
        critic_finding=finding,
        referee_decision=decision,
    )
    assert len(result.requests) == 1
    req = result.requests[0]
    assert req.rationale == "referee rationale"
    assert req.prior_attempts == 1
    assert req.items[0].required_change == "referee change"
    assert req.items[0].criterion_id == "AC2"


def test_append_from_review_falls_back_to_critic_structured_items_when_referee_has_none() -> None:
    """Falls back to critic revision_items when referee provides none."""
    finding = _finding(
        revision_items=[
            RevisionItem(criterion_id="AC3", required_change="critic structured", rationale="c")
        ]
    )
    result = RevisionHistory().append_from_review(
        rationale="rationale",
        prior_attempts=1,
        critic_finding=finding,
        referee_decision=_decision(),
    )
    assert result.requests[0].items[0].required_change == "critic structured"
    assert result.requests[0].items[0].criterion_id == "AC3"


def test_append_from_review_falls_back_to_critic_hints_when_no_structured_items() -> None:
    """Falls back to critic hints when neither referee nor critic have revision_items."""
    finding = _finding(hints=["add tests", "fix error handling"])
    result = RevisionHistory().append_from_review(
        rationale="needs improvement",
        prior_attempts=1,
        critic_finding=finding,
    )
    req = result.requests[0]
    assert len(req.items) == 2
    assert req.items[0].required_change == "add tests"
    assert req.items[1].required_change == "fix error handling"


def test_append_from_review_preserves_criterion_ids() -> None:
    """Criterion IDs on revision items survive the append_from_review path."""
    decision = _decision(
        rationale="tests required",
        revision_items=[
            RevisionItem(
                criterion_id="AC2", required_change="Add parser tests.", rationale="AC2 unmet."
            )
        ],
    )
    result = RevisionHistory().append_from_review(
        rationale="tests required",
        prior_attempts=1,
        critic_finding=_finding(),
        referee_decision=decision,
    )
    assert result.requests[0].items[0].criterion_id == "AC2"


def test_accumulates_multiple_revision_rounds() -> None:
    """Successive append_from_review calls accumulate all prior requests."""
    history = RevisionHistory()
    history = history.append_from_review(
        rationale="round 1",
        prior_attempts=1,
        critic_finding=_finding(rationale="round 1", hints=["fix 1"]),
    )
    history = history.append_from_review(
        rationale="round 2",
        prior_attempts=2,
        critic_finding=_finding(rationale="round 2", hints=["fix 2"]),
    )
    assert len(history.requests) == 2
    assert history.requests[0].prior_attempts == 1
    assert history.requests[1].prior_attempts == 2
    assert history.requests[0].items[0].required_change == "fix 1"
    assert history.requests[1].items[0].required_change == "fix 2"


def test_accumulation_is_immutable() -> None:
    """append_from_review does not mutate the original history."""
    original = RevisionHistory()
    updated = original.append_from_review(
        rationale="needs work",
        prior_attempts=1,
        critic_finding=_finding(hints=["add test"]),
    )
    assert len(original.requests) == 0
    assert len(updated.requests) == 1


def test_append_inserts_request_directly() -> None:
    """append accepts a pre-built RevisionRequest without extracting items."""
    request = RevisionRequest(
        rationale="manual",
        prior_attempts=0,
        items=[RevisionItem(required_change="do this", rationale="because")],
    )
    history = RevisionHistory().append(request)
    assert len(history.requests) == 1
    assert history.requests[0].rationale == "manual"


def test_initialized_with_existing_requests() -> None:
    """RevisionHistory can be seeded with an initial list of requests."""
    initial = RevisionRequest(
        rationale="initial",
        items=[RevisionItem(required_change="do this", rationale="")],
        prior_attempts=0,
    )
    history = RevisionHistory([initial])
    assert len(history.requests) == 1
    assert history.requests[0].rationale == "initial"


def test_render_starts_with_required_revision_header() -> None:
    """render output always opens with the REQUIRED REVISION header."""
    history = RevisionHistory().append_from_review(
        rationale="needs work",
        prior_attempts=1,
        critic_finding=_finding(hints=["fix it"]),
    )
    rendered = history.render("implementation", "")
    assert rendered.startswith("REQUIRED REVISION")


def test_render_includes_revise_noun_line() -> None:
    """render uses the output_noun argument in the closing revision directive."""
    history = RevisionHistory().append_from_review(
        rationale="needs work",
        prior_attempts=1,
        critic_finding=_finding(hints=["fix it"]),
    )
    assert "Revise your implementation now." in history.render("implementation", "")
    assert "Revise your plan now." in history.render("plan", "")


def test_render_includes_final_output_reminder() -> None:
    """render appends the final_output_reminder block when non-empty."""
    history = RevisionHistory().append_from_review(
        rationale="improve",
        prior_attempts=1,
        critic_finding=_finding(hints=["do more"]),
    )
    rendered = history.render("implementation", "FINAL OUTPUT FORMAT REMINDER\nReturn JSON.")
    assert "FINAL OUTPUT FORMAT REMINDER" in rendered
    assert "Return JSON." in rendered


def test_render_omits_final_reminder_when_empty() -> None:
    """render does not add a blank final_output_reminder section."""
    history = RevisionHistory().append_from_review(
        rationale="improve",
        prior_attempts=1,
        critic_finding=_finding(hints=["do more"]),
    )
    assert "FINAL OUTPUT FORMAT REMINDER" not in history.render("implementation", "")


def test_render_compacts_repeated_contract_block() -> None:
    """Embedded AgentRequest contract text in rationale is replaced with a marker."""
    contract_block = "AgentRequest contract:\nobjective: do X\nsuccess_condition: done"
    rationale = f"Fix the issue.\n\n{contract_block}\n\nApply these changes."
    history = RevisionHistory().append_from_review(
        rationale=rationale,
        prior_attempts=1,
        critic_finding=_finding(rationale=rationale, hints=["fix it"]),
    )
    rendered = history.render("implementation", "")
    assert "[omitted repeated AgentRequest contract" in rendered
    assert "Apply these changes." in rendered


def test_render_compacts_repeated_plugin_guidance() -> None:
    """Embedded Language plugin guidance text in rationale is replaced with a marker."""
    plugin_block = "Language plugin guidance:\nrule 1\nrule 2"
    rationale = f"The output violated rules.\n\n{plugin_block}\n\nPlease fix."
    history = RevisionHistory().append_from_review(
        rationale=rationale,
        prior_attempts=1,
        critic_finding=_finding(rationale=rationale, hints=["apply rules"]),
    )
    rendered = history.render("implementation", "")
    assert "[omitted repeated language plugin guidance" in rendered
    assert "Please fix." in rendered


def test_render_includes_prior_attempts_count() -> None:
    """render shows the prior_attempts count in each revision request header."""
    history = RevisionHistory().append_from_review(
        rationale="needs more",
        prior_attempts=2,
        critic_finding=_finding(hints=["expand"]),
    )
    assert "after 2 prior attempt(s)" in history.render("implementation", "")


def test_render_multiple_rounds_shows_all_revision_requests() -> None:
    """render includes a header for each accumulated revision round."""
    history = RevisionHistory()
    history = history.append_from_review(
        rationale="round 1",
        prior_attempts=1,
        critic_finding=_finding(hints=["fix A"]),
    )
    history = history.append_from_review(
        rationale="round 2",
        prior_attempts=2,
        critic_finding=_finding(hints=["fix B"]),
    )
    rendered = history.render("implementation", "")
    assert "Revision request 1 (after 1 prior attempt(s))" in rendered
    assert "Revision request 2 (after 2 prior attempt(s))" in rendered
    assert "1. Required change: fix A" in rendered
    assert "1. Required change: fix B" in rendered


def test_no_critic_no_items_falls_back_to_rationale_as_change() -> None:
    """With no finding and no decision, rationale becomes the required_change text."""
    result = RevisionHistory().append_from_review(
        rationale="please produce output", prior_attempts=1
    )
    req = result.requests[0]
    assert req.items[0].required_change == "please produce output"
