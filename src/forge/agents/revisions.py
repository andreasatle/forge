"""RevisionHistory — accumulated producer revision requirements for PWC retry loops."""

import re
from collections.abc import Sequence

from forge.core.models import CriticFinding, RefereeDecision, RevisionItem, RevisionRequest

_MAX_REVISION_RATIONALE_CHARS = 1200
_MAX_REVISION_CHANGE_CHARS = 1600
_MAX_REVISION_ITEM_RATIONALE_CHARS = 900
_REPEATED_CONTRACT_MARKER = (
    "[omitted repeated AgentRequest contract; apply the contract block above]"
)
_REPEATED_PLUGIN_GUIDANCE_MARKER = (
    "[omitted repeated language plugin guidance; apply the binding language constraints above]"
)


def _strip_repeated_block(text: str, heading: str, marker: str) -> str:
    start = text.find(heading)
    if start == -1:
        return text
    before = text[:start].rstrip()
    rest = text[start + len(heading) :]
    match = re.search(r"\n\s*\n", rest)
    after = rest[match.end() :].lstrip() if match else ""
    parts = [part for part in (before, marker, after) if part]
    return "\n".join(parts)


def _truncate_revision_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 15].rstrip()} ...[truncated]"


def _compact_revision_text(text: str | None, limit: int) -> str:
    if not text:
        return ""
    compact = _strip_repeated_block(
        text,
        "AgentRequest contract:",
        _REPEATED_CONTRACT_MARKER,
    )
    compact = _strip_repeated_block(
        compact,
        "Language plugin guidance:",
        _REPEATED_PLUGIN_GUIDANCE_MARKER,
    )
    return _truncate_revision_text(compact, limit)


def _revision_items_from_hints(hints: list[str], rationale: str) -> list[RevisionItem]:
    return [
        RevisionItem(required_change=hint, rationale=rationale) for hint in hints if hint.strip()
    ]


def _revision_items_from_finding(finding: CriticFinding) -> list[RevisionItem]:
    if finding.revision_items:
        return finding.revision_items
    return _revision_items_from_hints(finding.hints, finding.rationale)


def _build_revision_request(
    *,
    rationale: str,
    prior_attempts: int,
    items: list[RevisionItem],
) -> RevisionRequest:
    if not items:
        items = [RevisionItem(required_change=rationale, rationale=rationale)]
    return RevisionRequest(
        disposition="revise",
        rationale=rationale,
        items=items,
        prior_attempts=prior_attempts,
    )


def _render_revision_requests(
    revision_requests: list[RevisionRequest],
    output_noun: str,
    final_output_reminder: str,
) -> str:
    lines = [
        "REQUIRED REVISION",
        "You must revise your next output against the same AgentRequest contract above.",
        "The next output must address every required change listed below.",
        "This revision block does not replace the system prompt's tool-call syntax.",
    ]
    for request_index, revision_request in enumerate(revision_requests, start=1):
        lines.extend(
            [
                "",
                f"Revision request {request_index} "
                f"(after {revision_request.prior_attempts} prior attempt(s)):",
                f"Previous disposition: {revision_request.disposition}",
                "Rationale: "
                f"{_compact_revision_text(revision_request.rationale, _MAX_REVISION_RATIONALE_CHARS)}",
                "Required changes:",
            ]
        )
        for item_index, item in enumerate(revision_request.items, start=1):
            criterion = f" [{item.criterion_id}]" if item.criterion_id else ""
            lines.append(
                f"{item_index}. Required change{criterion}: "
                f"{_compact_revision_text(item.required_change, _MAX_REVISION_CHANGE_CHARS)}"
            )
            if item.rationale:
                lines.append(
                    "   Rationale: "
                    f"{_compact_revision_text(item.rationale, _MAX_REVISION_ITEM_RATIONALE_CHARS)}"
                )
    lines.extend(
        [
            "",
            f"Revise your {output_noun} now.",
            "Do not repeat the previous output unless it has been changed to address every required change.",
        ]
    )
    if final_output_reminder:
        lines.extend(["", final_output_reminder])
    return "\n".join(lines)


class RevisionHistory:
    """Accumulated producer revision requirements across PWC attempts."""

    def __init__(self, requests: Sequence[RevisionRequest] = ()) -> None:
        self._requests = tuple(requests)

    @property
    def requests(self) -> tuple[RevisionRequest, ...]:
        """Return the accumulated revision requests in order."""
        return self._requests

    def append(self, request: RevisionRequest) -> "RevisionHistory":
        """Return new history with request appended."""
        return RevisionHistory((*self._requests, request))

    def append_from_review(
        self,
        rationale: str,
        prior_attempts: int,
        critic_finding: CriticFinding | None = None,
        referee_decision: RefereeDecision | None = None,
    ) -> "RevisionHistory":
        """Return new history with a RevisionRequest derived from critic/referee output.

        Prefers referee revision_items when present; falls back to critic items or hints.
        """
        if referee_decision is not None and referee_decision.revision_items:
            items: list[RevisionItem] = list(referee_decision.revision_items)
        elif critic_finding is not None:
            items = _revision_items_from_finding(critic_finding)
        else:
            items = []
        return self.append(
            _build_revision_request(rationale=rationale, prior_attempts=prior_attempts, items=items)
        )

    def render(self, output_noun: str, final_output_reminder: str) -> str:
        """Render accumulated revision requests as a producer retry block."""
        return _render_revision_requests(list(self._requests), output_noun, final_output_reminder)
