"""AttemptTelemetryReporter — projects PWC execution events into telemetry."""

from typing import cast
from uuid import UUID

from pydantic import BaseModel

from forge.core.models import (
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    PlanResponse,
    RefereeDecision,
    RevisionRequest,
    WorkOutput,
)
from forge.core.telemetry import TelemetryEvent, TelemetrySink, safe_append_telemetry


def _preview(text: str | None, limit: int = 500) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 15].rstrip()} ...[truncated]"


def _model_data(model: BaseModel) -> dict[str, object]:
    return cast(dict[str, object], model.model_dump(mode="json"))


def _work_output_summary(output: WorkOutput, dispatch_sha: str = "") -> dict[str, object]:
    result: dict[str, object] = {"summary": output.summary}
    if dispatch_sha:
        result["dispatch_sha"] = dispatch_sha
    return result


def _plan_summary(plan: PlanResponse) -> dict[str, object]:
    return {
        "task_count": len(plan.tasks),
        "tasks": [
            {
                "objective": task.objective,
                "adapter": task.adapter,
                "artifact": task.artifact,
                "language": task.language,
                "depends_on": list(task.depends_on),
            }
            for task in plan.tasks
        ],
    }


def _producer_response_summary(
    response: AgentResponse, dispatch_sha: str = ""
) -> dict[str, object]:
    output_type = type(response.output).__name__ if response.output is not None else None
    data: dict[str, object] = {
        "status": response.status.value,
        "failure_kind": response.failure_kind.value if response.failure_kind else None,
        "error": _preview(response.error),
        "output_type": output_type,
        "ran_tests_and_passed": response.ran_tests_and_passed,
    }
    if isinstance(response.output, WorkOutput):
        data["work_output"] = _work_output_summary(response.output, dispatch_sha)
    elif isinstance(response.output, PlanResponse):
        data["plan"] = _plan_summary(response.output)
    if response.diagnostics:
        data["diagnostics"] = [_model_data(d) for d in response.diagnostics]
    return data


class AttemptTelemetryReporter:
    """Projects PWC execution events into telemetry."""

    def __init__(
        self,
        sink: TelemetrySink | None,
        run_id: UUID | None,
        node_id: UUID,
        agent_type: AgentType,
    ) -> None:
        self._sink = sink
        self._run_id = run_id
        self._node_id = node_id
        self._agent_type = agent_type

    def _emit(
        self,
        *,
        attempt_number: int | None,
        role: str,
        phase: str,
        event_type: str,
        status: str | None = None,
        summary: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        if self._run_id is None:
            return
        safe_append_telemetry(
            self._sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=self._node_id,
                request_id=self._node_id,
                agent_type=self._agent_type.value,
                attempt_number=attempt_number,
                role=role,
                phase=phase,
                event_type=event_type,
                status=status,
                summary=summary,
                data=data or {},
            ),
        )

    def attempt_started(self, attempt_number: int, max_attempts: int) -> None:
        """Emit pwc.attempt.started for the given attempt number."""
        self._emit(
            attempt_number=attempt_number,
            role="producer",
            phase="pwc",
            event_type="pwc.attempt.started",
            status="started",
            summary=f"attempt {attempt_number}/{max_attempts} started",
            data={"max_attempts": max_attempts},
        )

    def producer_response_parsed(
        self, attempt_number: int, response: AgentResponse, dispatch_sha: str = ""
    ) -> None:
        """Emit producer.response.parsed with the parsed response summary."""
        self._emit(
            attempt_number=attempt_number,
            role="producer",
            phase="producer",
            event_type="producer.response.parsed",
            status=response.status.value,
            summary=f"producer returned {response.status.value}",
            data=_producer_response_summary(response, dispatch_sha),
        )

    def critic_finding_parsed(self, attempt_number: int, finding: CriticFinding) -> None:
        """Emit critic.finding.parsed with the critic disposition and rationale."""
        self._emit(
            attempt_number=attempt_number,
            role="critic",
            phase="critic",
            event_type="critic.finding.parsed",
            status=finding.disposition.value,
            summary=_preview(finding.rationale),
            data={"critic_finding": _model_data(finding)},
        )

    def referee_decision_parsed(self, attempt_number: int, decision: RefereeDecision) -> None:
        """Emit referee.decision.parsed with the referee disposition and rationale."""
        self._emit(
            attempt_number=attempt_number,
            role="referee",
            phase="referee",
            event_type="referee.decision.parsed",
            status=decision.disposition.value,
            summary=_preview(decision.rationale),
            data={"referee_decision": _model_data(decision)},
        )

    def decompose_requested(self, attempt_number: int, decision: RefereeDecision) -> None:
        """Emit pwc.decompose.requested when the referee requests task decomposition."""
        self._emit(
            attempt_number=attempt_number,
            role="referee",
            phase="referee",
            event_type="pwc.decompose.requested",
            status="decompose",
            summary=_preview(decision.rationale),
            data={"referee_decision": _model_data(decision)},
        )

    def revision_appended(self, attempt_number: int, request: RevisionRequest) -> None:
        """Emit pwc.revision.appended when a revision request is queued for the next attempt."""
        self._emit(
            attempt_number=attempt_number,
            role="revision_loop",
            phase="pwc",
            event_type="pwc.revision.appended",
            status="revise",
            summary="revision request appended",
            data={"revision_request": _model_data(request)},
        )

    def exhausted(self, attempts: int, reason: str) -> None:
        """Emit pwc.exhausted when all attempts are consumed without acceptance."""
        self._emit(
            attempt_number=attempts,
            role="revision_loop",
            phase="pwc",
            event_type="pwc.exhausted",
            status="failed",
            summary=reason,
            data={
                "attempt_count": attempts,
                "final_disposition": CriticDisposition.REVISE.value,
                "error": reason,
            },
        )
