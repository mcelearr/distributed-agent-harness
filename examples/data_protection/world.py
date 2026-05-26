"""
DataProtectionWorldEnvironment

Models the full lifecycle of a data protection engagement at a law firm,
from pitching for the business through to managing data breaches.

Lifecycle:

    Pitch ──► Contract ──► Privacy Policy ──► [ Data Subject Requests ]
                                            └► [ Data Breaches        ]

Each @action method is one auditable step in this lifecycle. No other
mutations to the project state are possible — there are no arbitrary
shell commands or free-form tool calls.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel

from distributed_agent_harness.world import BaseWorldEnvironment, action

from .models import (
    BreachSeverity,
    BreachStatus,
    Client,
    Contract,
    DataBreach,
    DataSubject,
    DataSubjectRequest,
    DSRStatus,
    DSRType,
    EngagementStatus,
    PolicyStatus,
    PrivacyPolicy,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# State schema                                                                 #
# --------------------------------------------------------------------------- #

class DataProtectionState(BaseModel):
    """
    Complete, serialisable state for a data protection engagement.

    Stored as ``state.json`` in the Project Namespace so both agents and
    human solicitors can read the current position at any time.
    """
    status: EngagementStatus = EngagementStatus.PITCH
    client: Optional[Client] = None
    contract: Optional[Contract] = None
    privacy_policies: list[PrivacyPolicy] = []
    data_subjects: list[DataSubject] = []
    data_subject_requests: list[DataSubjectRequest] = []
    data_breaches: list[DataBreach] = []


# --------------------------------------------------------------------------- #
# World Environment                                                            #
# --------------------------------------------------------------------------- #

class DataProtectionWorldEnvironment(BaseWorldEnvironment):
    """
    World environment for a data protection law firm engagement.

    Covers the complete UK GDPR compliance lifecycle:

    1. **Pitch & win** the client engagement
    2. **Agree the contract** (with DPA included as standard)
    3. **Draft, approve, and publish** the client's privacy policy
    4. **Register data subjects** and the lawful basis for processing their data
    5. **Handle Data Subject Requests** — erasure, access, portability, etc.
       (30-day response obligation under UK GDPR Art. 12)
    6. **Manage data breaches** — assess, notify the ICO within 72 hours
       (Art. 33), notify affected subjects for high-risk breaches (Art. 34)
    """

    State = DataProtectionState

    # ----------------------------------------------------------------------- #
    # Summary rendering — domain-specific "card view"                          #
    # ----------------------------------------------------------------------- #

    def render_summary(self) -> str:
        """
        Produce the narrative summary that lives at ``<project>/summary.md``.

        Designed for a partner or compliance officer to read at a glance:
        engagement status, the live privacy policy, current data subjects,
        and the state of any open DSRs or breaches.
        """
        s = self.state
        client_line = (
            f"**{s.client.name}** — {s.client.industry}"
            if s.client
            else "_no client on record_"
        )

        lines: list[str] = [
            f"# Data Protection Engagement — {s.client.name if s.client else '(unnamed)'}",
            "",
            f"**Status:** `{s.status.value}`",
            f"**Client:** {client_line}",
        ]
        if s.client:
            lines.append(f"**Scope:** {s.client.scope_of_work}")

        # Contract
        lines.append("")
        lines.append("## Contract")
        if s.contract:
            fee = (
                f"£{s.contract.retainer_fee_gbp:,.0f}/month retainer"
                if s.contract.retainer_fee_gbp is not None
                else "no retainer recorded"
            )
            lines.append(
                f"- Signed {s.contract.signed_date:%Y-%m-%d} — {fee}"
            )
            lines.append(
                f"- DPA included: **{'yes' if s.contract.dpa_included else 'no'}**"
            )
        else:
            lines.append("_no contract yet_")

        # Privacy policies
        lines.append("")
        lines.append("## Privacy Policy")
        if s.privacy_policies:
            for p in s.privacy_policies:
                lines.append(f"- v{p.version} — `{p.status.value}`")
        else:
            lines.append("_no policy drafted_")

        # Data subjects
        active = [ds for ds in s.data_subjects if not ds.erased]
        erased = [ds for ds in s.data_subjects if ds.erased]
        lines.append("")
        lines.append(f"## Data Subjects ({len(s.data_subjects)})")
        lines.append(f"- {len(active)} active, {len(erased)} erased")

        # DSRs
        if s.data_subject_requests:
            open_dsrs = [
                r for r in s.data_subject_requests
                if r.status not in (DSRStatus.COMPLETED, DSRStatus.REFUSED)
            ]
            lines.append("")
            lines.append(f"## Data Subject Requests ({len(s.data_subject_requests)})")
            for r in s.data_subject_requests:
                deadline = (
                    f", deadline {r.deadline:%Y-%m-%d}"
                    if r.deadline
                    else ""
                )
                lines.append(
                    f"- `{r.id}` — {r.request_type.value} from {r.subject_email} "
                    f"— `{r.status.value}`{deadline}"
                )
            if open_dsrs:
                lines.append("")
                lines.append(f"_Note: {len(open_dsrs)} open DSR(s) require action._")

        # Breaches
        if s.data_breaches:
            open_breaches = [
                b for b in s.data_breaches if b.status != BreachStatus.RESOLVED
            ]
            lines.append("")
            lines.append(f"## Data Breaches ({len(s.data_breaches)})")
            for b in s.data_breaches:
                sev = b.severity.value if b.severity else "unassessed"
                ico = f", ICO ref {b.ico_reference}" if b.ico_reference else ""
                lines.append(
                    f"- `{b.id}` — {sev}, `{b.status.value}` "
                    f"({b.estimated_subjects_affected} subjects affected){ico}"
                )
            if open_breaches:
                lines.append("")
                lines.append(f"_Note: {len(open_breaches)} open breach(es) require action._")

        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------------------- #
    # Phase 1 — Engagement                                                     #
    # ----------------------------------------------------------------------- #

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.PITCH,
    )
    def submit_pitch(
        self,
        client_name: str,
        contact_name: str,
        contact_email: str,
        industry: str,
        scope_of_work: str,
    ) -> Client:
        """
        Record a pitch submission for a prospective data protection client.

        Sets engagement status to PITCH and stores the client details.
        Follow up with win_pitch() or lose_pitch() to progress the engagement.
        """
        client = Client(
            name=client_name,
            contact_name=contact_name,
            contact_email=contact_email,
            industry=industry,
            scope_of_work=scope_of_work,
        )
        self.state.client = client
        self.state.status = EngagementStatus.PITCH
        return client

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.PITCH,
    )
    def win_pitch(
        self,
        terms_summary: str,
        retainer_fee_gbp: Optional[float] = None,
        review_date: Optional[datetime] = None,
    ) -> Contract:
        """
        Record that the pitch was won and a contract has been agreed.

        Advances the engagement to CONTRACTED status. A Data Processing
        Agreement (DPA) is included as standard under UK GDPR Art. 28.
        submit_pitch() must have been called first.
        """
        # The PITCH-state precondition is enforced by the decorator.
        # We still need to ensure a client was actually recorded.
        if not self.state.client:
            raise ValueError("No client on record — call submit_pitch() first")

        contract = Contract(
            terms_summary=terms_summary,
            retainer_fee_gbp=retainer_fee_gbp,
            signed_date=_now(),
            dpa_included=True,
            review_date=review_date,
        )
        self.state.contract = contract
        self.state.status = EngagementStatus.CONTRACTED
        return contract

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.PITCH,
    )
    def lose_pitch(self, reason: str) -> None:
        """
        Record that the pitch was unsuccessful and close the engagement.

        The client record is retained for future reference.
        reason should describe why the pitch was not won.
        """
        self.state.status = EngagementStatus.CLOSED

    # ----------------------------------------------------------------------- #
    # Phase 2 — Privacy Policy                                                 #
    # ----------------------------------------------------------------------- #

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.CONTRACTED,
    )
    def draft_privacy_policy(
        self,
        version: str,
        content: str,
        data_categories: list[str],
        processing_purposes: list[str],
        retention_periods: dict[str, str],
    ) -> PrivacyPolicy:
        """
        Create a draft privacy policy for the client.

        The policy moves through DRAFT → APPROVED → PUBLISHED before going live.
        Multiple draft versions may exist simultaneously.

        data_categories: personal data processed, e.g. ["name", "email", "health data"]
        processing_purposes: reasons for processing, e.g. ["payroll", "marketing"]
        retention_periods: mapping of category to retention duration, e.g. {"email": "3 years"}
        """
        policy = PrivacyPolicy(
            version=version,
            content=content,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            retention_periods=retention_periods,
            created_at=_now(),
        )
        self.state.privacy_policies.append(policy)
        return policy

    @action
    def approve_privacy_policy(self, version: str, approved_by: str) -> PrivacyPolicy:
        """
        Mark a draft privacy policy as approved by a named solicitor.

        approved_by should contain the name or role of the approving solicitor.
        An approved policy can then be published with publish_privacy_policy().
        """
        policy = self._get_policy(version)
        if policy.status != PolicyStatus.DRAFT:
            raise ValueError(
                f"Policy v{version} has status {policy.status} — only DRAFT policies can be approved"
            )
        policy.status = PolicyStatus.APPROVED
        policy.approved_at = _now()
        policy.approved_by = approved_by
        return policy

    @action
    def publish_privacy_policy(self, version: str) -> PrivacyPolicy:
        """
        Mark an approved privacy policy as published (live).

        The published policy is the governing document for the client's data
        processing activities. Data subjects must be able to access it.
        """
        policy = self._get_policy(version)
        if policy.status != PolicyStatus.APPROVED:
            raise ValueError(
                f"Policy v{version} must be APPROVED before it can be published "
                f"(current: {policy.status})"
            )
        policy.status = PolicyStatus.PUBLISHED
        policy.published_at = _now()
        return policy

    # ----------------------------------------------------------------------- #
    # Phase 3 — Data Subjects                                                  #
    # ----------------------------------------------------------------------- #

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.CONTRACTED,
    )
    def register_data_subject(
        self,
        name: str,
        email: str,
        data_categories_held: list[str],
        lawful_basis: str,
    ) -> DataSubject:
        """
        Register an individual whose personal data is held by the client.

        lawful_basis must be one of the UK GDPR Art. 6 bases:
        "consent", "contract", "legal obligation", "vital interests",
        "public task", or "legitimate interests".

        This record establishes the basis for handling any future Data Subject Requests
        from or relating to this individual.
        """
        subject = DataSubject(
            name=name,
            email=email,
            data_categories_held=data_categories_held,
            lawful_basis=lawful_basis,
            registered_at=_now(),
        )
        self.state.data_subjects.append(subject)
        return subject

    # ----------------------------------------------------------------------- #
    # Phase 4 — Data Subject Requests                                          #
    # ----------------------------------------------------------------------- #

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.CONTRACTED,
    )
    def submit_dsr(
        self,
        subject_email: str,
        request_type: DSRType,
        description: str,
    ) -> DataSubjectRequest:
        """
        Log a Data Subject Request (DSR) received from an individual.

        UK GDPR grants individuals the right to erasure (Art. 17), access
        (Art. 15), portability (Art. 20), rectification (Art. 16), and
        restriction (Art. 18).

        The 30-day response deadline (UK GDPR Art. 12(3)) is set automatically.
        Follow with acknowledge_dsr(), then complete_dsr() or refuse_dsr().

        subject_email: email of the requesting individual
        request_type: DSRType.ERASURE | ACCESS | PORTABILITY | RECTIFICATION | RESTRICTION
        """
        subject = next(
            (s for s in self.state.data_subjects if s.email == subject_email),
            None,
        )
        dsr = DataSubjectRequest(
            subject_id=subject.id if subject else "unregistered",
            subject_email=subject_email,
            request_type=request_type,
            description=description,
            submitted_at=_now(),
            deadline=_now() + timedelta(days=30),
        )
        self.state.data_subject_requests.append(dsr)
        return dsr

    @action(
        relevance=lambda s, e: any(
            r.status == DSRStatus.SUBMITTED for r in s.data_subject_requests
        ),
    )
    def acknowledge_dsr(self, dsr_id: str) -> DataSubjectRequest:
        """
        Confirm receipt of a Data Subject Request.

        Under UK GDPR Art. 12, controllers must acknowledge receipt without
        undue delay and confirm the deadline for a full response.
        Advances the DSR to IN_PROGRESS status.
        """
        dsr = self._get_dsr(dsr_id)
        if dsr.status != DSRStatus.SUBMITTED:
            raise ValueError(
                f"DSR {dsr_id} has status {dsr.status} — only SUBMITTED DSRs can be acknowledged"
            )
        dsr.status = DSRStatus.IN_PROGRESS
        dsr.acknowledged_at = _now()
        return dsr

    @action(
        relevance=lambda s, e: any(
            r.status in (DSRStatus.SUBMITTED, DSRStatus.IN_PROGRESS)
            for r in s.data_subject_requests
        ),
    )
    def complete_dsr(
        self,
        dsr_id: str,
        action_taken: str,
        data_deleted: bool = False,
        exemptions_applied: Optional[list[str]] = None,
    ) -> DataSubjectRequest:
        """
        Record the completion of a Data Subject Request.

        action_taken: clear description of what was done — what data was deleted,
            provided, rectified, or restricted.
        data_deleted: True if personal data was deleted as part of this action.
        exemptions_applied: any Art. 17(3) exemptions that limited the scope of
            erasure, e.g. ["legal claims defence", "compliance with legal obligation"].

        If data_deleted is True and the request is for erasure, the corresponding
        DataSubject record is automatically marked as erased.
        """
        dsr = self._get_dsr(dsr_id)
        if dsr.status not in (DSRStatus.SUBMITTED, DSRStatus.IN_PROGRESS):
            raise ValueError(
                f"DSR {dsr_id} cannot be completed from status {dsr.status}"
            )
        dsr.status = DSRStatus.COMPLETED
        dsr.action_taken = action_taken
        dsr.data_deleted = data_deleted
        dsr.exemptions_applied = exemptions_applied or []
        dsr.completed_at = _now()

        # Mark the subject record as erased if applicable
        if data_deleted and dsr.request_type == DSRType.ERASURE:
            subject = next(
                (s for s in self.state.data_subjects if s.id == dsr.subject_id),
                None,
            )
            if subject:
                subject.erased = True
                subject.erased_at = _now()

        return dsr

    @action(
        relevance=lambda s, e: any(
            r.status in (DSRStatus.SUBMITTED, DSRStatus.IN_PROGRESS)
            for r in s.data_subject_requests
        ),
    )
    def refuse_dsr(
        self,
        dsr_id: str,
        refusal_reason: str,
        exemptions_applied: Optional[list[str]] = None,
    ) -> DataSubjectRequest:
        """
        Record a refusal to action a Data Subject Request.

        A DSR may be refused where a lawful exemption applies (e.g. Art. 17(3):
        legal claims, legal obligation, public interest). The refusal_reason must
        clearly state the exemption applied and inform the individual of their
        right to complain to the ICO.

        Under UK GDPR Art. 12(4), refusals must be communicated within one month.
        """
        dsr = self._get_dsr(dsr_id)
        dsr.status = DSRStatus.REFUSED
        dsr.refusal_reason = refusal_reason
        dsr.exemptions_applied = exemptions_applied or []
        dsr.completed_at = _now()
        return dsr

    # ----------------------------------------------------------------------- #
    # Phase 5 — Data Breaches                                                  #
    # ----------------------------------------------------------------------- #

    @action(
        precondition=lambda s, e: s.status == EngagementStatus.CONTRACTED,
    )
    def report_breach(
        self,
        description: str,
        data_categories_affected: list[str],
        estimated_subjects_affected: int,
        discovered_at: datetime,
    ) -> DataBreach:
        """
        Report a personal data breach internally.

        Call this as soon as a breach is discovered. The 72-hour clock for
        ICO notification (UK GDPR Art. 33) starts from discovered_at.

        Follow with assess_breach() to determine notifiability. If notifiable,
        call notify_ico() — it must be within 72 hours of discovered_at.

        data_categories_affected: e.g. ["email addresses", "health records"]
        estimated_subjects_affected: best estimate of individuals affected
        """
        breach = DataBreach(
            description=description,
            data_categories_affected=data_categories_affected,
            estimated_subjects_affected=estimated_subjects_affected,
            discovered_at=discovered_at,
            reported_internally_at=_now(),
        )
        self.state.data_breaches.append(breach)
        return breach

    @action(
        relevance=lambda s, e: any(
            b.status == BreachStatus.REPORTED for b in s.data_breaches
        ),
    )
    def assess_breach(
        self,
        breach_id: str,
        severity: BreachSeverity,
        is_notifiable: bool,
        reasoning: str,
    ) -> DataBreach:
        """
        Record the outcome of the breach risk assessment.

        severity:
            LOW    — unlikely to result in risk; no ICO notification required
            MEDIUM — likely to result in risk; notify ICO within 72h (Art. 33)
            HIGH   — high risk to individuals; notify ICO AND subjects (Art. 34)

        is_notifiable: True if the breach must be reported to the ICO.
        reasoning: documented justification for the severity and notifiability decision.

        If is_notifiable is True, call notify_ico() within 72 hours of discovery.
        If severity is HIGH, also call notify_affected_subjects().
        """
        breach = self._get_breach(breach_id)
        breach.severity = severity
        breach.is_notifiable = is_notifiable
        breach.notifiability_reasoning = reasoning
        breach.status = BreachStatus.ASSESSED
        return breach

    @action(
        relevance=lambda s, e: any(
            b.is_notifiable and b.ico_notified_at is None
            for b in s.data_breaches
        ),
    )
    def notify_ico(
        self,
        breach_id: str,
        notification_details: str,
        ico_reference: Optional[str] = None,
    ) -> DataBreach:
        """
        Record that the ICO has been notified of a personal data breach.

        Under UK GDPR Art. 33, notifiable breaches must be reported to the ICO
        without undue delay and within 72 hours of discovery. A warning is
        raised if this call is made after the 72-hour window.

        notification_details: summary of what was reported to the ICO, including
            any reason for late notification if beyond 72 hours.
        ico_reference: the reference number provided by the ICO.
        """
        breach = self._get_breach(breach_id)
        if breach.status == BreachStatus.REPORTED:
            raise ValueError(
                f"Breach {breach_id} has not been assessed yet — call assess_breach() first"
            )

        hours_elapsed = (_now() - breach.discovered_at).total_seconds() / 3600
        if hours_elapsed > 72:
            warnings.warn(
                f"Breach {breach_id}: ICO notified {hours_elapsed:.1f}h after discovery "
                f"(72h threshold). Ensure the reason for delay is documented in "
                f"notification_details as required by UK GDPR Art. 33(1).",
                UserWarning,
                stacklevel=2,
            )

        breach.ico_notified_at = _now()
        breach.ico_reference = ico_reference
        breach.status = BreachStatus.ICO_NOTIFIED
        return breach

    @action(
        relevance=lambda s, e: any(
            b.severity == BreachSeverity.HIGH and b.subjects_notified_at is None
            for b in s.data_breaches
        ),
    )
    def notify_affected_subjects(
        self,
        breach_id: str,
        notification_method: str,
        notification_summary: str,
    ) -> DataBreach:
        """
        Record that affected data subjects have been notified of the breach.

        Required under UK GDPR Art. 34 when a breach is likely to result in a
        HIGH risk to the rights and freedoms of individuals. Must be done
        without undue delay.

        notification_method: how subjects were notified — e.g. "email", "letter",
            "prominent website notice".
        notification_summary: description of what was communicated, including
            the nature of the breach and recommended protective measures.
        """
        breach = self._get_breach(breach_id)
        if breach.severity != BreachSeverity.HIGH:
            raise ValueError(
                f"Subject notification is only required for HIGH severity breaches "
                f"(this breach severity is {breach.severity})"
            )
        breach.subjects_notified_at = _now()
        breach.subjects_notification_method = notification_method
        breach.status = BreachStatus.SUBJECTS_NOTIFIED
        return breach

    @action(
        relevance=lambda s, e: any(
            b.status != BreachStatus.RESOLVED for b in s.data_breaches
        ),
    )
    def resolve_breach(
        self,
        breach_id: str,
        remediation_steps: str,
        lessons_learned: str,
    ) -> DataBreach:
        """
        Mark a data breach as resolved after all required actions are complete.

        Will raise an error if the breach is notifiable but the ICO has not
        yet been notified.

        remediation_steps: concrete steps taken to contain the breach and
            prevent recurrence.
        lessons_learned: post-incident review findings to improve future response.
        """
        breach = self._get_breach(breach_id)
        if breach.is_notifiable and breach.ico_notified_at is None:
            raise ValueError(
                f"Breach {breach_id} is notifiable but the ICO has not been notified. "
                f"Call notify_ico() before resolving."
            )
        breach.remediation_steps = remediation_steps
        breach.lessons_learned = lessons_learned
        breach.status = BreachStatus.RESOLVED
        breach.resolved_at = _now()
        return breach

    # ----------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ----------------------------------------------------------------------- #

    def _get_policy(self, version: str) -> PrivacyPolicy:
        policy = next(
            (p for p in self.state.privacy_policies if p.version == version), None
        )
        if not policy:
            raise ValueError(f"No privacy policy found with version '{version}'")
        return policy

    def _get_dsr(self, dsr_id: str) -> DataSubjectRequest:
        dsr = next(
            (r for r in self.state.data_subject_requests if r.id == dsr_id), None
        )
        if not dsr:
            raise ValueError(f"No DSR found with id '{dsr_id}'")
        return dsr

    def _get_breach(self, breach_id: str) -> DataBreach:
        breach = next(
            (b for b in self.state.data_breaches if b.id == breach_id), None
        )
        if not breach:
            raise ValueError(f"No data breach found with id '{breach_id}'")
        return breach
