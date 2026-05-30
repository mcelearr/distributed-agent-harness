"""
Tests for the DataProtectionWorldEnvironment example.

Covers the full GDPR engagement lifecycle:
pitch → contract → privacy policy → DSR → breach.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.eventlog import InMemoryEventLog
from distributed_agent_harness.world import ActionNotAvailable

from examples.use_cases.data_protection.models import (
    BreachSeverity,
    BreachStatus,
    DSRStatus,
    DSRType,
    EngagementStatus,
    PolicyStatus,
)
from examples.use_cases.data_protection.world import DataProtectionWorldEnvironment


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def env() -> DataProtectionWorldEnvironment:
    return DataProtectionWorldEnvironment(
        project_id="test-dp",
        namespace=InMemoryNamespace(),
        eventlog=InMemoryEventLog(),
    )


@pytest.fixture
def contracted_env(env: DataProtectionWorldEnvironment) -> DataProtectionWorldEnvironment:
    """Env already in CONTRACTED state with a client and contract."""
    env.submit_pitch(
        client_name="Test Client Ltd",
        contact_name="Alice",
        contact_email="alice@test.example",
        industry="Healthcare",
        scope_of_work="GDPR audit",
    )
    env.win_pitch(terms_summary="Standard terms", retainer_fee_gbp=2000.0)
    return env


# --------------------------------------------------------------------------- #
# Phase 1 — Pitch & Contract                                                   #
# --------------------------------------------------------------------------- #

class TestEngagement:
    def test_submit_pitch_sets_client(self, env: DataProtectionWorldEnvironment) -> None:
        client = env.submit_pitch(
            client_name="Acme",
            contact_name="Bob",
            contact_email="bob@acme.example",
            industry="Retail",
            scope_of_work="DPO retainer",
        )
        assert env.state.client is not None
        assert env.state.client.name == "Acme"
        assert env.state.status == EngagementStatus.PITCH
        assert client.name == "Acme"

    def test_win_pitch_creates_contract(self, env: DataProtectionWorldEnvironment) -> None:
        env.submit_pitch("X", "Y", "y@x.com", "Tech", "Audit")
        contract = env.win_pitch(terms_summary="Monthly retainer")
        assert env.state.status == EngagementStatus.CONTRACTED
        assert env.state.contract is not None
        assert env.state.contract.dpa_included is True
        assert contract.signed_date is not None

    def test_lose_pitch_closes_engagement(self, env: DataProtectionWorldEnvironment) -> None:
        env.submit_pitch("X", "Y", "y@x.com", "Tech", "Audit")
        env.lose_pitch(reason="Client selected a competitor")
        assert env.state.status == EngagementStatus.CLOSED

    def test_win_pitch_without_submit_raises(self, env: DataProtectionWorldEnvironment) -> None:
        with pytest.raises(ValueError, match="No client"):
            env.win_pitch(terms_summary="Terms")

    def test_win_pitch_when_already_contracted_raises(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        # The PITCH-state guard is enforced by show_when on the @action,
        # so a contracted env raises ActionNotAvailable rather than ValueError.
        with pytest.raises(ActionNotAvailable, match="win_pitch"):
            contracted_env.win_pitch(terms_summary="Terms again")


# --------------------------------------------------------------------------- #
# Phase 2 — Privacy Policy                                                     #
# --------------------------------------------------------------------------- #

class TestPrivacyPolicy:
    def test_draft_creates_policy(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        policy = contracted_env.draft_privacy_policy(
            version="1.0",
            content="Policy text",
            data_categories=["name", "email"],
            processing_purposes=["service delivery"],
            retention_periods={"email": "3 years"},
        )
        assert policy.status == PolicyStatus.DRAFT
        assert len(contracted_env.state.privacy_policies) == 1

    def test_approve_policy(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        contracted_env.draft_privacy_policy(
            version="1.0", content="...", data_categories=[],
            processing_purposes=[], retention_periods={},
        )
        policy = contracted_env.approve_privacy_policy("1.0", approved_by="Partner A")
        assert policy.status == PolicyStatus.APPROVED
        assert policy.approved_by == "Partner A"
        assert policy.approved_at is not None

    def test_publish_policy(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        contracted_env.draft_privacy_policy(
            version="1.0", content="...", data_categories=[],
            processing_purposes=[], retention_periods={},
        )
        contracted_env.approve_privacy_policy("1.0", approved_by="Partner A")
        policy = contracted_env.publish_privacy_policy("1.0")
        assert policy.status == PolicyStatus.PUBLISHED
        assert policy.published_at is not None

    def test_cannot_draft_before_contract(
        self, env: DataProtectionWorldEnvironment
    ) -> None:
        # The CONTRACTED-state guard is enforced by show_when on the @action.
        with pytest.raises(ActionNotAvailable, match="draft_privacy_policy"):
            env.draft_privacy_policy(
                version="1.0", content="...", data_categories=[],
                processing_purposes=[], retention_periods={},
            )

    def test_cannot_publish_unapproved_policy(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        contracted_env.draft_privacy_policy(
            version="1.0", content="...", data_categories=[],
            processing_purposes=[], retention_periods={},
        )
        with pytest.raises(ValueError, match="APPROVED"):
            contracted_env.publish_privacy_policy("1.0")

    def test_unknown_version_raises(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        with pytest.raises(ValueError, match="version '9.9'"):
            contracted_env.approve_privacy_policy("9.9", approved_by="X")


# --------------------------------------------------------------------------- #
# Phase 3 — Data Subjects                                                      #
# --------------------------------------------------------------------------- #

class TestDataSubjects:
    def test_register_data_subject(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        subject = contracted_env.register_data_subject(
            name="Bob Jones",
            email="bob@jones.example",
            data_categories_held=["name", "email"],
            lawful_basis="contract",
        )
        assert len(contracted_env.state.data_subjects) == 1
        assert subject.id is not None
        assert subject.erased is False


# --------------------------------------------------------------------------- #
# Phase 4 — Data Subject Requests                                              #
# --------------------------------------------------------------------------- #

class TestDSR:
    @pytest.fixture
    def env_with_subject(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> DataProtectionWorldEnvironment:
        contracted_env.register_data_subject(
            name="Bob Jones",
            email="bob@jones.example",
            data_categories_held=["name", "email"],
            lawful_basis="contract",
        )
        return contracted_env

    def test_submit_dsr_sets_deadline(
        self, env_with_subject: DataProtectionWorldEnvironment
    ) -> None:
        dsr = env_with_subject.submit_dsr(
            subject_email="bob@jones.example",
            request_type=DSRType.ERASURE,
            description="Delete my data",
        )
        assert dsr.status == DSRStatus.SUBMITTED
        assert dsr.deadline is not None
        delta = dsr.deadline - dsr.submitted_at
        assert 29 <= delta.days <= 30

    def test_acknowledge_dsr(
        self, env_with_subject: DataProtectionWorldEnvironment
    ) -> None:
        dsr_id = env_with_subject.submit_dsr("bob@jones.example", DSRType.ACCESS, "SAR").id
        env_with_subject.acknowledge_dsr(dsr_id)
        # Re-read from state — _hydrate() replaces self.state on each @action call
        dsr = next(r for r in env_with_subject.state.data_subject_requests if r.id == dsr_id)
        assert dsr.status == DSRStatus.IN_PROGRESS
        assert dsr.acknowledged_at is not None

    def test_complete_erasure_dsr_marks_subject_erased(
        self, env_with_subject: DataProtectionWorldEnvironment
    ) -> None:
        dsr_id = env_with_subject.submit_dsr(
            "bob@jones.example", DSRType.ERASURE, "Delete me"
        ).id
        env_with_subject.complete_dsr(
            dsr_id=dsr_id,
            action_taken="All data deleted",
            data_deleted=True,
        )
        dsr = next(r for r in env_with_subject.state.data_subject_requests if r.id == dsr_id)
        assert dsr.status == DSRStatus.COMPLETED
        subject = env_with_subject.state.data_subjects[0]
        assert subject.erased is True
        assert subject.erased_at is not None

    def test_refuse_dsr(
        self, env_with_subject: DataProtectionWorldEnvironment
    ) -> None:
        dsr_id = env_with_subject.submit_dsr(
            "bob@jones.example", DSRType.ERASURE, "Delete me"
        ).id
        env_with_subject.refuse_dsr(
            dsr_id=dsr_id,
            refusal_reason="Data retained for legal claims defence (Art. 17(3)(e))",
            exemptions_applied=["legal claims defence"],
        )
        dsr = next(r for r in env_with_subject.state.data_subject_requests if r.id == dsr_id)
        assert dsr.status == DSRStatus.REFUSED
        assert dsr.refusal_reason is not None

    def test_unknown_dsr_raises(
        self, env_with_subject: DataProtectionWorldEnvironment
    ) -> None:
        # show_when on acknowledge_dsr requires at least one SUBMITTED DSR,
        # so we first submit one to make the action callable; then the
        # in-method lookup raises ValueError on an unknown id.
        env_with_subject.submit_dsr(
            "bob@jones.example", DSRType.ACCESS, "SAR"
        )
        with pytest.raises(ValueError, match="DSR"):
            env_with_subject.acknowledge_dsr("nonexistent")


# --------------------------------------------------------------------------- #
# Phase 5 — Data Breaches                                                      #
# --------------------------------------------------------------------------- #

class TestSummaryDoc:
    def test_summary_reflects_engagement_progress(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        summary = contracted_env._namespace.read_doc("test-dp/summary.md")
        assert summary is not None
        # Client + scope appear
        assert "Test Client Ltd" in summary
        assert "GDPR audit" in summary
        # Status is contracted
        assert "contracted" in summary.lower()
        # Sections exist
        assert "## Contract" in summary
        assert "## Privacy Policy" in summary

    def test_summary_flags_open_dsrs(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        contracted_env.register_data_subject(
            name="Bob",
            email="bob@example.com",
            data_categories_held=["email"],
            lawful_basis="consent",
        )
        contracted_env.submit_dsr("bob@example.com", DSRType.ERASURE, "delete me")
        summary = contracted_env._namespace.read_doc("test-dp/summary.md")
        assert summary is not None
        assert "Data Subject Requests (1)" in summary
        assert "open DSR" in summary

    def test_summary_flags_unresolved_breach(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        contracted_env.report_breach(
            description="bucket leak",
            data_categories_affected=["email"],
            estimated_subjects_affected=10,
            discovered_at=_now(),
        )
        summary = contracted_env._namespace.read_doc("test-dp/summary.md")
        assert summary is not None
        assert "Data Breaches (1)" in summary
        assert "open breach" in summary


class TestDataBreach:
    @pytest.fixture
    def env_with_breach(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> DataProtectionWorldEnvironment:
        contracted_env.report_breach(
            description="S3 bucket misconfiguration exposed email addresses",
            data_categories_affected=["email addresses"],
            estimated_subjects_affected=500,
            discovered_at=_now(),
        )
        return contracted_env

    def test_report_breach(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        breach = contracted_env.report_breach(
            description="Test breach",
            data_categories_affected=["email"],
            estimated_subjects_affected=100,
            discovered_at=_now(),
        )
        assert breach.status == BreachStatus.REPORTED
        assert len(contracted_env.state.data_breaches) == 1

    def test_assess_breach(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        breach_id = env_with_breach.state.data_breaches[0].id
        env_with_breach.assess_breach(
            breach_id=breach_id,
            severity=BreachSeverity.MEDIUM,
            is_notifiable=True,
            reasoning="Email addresses exposed — likely risk",
        )
        breach = next(b for b in env_with_breach.state.data_breaches if b.id == breach_id)
        assert breach.status == BreachStatus.ASSESSED
        assert breach.is_notifiable is True

    def test_notify_ico(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        breach_id = env_with_breach.state.data_breaches[0].id
        env_with_breach.assess_breach(breach_id, BreachSeverity.MEDIUM, True, "Reasoning")
        env_with_breach.notify_ico(
            breach_id=breach_id,
            notification_details="Reported via ICO portal",
            ico_reference="ICO-TEST-001",
        )
        breach = next(b for b in env_with_breach.state.data_breaches if b.id == breach_id)
        assert breach.status == BreachStatus.ICO_NOTIFIED
        assert breach.ico_reference == "ICO-TEST-001"
        assert breach.ico_notified_at is not None

    def test_cannot_notify_ico_without_assessment(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        # show_when on notify_ico requires a notifiable, unnotified breach;
        # an unassessed breach has is_notifiable=None, so the action is
        # hidden and uncallable until assess_breach() runs.
        breach = env_with_breach.state.data_breaches[0]
        with pytest.raises(ActionNotAvailable, match="notify_ico"):
            env_with_breach.notify_ico(breach.id, "Details")

    def test_late_ico_notification_raises_warning(
        self, contracted_env: DataProtectionWorldEnvironment
    ) -> None:
        discovered = _now() - timedelta(hours=80)  # 80h ago — past 72h window
        breach = contracted_env.report_breach(
            description="Old breach",
            data_categories_affected=["names"],
            estimated_subjects_affected=10,
            discovered_at=discovered,
        )
        contracted_env.assess_breach(breach.id, BreachSeverity.MEDIUM, True, "Risk")
        with pytest.warns(UserWarning, match="72h"):
            contracted_env.notify_ico(breach.id, "Late notification with reason")

    def test_notify_affected_subjects_requires_high_severity(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        # show_when on notify_affected_subjects requires a HIGH-severity
        # breach with no prior subject notification; a MEDIUM breach is
        # not visible to the agent at all.
        breach = env_with_breach.state.data_breaches[0]
        env_with_breach.assess_breach(
            breach.id, BreachSeverity.MEDIUM, True, "Medium risk"
        )
        env_with_breach.notify_ico(breach.id, "Notified ICO")
        with pytest.raises(ActionNotAvailable, match="notify_affected_subjects"):
            env_with_breach.notify_affected_subjects(
                breach.id, "email", "We had a breach"
            )

    def test_resolve_breach(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        breach_id = env_with_breach.state.data_breaches[0].id
        env_with_breach.assess_breach(breach_id, BreachSeverity.MEDIUM, True, "Risk reasoning")
        env_with_breach.notify_ico(breach_id, "ICO notified", "ICO-REF-001")
        env_with_breach.resolve_breach(
            breach_id=breach_id,
            remediation_steps="Bucket locked down",
            lessons_learned="Implement automated S3 audits",
        )
        breach = next(b for b in env_with_breach.state.data_breaches if b.id == breach_id)
        assert breach.status == BreachStatus.RESOLVED
        assert breach.resolved_at is not None

    def test_cannot_resolve_notifiable_breach_without_ico_notification(
        self, env_with_breach: DataProtectionWorldEnvironment
    ) -> None:
        breach = env_with_breach.state.data_breaches[0]
        env_with_breach.assess_breach(
            breach.id, BreachSeverity.MEDIUM, True, "Risk"
        )
        with pytest.raises(ValueError, match="ICO has not been notified"):
            env_with_breach.resolve_breach(breach.id, "Steps", "Lessons")
