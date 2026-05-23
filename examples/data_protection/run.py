"""
Data protection demo — walk through the full GDPR engagement lifecycle.

Run with:
    python -m examples.data_protection.run
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.concurrency_handlers import InProcessLock
from distributed_agent_harness.prompt_builder import PromptBuilder

from .models import BreachSeverity, DSRType
from .world import DataProtectionWorldEnvironment


def main() -> None:
    # ------------------------------------------------------------------ #
    # Bootstrap the harness with zero-dependency in-memory backends       #
    # ------------------------------------------------------------------ #
    namespace = InMemoryNamespace()
    concurrency = InProcessLock()

    env = DataProtectionWorldEnvironment(
        project_id="acme-gdpr-2025",
        namespace=namespace,
        concurrency=concurrency,
    )

    # ------------------------------------------------------------------ #
    # Show the generated agent prompt                                      #
    # ------------------------------------------------------------------ #
    builder = PromptBuilder(DataProtectionWorldEnvironment, include_source=False)
    print("=" * 70)
    print("GENERATED AGENT PROMPT (actions only)")
    print("=" * 70)
    print(builder.build_actions_prompt())
    print()

    # ------------------------------------------------------------------ #
    # Phase 1: Win the engagement                                          #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("PHASE 1 — Pitch & Contract")
    print("=" * 70)

    env.submit_pitch(
        client_name="Acme Corp Ltd",
        contact_name="Jane Smith",
        contact_email="jane.smith@acme.example",
        industry="E-commerce",
        scope_of_work="Full GDPR compliance audit and ongoing outsourced DPO retainer",
    )
    print("✓ Pitch submitted")

    contract = env.win_pitch(
        terms_summary=(
            "Monthly DPO retainer at £3,500/month. Client responsible for "
            "providing access to internal systems. 3-month notice period."
        ),
        retainer_fee_gbp=3500.0,
    )
    print(f"✓ Contract signed: {contract.signed_date.strftime('%Y-%m-%d %H:%M UTC')}")

    # ------------------------------------------------------------------ #
    # Phase 2: Privacy Policy                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 2 — Privacy Policy")
    print("=" * 70)

    env.draft_privacy_policy(
        version="1.0",
        content=(
            "Acme Corp Ltd Privacy Policy v1.0\n\n"
            "1. Who we are: Acme Corp Ltd, a UK-registered company...\n"
            "2. What data we collect: [full text omitted for brevity]\n"
        ),
        data_categories=["name", "email address", "postal address", "purchase history", "IP address"],
        processing_purposes=["order fulfilment", "customer support", "marketing", "fraud prevention"],
        retention_periods={
            "purchase history": "7 years",
            "email address": "3 years after last purchase",
            "IP address": "12 months",
        },
    )
    print("✓ Privacy policy v1.0 drafted")

    env.approve_privacy_policy(version="1.0", approved_by="Senior Partner — Data Protection")
    print("✓ Privacy policy v1.0 approved")

    env.publish_privacy_policy(version="1.0")
    print("✓ Privacy policy v1.0 published (live)")

    # ------------------------------------------------------------------ #
    # Phase 3: Register data subjects                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 3 — Data Subjects")
    print("=" * 70)

    bob = env.register_data_subject(
        name="Bob Johnson",
        email="bob.johnson@example.com",
        data_categories_held=["name", "email address", "purchase history"],
        lawful_basis="contract",
    )
    print(f"✓ Registered Bob Johnson (id: {bob.id})")

    env.register_data_subject(
        name="Alice Williams",
        email="alice.w@example.com",
        data_categories_held=["name", "email address", "purchase history", "IP address"],
        lawful_basis="legitimate interests",
    )
    print("✓ Registered Alice Williams")

    # ------------------------------------------------------------------ #
    # Phase 4: Data Subject Request — erasure                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 4 — Data Subject Request (right to erasure)")
    print("=" * 70)

    dsr = env.submit_dsr(
        subject_email="bob.johnson@example.com",
        request_type=DSRType.ERASURE,
        description="Please delete all personal data you hold about me.",
    )
    print(f"✓ DSR submitted (id: {dsr.id}, deadline: {dsr.deadline.strftime('%Y-%m-%d')})")

    env.acknowledge_dsr(dsr_id=dsr.id)
    print("✓ DSR acknowledged")

    env.complete_dsr(
        dsr_id=dsr.id,
        action_taken=(
            "All personal data for bob.johnson@example.com deleted from CRM, "
            "email marketing platform, and order management system. "
            "Record suppressed to prevent re-addition to marketing lists."
        ),
        data_deleted=True,
    )
    print(f"✓ DSR completed — data deleted: {dsr.data_deleted}")

    # ------------------------------------------------------------------ #
    # Phase 5: Data Breach                                                 #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 5 — Data Breach")
    print("=" * 70)

    discovered = datetime.now(timezone.utc)
    breach = env.report_breach(
        description=(
            "Misconfigured AWS S3 bucket exposed a CSV export of customer email "
            "addresses publicly for approximately 4 hours before being detected."
        ),
        data_categories_affected=["email addresses"],
        estimated_subjects_affected=1_247,
        discovered_at=discovered,
    )
    print(f"✓ Breach reported (id: {breach.id})")

    env.assess_breach(
        breach_id=breach.id,
        severity=BreachSeverity.MEDIUM,
        is_notifiable=True,
        reasoning=(
            "Email addresses exposed. Likely to result in risk (spam, phishing). "
            "ICO notification required within 72h under UK GDPR Art. 33."
        ),
    )
    print(f"✓ Breach assessed: severity={breach.severity}, notifiable={breach.is_notifiable}")

    env.notify_ico(
        breach_id=breach.id,
        notification_details=(
            "Notified via ICO self-reporting portal. Described: nature of breach, "
            "categories and volume of data affected, likely consequences, and "
            "remediation measures taken."
        ),
        ico_reference="ICO-2025-BR-00142",
    )
    print(f"✓ ICO notified (ref: {breach.ico_reference})")

    env.resolve_breach(
        breach_id=breach.id,
        remediation_steps=(
            "1. S3 bucket access restricted immediately upon discovery. "
            "2. S3 Block Public Access setting enabled org-wide. "
            "3. Full security audit of all S3 buckets completed. "
            "4. Automated misconfiguration alerts deployed via AWS Config."
        ),
        lessons_learned=(
            "All S3 bucket configuration changes now require dual approval. "
            "Monthly automated S3 ACL audit added to security runbook."
        ),
    )
    print(f"✓ Breach resolved")

    # ------------------------------------------------------------------ #
    # Show persisted documents                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PERSISTED NAMESPACE DOCUMENTS")
    print("=" * 70)
    for path in namespace.list_docs():
        print(f"  {path}")

    print("\n" + "=" * 70)
    print("AUDIT LOG")
    print("=" * 70)
    audit_raw = namespace.read_doc("acme-gdpr-2025/audit.jsonl")
    if audit_raw:
        for line in audit_raw.strip().split("\n"):
            entry = json.loads(line)
            args_str = ", ".join(entry.get("args", []))
            status = "ERROR" if "error" in entry else "OK"
            print(f"  [{status}] {entry['timestamp']}  {entry['method']}({args_str})")


if __name__ == "__main__":
    main()
