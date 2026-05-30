"""
Pydantic domain models for the data protection law firm example.

These represent the entities that make up the state of a GDPR engagement:
clients, contracts, privacy policies, data subjects, subject requests, and breaches.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    """Generate a short unique ID."""
    return str(uuid.uuid4())[:8]


# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #

class EngagementStatus(str, Enum):
    PITCH = "pitch"
    CONTRACTED = "contracted"
    CLOSED = "closed"


class PolicyStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    PUBLISHED = "published"


class DSRType(str, Enum):
    """UK GDPR individual rights that can be exercised via a Data Subject Request."""
    ERASURE = "erasure"              # Right to be forgotten — Art. 17
    ACCESS = "access"                # Subject Access Request — Art. 15
    PORTABILITY = "portability"      # Right to data portability — Art. 20
    RECTIFICATION = "rectification"  # Right to rectification — Art. 16
    RESTRICTION = "restriction"      # Right to restriction of processing — Art. 18


class DSRStatus(str, Enum):
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REFUSED = "refused"


class BreachSeverity(str, Enum):
    LOW = "low"        # Unlikely to result in risk — no ICO notification required
    MEDIUM = "medium"  # Likely to result in risk — must notify ICO (Art. 33)
    HIGH = "high"      # High risk to individuals — notify ICO AND subjects (Art. 34)


class BreachStatus(str, Enum):
    REPORTED = "reported"
    ASSESSED = "assessed"
    ICO_NOTIFIED = "ico_notified"
    SUBJECTS_NOTIFIED = "subjects_notified"
    RESOLVED = "resolved"


# --------------------------------------------------------------------------- #
# Domain models                                                                #
# --------------------------------------------------------------------------- #

class Client(BaseModel):
    name: str
    contact_name: str
    contact_email: str
    industry: str
    scope_of_work: str


class Contract(BaseModel):
    terms_summary: str
    retainer_fee_gbp: Optional[float] = None
    signed_date: datetime
    dpa_included: bool = True   # Data Processing Agreement included as standard
    review_date: Optional[datetime] = None


class PrivacyPolicy(BaseModel):
    id: str = Field(default_factory=_new_id)
    version: str
    content: str
    data_categories: list[str]           # e.g. ["name", "email", "health data"]
    processing_purposes: list[str]        # e.g. ["payroll", "marketing"]
    retention_periods: dict[str, str]     # e.g. {"email address": "3 years"}
    status: PolicyStatus = PolicyStatus.DRAFT
    created_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    published_at: Optional[datetime] = None


class DataSubject(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    email: str
    data_categories_held: list[str]
    lawful_basis: str   # UK GDPR Art. 6 basis, e.g. "consent", "contract"
    registered_at: datetime
    erased: bool = False
    erased_at: Optional[datetime] = None


class DataSubjectRequest(BaseModel):
    id: str = Field(default_factory=_new_id)
    subject_id: str
    subject_email: str
    request_type: DSRType
    description: str
    status: DSRStatus = DSRStatus.SUBMITTED
    submitted_at: datetime
    acknowledged_at: Optional[datetime] = None
    deadline: Optional[datetime] = None       # 30 days from submission — UK GDPR Art. 12(3)
    completed_at: Optional[datetime] = None
    action_taken: Optional[str] = None
    data_deleted: bool = False
    exemptions_applied: list[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None


class DataBreach(BaseModel):
    id: str = Field(default_factory=_new_id)
    description: str
    data_categories_affected: list[str]
    estimated_subjects_affected: int
    discovered_at: datetime
    reported_internally_at: datetime
    severity: Optional[BreachSeverity] = None
    is_notifiable: Optional[bool] = None          # Must notify ICO if True
    notifiability_reasoning: Optional[str] = None
    status: BreachStatus = BreachStatus.REPORTED
    # ICO notification — must be within 72 hours of discovery (Art. 33)
    ico_notified_at: Optional[datetime] = None
    ico_reference: Optional[str] = None
    # Subject notification — required for HIGH severity breaches (Art. 34)
    subjects_notified_at: Optional[datetime] = None
    subjects_notification_method: Optional[str] = None
    # Resolution
    remediation_steps: Optional[str] = None
    lessons_learned: Optional[str] = None
    resolved_at: Optional[datetime] = None
