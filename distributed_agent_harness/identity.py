"""
AgentIdentity — first-class identity for every actor that touches a project.

Threaded through the system in four places:

- ``TriggerEvent.identity``         — the principal that fired the run
- ``ActionContext.identity``        — passed to every hook
- ``SubagentContext.identity``      — passed to subagent hooks
- ``Event.identity``                — stamped on every audit-log row

Identity is deliberately structured (not a free-form string) so that:

- Policy hooks can branch on ``principal`` / ``roles`` without parsing.
- The audit log records who did what at the granularity the deployment
  defines (a single ``agent`` string is enough for a demo; a
  ``human:rorymcelearney@gmail.com#<instance_id>`` is what a regulated
  deployment needs).
- A cryptographic spine (``pubkey_fingerprint``) can be added by the
  surrounding deployment without changing any application code — DAH itself
  stays agnostic about *how* the identity was established.

Backwards compatibility
-----------------------
``Event.actor: str`` is preserved as a derived field — when an
``AgentIdentity`` is supplied it is rendered to a stable string via
``identity.label``; when only the legacy ``actor`` string is supplied the
``identity`` field stays ``None``. Existing event-log search by ``actor``
keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentIdentity:
    """One actor's identity claim.

    Attributes
    ----------
    principal:
        Stable identifier for *who* this is — e.g. ``"agent"``,
        ``"human:rory@example.com"``, ``"subagent:legal_research"``. Defaults
        to ``"agent"`` to keep zero-config use cases simple.
    instance_id:
        Optional unique id for this *running instance* — distinguishes two
        copies of the same logical principal. Recommended to be a uuid hex.
        When set, ``label`` includes the first eight characters.
    pubkey_fingerprint:
        Hex SHA-256 of the credential's public-key material (or any other
        cryptographic root the deployment chooses). DAH does not verify it;
        deployments that care plug verification into a ``pre_trigger`` hook.
    roles:
        Tuple of role names for RBAC. Hook authors can compare against
        this directly.
    """
    principal: str = "agent"
    instance_id: str | None = None
    pubkey_fingerprint: str | None = None
    roles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """Compact string form for logs and ``Event.actor`` derivation.

        Format: ``principal[#instance-id-prefix]``. Stable, sortable,
        grep-friendly. The 8-char id prefix is enough to distinguish
        co-tenant instances without bloating every event-log line.
        """
        if self.instance_id:
            return f"{self.principal}#{self.instance_id[:8]}"
        return self.principal

    def has_role(self, role: str) -> bool:
        """Convenience for policy hooks."""
        return role in self.roles

    @classmethod
    def anonymous_agent(cls) -> "AgentIdentity":
        """Default identity used when no caller-supplied identity exists.

        Carries no instance_id and no roles — equivalent to the legacy
        ``actor="agent"`` string. Use only for tests and single-actor demos.
        """
        return cls(principal="agent")
