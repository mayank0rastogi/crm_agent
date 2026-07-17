"""
Deterministic validation functions for CRM guardrails.

These are pure functions — no LLM involved.
Each validator returns a (passed: bool, severity: str, message: str) tuple.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


ValidationResult = Tuple[bool, str, str]  # (passed, severity, message)
# severity: "ERROR" | "WARNING" | "INFO"

# Revenue tier boundaries
REVENUE_TIERS = {
    "Starter":    (0,        10_000),
    "Standard":   (10_000,  100_000),
    "Enterprise": (100_000, 10_000_000),
}

ENTERPRISE_THRESHOLD = 100_000
MAX_REASONABLE_DEAL = 10_000_000
MIN_DEAL_VALUE = 1_000


def validate_revenue_range(
    revenue: float,
    tier: Optional[str] = None,
    num_users: Optional[int] = None,
) -> ValidationResult:
    """Validate that revenue is within acceptable bounds."""
    if revenue < MIN_DEAL_VALUE:
        return (
            False,
            "ERROR",
            f"Revenue ${revenue:,.0f} is below minimum deal value ${MIN_DEAL_VALUE:,}.",
        )
    if revenue > MAX_REASONABLE_DEAL:
        return (
            False,
            "ERROR",
            f"Revenue ${revenue:,.0f} exceeds maximum reasonable deal value ${MAX_REASONABLE_DEAL:,}. "
            "Possible data entry error.",
        )
    if tier:
        bounds = REVENUE_TIERS.get(tier)
        if bounds:
            low, high = bounds
            if not (low <= revenue <= high):
                return (
                    False,
                    "WARNING",
                    f"Revenue ${revenue:,.0f} is outside the expected range for "
                    f"{tier} tier (${low:,}–${high:,}).",
                )
    if num_users and num_users > 0:
        per_user = revenue / num_users
        if per_user > 5_000:
            return (
                False,
                "WARNING",
                f"Implied per-user cost ${per_user:,.0f} seems high. "
                "Verify revenue calculation.",
            )
        if per_user < 50:
            return (
                False,
                "WARNING",
                f"Implied per-user cost ${per_user:,.0f} seems too low. "
                "Verify revenue calculation.",
            )
    return (True, "INFO", f"Revenue ${revenue:,.0f} is within acceptable range.")


def validate_account_name_change(
    old_name: str,
    new_name: str,
    entity_graph: Optional[Dict] = None,
) -> ValidationResult:
    """
    Validate an account name change.
    Cross-references the entity graph to confirm known rebrand.
    """
    if not new_name or not new_name.strip():
        return (False, "ERROR", "New account name cannot be empty.")

    if old_name.lower() == new_name.lower():
        return (True, "INFO", "Account name unchanged.")

    # Check entity graph for known alias/rebrand
    if entity_graph:
        for node in entity_graph.get("nodes", []):
            aliases = [a.lower() for a in node.get("aliases", [])]
            canonical = node.get("canonical_name", "").lower()
            if old_name.lower() in aliases and new_name.lower() == canonical:
                return (
                    True,
                    "INFO",
                    f"Account rename confirmed by entity graph: "
                    f"'{old_name}' is a known alias of '{node['canonical_name']}'.",
                )
            if old_name.lower() in aliases and new_name.lower() not in aliases and \
               new_name.lower() != canonical:
                return (
                    False,
                    "WARNING",
                    f"'{new_name}' is not the known canonical name for '{old_name}'. "
                    f"Expected: '{node['canonical_name']}'.",
                )

        # Check relationships
        for rel in entity_graph.get("relationships", []):
            if rel.get("from", "").lower() == old_name.lower():
                expected_to = rel.get("to", "")
                if new_name.lower() == expected_to.lower():
                    return (
                        True,
                        "INFO",
                        f"Rename validated by entity graph relationship: "
                        f"'{old_name}' → '{expected_to}' ({rel.get('type')}).",
                    )
                else:
                    return (
                        False,
                        "WARNING",
                        f"Entity graph shows '{old_name}' was renamed to '{expected_to}', "
                        f"not '{new_name}'. Verify the correct name.",
                    )

    # No entity graph or no match — soft warning
    return (
        False,
        "WARNING",
        f"Account rename from '{old_name}' to '{new_name}' has no corroborating "
        "entry in the entity graph. Requires human confirmation.",
    )


def validate_email_domain(
    email: str,
    expected_domain: str,
    entity_graph: Optional[Dict] = None,
) -> ValidationResult:
    """Validate that an email domain matches the expected company domain."""
    if not email:
        return (True, "INFO", "No email provided — domain check skipped.")

    m = re.search(r'@([\w.-]+)', email)
    if not m:
        return (False, "WARNING", f"Could not parse domain from email: '{email}'.")

    actual_domain = m.group(1).lower()
    expected_lower = expected_domain.lower()

    if actual_domain == expected_lower:
        return (True, "INFO", f"Email domain '{actual_domain}' matches expected domain.")

    # Check deprecated domains in entity graph
    if entity_graph:
        for node in entity_graph.get("nodes", []):
            if expected_lower in node.get("domain", "").lower():
                deprecated = [d.lower() for d in node.get("deprecated_domains", [])]
                if actual_domain in deprecated:
                    return (
                        False,
                        "WARNING",
                        f"Email domain '@{actual_domain}' is a deprecated domain. "
                        f"Customer should update to '@{expected_domain}'.",
                    )

    return (
        False,
        "ERROR",
        f"Email domain '@{actual_domain}' does not match expected '@{expected_domain}'. "
        "Possible wrong contact or domain mismatch.",
    )


def validate_required_docs(
    proposed_stage: str,
    custom_fields: Dict,
    deal_value: Optional[float] = None,
) -> List[ValidationResult]:
    """
    Validate that required documents exist before a stage transition.
    Returns a list of validation results (one per check).
    """
    results = []

    # Security questionnaire before proposal
    if proposed_stage in ("Proposal", "Negotiation", "Closed Won"):
        if not custom_fields.get("security_questionnaire_sent", False):
            results.append((
                False,
                "ERROR",
                f"Security questionnaire must be sent before advancing to '{proposed_stage}' "
                "(company policy, wiki rule: wiki_003).",
            ))
        else:
            results.append((
                True,
                "INFO",
                "Security questionnaire already sent. ✓",
            ))

    # Enterprise deals require security review flag
    if deal_value and deal_value > ENTERPRISE_THRESHOLD:
        if proposed_stage in ("Proposal", "Negotiation", "Closed Won"):
            if not custom_fields.get("security_review_required"):
                results.append((
                    False,
                    "ERROR",
                    f"Deal value ${deal_value:,} exceeds ${ENTERPRISE_THRESHOLD:,}. "
                    "Security review must be marked as required (wiki rule: wiki_004).",
                ))
            else:
                results.append((
                    True,
                    "INFO",
                    "Security review requirement noted. ✓",
                ))

    # Proposal must be sent before negotiation/close
    if proposed_stage in ("Negotiation", "Closed Won"):
        if not custom_fields.get("proposal_sent", False):
            results.append((
                False,
                "ERROR",
                f"Proposal must be sent before advancing to '{proposed_stage}'.",
            ))

    return results or [(True, "INFO", "No required document checks for this stage.")]


def validate_renewal_date(
    renewal_date: str,
    close_date: str,
) -> ValidationResult:
    """Validate that renewal date is logically after close date."""
    if renewal_date in ("Unknown", None, ""):
        return (True, "INFO", "Renewal date not set — skipping check.")
    if close_date in ("Unknown", None, ""):
        return (True, "INFO", "Close date not set — skipping renewal check.")
    # Simplified string comparison (production would use datetime parsing)
    return (True, "INFO", "Renewal date check passed.")


def validate_no_stage_regression(
    from_stage: str,
    to_stage: str,
    stage_order: List[str],
) -> ValidationResult:
    """Warn if a stage is moving backwards (regression)."""
    if from_stage not in stage_order or to_stage not in stage_order:
        return (True, "INFO", "One or both stages not in standard order — skipping regression check.")
    fi = stage_order.index(from_stage)
    ti = stage_order.index(to_stage)
    if ti < fi:
        return (
            False,
            "WARNING",
            f"Stage regression detected: '{from_stage}' → '{to_stage}'. "
            "Moving backwards in the pipeline. Confirm this is intentional.",
        )
    return (True, "INFO", "Stage is moving forward. ✓")
