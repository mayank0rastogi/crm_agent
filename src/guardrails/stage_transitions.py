"""
Stage transition rules and valid CRM pipeline states.

This is a pure-Python deterministic module — no LLM involved.
All transition logic is encoded as data, making it auditable and testable.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

# Ordered list of forward-moving stages
STAGE_ORDER: List[str] = [
    "Qualification",
    "Discovery",
    "Demo",
    "Proposal",
    "Negotiation",
    "Closed Won",
]

TERMINAL_STAGES: List[str] = ["Closed Won", "Closed Lost"]

# Allowed transitions: {from_stage: [allowed_to_stages]}
VALID_TRANSITIONS: Dict[str, List[str]] = {
    "Qualification":  ["Discovery", "Closed Lost"],
    "Discovery":      ["Demo", "Proposal", "Closed Lost"],
    "Demo":           ["Proposal", "Discovery", "Closed Lost"],
    "Proposal":       ["Negotiation", "Demo", "Closed Lost"],
    "Negotiation":    ["Closed Won", "Proposal", "Closed Lost"],
    "Closed Won":     [],  # Terminal — no further transitions
    "Closed Lost":    ["Qualification"],  # Re-open is allowed with justification
}

# Stages that require a security review (for enterprise deals)
REQUIRES_SECURITY_REVIEW: List[str] = ["Proposal", "Negotiation", "Closed Won"]

# Required fields before entering a stage
STAGE_ENTRY_REQUIREMENTS: Dict[str, List[str]] = {
    "Proposal": [
        "security_questionnaire_sent",
        "security_review_required_check",
    ],
    "Negotiation": [
        "proposal_sent",
    ],
    "Closed Won": [
        "proposal_sent",
    ],
}


# ---------------------------------------------------------------------------
# Transition validator
# ---------------------------------------------------------------------------

class StageTransitionValidator:
    """
    Validates CRM stage transitions deterministically.
    Returns a structured result with the reason for any failure.
    """

    def validate(
        self,
        from_stage: str,
        to_stage: str,
        custom_fields: Optional[Dict] = None,
        deal_value: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Validate a stage transition.

        Returns
        -------
        (is_valid: bool, message: str)
        """
        custom_fields = custom_fields or {}

        # Unknown stages
        all_stages = list(VALID_TRANSITIONS.keys()) + ["Closed Lost"]
        if from_stage not in all_stages:
            return False, f"Unknown source stage: '{from_stage}'"
        if to_stage not in all_stages:
            return False, f"Unknown target stage: '{to_stage}'"

        # No-op transition
        if from_stage == to_stage:
            return True, "No stage change."

        # Terminal stage block
        if from_stage in TERMINAL_STAGES and to_stage not in VALID_TRANSITIONS.get(from_stage, []):
            return False, (
                f"Stage '{from_stage}' is terminal. "
                f"Cannot transition to '{to_stage}' without explicit re-open."
            )

        # Check allowed transitions
        allowed = VALID_TRANSITIONS.get(from_stage, [])
        if to_stage not in allowed:
            return False, (
                f"Invalid transition: '{from_stage}' → '{to_stage}'. "
                f"Allowed: {allowed}"
            )

        # Check stage-entry requirements
        entry_reqs = STAGE_ENTRY_REQUIREMENTS.get(to_stage, [])
        for req in entry_reqs:
            if req == "security_questionnaire_sent":
                if not custom_fields.get("security_questionnaire_sent", False):
                    return False, (
                        f"Cannot advance to '{to_stage}': "
                        "security questionnaire has not been sent (required by wiki rule)."
                    )
            elif req == "security_review_required_check":
                if deal_value and deal_value > 100_000:
                    if not custom_fields.get("security_review_required", False):
                        return False, (
                            f"Cannot advance to '{to_stage}': "
                            f"deal value ${deal_value:,} exceeds $100k — "
                            "security review must be marked as required."
                        )
            elif req == "proposal_sent":
                if not custom_fields.get("proposal_sent", False):
                    return False, (
                        f"Cannot advance to '{to_stage}': proposal has not been sent."
                    )

        return True, f"Transition '{from_stage}' → '{to_stage}' is valid."

    def get_next_valid_stages(self, current_stage: str) -> List[str]:
        """Return all valid next stages from the current one."""
        return VALID_TRANSITIONS.get(current_stage, [])

    def is_forward_move(self, from_stage: str, to_stage: str) -> bool:
        """Return True if this is a forward progression in the pipeline."""
        if from_stage not in STAGE_ORDER or to_stage not in STAGE_ORDER:
            return False
        return STAGE_ORDER.index(to_stage) > STAGE_ORDER.index(from_stage)

    def stages_skipped(self, from_stage: str, to_stage: str) -> List[str]:
        """Return any stages that would be skipped in this transition."""
        if from_stage not in STAGE_ORDER or to_stage not in STAGE_ORDER:
            return []
        fi = STAGE_ORDER.index(from_stage)
        ti = STAGE_ORDER.index(to_stage)
        if ti <= fi:
            return []
        return STAGE_ORDER[fi + 1: ti]
