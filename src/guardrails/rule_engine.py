"""
Declarative Rule Engine for CRM guardrails.

Rules are loaded from a JSON config and evaluated against the pipeline state.
This decouples business logic from code — new rules can be added without deployment.

Rule structure:
{
  "id": "rule_001",
  "name": "Security before proposal",
  "condition": {"field": "proposed_stage", "operator": "in", "value": ["Proposal", "Negotiation"]},
  "check": {"field": "custom_fields.security_questionnaire_sent", "operator": "eq", "value": true},
  "severity": "ERROR",
  "message": "Security questionnaire must be sent before advancing to Proposal."
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Rule data model
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str
    name: str
    condition: Dict       # When to apply this rule
    check: Dict           # What to verify
    severity: str         # ERROR | WARNING | INFO
    message: str
    wiki_ref: str = ""    # Reference to wiki entry that defines this rule
    enabled: bool = True


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    passed: bool
    severity: str
    message: str
    wiki_ref: str = ""


# ---------------------------------------------------------------------------
# Built-in rules (also loadable from JSON)
# ---------------------------------------------------------------------------

BUILTIN_RULES: List[Dict] = [
    {
        "id": "rule_001",
        "name": "Security questionnaire before proposal",
        "condition": {
            "field": "proposed_stage",
            "operator": "in",
            "value": ["Proposal", "Negotiation", "Closed Won"],
        },
        "check": {
            "field": "custom_fields.security_questionnaire_sent",
            "operator": "eq",
            "value": True,
        },
        "severity": "ERROR",
        "message": "Security questionnaire must be sent before advancing to Proposal/Negotiation stage.",
        "wiki_ref": "wiki_003",
        "enabled": True,
    },
    {
        "id": "rule_002",
        "name": "Enterprise deal requires security review",
        "condition": {
            "field": "expected_revenue",
            "operator": "gt",
            "value": 100000,
        },
        "check": {
            "field": "custom_fields.security_review_required",
            "operator": "eq",
            "value": True,
        },
        "severity": "ERROR",
        "message": "Enterprise opportunities above $100,000 require Security Review.",
        "wiki_ref": "wiki_004",
        "enabled": True,
    },
    {
        "id": "rule_003",
        "name": "Account name should match entity graph",
        "condition": {
            "field": "account_name_changed",
            "operator": "eq",
            "value": True,
        },
        "check": {
            "field": "account_name_validated",
            "operator": "eq",
            "value": True,
        },
        "severity": "WARNING",
        "message": "Account name change should be validated against entity knowledge graph.",
        "wiki_ref": "wiki_001",
        "enabled": True,
    },
    {
        "id": "rule_004",
        "name": "Proposal sent before negotiation",
        "condition": {
            "field": "proposed_stage",
            "operator": "in",
            "value": ["Negotiation", "Closed Won"],
        },
        "check": {
            "field": "custom_fields.proposal_sent",
            "operator": "eq",
            "value": True,
        },
        "severity": "ERROR",
        "message": "Proposal must be sent before advancing to Negotiation or Closed Won.",
        "wiki_ref": "",
        "enabled": True,
    },
    {
        "id": "rule_005",
        "name": "EU data residency must be flagged",
        "condition": {
            "field": "concerns_contains_data_residency",
            "operator": "eq",
            "value": True,
        },
        "check": {
            "field": "custom_fields.data_residency_requirement",
            "operator": "not_null",
            "value": None,
        },
        "severity": "ERROR",
        "message": "EU data residency requirement must be flagged in CRM custom fields.",
        "wiki_ref": "wiki_006",
        "enabled": True,
    },
    {
        "id": "rule_006",
        "name": "Revenue within reasonable range",
        "condition": {
            "field": "expected_revenue",
            "operator": "gt",
            "value": 0,
        },
        "check": {
            "field": "expected_revenue",
            "operator": "lte",
            "value": 10_000_000,
        },
        "severity": "ERROR",
        "message": "Expected revenue exceeds $10M — likely a data entry error.",
        "wiki_ref": "",
        "enabled": True,
    },
]


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Evaluates declarative rules against a pipeline context dictionary.

    The context is a flat/nested dict with keys like:
      - proposed_stage
      - expected_revenue
      - custom_fields.security_questionnaire_sent
      - account_name_changed
      - concerns_contains_data_residency
    """

    def __init__(self, extra_rules_path: Optional[str] = None):
        self.rules: List[Rule] = [Rule(**r) for r in BUILTIN_RULES]

        if extra_rules_path and os.path.exists(extra_rules_path):
            with open(extra_rules_path, "r") as f:
                extra = json.load(f)
            for r in extra:
                self.rules.append(Rule(**r))

    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate all enabled rules against the given context."""
        results: List[RuleResult] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check if this rule's condition applies
            if not self._evaluate_condition(rule.condition, context):
                # Condition not met — rule doesn't apply, skip
                continue

            # Condition applies — now evaluate the check
            check_passed = self._evaluate_condition(rule.check, context)

            results.append(RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=check_passed,
                severity=rule.severity if not check_passed else "INFO",
                message=rule.message if not check_passed else f"✓ {rule.name}",
                wiki_ref=rule.wiki_ref,
            ))

        return results

    def has_blocking_errors(self, results: List[RuleResult]) -> bool:
        """Return True if any ERROR-severity rule failed."""
        return any(not r.passed and r.severity == "ERROR" for r in results)

    def get_failures(self, results: List[RuleResult]) -> List[RuleResult]:
        return [r for r in results if not r.passed]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_field(self, field_path: str, context: Dict) -> Any:
        """Resolve a dot-notation field path in the context dict."""
        parts = field_path.split(".")
        val = context
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return None
        return val

    def _evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """Evaluate a single condition/check against the context."""
        field_path = condition.get("field", "")
        operator = condition.get("operator", "eq")
        expected = condition.get("value")

        actual = self._get_field(field_path, context)

        if operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return actual is not None and actual > expected
        elif operator == "gte":
            return actual is not None and actual >= expected
        elif operator == "lt":
            return actual is not None and actual < expected
        elif operator == "lte":
            return actual is not None and actual <= expected
        elif operator == "in":
            return actual in (expected or [])
        elif operator == "not_in":
            return actual not in (expected or [])
        elif operator == "not_null":
            return actual is not None and actual != "" and actual != "Unknown"
        elif operator == "is_null":
            return actual is None or actual == "" or actual == "Unknown"
        elif operator == "contains":
            return expected in (actual or "")
        else:
            return False

    def add_rule(self, rule_dict: Dict) -> None:
        """Dynamically add a new rule at runtime."""
        self.rules.append(Rule(**rule_dict))

    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                rule.enabled = False

    def summary(self, results: List[RuleResult]) -> Dict:
        """Return a summary of rule evaluation results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errors = [r for r in results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in results if not r.passed and r.severity == "WARNING"]

        return {
            "total_rules_evaluated": total,
            "passed": passed,
            "failed": total - passed,
            "errors": len(errors),
            "warnings": len(warnings),
            "blocking": self.has_blocking_errors(results),
            "details": [
                {
                    "rule_id": r.rule_id,
                    "name": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                    "wiki_ref": r.wiki_ref,
                }
                for r in results
            ],
        }
