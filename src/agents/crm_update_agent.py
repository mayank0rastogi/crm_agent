"""
CRM Update Agent — Part 2 of the pipeline.

Responsibilities:
  1. Take the ExtractionResult from the Extraction Agent
  2. Load the current CRM record from the customer-centric store
  3. Propose CRM field updates
  4. Run guardrail checks (rule engine + validators)
  5. Flag low-confidence updates for human review
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.agents.base_agent import AgentResult, BaseAgent
from src.crm.customer_crm_store import CustomerCRMStore
from src.guardrails.rule_engine import RuleEngine
from src.guardrails.stage_transitions import STAGE_ORDER, StageTransitionValidator
from src.guardrails.validators import (
    validate_account_name_change,
    validate_no_stage_regression,
    validate_required_docs,
    validate_revenue_range,
)
from src.llm_mock.mock_llm import CRMUpdateProposal, ExtractionResult, MockLLM

HUMAN_REVIEW_THRESHOLD = 0.65
AUTO_APPLY_THRESHOLD = 0.80


class CRMUpdateAgent(BaseAgent):

    def __init__(
        self,
        llm: MockLLM,
        crm_store: CustomerCRMStore,
        rule_engine: RuleEngine,
        entity_graph: Optional[Dict] = None,
    ):
        super().__init__("CRMUpdateAgent")
        self.llm = llm
        self.crm = crm_store
        self.rules = rule_engine
        self.stage_validator = StageTransitionValidator()
        self.entity_graph = entity_graph or {}

    def run(
        self,
        extraction_result: ExtractionResult,
        wiki_context: List[Dict],
        customer_id: str,
        run_id: Optional[str] = None,
    ) -> AgentResult:
        result = self._new_result(run_id)
        start = time.time()

        try:
            crm = self.crm.get_crm(customer_id)
            if not crm:
                result.status = "error"
                result.error = f"No CRM record for customer '{customer_id}'."
                return result

            profile = self.crm.get_profile(customer_id) or {}
            company_name = crm.get("company_name", profile.get("company_name", ""))
            tier = profile.get("tier")

            result.add_step("crm_load", f"Loaded CRM for {company_name} ({customer_id})", {
                "stage": crm.get("stage"),
                "expected_revenue": crm.get("expected_revenue"),
                "conversion_likelihood": crm.get("conversion_likelihood"),
            })

            crm_view = {**crm, "account_name": company_name}

            proposals: List[CRMUpdateProposal] = self.llm.propose_crm_updates(
                extraction_result, crm_view, wiki_context
            )
            result.add_step("llm_proposals", f"{len(proposals)} proposals generated", [
                {"field": p.field_name, "new_value": p.new_value, "confidence": round(p.confidence, 2)}
                for p in proposals
            ])

            annotated_proposals = []
            guardrail_log = []

            # Build merged custom_fields BEFORE the guardrail loop.
            # This combines the current CRM state with ALL custom_fields proposed in
            # this call, so the stage validator sees the correct "will-be" state.
            _pre_proposed_cf = {
                p.field_name.replace("custom_fields.", ""): p.new_value
                for p in proposals if p.field_name.startswith("custom_fields.")
            }
            _pre_actions = next(
                (p.new_value for p in proposals if p.field_name == "follow_up_tasks"), []
            ) or []
            merged_cf = {
                **crm.get("custom_fields", {}),
                **_pre_proposed_cf,
                "security_questionnaire_sent": (
                    crm.get("custom_fields", {}).get("security_questionnaire_sent", False) or
                    _pre_proposed_cf.get("security_questionnaire_sent", False) or
                    "send_security_questionnaire" in _pre_actions
                ),
                "proposal_sent": (
                    crm.get("custom_fields", {}).get("proposal_sent", False) or
                    _pre_proposed_cf.get("proposal_sent", False) or
                    "send_proposal" in _pre_actions
                ),
            }

            for proposal in proposals:
                issues: List[str] = []
                blocked = False

                if proposal.field_name == "stage":
                    from_stage = crm.get("stage", "")
                    to_stage = proposal.new_value
                    valid, msg = self.stage_validator.validate(
                        from_stage, to_stage,
                        custom_fields=merged_cf,          # ← merged, not stale CRM
                        deal_value=crm.get("expected_revenue"),
                    )
                    if not valid:
                        issues.append(f"STAGE: {msg}")
                        blocked = True
                    skipped = self.stage_validator.stages_skipped(from_stage, to_stage)
                    if skipped:
                        issues.append(f"STAGES SKIPPED: {skipped}")
                    reg_ok, _, reg_msg = validate_no_stage_regression(from_stage, to_stage, STAGE_ORDER)
                    if not reg_ok:
                        issues.append(f"REGRESSION: {reg_msg}")
                    guardrail_log.append({"field": "stage", "from": from_stage, "to": to_stage, "valid": valid, "msg": msg})

                if proposal.field_name == "expected_revenue":
                    num_users = extraction_result.num_users.value if extraction_result.num_users else None
                    rev_ok, rev_sev, rev_msg = validate_revenue_range(proposal.new_value, tier=tier, num_users=num_users)
                    if not rev_ok:
                        issues.append(f"{'ERROR' if rev_sev == 'ERROR' else 'WARNING'}: {rev_msg}")
                        if rev_sev == "ERROR":
                            blocked = True
                    guardrail_log.append({"field": "expected_revenue", "valid": rev_ok, "msg": rev_msg})

                if proposal.field_name == "account_name":
                    old_name = company_name
                    name_ok, name_sev, name_msg = validate_account_name_change(
                        old_name, proposal.new_value, self.entity_graph
                    )
                    if not name_ok and name_sev == "ERROR":
                        issues.append(f"ACCOUNT NAME: {name_msg}")
                        blocked = True
                    elif not name_ok:
                        issues.append(f"ACCOUNT NAME WARNING: {name_msg}")
                    else:
                        wiki_ref = next(
                            (e["id"] for e in wiki_context
                             if e.get("category") == "company_rename"
                             and proposal.new_value.lower() in e.get("content", "").lower()),
                            None,
                        )
                        wiki_note = f" (wiki ref: {wiki_ref})" if wiki_ref else ""
                        proposal.reasoning = (
                            f"Confirmed by entity graph{wiki_note}: '{old_name}' is a known alias "
                            f"of '{proposal.new_value}'. Company rename validated with high confidence."
                        )
                        proposal.confidence = min(0.99, proposal.confidence + 0.10)
                    guardrail_log.append({"field": "account_name", "valid": name_ok, "msg": name_msg})

                    # Email domain mismatch check (when domain info available in entity graph)
                    new_canonical = proposal.new_value
                    for node in self.entity_graph.get("nodes", []):
                        if node.get("canonical_name", "").lower() == new_canonical.lower():
                            expected_domain = node.get("domain", "")
                            deprecated = node.get("deprecated_domains", [])
                            if expected_domain and deprecated:
                                issues.append(
                                    f"DOMAIN NOTE: '{old_name}' used deprecated domain(s) {deprecated}. "
                                    f"Ensure contacts updated to @{expected_domain}."
                                )
                            break

                if proposal.field_name == "stage":
                    doc_results = validate_required_docs(
                        proposal.new_value,
                        merged_cf,                        # ← merged, not stale CRM
                        deal_value=crm.get("expected_revenue"),
                    )
                    for doc_ok, doc_sev, doc_msg in doc_results:
                        if not doc_ok:
                            issues.append(f"REQUIRED DOCS: {doc_msg}")
                            if doc_sev == "ERROR":
                                blocked = True

                proposal.blocked_by_guardrail = blocked
                proposal.guardrail_message = "; ".join(issues)
                if proposal.confidence < HUMAN_REVIEW_THRESHOLD and not blocked:
                    proposal.requires_human_review = True
                annotated_proposals.append(proposal)

            result.add_step(
                "guardrail_checks",
                f"{sum(1 for p in annotated_proposals if p.blocked_by_guardrail)} blocked, "
                f"{sum(1 for p in annotated_proposals if p.requires_human_review)} need human review",
                guardrail_log,
            )

            new_stage = next((p.new_value for p in annotated_proposals if p.field_name == "stage"), crm.get("stage"))
            new_revenue = next((p.new_value for p in annotated_proposals if p.field_name == "expected_revenue"), crm.get("expected_revenue", 0))
            concerns = extraction_result.customer_concerns.value if extraction_result.customer_concerns else []
            actions = next((p.new_value for p in annotated_proposals if p.field_name == "follow_up_tasks"), [])

            # Collect all custom_fields proposed values (current CRM + this call's proposals)
            proposed_cf = {
                p.field_name.replace("custom_fields.", ""): p.new_value
                for p in annotated_proposals
                if p.field_name.startswith("custom_fields.") and not p.blocked_by_guardrail
            }

            rule_context = {
                "proposed_stage": new_stage,
                "expected_revenue": new_revenue,
                "custom_fields": {
                    **crm.get("custom_fields", {}),
                    **proposed_cf,   # proposed values override current CRM values
                    # Also count "send_security_questionnaire" in follow_up actions
                    "security_questionnaire_sent": (
                        crm.get("custom_fields", {}).get("security_questionnaire_sent", False) or
                        proposed_cf.get("security_questionnaire_sent", False) or
                        "send_security_questionnaire" in (actions or [])
                    ),
                    "proposal_sent": (
                        crm.get("custom_fields", {}).get("proposal_sent", False) or
                        proposed_cf.get("proposal_sent", False) or
                        "send_proposal" in (actions or [])
                    ),
                },
                "account_name_changed": any(p.field_name == "account_name" for p in annotated_proposals),
                "account_name_validated": all(
                    not (p.field_name == "account_name" and p.blocked_by_guardrail)
                    for p in annotated_proposals
                ),
                "concerns_contains_data_residency": "data_residency" in (concerns or []),
            }

            rule_results = self.rules.evaluate(rule_context)
            rule_summary = self.rules.summary(rule_results)

            result.add_step(
                "rule_engine",
                f"{rule_summary['passed']}/{rule_summary['total_rules_evaluated']} rules passed, "
                f"{rule_summary['errors']} errors",
                rule_summary,
            )

            if rule_summary["blocking"]:
                blocking_rules = [r for r in rule_results if not r.passed and r.severity == "ERROR"]
                for proposal in annotated_proposals:
                    if proposal.field_name == "stage":
                        if any("security" in r.rule_name.lower() or "proposal" in r.rule_name.lower()
                               for r in blocking_rules):
                            proposal.blocked_by_guardrail = True
                            proposal.guardrail_message += " " + "; ".join(r.message for r in blocking_rules)

            result.output = {
                "proposals": annotated_proposals,
                "rule_summary": rule_summary,
                "current_crm": crm,
                "customer_id": customer_id,
            }
            result.status = "success"

        except Exception as exc:
            import traceback
            result.status = "error"
            result.error = f"{exc}\n{traceback.format_exc()}"

        result.duration_ms = (time.time() - start) * 1000
        return result

