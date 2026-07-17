"""
Review Agent — Part 3 of the pipeline (Critique / Second Opinion).

Responsibilities:
  1. Receive the proposed CRM updates from CRMUpdateAgent
  2. Critically evaluate each proposal for:
     - Hallucinated values (not supported by transcript evidence)
     - Missing evidence
     - Violated business rules
     - Inconsistent company information
     - Missing required follow-up actions
  3. Issue a verdict per proposal: APPROVED | REVISED | REJECTED | NEEDS_HUMAN_REVIEW
  4. Produce an overall verdict: APPROVED | APPROVED_WITH_WARNINGS | REVISED | REJECTED
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.agents.base_agent import AgentResult, BaseAgent
from src.llm_mock.mock_llm import CRMUpdateProposal, ExtractionResult, MockLLM


# Confidence below this → reviewer downgrades to NEEDS_HUMAN_REVIEW
REVIEW_CONFIDENCE_THRESHOLD = 0.65

# Fields that MUST have transcript evidence
EVIDENCE_REQUIRED_FIELDS = {
    "account_name", "stage", "expected_revenue",
    "close_date", "custom_fields.num_users",
    "custom_fields.data_residency_requirement",
}


class ReviewAgent(BaseAgent):
    """
    Independent critique agent that evaluates the CRMUpdateAgent's proposals.

    Acts as a second-pass quality gate before updates are applied.
    """

    def __init__(self, llm: MockLLM):
        super().__init__("ReviewAgent")
        self.llm = llm

    def run(
        self,
        proposals: List[CRMUpdateProposal],
        extraction: ExtractionResult,
        transcript: str,
        wiki_context: List[Dict],
        run_id: Optional[str] = None,
    ) -> AgentResult:
        result = self._new_result(run_id)
        start = time.time()

        try:
            result.add_step(
                "review_start",
                f"Reviewing {len(proposals)} proposals",
                {"proposal_fields": [p.field_name for p in proposals]},
            )

            # Run the mock LLM's review logic
            llm_review = self.llm.review_proposals(
                proposals, extraction, transcript, wiki_context
            )

            # Build a per-proposal verdict map
            per_proposal_map = {
                v["field"]: v for v in llm_review.get("per_proposal", [])
            }

            final_proposals = []
            review_notes = list(llm_review.get("issues_found", []))

            for proposal in proposals:
                field = proposal.field_name
                llm_verdict = per_proposal_map.get(field, {})
                verdict = llm_verdict.get("verdict", "APPROVED")
                issues = llm_verdict.get("issues", [])

                # Additional deterministic checks

                # 1. Blocked by guardrail → REJECTED (non-negotiable)
                if proposal.blocked_by_guardrail:
                    verdict = "REJECTED"
                    issues.append(f"Blocked by guardrail: {proposal.guardrail_message}")

                # 2. Already flagged for human review
                elif proposal.requires_human_review:
                    verdict = "NEEDS_HUMAN_REVIEW"
                    issues.append(
                        f"Low confidence ({proposal.confidence:.2f}) — flagged for human review."
                    )

                # 3. Required evidence fields must have evidence
                elif field in EVIDENCE_REQUIRED_FIELDS and not proposal.evidence:
                    verdict = "REJECTED"
                    issues.append(f"No supporting evidence for '{field}' update.")

                # 4. Hallucinated flag from LLM
                #    (Mock LLM can set proposal.evidence to a fake value)
                if field == "expected_revenue" and proposal.new_value:
                    import re
                    # Check both raw (375000) and comma-formatted (375,000) forms
                    rev_raw = str(int(proposal.new_value))
                    rev_formatted = f"{int(proposal.new_value):,}"
                    found_in_transcript = (
                        re.search(r'\$?\s*' + re.escape(rev_raw), transcript) or
                        re.search(r'\$?\s*' + re.escape(rev_formatted), transcript)
                    )
                    if not found_in_transcript:
                        if verdict not in ("REJECTED",):
                            issues.append(
                                f"Revenue ${proposal.new_value:,} not found verbatim in transcript — "
                                "possible hallucination. Flagging for human review."
                            )
                            verdict = "NEEDS_HUMAN_REVIEW"

                # Map verdict to status used by CRMStore
                status_map = {
                    "APPROVED": "APPROVED",
                    "REVISED": "REVISED",
                    "REJECTED": "REJECTED",
                    "NEEDS_HUMAN_REVIEW": "NEEDS_HUMAN_REVIEW",
                    "APPROVED_WITH_WARNINGS": "APPROVED",
                }

                final_proposals.append({
                    "field": field,
                    "old_value": proposal.old_value,
                    "new_value": proposal.new_value,
                    "confidence": proposal.confidence,
                    "verdict": verdict,
                    "status": status_map.get(verdict, "APPROVED"),
                    "issues": issues,
                    "reasoning": proposal.reasoning,
                    "evidence": proposal.evidence,
                    "guardrail_blocked": proposal.blocked_by_guardrail,
                    "guardrail_message": proposal.guardrail_message,
                })

            # Overall verdict
            verdicts = [p["verdict"] for p in final_proposals]
            if any(v == "REJECTED" for v in verdicts):
                overall = "APPROVED_WITH_REJECTIONS"
            elif any(v == "NEEDS_HUMAN_REVIEW" for v in verdicts):
                overall = "NEEDS_HUMAN_REVIEW"
            elif review_notes:
                overall = "APPROVED_WITH_WARNINGS"
            else:
                overall = "APPROVED"

            result.add_step(
                "review_verdict",
                f"Overall verdict: {overall}",
                {
                    "overall": overall,
                    "approved": sum(1 for p in final_proposals if p["verdict"] in ("APPROVED", "REVISED")),
                    "rejected": sum(1 for p in final_proposals if p["verdict"] == "REJECTED"),
                    "human_review": sum(1 for p in final_proposals if p["verdict"] == "NEEDS_HUMAN_REVIEW"),
                    "issues": review_notes,
                },
            )

            result.output = {
                "overall_verdict": overall,
                "final_proposals": final_proposals,
                "review_notes": review_notes,
                "reviewer": self.name,
            }
            result.status = "success"

        except Exception as exc:
            import traceback
            result.status = "error"
            result.error = f"{exc}\n{traceback.format_exc()}"

        result.duration_ms = (time.time() - start) * 1000
        return result
