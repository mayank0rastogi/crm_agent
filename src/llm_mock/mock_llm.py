"""
Mock LLM — simulates Large Language Model behavior without any external API.

Design philosophy:
  - Pattern matching + weighted keyword scoring for entity extraction
  - Decision trees for classification (stage, intent, sentiment)
  - Template-based response generation for explainability
  - Configurable hallucination injection to stress-test the Review Agent
  - Confidence scoring per field (0.0–1.0)

In production: swap MockLLM for any real LLM client by implementing BaseLLM.
"""

from __future__ import annotations

import re
import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMField:
    """A single extracted or generated field with provenance."""
    value: Any
    confidence: float          # 0.0 – 1.0
    evidence: str              # snippet from transcript supporting this value
    reasoning: str             # explanation of why this value was chosen
    hallucinated: bool = False # True if injected for testing


@dataclass
class ExtractionResult:
    """Structured output from the extraction step."""
    company_name: Optional[LLMField] = None
    contact_names: Optional[LLMField] = None
    num_users: Optional[LLMField] = None
    revenue_estimate: Optional[LLMField] = None
    timeline: Optional[LLMField] = None
    close_deadline: Optional[LLMField] = None
    deal_duration_years: Optional[LLMField] = None
    renewal_date: Optional[LLMField] = None
    buying_intent: Optional[LLMField] = None      # HIGH / MEDIUM / LOW / LOST
    opportunity_stage: Optional[LLMField] = None
    requested_actions: Optional[LLMField] = None   # list of strings
    customer_concerns: Optional[LLMField] = None   # list of strings
    sentiment: Optional[LLMField] = None
    meeting_summary: Optional[LLMField] = None
    raw_fields: Dict[str, LLMField] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for attr in [
            "company_name", "contact_names", "num_users", "revenue_estimate",
            "timeline", "close_deadline", "deal_duration_years", "renewal_date",
            "buying_intent", "opportunity_stage",
            "requested_actions", "customer_concerns", "sentiment", "meeting_summary",
        ]:
            f = getattr(self, attr)
            if f is not None:
                result[attr] = {
                    "value": f.value,
                    "confidence": round(f.confidence, 3),
                    "evidence": f.evidence,
                    "reasoning": f.reasoning,
                    "hallucinated": f.hallucinated,
                }
        return result


@dataclass
class CRMUpdateProposal:
    """A single proposed change to a CRM field."""
    field_name: str
    old_value: Any
    new_value: Any
    confidence: float
    reasoning: str
    evidence: str
    requires_human_review: bool = False
    blocked_by_guardrail: bool = False
    guardrail_message: str = ""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Interface that any LLM implementation must satisfy."""

    @abstractmethod
    def extract_entities(self, transcript: str, context: Dict) -> ExtractionResult:
        ...

    @abstractmethod
    def propose_crm_updates(
        self,
        extraction: ExtractionResult,
        current_crm: Dict,
        wiki_context: List[Dict],
    ) -> List[CRMUpdateProposal]:
        ...

    @abstractmethod
    def review_proposals(
        self,
        proposals: List[CRMUpdateProposal],
        extraction: ExtractionResult,
        transcript: str,
        wiki_context: List[Dict],
    ) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Pattern library — building blocks for mock extraction
# ---------------------------------------------------------------------------

class _Patterns:
    """Compiled regex patterns used across extraction tasks."""

    COMPANY_NAME = [
        r"(?:we(?:'re| are)|our company(?:'s)?|from)\s+([A-Z][A-Za-z\s&]+?)(?:\s+in\b|\s+has\b|\s+would|\.|,|\s+is\b)",
        r"listed\s+as\s+([A-Z][A-Za-z\s&]+?)\s+in\s+your\s+system",
        r"rebranded\s+(?:from\s+)?([A-Z][A-Za-z\s&]+?)\s+to\s+([A-Z][A-Za-z\s&]+)",
        r"\bI(?:'m| am)\s+(?:calling\s+from\s+|with\s+)([A-Z][A-Za-z\s&]+?)(?:\.|,|\s+and\b)",
    ]

    USER_COUNT = [
        r"(\d[\d,]*)\s*(?:users?|seats?|licenses?|employees?|people)",
        r"(?:scaling|scale)\s+to\s+(?:around\s+)?(\d[\d,]*)\s*(?:users?|seats?)",
        r"(?:team|workforce)\s+of\s+(\d[\d,]*)",
    ]

    REVENUE = [
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:k\b|thousand\b)?",
        r"([\d,]+(?:\.\d+)?)\s*(?:USD|dollars)",
        r"budget(?:\s+is|\s+of)?\s+(?:around\s+)?\$\s*([\d,]+)",
        r"([\d,]+)\s*(?:k|K)\b",
    ]

    TIMELINE = [
        r"(Q[1-4](?:\s+\d{4}|\s+next\s+year|\s+this\s+year)?)",
        r"by\s+(end\s+of\s+(?:Q[1-4]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*)\s*\d{0,4})",
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)",
        r"within\s+([\d]+\s+(?:weeks?|months?))",
        r"(next\s+(?:month|quarter|year))",
    ]

    CLOSE_DATE = [
        # Explicit date with day number
        r"([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+at\s+the\s+latest)",
        r"deadline\s+(?:is\s+)?([A-Z][a-z]+\s+\d{1,2})",
        r"by\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
        r"no\s+later\s+than\s+([\w\s,]+?)(?:\.|$)",
        # "by end of <MonthName>" — only actual months, not 'this week'
        r"(?:signed|closed|done)\s+by\s+(end\s+of\s+(?:January|February|March|April|May|June"
        r"|July|August|September|October|November|December)(?:\s+\d{4})?)",
        r"by\s+(end\s+of\s+(?:January|February|March|April|May|June"
        r"|July|August|September|October|November|December)(?:\s+\d{4})?)",
    ]

    REBRAND = [
        r"rebranded?\s+from\s+([A-Z][A-Za-z\s&]+?)\s+to\s+([A-Z][A-Za-z\s&]+)",
        r"([A-Z][A-Za-z\s&]+?)\s+(?:is\s+now|has\s+been\s+renamed\s+to)\s+([A-Z][A-Za-z\s&]+)",
        r"you(?:'re|\s+are)\s+(?:still\s+)?(?:listing|recording|using)\s+(?:us\s+as\s+)?([A-Z][A-Za-z\s&]+)",
    ]

    # Action keywords → standard action codes
    ACTION_KEYWORDS = {
        "pricing": "send_pricing_details",
        "price": "send_pricing_details",
        "security document": "send_security_document",
        "security questionnaire": "send_security_questionnaire",
        "security doc": "send_security_document",
        "proposal": "send_proposal",
        "contract": "send_contract",
        "demo": "schedule_demo",
        "follow.up": "schedule_follow_up_call",
        "nda": "send_nda",
        "gdpr": "add_gdpr_clause",
        "data residency": "flag_data_residency_requirement",
        "eu region": "flag_data_residency_requirement",
        "integration": "confirm_integration_timeline",
    }

    CONCERN_KEYWORDS = {
        "integration": "integration_gaps",
        "data residency": "data_residency",
        "eu region": "data_residency",
        "security": "security_concerns",
        "pricing": "price_sensitivity",
        "competitor": "competitive_threat",
        "timeline": "timeline_pressure",
        "budget": "budget_constraint",
        "gdpr": "compliance_requirement",
    }

    STAGE_SIGNALS = {
        "Qualification": [
            "interested in", "looking at", "considering", "evaluating options",
            "initial", "outreach", "first meeting", "exploratory",
        ],
        "Discovery": [
            "told us about", "shared their needs", "requirements", "pain points",
            "current challenges", "scale to", "q1", "q2", "q3", "q4",
            "budget approved", "pricing details", "500 users", "500 employees",
            "modernise", "migrate", "before peak season",
        ],
        "Demo": [
            "demo", "showed", "walkthrough", "presentation", "trial",
        ],
        "Proposal": [
            "formal proposal", "send proposal", "proposal before", "pricing proposal",
            "quote", "ready to move to", "proposal drafted", "have the proposal",
            "comfortable with", "agreed on", "ready for proposal", "move to a formal",
            "proposal to you", "send over the proposal",
        ],
        "Negotiation": [
            "negotiate", "counter", "terms", "contract review", "legal",
            "pricing discussion", "final terms",
        ],
        "Closed Won": [
            "signed", "closed", "deal done", "contract signed", "moving forward",
            "purchase order", "signed contract", "board signed off",
            "board approved", "payment will be cleared", "kick off onboarding",
            "onboarding", "3-year", "confirmed on our end",
        ],
        "Closed Lost": [
            "went with a competitor", "chose another", "not moving forward",
            "lost the deal", "no longer interested", "decided against",
        ],
    }

    BUYING_INTENT = {
        "HIGH": [
            "budget approved", "ready to move", "want to close", "moving forward",
            "april 1st", "hard deadline", "q1", "purchase", "sign",
        ],
        "MEDIUM": [
            "evaluating", "considering", "interested", "looking at", "might",
            "could work", "pending approval",
        ],
        "LOW": [
            "not sure", "exploring", "early stages", "just browsing",
            "no timeline", "no budget",
        ],
        "LOST": [
            "went with a competitor", "chose another vendor", "no longer",
            "lost", "not moving forward",
        ],
    }

    SENTIMENT = {
        "positive": [
            "great", "excellent", "impressed", "excited", "moving forward",
            "strongest", "best", "love", "perfect", "happy", "appreciate",
        ],
        "neutral": [
            "ok", "fine", "maybe", "considering", "evaluating", "let's see",
        ],
        "negative": [
            "disappointed", "concerned", "not sure", "competitor", "lower price",
            "couldn't commit", "went with", "lost", "direct",
        ],
    }


# ---------------------------------------------------------------------------
# Mock LLM implementation
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """
    Simulates LLM behavior using pattern matching, keyword scoring,
    and decision trees.

    Parameters
    ----------
    hallucination_rate : float
        Probability (0–1) of injecting a hallucinated field per run.
        Default 0.0 for production-like runs; set > 0 to stress-test reviewer.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Print internal reasoning steps.
    """

    def __init__(
        self,
        hallucination_rate: float = 0.0,
        seed: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.hallucination_rate = hallucination_rate
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)
        self._p = _Patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, transcript: str, context: Dict) -> ExtractionResult:
        """Extract structured entities from a meeting transcript."""
        text = transcript
        result = ExtractionResult()

        result.company_name = self._extract_company(text, context)
        result.contact_names = self._extract_contacts(text, context)
        result.num_users = self._extract_num_users(text)
        result.revenue_estimate = self._extract_revenue(text, context)
        result.timeline = self._extract_timeline(text)
        result.close_deadline = self._extract_close_date(text)
        result.deal_duration_years = self._extract_deal_duration(text)
        result.renewal_date = self._extract_renewal_date(text)
        result.buying_intent = self._classify_buying_intent(text)
        result.opportunity_stage = self._classify_stage(text, context)
        result.requested_actions = self._extract_actions(text)
        result.customer_concerns = self._extract_concerns(text)
        result.sentiment = self._classify_sentiment(text)
        result.meeting_summary = self._generate_summary(text, result)

        # Optionally inject a hallucination to test the reviewer
        if random.random() < self.hallucination_rate:
            result = self._inject_hallucination(result, text)

        return result

    def propose_crm_updates(
        self,
        extraction: ExtractionResult,
        current_crm: Dict,
        wiki_context: List[Dict],
    ) -> List[CRMUpdateProposal]:
        """Generate CRM update proposals based on extracted entities."""
        proposals: List[CRMUpdateProposal] = []

        # Build a searchable text from all extraction evidence
        # (transcript is not passed here — reconstruct from extraction fields)
        evidence_text = " ".join(filter(None, [
            extraction.meeting_summary.value if extraction.meeting_summary else "",
            extraction.buying_intent.evidence if extraction.buying_intent else "",
            extraction.revenue_estimate.evidence if extraction.revenue_estimate else "",
            extraction.opportunity_stage.evidence if extraction.opportunity_stage else "",
            extraction.num_users.evidence if extraction.num_users else "",
            extraction.timeline.evidence if extraction.timeline else "",
            extraction.customer_concerns.evidence if extraction.customer_concerns else "",
            extraction.requested_actions.evidence if extraction.requested_actions else "",
        ]))
        text_lower = evidence_text.lower()

        # Account name update
        if extraction.company_name and extraction.company_name.value:
            new_name = extraction.company_name.value
            old_name = current_crm.get("account_name", current_crm.get("name", ""))
            if new_name.lower() != str(old_name).lower():
                proposals.append(CRMUpdateProposal(
                    field_name="account_name",
                    old_value=old_name,
                    new_value=new_name,
                    confidence=extraction.company_name.confidence,
                    reasoning=extraction.company_name.reasoning,
                    evidence=extraction.company_name.evidence,
                ))

        # Stage update
        if extraction.opportunity_stage and extraction.opportunity_stage.value:
            new_stage = extraction.opportunity_stage.value
            old_stage = current_crm.get("stage", "")
            if new_stage != old_stage:
                proposals.append(CRMUpdateProposal(
                    field_name="stage",
                    old_value=old_stage,
                    new_value=new_stage,
                    confidence=extraction.opportunity_stage.confidence,
                    reasoning=extraction.opportunity_stage.reasoning,
                    evidence=extraction.opportunity_stage.evidence,
                ))

        # Revenue update
        if extraction.revenue_estimate and extraction.revenue_estimate.value:
            new_rev = extraction.revenue_estimate.value
            old_rev = current_crm.get("expected_revenue", 0)
            if abs(new_rev - old_rev) > 1000:  # Only update if meaningfully different
                proposals.append(CRMUpdateProposal(
                    field_name="expected_revenue",
                    old_value=old_rev,
                    new_value=new_rev,
                    confidence=extraction.revenue_estimate.confidence,
                    reasoning=extraction.revenue_estimate.reasoning,
                    evidence=extraction.revenue_estimate.evidence,
                ))

        # Close date update
        if extraction.close_deadline and extraction.close_deadline.value:
            old_close = current_crm.get("close_date")
            if old_close != extraction.close_deadline.value:
                proposals.append(CRMUpdateProposal(
                    field_name="close_date",
                    old_value=old_close,
                    new_value=extraction.close_deadline.value,
                    confidence=extraction.close_deadline.confidence,
                    reasoning=extraction.close_deadline.reasoning,
                    evidence=extraction.close_deadline.evidence,
                ))

        # Deal duration update
        if extraction.deal_duration_years and extraction.deal_duration_years.value:
            old_dur = current_crm.get("deal_duration_years")
            if old_dur != extraction.deal_duration_years.value:
                proposals.append(CRMUpdateProposal(
                    field_name="deal_duration_years",
                    old_value=old_dur,
                    new_value=extraction.deal_duration_years.value,
                    confidence=extraction.deal_duration_years.confidence,
                    reasoning=extraction.deal_duration_years.reasoning,
                    evidence=extraction.deal_duration_years.evidence,
                ))

        # Renewal date update
        if extraction.renewal_date and extraction.renewal_date.value:
            old_renewal = current_crm.get("renewal_date")
            if old_renewal != extraction.renewal_date.value:
                proposals.append(CRMUpdateProposal(
                    field_name="renewal_date",
                    old_value=old_renewal,
                    new_value=extraction.renewal_date.value,
                    confidence=extraction.renewal_date.confidence,
                    reasoning=extraction.renewal_date.reasoning,
                    evidence=extraction.renewal_date.evidence,
                ))

        # User count update — top-level CRM field
        if extraction.num_users and extraction.num_users.value:
            old_users = current_crm.get("num_users")
            if old_users != extraction.num_users.value:
                proposals.append(CRMUpdateProposal(
                    field_name="num_users",
                    old_value=old_users,
                    new_value=extraction.num_users.value,
                    confidence=extraction.num_users.confidence,
                    reasoning=extraction.num_users.reasoning,
                    evidence=extraction.num_users.evidence,
                ))

        # Follow-up tasks from requested actions
        if extraction.requested_actions and extraction.requested_actions.value:
            actions = extraction.requested_actions.value
            if actions:
                proposals.append(CRMUpdateProposal(
                    field_name="follow_up_tasks",
                    old_value=[],
                    new_value=actions,
                    confidence=extraction.requested_actions.confidence,
                    reasoning="Actions requested by customer during meeting.",
                    evidence=extraction.requested_actions.evidence,
                ))

        # Data residency flag
        if extraction.customer_concerns and extraction.customer_concerns.value:
            concerns = extraction.customer_concerns.value
            if "data_residency" in concerns:
                proposals.append(CRMUpdateProposal(
                    field_name="custom_fields.data_residency_requirement",
                    old_value=None,
                    new_value="EU",
                    confidence=0.92,
                    reasoning="Customer explicitly stated EU data residency is a hard requirement.",
                    evidence=extraction.customer_concerns.evidence,
                ))

        # Meeting notes (append to history, don't overwrite)
        if extraction.meeting_summary and extraction.meeting_summary.value:
            existing_notes = current_crm.get("notes_history", []) or current_crm.get("notes", "")
            proposals.append(CRMUpdateProposal(
                field_name="notes_history",
                old_value=existing_notes,
                new_value=extraction.meeting_summary.value,
                confidence=extraction.meeting_summary.confidence,
                reasoning="Auto-generated meeting summary.",
                evidence="Full transcript",
            ))

        # ---- Intelligence fields ----
        # Conversion likelihood — driven by NBA, intent, stage progress, and deal signals.
        # This answers: "how likely is this deal to close, given what just happened?"
        intent = extraction.buying_intent.value if extraction.buying_intent else "MEDIUM"
        cf_proposed = {
            p.field_name.replace("custom_fields.", ""): p.new_value
            for p in proposals if p.field_name.startswith("custom_fields.")
        }
        contract_signed = (
            current_crm.get("custom_fields", {}).get("contract_signed", False) or
            cf_proposed.get("contract_signed", False)
        )

        if contract_signed or any(
            kw in text_lower for kw in ["board signed off", "signed contract", "confirmed on our end"]
        ):
            # Deal is confirmed closed
            likelihood = 1.0
            lik_reasoning = "Contract signed — deal confirmed closed."
        elif intent == "LOST":
            likelihood = 0.0
            lik_reasoning = "Deal lost — customer chose alternative."
        elif intent == "LOW":
            # Not dead yet, but needs nurturing. Likelihood drops but stays non-zero.
            likelihood = 0.15
            lik_reasoning = "Low intent — budget/decision-maker issue. Nurture required."
        else:
            # Base: current CRM stage tells us how far we've progressed
            current_stage = current_crm.get("stage", "Qualification")
            stage_base = {
                "Qualification": 0.45, "Discovery": 0.65,
                "Demo": 0.72, "Proposal": 0.82, "Negotiation": 0.90, "Closed Won": 1.0,
            }.get(current_stage, 0.50)

            # Modifier: did the deal advance this call?
            proposed_stage = next(
                (p.new_value for p in proposals if p.field_name == "stage"), None
            )
            from src.guardrails.stage_transitions import STAGE_ORDER
            advanced = (
                proposed_stage and
                proposed_stage in STAGE_ORDER and
                current_stage in STAGE_ORDER and
                STAGE_ORDER.index(proposed_stage) > STAGE_ORDER.index(current_stage)
            )

            # Modifier: key milestones hit this call
            security_cleared = cf_proposed.get("security_questionnaire_sent", False)
            proposal_sent = cf_proposed.get("proposal_sent", False)

            modifier = 0.0
            if advanced:           modifier += 0.08   # stage moved forward
            if security_cleared:   modifier += 0.05   # security hurdle cleared
            if proposal_sent:      modifier += 0.04   # proposal accepted
            if intent == "HIGH":   modifier += 0.02   # strong positive signals

            likelihood = round(min(0.97, stage_base + modifier), 2)
            lik_reasoning = (
                f"Stage={current_stage}, intent={intent}, "
                f"advanced={advanced}, security_cleared={security_cleared}."
            )

        old_likelihood = current_crm.get("conversion_likelihood", 0.5)
        if abs(likelihood - old_likelihood) >= 0.04:
            proposals.append(CRMUpdateProposal(
                field_name="conversion_likelihood",
                old_value=old_likelihood,
                new_value=likelihood,
                confidence=0.88,
                reasoning=lik_reasoning,
                evidence=f"Buying intent: {intent}.",
            ))
        # Mark security_review_required=True when deal > $100k
        if extraction.revenue_estimate and extraction.revenue_estimate.value:
            if extraction.revenue_estimate.value > 100_000:
                if not current_crm.get("custom_fields", {}).get("security_review_required", False):
                    proposals.append(CRMUpdateProposal(
                        field_name="custom_fields.security_review_required",
                        old_value=False,
                        new_value=True,
                        confidence=0.98,
                        reasoning="Deal value exceeds $100k — security review required per company policy (cwiki_004).",
                        evidence=f"Revenue ${extraction.revenue_estimate.value:,} exceeds $100k threshold.",
                    ))

        # Mark security_questionnaire_sent=True when call confirms it was cleared
        SECURITY_CLEARED = ["signed off", "security cleared", "security approved",
                            "good on the security", "no blockers", "cleared internally"]
        if any(sig in text_lower for sig in SECURITY_CLEARED):
            if not current_crm.get("custom_fields", {}).get("security_questionnaire_sent", False):
                proposals.append(CRMUpdateProposal(
                    field_name="custom_fields.security_questionnaire_sent",
                    old_value=False,
                    new_value=True,
                    confidence=0.92,
                    reasoning="Customer confirmed security questionnaire reviewed and approved.",
                    evidence="Security clearance confirmed in meeting transcript.",
                ))

        # Mark proposal_sent=True when contract signing implies proposal was accepted
        CONTRACT_SIGNALS = ["signed contract", "contract signed", "board signed off", "signed off yesterday"]
        if any(sig in text_lower for sig in CONTRACT_SIGNALS):
            if not current_crm.get("custom_fields", {}).get("proposal_sent", False):
                proposals.append(CRMUpdateProposal(
                    field_name="custom_fields.proposal_sent",
                    old_value=False,
                    new_value=True,
                    confidence=0.95,
                    reasoning="Contract signed — proposal must have been accepted.",
                    evidence="Contract signing confirmed in transcript.",
                ))
            if not current_crm.get("custom_fields", {}).get("contract_signed", False):
                proposals.append(CRMUpdateProposal(
                    field_name="custom_fields.contract_signed",
                    old_value=False,
                    new_value=True,
                    confidence=0.97,
                    reasoning="Contract signing explicitly confirmed by both parties.",
                    evidence="Contract signing confirmed in transcript.",
                ))
        # Next best action — derived using MERGED custom fields (current + proposed this call)
        proposed_custom = {
            p.field_name.replace("custom_fields.", ""): p.new_value
            for p in proposals
            if p.field_name.startswith("custom_fields.")
        }
        crm_for_nba = {
            **current_crm,
            "custom_fields": {**current_crm.get("custom_fields", {}), **proposed_custom},
        }
        nba = self._derive_next_best_action(extraction, crm_for_nba, wiki_context)
        if nba:
            proposals.append(CRMUpdateProposal(
                field_name="next_best_action",
                old_value=current_crm.get("next_best_action"),
                new_value=nba["action"],
                confidence=nba["confidence"],
                reasoning=nba["reasoning"],
                evidence=nba["evidence"],
            ))

        # Loss reason (if deal lost or very low intent)
        if intent in ("LOST", "LOW"):
            loss_reason = self._derive_loss_reason(extraction, evidence_text)
            if loss_reason:
                proposals.append(CRMUpdateProposal(
                    field_name="loss_reason",
                    old_value=current_crm.get("loss_reason"),
                    new_value=loss_reason["reason"],
                    confidence=loss_reason["confidence"],
                    reasoning=loss_reason["reasoning"],
                    evidence=loss_reason["evidence"],
                ))

        # Follow-up recommended flag
        follow_up_rec = self._derive_follow_up_recommendation(extraction, current_crm, wiki_context)
        if follow_up_rec is not None:
            proposals.append(CRMUpdateProposal(
                field_name="follow_up_recommended",
                old_value=current_crm.get("follow_up_recommended"),
                new_value=follow_up_rec["recommended"],
                confidence=follow_up_rec["confidence"],
                reasoning=follow_up_rec["rationale"],
                evidence=follow_up_rec["evidence"],
            ))
            if follow_up_rec.get("rationale"):
                proposals.append(CRMUpdateProposal(
                    field_name="follow_up_rationale",
                    old_value=current_crm.get("follow_up_rationale"),
                    new_value=follow_up_rec["rationale"],
                    confidence=follow_up_rec["confidence"],
                    reasoning="Follow-up rationale derived from call analysis.",
                    evidence=follow_up_rec["evidence"],
                ))

        # Actual revenue (if deal closed/won — signed contract detected)
        if intent == "HIGH" and any(kw in text_lower for kw in ["signed", "contract signed", "moving forward"]):
            if extraction.revenue_estimate and extraction.revenue_estimate.value:
                proposals.append(CRMUpdateProposal(
                    field_name="actual_revenue",
                    old_value=current_crm.get("actual_revenue"),
                    new_value=extraction.revenue_estimate.value,
                    confidence=0.95,
                    reasoning="Contract signed — actual revenue confirmed.",
                    evidence=extraction.revenue_estimate.evidence,
                ))
            # Set close_date to today (actual signing date) if not already set
            if not current_crm.get("close_date"):
                from datetime import datetime, timezone
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                proposals.append(CRMUpdateProposal(
                    field_name="close_date",
                    old_value=None,
                    new_value=today,
                    confidence=0.98,
                    reasoning="Contract signed — close date set to today.",
                    evidence="Contract signing confirmed in transcript.",
                ))

        return proposals

    # ------------------------------------------------------------------
    # Intelligence helpers (next best action, loss reason, follow-up)
    # ------------------------------------------------------------------

    def _derive_next_best_action(
        self, extraction: ExtractionResult, crm: Dict, wiki_context: List[Dict]
    ) -> Optional[Dict]:
        """Determine the recommended next best action."""
        intent = extraction.buying_intent.value if extraction.buying_intent else "MEDIUM"
        text_lower = (extraction.meeting_summary.value or "").lower() if extraction.meeting_summary else ""
        custom = crm.get("custom_fields", {})

        # Contract signed → onboarding (check both the flag and text signals)
        if custom.get("contract_signed") or any(
            kw in text_lower for kw in ["signed", "contract signed", "closed won", "send_contract"]
        ):
            return {
                "action": "INITIATE_ONBOARDING: Assign customer success manager and schedule kick-off within 48 hours.",
                "confidence": 0.97,
                "reasoning": "Contract signed. Onboarding is the mandatory next step.",
                "evidence": "Contract signing confirmed in transcript.",
            }

        # HIGH intent, security cleared, proposal needed
        if intent == "HIGH":
            if not custom.get("security_questionnaire_sent"):
                return {
                    "action": "SEND_SECURITY_QUESTIONNAIRE: Required before any commercial proposal.",
                    "confidence": 0.95,
                    "reasoning": "High-intent deal — security questionnaire is next required step before proposal.",
                    "evidence": "Company policy (cwiki_003).",
                }
            if custom.get("security_questionnaire_sent") and not custom.get("proposal_sent"):
                return {
                    "action": "SEND_FORMAL_PROPOSAL: Security cleared. Prepare and send commercial proposal.",
                    "confidence": 0.93,
                    "reasoning": "Security review complete. Proposal is the logical next step.",
                    "evidence": "Security clearance confirmed in transcript.",
                }
            return {
                "action": "SCHEDULE_CLOSING_CALL: Proposal sent. Follow up to address questions and close.",
                "confidence": 0.88,
                "reasoning": "Proposal already sent. Next step is closing or handling objections.",
                "evidence": "Pipeline progression logic.",
            }

        # LOW intent or not decision maker
        if intent == "LOW":
            return {
                "action": "NURTURE: Place in 90-day nurture sequence. Re-engage when budget/decision-maker situation changes.",
                "confidence": 0.80,
                "reasoning": "Low buying intent detected. Immediate pursuit not recommended.",
                "evidence": "Low intent signals in transcript.",
            }

        # MEDIUM intent
        return {
            "action": "SCHEDULE_DISCOVERY_FOLLOW_UP: Deepen qualification. Identify decision maker and budget approval path.",
            "confidence": 0.72,
            "reasoning": "Medium intent. Need stronger qualification before advancing.",
            "evidence": "Moderate engagement signals in transcript.",
        }

    def _derive_loss_reason(self, extraction: ExtractionResult, transcript: str) -> Optional[Dict]:
        """Extract or infer the reason for deal loss or low intent."""
        text_lower = transcript.lower()

        reasons = []
        if "budget" in text_lower and any(kw in text_lower for kw in ["$20,000", "$20k", "20,000", "tight", "constrained"]):
            reasons.append("Budget below minimum threshold")
        if "not the" in text_lower and "decision" in text_lower:
            reasons.append("Contact is not the decision maker")
        if "no timeline" in text_lower or "exploratory" in text_lower or "later in the year" in text_lower:
            reasons.append("No active timeline — exploratory inquiry only")
        if "competitor" in text_lower:
            reasons.append("Customer chose a competitor")
        if "integration" in text_lower and "couldn't" in text_lower:
            reasons.append("Integration gap — unable to meet technical requirements")

        if not reasons:
            return None

        reason_str = "; ".join(reasons)
        return {
            "reason": reason_str,
            "confidence": 0.82,
            "reasoning": f"Loss/stall reasons identified from transcript: {reason_str}.",
            "evidence": transcript[:120],
        }

    def _derive_follow_up_recommendation(
        self, extraction: ExtractionResult, crm: Dict, wiki_context: List[Dict]
    ) -> Optional[Dict]:
        """Recommend whether to follow up and when."""
        intent = extraction.buying_intent.value if extraction.buying_intent else "MEDIUM"

        if intent == "LOST":
            return {
                "recommended": False,
                "rationale": "Explicit rejection. Do not follow up.",
                "confidence": 0.90,
                "evidence": "Lost deal — customer chose alternative.",
            }
        if intent == "LOW":
            return {
                "recommended": True,
                "rationale": (
                    "Low intent but not a hard rejection. Re-engage in 90 days when budget cycle "
                    "may change or when decision maker is reachable. Monitor company news for triggers."
                ),
                "confidence": 0.78,
                "evidence": "Company policy (cwiki_005): one re-engagement attempt recommended after 90 days.",
            }
        if intent == "HIGH":
            return {
                "recommended": True,
                "rationale": "High intent. Continue active pursuit — do not allow deal to go cold.",
                "confidence": 0.92,
                "evidence": "High buying intent signals detected.",
            }
        return None

    def review_proposals(
        self,
        proposals: List[CRMUpdateProposal],
        extraction: ExtractionResult,
        transcript: str,
        wiki_context: List[Dict],
    ) -> Dict[str, Any]:
        """
        Second-pass review: critique each proposal for hallucinations,
        missing evidence, and rule violations.
        Returns a review report with per-proposal verdicts.
        """
        transcript_lower = transcript.lower()
        issues = []
        per_proposal_verdicts = []

        for prop in proposals:
            verdict = "APPROVED"
            prop_issues = []

            # Check 1: Is the evidence actually in the transcript?
            evidence_found = self._evidence_in_transcript(
                prop.evidence, transcript_lower
            )
            if not evidence_found and prop.field_name not in ("notes", "follow_up_tasks"):
                prop_issues.append(
                    f"Evidence '{prop.evidence[:60]}...' not found verbatim in transcript."
                )
                verdict = "NEEDS_REVISION"

            # Check 2: Hallucination — value not supported by extraction
            if prop.field_name == "expected_revenue":
                # Check both raw (375000) and comma-formatted (375,000) forms
                rev_raw = str(int(prop.new_value)) if isinstance(prop.new_value, (int, float)) else str(prop.new_value)
                rev_formatted = f"{int(prop.new_value):,}" if isinstance(prop.new_value, (int, float)) else rev_raw
                found = (
                    re.search(r'\$?\s*' + re.escape(rev_raw), transcript_lower) or
                    re.search(r'\$?\s*' + re.escape(rev_formatted), transcript_lower)
                )
                if not found:
                    prop_issues.append(
                        f"Revenue value ${prop.new_value:,} not explicitly mentioned in transcript."
                    )
                    if prop.confidence < 0.8:
                        verdict = "NEEDS_REVISION"

            # Check 3: Low confidence flagging
            if prop.confidence < 0.6:
                prop_issues.append(
                    f"Low confidence ({prop.confidence:.2f}) — requires human validation."
                )
                verdict = "NEEDS_HUMAN_REVIEW"

            # Check 4: Wiki consistency
            for entry in wiki_context:
                if entry.get("category") == "company_rename":
                    subject = entry.get("subject", "").lower()
                    if prop.field_name == "account_name":
                        if subject in str(prop.old_value).lower():
                            # Good — we're correcting this
                            pass

            issues.extend(prop_issues)
            per_proposal_verdicts.append({
                "field": prop.field_name,
                "verdict": verdict,
                "issues": prop_issues,
                "confidence": prop.confidence,
            })

        # Check for missing required actions
        missing_actions = self._check_missing_required_actions(
            proposals, extraction, wiki_context
        )
        if missing_actions:
            issues.extend(missing_actions)

        # Overall verdict
        verdicts = [v["verdict"] for v in per_proposal_verdicts]
        if any(v == "NEEDS_HUMAN_REVIEW" for v in verdicts):
            overall = "NEEDS_HUMAN_REVIEW"
        elif any(v == "NEEDS_REVISION" for v in verdicts):
            overall = "REVISED"
        elif issues:
            overall = "APPROVED_WITH_WARNINGS"
        else:
            overall = "APPROVED"

        return {
            "overall_verdict": overall,
            "issues_found": issues,
            "per_proposal": per_proposal_verdicts,
            "reviewer": "MockLLM_ReviewAgent_v1",
        }

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _extract_company(self, text: str, context: Dict) -> LLMField:
        """Extract company name, detecting rebrand signals."""
        # Check for explicit rebrand statement first
        for pat in self._p.REBRAND:
            m = re.search(pat, text, re.IGNORECASE)
            if m and len(m.groups()) >= 1:
                new_name = m.group(len(m.groups())).strip().rstrip(".")
                evidence = m.group(0)
                return LLMField(
                    value=new_name,
                    confidence=0.95,
                    evidence=evidence,
                    reasoning=f"Explicit rebrand detected in transcript. Customer stated '{evidence}'.",
                )

        # Parse participants — format: "Name (Role, Company)" or "Name (Company)"
        # Skip seller-side participants (AE/SE/CSM from TechServ)
        participants = context.get("participants", [])
        SELLER_MARKERS = ["(AE", "(SE)", "(CSM", "(AM,", "(AM)", "TechServ"]
        ROLE_WORDS = {"cto", "ceo", "vp", "cfo", "coo", "md", "director",
                      "manager", "procurement", "finance", "operations", "ae", "se"}

        for participant in participants:
            # Skip selling-side participant
            if any(marker in participant for marker in SELLER_MARKERS):
                continue
            # Extract content inside parentheses
            paren_match = re.search(r'\(([^)]+)\)', participant)
            if not paren_match:
                continue
            paren_content = paren_match.group(1).strip()  # e.g. "CTO, XYZ Air Services"
            # Split on FIRST comma to separate role from company
            if ',' in paren_content:
                first_comma = paren_content.index(',')
                company = paren_content[first_comma + 1:].strip()  # "XYZ Air Services"
            else:
                company = paren_content  # only a company name, no role prefix
            # Reject if the company part is itself a role word
            if company and not any(
                company.lower() == w or company.lower().startswith(w + " ")
                for w in ROLE_WORDS
            ):
                return LLMField(
                    value=company,
                    confidence=0.88,
                    evidence=f"Participant: {participant}",
                    reasoning="Company name extracted from customer participant in meeting.",
                )

        return LLMField(
            value=None,
            confidence=0.0,
            evidence="",
            reasoning="No company name found in transcript or context.",
        )

    def _extract_contacts(self, text: str, context: Dict) -> LLMField:
        """Extract customer contact names from participants (excludes the selling-side AE)."""
        participants = context.get("participants", [])
        SELLER_MARKERS = ["(AE", "(SE)", "(CSM", "(AM,", "(AM)", "TechServ"]
        customer_contacts = []
        for p in participants:
            if any(marker in p for marker in SELLER_MARKERS):
                continue  # skip seller-side participant
            name_match = re.match(r'^([A-Z][a-z]+ [A-Z][a-z]+)', p)
            if name_match:
                customer_contacts.append(name_match.group(1))

        if customer_contacts:
            return LLMField(
                value=customer_contacts,
                confidence=0.90,
                evidence=f"Participants: {', '.join(participants)}",
                reasoning="Customer contacts extracted from meeting participants.",
            )
        return LLMField(value=[], confidence=0.5, evidence="", reasoning="No contacts identified.")

    def _extract_num_users(self, text: str) -> LLMField:
        """Extract number of users/seats."""
        for pat in self._p.USER_COUNT:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "")
                try:
                    num = int(raw)
                    evidence = m.group(0)
                    conf = 0.90 if num > 10 else 0.60
                    return LLMField(
                        value=num,
                        confidence=conf,
                        evidence=evidence,
                        reasoning=f"User count '{num}' explicitly mentioned in transcript.",
                    )
                except ValueError:
                    pass
        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No user count found.")

    def _extract_revenue(self, text: str, context: Dict) -> LLMField:
        """
        Extract or infer revenue. Priority:
        1. Explicit dollar amount in transcript
        2. Calculated from users * price if users found
        """
        # Direct mention
        for pat in self._p.REVENUE:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "")
                try:
                    amount = float(raw)
                    # Handle shorthand like "$80k" or "80K"
                    if amount < 10000 and re.search(r'\d\s*[kK]\b', m.group(0)):
                        amount *= 1000
                    evidence = m.group(0)
                    return LLMField(
                        value=int(amount),
                        confidence=0.88,
                        evidence=evidence,
                        reasoning=f"Revenue '{evidence}' explicitly mentioned in transcript.",
                    )
                except ValueError:
                    pass

        # Infer from users × per-user price (approximate)
        users = self._extract_num_users(text)
        if users.value and users.value > 0:
            # Use enterprise pricing from wiki if available ($750/user/year)
            per_user = 750
            estimated = users.value * per_user
            return LLMField(
                value=estimated,
                confidence=0.62,  # Inferred, not explicit
                evidence=f"{users.value} users × ${per_user}/user/year (enterprise pricing)",
                reasoning=(
                    f"Revenue inferred from user count ({users.value}) × standard "
                    f"enterprise pricing (${per_user}/user). Not explicitly stated."
                ),
            )

        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No revenue information found.")

    def _extract_timeline(self, text: str) -> LLMField:
        """Extract go-live or purchase timeline."""
        for pat in self._p.TIMELINE:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                timeline = m.group(1).strip()
                return LLMField(
                    value=timeline,
                    confidence=0.85,
                    evidence=m.group(0),
                    reasoning=f"Timeline '{timeline}' explicitly mentioned in transcript.",
                )
        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No timeline found.")

    def _extract_close_date(self, text: str) -> LLMField:
        """Extract hard deadline or close date."""
        for pat in self._p.CLOSE_DATE:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                date = m.group(1).strip()
                return LLMField(
                    value=date,
                    confidence=0.82,
                    evidence=m.group(0),
                    reasoning=f"Hard deadline '{date}' stated in transcript.",
                )
        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No close date found.")

    def _extract_deal_duration(self, text: str) -> LLMField:
        """Extract deal duration in years (e.g. 3-year term)."""
        patterns = [
            r'(\d+)\s*-\s*year\s+(?:deal|term|contract|subscription)',
            r'(\d+)-year\s+(?:deal|term)',
            r'(\d+)\s+year\s+(?:deal|term|contract)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                years = int(m.group(1))
                return LLMField(
                    value=years,
                    confidence=0.93,
                    evidence=m.group(0),
                    reasoning=f"{years}-year deal term explicitly stated in transcript.",
                )
        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No deal duration found.")

    def _extract_renewal_date(self, text: str) -> LLMField:
        """Extract renewal date from transcript."""
        patterns = [
            r'renewal\s+(?:would\s+)?(?:come\s+up\s+)?(?:in\s+)?'
            r'((?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{4})',
            r'renew(?:al|s)?\s+(?:in|on|by)\s+'
            r'((?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{4})',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                date = m.group(1).strip()
                return LLMField(
                    value=date,
                    confidence=0.92,
                    evidence=m.group(0),
                    reasoning=f"Renewal date '{date}' explicitly stated in transcript.",
                )
        return LLMField(value=None, confidence=0.0, evidence="", reasoning="No renewal date found.")

    def _classify_buying_intent(self, text: str) -> LLMField:
        """Classify buying intent: HIGH / MEDIUM / LOW / LOST."""
        text_lower = text.lower()
        scores: Dict[str, int] = {level: 0 for level in self._p.BUYING_INTENT}

        for level, keywords in self._p.BUYING_INTENT.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[level] += 1

        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values()) or 1
        confidence = min(0.95, 0.5 + (scores[best] / total) * 0.5)

        matched_keywords = [kw for kw in self._p.BUYING_INTENT[best] if kw in text_lower]
        evidence = f"Matched: {matched_keywords[:3]}"

        return LLMField(
            value=best,
            confidence=round(confidence, 3),
            evidence=evidence,
            reasoning=(
                f"Buying intent classified as {best} based on keyword signals "
                f"(score: {scores[best]}/{total} weighted matches)."
            ),
        )

    def _classify_stage(self, text: str, context: Dict) -> LLMField:
        """Classify recommended CRM stage, respecting current stage and priority signals."""
        text_lower = text.lower()
        scores: Dict[str, int] = {stage: 0 for stage in self._p.STAGE_SIGNALS}

        for stage, signals in self._p.STAGE_SIGNALS.items():
            for sig in signals:
                if sig.lower() in text_lower:
                    scores[stage] += 1

        # ---- Priority 1: Closed Lost (explicit loss) ----
        if scores["Closed Lost"] > 0:
            return LLMField(
                value="Closed Lost",
                confidence=0.95,
                evidence="Competitor loss signals detected.",
                reasoning="Customer explicitly stated they chose a competitor.",
            )

        # ---- Priority 2: Closed Won (strong closing signals) ----
        CLOSED_WON_PRIORITY = [
            "signed contract", "board signed off", "board approved",
            "kick off onboarding", "payment will be cleared", "confirmed on our end",
        ]
        if any(sig in text_lower for sig in CLOSED_WON_PRIORITY) or scores["Closed Won"] >= 2:
            return LLMField(
                value="Closed Won",
                confidence=0.95,
                evidence="Contract/signing signals detected: " +
                         ", ".join(s for s in CLOSED_WON_PRIORITY if s in text_lower)[:60],
                reasoning="Strong contract-closing signals in transcript.",
            )

        # ---- Priority 3: Proposal (explicit proposal readiness) ----
        PROPOSAL_PRIORITY = [
            "move to a formal proposal", "ready to move to", "proposal drafted",
            "have the proposal", "send over the proposal", "agreed on",
        ]
        if any(sig in text_lower for sig in PROPOSAL_PRIORITY):
            return LLMField(
                value="Proposal",
                confidence=0.90,
                evidence="Proposal-readiness signals: " +
                         ", ".join(s for s in PROPOSAL_PRIORITY if s in text_lower)[:60],
                reasoning="Customer is ready to receive a formal proposal.",
            )

        # ---- General scoring ----
        del scores["Closed Lost"]
        if not any(scores.values()):
            return LLMField(
                value=None,
                confidence=0.0,
                evidence="",
                reasoning="Insufficient signals to determine stage.",
            )

        # Use current_stage from context to prevent backward regression
        from src.guardrails.stage_transitions import STAGE_ORDER
        current_stage = context.get("current_stage", "Qualification")
        current_idx = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else 0

        # Only consider stages at or above current stage (no regression)
        valid = {s: v for s, v in scores.items()
                 if s in STAGE_ORDER and STAGE_ORDER.index(s) >= current_idx}
        if not valid or not any(valid.values()):
            valid = scores  # fallback: use all scores

        best = max(valid, key=lambda k: valid[k])
        total = max(sum(valid.values()), 1)
        confidence = min(0.93, 0.55 + (valid[best] / total) * 0.45)

        return LLMField(
            value=best,
            confidence=round(confidence, 3),
            evidence=f"Stage signals matched: {[s for s in self._p.STAGE_SIGNALS.get(best, []) if s.lower() in text_lower][:3]}",
            reasoning=(
                f"Stage '{best}' recommended based on {valid[best]} matched signals "
                f"out of {total} total (current: {current_stage})."
            ),
        )

    def _extract_actions(self, text: str) -> LLMField:
        """Extract requested follow-up actions."""
        text_lower = text.lower()
        found_actions = []
        evidence_snippets = []

        for keyword, action_code in self._p.ACTION_KEYWORDS.items():
            if keyword.lower() in text_lower:
                if action_code not in found_actions:
                    found_actions.append(action_code)
                    # Find surrounding context
                    idx = text_lower.find(keyword)
                    snippet = text[max(0, idx - 20): idx + len(keyword) + 30].strip()
                    evidence_snippets.append(snippet)

        conf = min(0.95, 0.65 + len(found_actions) * 0.05)
        return LLMField(
            value=found_actions,
            confidence=round(conf, 3),
            evidence="; ".join(evidence_snippets[:3]),
            reasoning=f"Found {len(found_actions)} action keywords in transcript.",
        )

    def _extract_concerns(self, text: str) -> LLMField:
        """Extract customer concerns or blockers."""
        text_lower = text.lower()
        found_concerns = []
        evidence_snippets = []

        for keyword, concern_code in self._p.CONCERN_KEYWORDS.items():
            if keyword.lower() in text_lower:
                if concern_code not in found_concerns:
                    found_concerns.append(concern_code)
                    idx = text_lower.find(keyword)
                    snippet = text[max(0, idx - 10): idx + len(keyword) + 40].strip()
                    evidence_snippets.append(snippet)

        conf = min(0.92, 0.60 + len(found_concerns) * 0.05)
        return LLMField(
            value=found_concerns,
            confidence=round(conf, 3),
            evidence="; ".join(evidence_snippets[:3]),
            reasoning=f"Found {len(found_concerns)} concern indicators in transcript.",
        )

    def _classify_sentiment(self, text: str) -> LLMField:
        """Classify overall meeting sentiment."""
        text_lower = text.lower()
        scores = {s: 0 for s in ["positive", "neutral", "negative"]}

        for sentiment, keywords in self._p.SENTIMENT.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[sentiment] += 1

        best = max(scores, key=lambda k: scores[k])
        total = max(sum(scores.values()), 1)
        confidence = min(0.90, 0.60 + (scores[best] / total) * 0.35)

        return LLMField(
            value=best,
            confidence=round(confidence, 3),
            evidence=f"Matched: {[kw for kw in self._p.SENTIMENT[best] if kw in text_lower][:3]}",
            reasoning=f"Sentiment classified as {best} with {scores[best]} matched signals.",
        )

    def _generate_summary(self, text: str, extraction: ExtractionResult) -> LLMField:
        """Generate a structured meeting summary from extracted entities."""
        parts = []

        if extraction.company_name and extraction.company_name.value:
            parts.append(f"Meeting with {extraction.company_name.value}.")

        if extraction.buying_intent and extraction.buying_intent.value:
            parts.append(f"Buying intent: {extraction.buying_intent.value}.")

        if extraction.num_users and extraction.num_users.value:
            parts.append(f"Interested in {extraction.num_users.value} users.")

        if extraction.revenue_estimate and extraction.revenue_estimate.value:
            parts.append(f"Estimated deal value: ${extraction.revenue_estimate.value:,}.")

        if extraction.timeline and extraction.timeline.value:
            parts.append(f"Target timeline: {extraction.timeline.value}.")

        if extraction.close_deadline and extraction.close_deadline.value:
            parts.append(f"Hard deadline: {extraction.close_deadline.value}.")

        if extraction.requested_actions and extraction.requested_actions.value:
            actions = extraction.requested_actions.value
            parts.append(f"Required actions: {', '.join(actions)}.")

        if extraction.customer_concerns and extraction.customer_concerns.value:
            concerns = extraction.customer_concerns.value
            parts.append(f"Customer concerns: {', '.join(concerns)}.")

        summary = " ".join(parts) if parts else "No structured summary available."

        return LLMField(
            value=summary,
            confidence=0.85,
            evidence="Full transcript",
            reasoning="Auto-generated from extracted entities.",
        )

    def _inject_hallucination(self, result: ExtractionResult, text: str) -> ExtractionResult:
        """Inject a hallucinated field to test review agent robustness."""
        hallucinated_revenue = 999999
        if result.revenue_estimate:
            original = result.revenue_estimate.value
            result.revenue_estimate = LLMField(
                value=hallucinated_revenue,
                confidence=0.85,
                evidence="Customer mentioned $999,999 annual contract value",
                reasoning="Revenue extracted from transcript (HALLUCINATED FOR TESTING)",
                hallucinated=True,
            )
            if self.verbose:
                print(f"  [MockLLM] HALLUCINATION INJECTED: revenue {original} → {hallucinated_revenue}")
        return result

    # ------------------------------------------------------------------
    # Review helpers
    # ------------------------------------------------------------------

    def _evidence_in_transcript(self, evidence: str, transcript_lower: str) -> bool:
        """Check if key evidence terms appear in the transcript."""
        if not evidence or evidence in ("Full transcript", ""):
            return True
        # Skip checks for internal policy refs, wiki refs, and pipeline logic strings
        SKIP_PREFIXES = [
            "company policy", "pipeline progression", "cwiki_",
            "high buying intent", "low intent signals", "moderate engagement",
            "revenue ", "stage signals", "participant:",
        ]
        ev_lower = evidence.lower()
        if any(ev_lower.startswith(p) for p in SKIP_PREFIXES):
            return True
        # Extract key nouns/numbers from evidence string
        key_terms = re.findall(r'\b\w{4,}\b', ev_lower)
        if not key_terms:
            return True
        matched = sum(1 for t in key_terms if t in transcript_lower)
        return matched / len(key_terms) >= 0.4  # 40% of evidence terms found

    def _check_missing_required_actions(
        self,
        proposals: List[CRMUpdateProposal],
        extraction: ExtractionResult,
        wiki_context: List[Dict],
    ) -> List[str]:
        """Check for required actions missing from proposals."""
        missing = []
        proposed_fields = {p.field_name for p in proposals}
        proposed_actions = []
        for p in proposals:
            if p.field_name == "follow_up_tasks":
                proposed_actions = p.new_value or []

        # Rule: enterprise deals > $100k require security review
        rev = extraction.revenue_estimate
        if rev and rev.value and rev.value > 100000:
            if "send_security_questionnaire" not in proposed_actions:
                missing.append(
                    "MISSING ACTION: Enterprise deal >$100k — security questionnaire must be sent (wiki rule)."
                )

        # Rule from wiki: security doc before proposal
        if "send_proposal" in proposed_actions:
            if "send_security_questionnaire" not in proposed_actions and \
               "send_security_document" not in proposed_actions:
                missing.append(
                    "RULE VIOLATION: Proposal cannot be sent before security questionnaire (wiki rule)."
                )

        return missing