"""
Pipeline Orchestrator — wires all agents and components together.

Usage:
    pipeline = CRMAgentPipeline()
    result = pipeline.run(transcript_id="transcript_001")
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from src.agents.crm_update_agent import CRMUpdateAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.review_agent import ReviewAgent
from src.crm.customer_crm_store import CustomerCRMStore
from src.guardrails.rule_engine import RuleEngine
from src.knowledge.dual_wiki_manager import DualWikiManager
from src.knowledge.entity_resolver import EntityResolver
from src.llm_mock.mock_llm import MockLLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


class CRMAgentPipeline:
    """
    Full autonomous CRM agent pipeline.

    Components:
      - ExtractionAgent  → extract entities from transcript
      - CRMUpdateAgent   → propose CRM updates + run guardrails
      - ReviewAgent      → critique and approve/reject proposals
      - CRMStore         → apply approved updates
    """

    def __init__(
        self,
        hallucination_rate: float = 0.0,
        verbose: bool = False,
        human_in_the_loop: bool = True,
    ):
        self.verbose = verbose
        self.human_in_the_loop = human_in_the_loop

        # Infrastructure
        self.wiki = DualWikiManager()
        self.entity_resolver = EntityResolver(
            entity_graph_path=os.path.join(DATA_DIR, "wiki", "entity_graph.json"),
        )
        self.crm = CustomerCRMStore()
        self.rule_engine = RuleEngine()

        # Load entity graph for validators
        entity_graph_path = os.path.join(DATA_DIR, "wiki", "entity_graph.json")
        self.entity_graph = {}
        if os.path.exists(entity_graph_path):
            with open(entity_graph_path) as f:
                self.entity_graph = json.load(f)

        # Mock LLM (shared instance)
        self.llm = MockLLM(
            hallucination_rate=hallucination_rate,
            verbose=verbose,
        )

        # Agents
        self.extraction_agent = ExtractionAgent(
            llm=self.llm,
            entity_resolver=self.entity_resolver,
            wiki_manager=self.wiki,
        )
        self.crm_update_agent = CRMUpdateAgent(
            llm=self.llm,
            crm_store=self.crm,
            rule_engine=self.rule_engine,
            entity_graph=self.entity_graph,
        )
        self.review_agent = ReviewAgent(llm=self.llm)

        # Load all transcripts from the new per-customer structure
        self._transcripts: Dict[str, Dict] = {}
        customers_dir = os.path.join(DATA_DIR, "customers")
        if os.path.exists(customers_dir):
            for cid in os.listdir(customers_dir):
                t_dir = os.path.join(customers_dir, cid, "transcripts")
                if os.path.isdir(t_dir):
                    for fname in os.listdir(t_dir):
                        if fname.endswith(".json"):
                            with open(os.path.join(t_dir, fname)) as f:
                                t = json.load(f)
                            self._transcripts[t["call_id"]] = {
                                **t,
                                "opportunity_id": t.get("customer_id", cid),
                                "id": t["call_id"],
                            }

    def run(
        self,
        transcript_id: str,
        apply_updates: bool = True,
        run_id: Optional[str] = None,
    ) -> Dict:
        """
        Run the full pipeline for a given transcript.

        Returns a complete pipeline result dict with all intermediate outputs.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        transcript_data = self._transcripts.get(transcript_id)
        if not transcript_data:
            return {"error": f"Transcript '{transcript_id}' not found."}

        transcript_text = transcript_data["raw_text"]
        opportunity_id = transcript_data.get("customer_id") or transcript_data.get("opportunity_id")
        context = {
            "participants": transcript_data.get("participants", []),
            "meeting_date": transcript_data.get("meeting_date"),
            "customer_id": opportunity_id,
        }

        # Reload wiki
        if hasattr(self.wiki, "reload"):
            self.wiki.reload()

        # Build wiki context
        if hasattr(self.wiki, "get_context_for_call"):
            wiki_ctx = self.wiki.get_context_for_call(opportunity_id)
            wiki_context = wiki_ctx.get("company_wiki", []) + wiki_ctx.get("customer_wiki", [])
        else:
            wiki_context = []

        # ----------------------------------------------------------------
        # Stage 1: Extraction
        # ----------------------------------------------------------------
        extraction_result = self.extraction_agent.run(
            transcript=transcript_text,
            context=context,
            run_id=run_id,
        )
        if extraction_result.status == "error":
            return {"error": f"ExtractionAgent failed: {extraction_result.error}"}

        extraction_output = extraction_result.output
        extracted = extraction_output["extraction"]

        # ----------------------------------------------------------------
        # Stage 2: CRM Update Proposals + Guardrails
        # ----------------------------------------------------------------
        update_result = self.crm_update_agent.run(
            extraction_result=extracted,
            wiki_context=wiki_context,
            customer_id=opportunity_id,
            run_id=run_id,
        )
        if update_result.status == "error":
            return {"error": update_result.error}

        proposals = update_result.output.get("proposals", [])
        rule_summary = update_result.output.get("rule_summary", {})

        # ----------------------------------------------------------------
        # Stage 3: Review Agent
        # ----------------------------------------------------------------
        review_result = self.review_agent.run(
            proposals=proposals,
            extraction=extracted,
            transcript=transcript_text,
            wiki_context=wiki_context,
            run_id=run_id,
        )
        if review_result.status == "error":
            return {"error": f"ReviewAgent failed: {review_result.error}"}

        review_output = review_result.output
        final_proposals = review_output["final_proposals"]

        # ----------------------------------------------------------------
        # Stage 4: Human-in-the-Loop (simulated)
        # ----------------------------------------------------------------
        hitl_decisions = {}
        if self.human_in_the_loop:
            final_proposals, hitl_decisions = self._simulate_human_review(final_proposals)

        # ----------------------------------------------------------------
        # Stage 5: Apply Updates
        # ----------------------------------------------------------------
        applied_result = {}
        if apply_updates:
            updates_to_apply = [
                {
                    "field": p["field"],
                    "new_value": p["new_value"],
                    "confidence": p["confidence"],
                    "status": p["status"],
                }
                for p in final_proposals
                if p["status"] not in ("REJECTED",)
            ]
            applied_result = self.crm.apply_updates(
                customer_id=opportunity_id,
                updates=updates_to_apply,
                run_id=run_id,
            )

        return {
            "run_id": run_id,
            "transcript_id": transcript_id,
            "opportunity_id": opportunity_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extraction": extracted.to_dict(),
            "wiki_context_ids": [e.get("id", "") for e in wiki_context],
            "rule_summary": rule_summary,
            "final_proposals": final_proposals,
            "review_output": review_output,
            "hitl_decisions": hitl_decisions,
            "applied_result": applied_result,
            "reasoning_traces": {
                "extraction_agent": [s.__dict__ for s in extraction_result.reasoning_trace],
                "crm_update_agent": [s.__dict__ for s in update_result.reasoning_trace],
                "review_agent": [s.__dict__ for s in review_result.reasoning_trace],
            },
            "timing": {
                "extraction_ms": extraction_result.duration_ms,
                "update_ms": update_result.duration_ms,
                "review_ms": review_result.duration_ms,
            },
        }

    def run_inline(
        self,
        transcript_text: str,
        context: Dict,
        current_crm: Dict,
        opportunity_id: str,
        apply_updates: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict:
        """
        Run the pipeline with inline inputs (no file lookup).
        Used by the evaluation framework for test cases.
        """
        import tempfile, shutil
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"

        # Temporarily write test CRM data to the customers dir
        customers_dir = self.crm.customers_dir
        test_dir = os.path.join(customers_dir, opportunity_id)
        test_crm_path = os.path.join(test_dir, "crm_opportunity.json")
        test_profile_path = os.path.join(test_dir, "profile.json")
        created_temp = not os.path.exists(test_dir)

        # Build a normalized crm record (handle both "id" and "customer_id" schemas)
        normalized_crm = {
            "customer_id": opportunity_id,
            "company_name": current_crm.get("account_name") or current_crm.get("company_name", "Test Co"),
            "stage": current_crm.get("stage", "Qualification"),
            "expected_revenue": current_crm.get("expected_revenue", 0),
            "actual_revenue": current_crm.get("actual_revenue"),
            "num_users": current_crm.get("num_users"),
            "close_date": current_crm.get("close_date"),
            "renewal_date": current_crm.get("renewal_date"),
            "conversion_likelihood": current_crm.get("conversion_likelihood", 0.5),
            "loss_reason": current_crm.get("loss_reason"),
            "next_best_action": current_crm.get("next_best_action"),
            "follow_up_recommended": current_crm.get("follow_up_recommended"),
            "follow_up_tasks": current_crm.get("follow_up_tasks", []),
            "notes_history": current_crm.get("notes_history", []),
            "custom_fields": current_crm.get("custom_fields", {}),
        }

        if created_temp:
            os.makedirs(test_dir, exist_ok=True)
            with open(test_profile_path, "w") as f:
                json.dump({"customer_id": opportunity_id,
                           "company_name": normalized_crm["company_name"],
                           "tier": current_crm.get("tier", "Standard")}, f)
            with open(test_crm_path, "w") as f:
                json.dump(normalized_crm, f)

        try:
            wiki_ctx = self.wiki.get_context_for_call(opportunity_id)
            wiki_context = wiki_ctx.get("company_wiki", []) + wiki_ctx.get("customer_wiki", [])

            extraction_result = self.extraction_agent.run(
                transcript=transcript_text,
                context={**context, "customer_id": opportunity_id},
                run_id=run_id,
            )
            extraction_output = extraction_result.output or {}
            extracted = extraction_output.get("extraction")
            if not extracted:
                return {"error": "Extraction failed"}

            update_result = self.crm_update_agent.run(
                extraction_result=extracted,
                wiki_context=wiki_context,
                customer_id=opportunity_id,
                run_id=run_id,
            )
            if update_result.status == "error":
                return {"error": update_result.error}

            proposals = update_result.output.get("proposals", [])
            rule_summary = update_result.output.get("rule_summary", {})

            review_result = self.review_agent.run(
                proposals=proposals,
                extraction=extracted,
                transcript=transcript_text,
                wiki_context=wiki_context,
                run_id=run_id,
            )
            review_output = review_result.output or {}
            final_proposals = review_output.get("final_proposals", [])

            return {
                "run_id": run_id,
                "extraction": extracted.to_dict(),
                "final_proposals": final_proposals,
                "rule_summary": rule_summary,
                "review_output": review_output,
            }
        finally:
            if created_temp:
                shutil.rmtree(test_dir, ignore_errors=True)

    def _simulate_human_review(self, proposals):
        """
        Simulate human-in-the-loop approval for NEEDS_HUMAN_REVIEW proposals.

        In the demo, auto-approves all. In production, this would block
        until a human responds via the UI/API.
        """
        decisions = {}
        for p in proposals:
            if p["status"] == "NEEDS_HUMAN_REVIEW":
                # Simulated: human approves if confidence > 0.5
                human_approved = p["confidence"] >= 0.5
                p["status"] = "APPROVED" if human_approved else "REJECTED"
                decisions[p["field"]] = {
                    "action": "approved" if human_approved else "rejected",
                    "simulated": True,
                    "reason": f"Confidence {p['confidence']:.2f} {'≥' if human_approved else '<'} 0.50",
                }
        return proposals, decisions
