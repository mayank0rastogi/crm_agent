"""
Interactive CRM Agent Simulation.

Orchestrates the step-by-step journey of two customers through TechServ
Solutions' sales pipeline, with:
  - Wiki starting empty and growing after each call
  - HITL prompts for low-confidence updates
  - Customer-specific + company wiki separation
  - Visualization of the customer journey graph
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.agents.crm_update_agent import CRMUpdateAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.review_agent import ReviewAgent
from src.crm.customer_crm_store import CustomerCRMStore
from src.guardrails.rule_engine import RuleEngine
from src.knowledge.dual_wiki_manager import DualWikiManager
from src.knowledge.entity_resolver import EntityResolver
from src.llm_mock.mock_llm import MockLLM
from src.visualization import CustomerJourneyGraph, save_combined_graph

DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPHS_DIR = os.path.join(BASE_DIR, "output", "graphs")

try:
    _tty = sys.stdout.isatty()
except Exception:
    _tty = False

def _c(text, code): return f"\033[{code}m{text}\033[0m" if _tty else text
def bold(t):   return _c(t, "1")
def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def dim(t):    return _c(t, "2")


class CRMSimulation:
    """Drives the interactive demo simulation — one selling company, many customers."""

    def __init__(self, hallucination_rate: float = 0.0):
        self.wiki = DualWikiManager()
        self.crm = CustomerCRMStore()
        self.rule_engine = RuleEngine()

        entity_graph_path = os.path.join(DATA_DIR, "wiki", "entity_graph.json")
        self.entity_graph = {}
        if os.path.exists(entity_graph_path):
            with open(entity_graph_path) as f:
                self.entity_graph = json.load(f)

        self.entity_resolver = EntityResolver(entity_graph_path)
        self.llm = MockLLM(hallucination_rate=hallucination_rate, seed=42)

        self.extraction_agent = ExtractionAgent(
            llm=self.llm,
            entity_resolver=self.entity_resolver,
            wiki_manager=self.wiki,      # type: ignore[arg-type]
        )
        self.crm_update_agent = CRMUpdateAgent(
            llm=self.llm,
            crm_store=self.crm,
            rule_engine=self.rule_engine,
            entity_graph=self.entity_graph,
        )
        self.review_agent = ReviewAgent(llm=self.llm)
        self._journey_graphs: Dict[str, CustomerJourneyGraph] = {}

    def run_single_call(
        self,
        customer_id: str,
        call_number: int,
        interactive: bool = True,
    ) -> Optional[Dict]:
        """
        Process one specific call for a customer and return the updated CRM state.
        Builds the journey graph incrementally — call this in sequence.
        """
        transcripts = self.crm.get_transcripts(customer_id)
        transcript = next(
            (t for t in transcripts if t.get("call_number") == call_number), None
        )
        if not transcript:
            print(red(f"  Call {call_number} not found for {customer_id}"))
            return None

        # Ensure journey graph exists for this customer
        if customer_id not in self._journey_graphs:
            profile = self.crm.get_profile(customer_id) or {}
            company_name = profile.get("company_name", customer_id)
            self._journey_graphs[customer_id] = CustomerJourneyGraph(customer_id, company_name)

        self._process_call(
            customer_id,
            transcript,
            self._journey_graphs[customer_id],
            interactive,
        )
        return self.crm.get_crm(customer_id)

    def get_crm_state(self, customer_id: str) -> Optional[Dict]:
        return self.crm.get_crm(customer_id)

    def get_all_crm_states(self) -> Dict[str, Dict]:
        result = {}
        for cid in self.crm.list_customers():
            crm = self.crm.get_crm(cid)
            if crm:
                result[cid] = crm
        return result

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
        Run the full pipeline against inline inputs — no transcript file needed.
        Used by the evaluation framework (run_evaluation.py).

        Temporarily writes the test CRM to disk so CustomerCRMStore can read it,
        runs extraction → update proposals → review, then cleans up.
        """
        import shutil
        import uuid as _uuid
        run_id = run_id or f"run_{_uuid.uuid4().hex[:8]}"

        # Write a temporary customer directory with the test CRM data
        test_dir = os.path.join(self.crm.customers_dir, opportunity_id)
        crm_path = os.path.join(test_dir, "crm_opportunity.json")
        profile_path = os.path.join(test_dir, "profile.json")
        wiki_path = os.path.join(test_dir, "wiki.json")
        created_temp = not os.path.exists(test_dir)

        # Normalize the incoming CRM dict to our schema
        normalized = {
            "customer_id": opportunity_id,
            "company_name": current_crm.get("account_name") or current_crm.get("company_name", "Test Co"),
            "stage": current_crm.get("stage", "Qualification"),
            "expected_revenue": current_crm.get("expected_revenue", 0),
            "actual_revenue": current_crm.get("actual_revenue"),
            "num_users": current_crm.get("num_users"),
            "deal_duration_years": current_crm.get("deal_duration_years"),
            "close_date": current_crm.get("close_date"),
            "renewal_date": current_crm.get("renewal_date"),
            "conversion_likelihood": current_crm.get("conversion_likelihood", 0.5),
            "loss_reason": current_crm.get("loss_reason"),
            "next_best_action": current_crm.get("next_best_action"),
            "follow_up_recommended": current_crm.get("follow_up_recommended"),
            "follow_up_rationale": current_crm.get("follow_up_rationale"),
            "follow_up_tasks": current_crm.get("follow_up_tasks", []),
            "notes_history": current_crm.get("notes_history", []),
            "custom_fields": current_crm.get("custom_fields", {}),
        }

        try:
            if created_temp:
                os.makedirs(test_dir, exist_ok=True)
                with open(profile_path, "w") as f:
                    import json as _json
                    _json.dump({
                        "customer_id": opportunity_id,
                        "company_name": normalized["company_name"],
                        "tier": current_crm.get("tier", "Enterprise"),
                    }, f)
                with open(crm_path, "w") as f:
                    import json as _json
                    _json.dump(normalized, f)
                with open(wiki_path, "w") as f:
                    import json as _json
                    _json.dump({"version": "1.0", "customer_id": opportunity_id, "entries": []}, f)

            # Build wiki context from company wiki only (no customer wiki for eval)
            wiki_ctx = self.wiki.get_context_for_call(opportunity_id)
            all_wiki = wiki_ctx.get("company_wiki", []) + wiki_ctx.get("customer_wiki", [])

            # Extraction
            eval_context = {**context, "customer_id": opportunity_id,
                            "current_stage": normalized["stage"]}
            er = self.extraction_agent.run(transcript_text, eval_context, run_id)
            if er.status == "error":
                return {"error": er.error}
            extracted = er.output["extraction"]

            # CRM update proposals + guardrails
            ur = self.crm_update_agent.run(extracted, all_wiki, opportunity_id, run_id)
            if ur.status == "error":
                return {"error": ur.error}
            proposals = ur.output["proposals"]
            rule_summary = ur.output["rule_summary"]

            # Review
            rr = self.review_agent.run(proposals, extracted, transcript_text, all_wiki, run_id)
            if rr.status == "error":
                return {"error": rr.error}
            review_output = rr.output or {}
            final_proposals = review_output.get("final_proposals", [])

            return {
                "run_id": run_id,
                "extraction": extracted.to_dict(),
                "final_proposals": final_proposals,
                "rule_summary": rule_summary,
                "review_output": review_output,
            }
        finally:
            # Clean up temporary files
            if created_temp:
                shutil.rmtree(test_dir, ignore_errors=True)

    def finalize_journey(self, customer_id: str) -> None:
        """Print the journey narrative after all calls are processed."""
        if customer_id not in self._journey_graphs:
            return
        self._journey_graphs[customer_id].print_ascii()

    def run_customer(self, customer_id: str, interactive: bool = True) -> None:
        profile = self.crm.get_profile(customer_id)
        if not profile:
            print(red(f"  No profile for {customer_id}")); return
        company_name = profile.get("company_name", customer_id)
        transcripts = self.crm.get_transcripts(customer_id)
        if not transcripts:
            print(red(f"  No transcripts for {customer_id}")); return

        print(f"\n{cyan('═' * 68)}")
        print(f"  {bold('CUSTOMER:')} {company_name}  ({customer_id})")
        print(f"  {bold('Calls:')} {len(transcripts)}  |  "
              f"{bold('Customer wiki:')} {self.wiki.get_customer_wiki_entry_count(customer_id)} entries  |  "
              f"{bold('Company wiki:')} {self.wiki.get_company_wiki_entry_count()} entries")
        print(cyan('═' * 68))

        journey = CustomerJourneyGraph(customer_id, company_name)
        self._journey_graphs[customer_id] = journey

        for transcript in transcripts:
            self._process_call(customer_id, transcript, journey, interactive)

        journey.print_ascii()

    def run_all(self, interactive: bool = True) -> None:
        customer_ids = sorted(self.crm.list_customers())
        print(f"\n{bold('TechServ Solutions — CRM Agent Simulation')}")
        print(f"  Company wiki: {self.wiki.get_company_wiki_entry_count()} entries loaded")
        print(f"  Customers: {', '.join(customer_ids)}\n")

        for cid in customer_ids:
            if interactive:
                input(f"  {yellow('Press ENTER to start:')} {cid}  ...")
            self.run_customer(cid, interactive=interactive)

        journeys = [self._journey_graphs[c] for c in customer_ids if c in self._journey_graphs]
        self._print_dashboard(customer_ids)

    def _process_call(self, customer_id, transcript, journey, interactive):
        call_number = transcript.get("call_number", "?")
        call_id = transcript.get("call_id", "?")
        text = transcript.get("raw_text", "")

        print(f"\n  {cyan('─' * 64)}")
        print(f"  {bold(f'CALL {call_number}')}  [{call_id}]  {dim(transcript.get('meeting_date',''))}")
        print(f"  {cyan('─' * 64)}")

        crm_before = self.crm.get_crm(customer_id) or {}
        stage_before = crm_before.get("stage", "Qualification")
        wiki_ctx = self.wiki.get_context_for_call(customer_id)
        all_wiki = wiki_ctx["company_wiki"] + wiki_ctx["customer_wiki"]
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        # Extract 1: run extraction with current stage context
        context = {
            "participants": transcript.get("participants", []),
            "customer_id": customer_id,
            "current_stage": stage_before,   # lets _classify_stage avoid backward regression
        }
        print(f"\n  {bold('[1] EXTRACTION')}")
        er = self.extraction_agent.run(text, context, run_id)
        if er.status == "error": print(red(er.error)); return
        extracted = er.output["extraction"]
        self._print_extraction(extracted)

        # 2. Proposals + guardrails
        print(f"\n  {bold('[2] GUARDRAILS + PROPOSALS')}")
        ur = self.crm_update_agent.run(extracted, all_wiki, customer_id, run_id)
        if ur.status == "error": print(red(ur.error)); return
        proposals = ur.output["proposals"]
        self._print_rule_summary(ur.output["rule_summary"])
        self._print_proposals(proposals)

        # 3. Review
        print(f"\n  {bold('[3] REVIEW AGENT')}")
        rr = self.review_agent.run(proposals, extracted, text, all_wiki, run_id)
        if rr.status == "error": print(red(rr.error)); return
        final_proposals = rr.output["final_proposals"]
        verdict = rr.output.get("overall_verdict", "?")
        v_color = green if "APPROVED" in verdict and "REJECTION" not in verdict else yellow
        print(f"  Verdict: {bold(v_color(verdict))}")
        for note in rr.output.get("review_notes", []):
            print(f"  {yellow('⚠')} {note}")

        # 4. HITL
        print(f"\n  {bold('[4] HUMAN-IN-THE-LOOP')}")
        final_proposals, hitl_log = self._run_hitl(final_proposals, interactive)

        # Wiki updates from HITL decisions
        wiki_entries_added = []
        for dec in hitl_log:
            if dec.get("approved") and dec.get("wiki_update"):
                wu = dec["wiki_update"]
                e = self.wiki.add_customer_wiki_entry(customer_id, wu["category"], wu["subject"], wu["content"], "hitl")
                wiki_entries_added.append(e["subject"])
                print(f"  {green('✓')} Wiki+ : {e['subject']}")

        # Auto wiki
        wiki_entries_added += self._auto_wiki(customer_id, extracted, transcript, final_proposals)

        # 5. Apply
        print(f"\n  {bold('[5] APPLYING TO CRM')}")
        updates = [{"field": p["field"], "new_value": p["new_value"],
                    "confidence": p["confidence"], "status": p["status"]}
                   for p in final_proposals if p["status"] != "REJECTED"]
        applied = self.crm.apply_updates(customer_id, updates, run_id)
        for u in applied.get("applied_updates", []):
            v = f"${u['new_value']:,}" if u["field"] == "expected_revenue" and isinstance(u["new_value"], (int, float)) else str(u["new_value"])[:60]
            print(f"  {green('✓')} {bold(u['field'])} → {green(v)}")
        for u in applied.get("deferred_updates", []):
            print(f"  {yellow('⏳')} {u['field']} deferred")

        crm_after = self.crm.get_crm(customer_id) or {}
        journey.add_call(
            call_number=call_number,
            call_id=call_id,
            meeting_date=transcript.get("meeting_date", ""),
            meeting_type=transcript.get("meeting_type", ""),
            stage_before=stage_before,
            stage_after=crm_after.get("stage", stage_before),
            wiki_entries_added=wiki_entries_added,
            crm_fields_updated=[u["field"] for u in applied.get("applied_updates", [])],
            conversion_likelihood=crm_after.get("conversion_likelihood", 0.5),
            next_best_action=crm_after.get("next_best_action"),
            key_facts={
                "users":    crm_after.get("num_users"),
                "revenue":  crm_after.get("actual_revenue") or crm_after.get("expected_revenue"),
                "duration": crm_after.get("deal_duration_years"),
                "renewal":  crm_after.get("renewal_date"),
                "intent":   extracted.buying_intent.value if extracted.buying_intent else None,
                "sentiment":extracted.sentiment.value if extracted.sentiment else None,
                "contract_signed": crm_after.get("custom_fields", {}).get("contract_signed", False),
            },
        )

    def _run_hitl(self, proposals, interactive):
        hitl_items = [p for p in proposals if p.get("status") == "NEEDS_HUMAN_REVIEW"]
        log = []
        if not hitl_items:
            print(f"  {dim('No items require review.')}"); return proposals, log

        for p in proposals:
            if p.get("status") != "NEEDS_HUMAN_REVIEW": continue
            field, val, conf = p["field"], p["new_value"], p.get("confidence", 0)
            val_str = f"${val:,}" if field == "expected_revenue" and isinstance(val, (int, float)) else str(val)[:80]
            print(f"\n  {yellow('⚠  REVIEW NEEDED')}")
            print(f"  Field: {bold(field)}  │  Value: {bold(val_str)}  │  Confidence: {yellow(f'{conf:.2f}')}")
            print(f"  Reason: {p.get('reasoning','')[:100]}")
            for issue in p.get("issues", []):
                print(f"  {yellow('⚠')} {issue}")

            choice = input(f"  [A]pprove / [R]eject / [E]dit: ").strip().lower() if interactive else "a"
            if not interactive: print(f"  {dim('[auto-approved]')}")

            if choice in ("a", ""):
                p["status"] = "APPROVED"
                print(f"  {green('✓ Approved')}")
                wu = self._offer_wiki_save(field, val, p.get("reasoning",""), interactive)
                log.append({"approved": True, "field": field, "wiki_update": wu})
            elif choice == "e":
                new_val = input(f"  Enter corrected value: ").strip()
                if new_val:
                    p["new_value"] = new_val
                    p["status"] = "APPROVED"
                    print(f"  {green(f'✓ Edited: {new_val}')}")
                    log.append({"approved": True, "field": field})
            else:
                p["status"] = "REJECTED"
                print(f"  {red('✗ Rejected')}")
                log.append({"approved": False, "field": field})
        return proposals, log

    def _offer_wiki_save(self, field, value, reasoning, interactive):
        saveable = {"expected_revenue", "num_users", "close_date", "conversion_likelihood"}
        if field not in saveable: return None
        save = input(f"  Save '{field}={value}' to customer wiki? [y/n]: ").strip().lower() if interactive else "y"
        if save in ("y", ""):
            return {"category": "customer_fact", "subject": field,
                    "content": f"{field} confirmed: {value}. {reasoning[:80]}"}
        return None

    def _auto_wiki(self, customer_id, extracted, transcript, final_proposals):
        added = []
        call_n = transcript.get("call_number", "")
        def _add(cat, subj, content, conf=0.9):
            self.wiki.add_customer_wiki_entry(customer_id, cat, subj, content,
                                               f"auto_call_{call_n}", conf)
            added.append(subj)

        if extracted.contact_names and extracted.contact_names.value and extracted.contact_names.confidence >= 0.85:
            _add("contacts", f"call_{call_n}_participants",
                 f"Call {call_n}: {', '.join(extracted.contact_names.value)}")
        if extracted.sentiment and extracted.sentiment.value and extracted.sentiment.confidence >= 0.75:
            _add("meeting_outcome", f"call_{call_n}_sentiment",
                 f"Sentiment: {extracted.sentiment.value} — {transcript.get('meeting_type','')}")
        if extracted.buying_intent and extracted.buying_intent.value and extracted.buying_intent.confidence >= 0.75:
            _add("intent", f"call_{call_n}_intent",
                 f"Buying intent: {extracted.buying_intent.value} after call {call_n}.",
                 conf=extracted.buying_intent.confidence)
        for p in final_proposals:
            if p["field"] == "loss_reason" and p["status"] != "REJECTED" and p.get("new_value"):
                _add("deal_outcome", "loss_reason", f"Stall/loss reason: {p['new_value']}")
            if p["field"] == "next_best_action" and p["status"] != "REJECTED" and p.get("new_value"):
                _add("next_action", "recommended_nba", f"NBA: {p['new_value']}")
        return added

    def _print_extraction(self, extracted):
        fields = [("company_name","Company"), ("num_users","Users"), ("revenue_estimate","Revenue"),
                  ("buying_intent","Intent"), ("opportunity_stage","Stage"), ("sentiment","Sentiment")]
        for attr, label in fields:
            f = getattr(extracted, attr, None)
            if f and f.value is not None:
                color = green if f.confidence >= 0.85 else yellow if f.confidence >= 0.65 else red
                val = f"${f.value:,}" if attr == "revenue_estimate" and isinstance(f.value, (int,float)) else str(f.value)
                print(f"  {label:<14} {bold(val)}  {color(f'{f.confidence:.2f}')}")

    def _print_rule_summary(self, rule_summary):
        passed = rule_summary.get("passed", 0)
        total = rule_summary.get("total_rules_evaluated", 0)
        blocking = rule_summary.get("blocking", False)
        status = red("BLOCKING") if blocking else green("OK")
        print(f"  Rules: {passed}/{total} passed  {status}")
        for r in rule_summary.get("details", []):
            if not r["passed"]:
                icon = red("✗") if r["severity"] == "ERROR" else yellow("⚠")
                ref = f"  [{dim(r['wiki_ref'])}]" if r.get("wiki_ref") else ""
                print(f"  {icon} {r['message']}{ref}")

    def _print_proposals(self, proposals):
        for p in proposals:
            val = p.new_value
            if p.field_name == "expected_revenue" and isinstance(val, (int,float)):
                vs = f"${val:,}"
            elif isinstance(val, list):
                vs = "[" + ", ".join(str(v) for v in val[:2]) + (", ..." if len(val) > 2 else "") + "]"
            elif isinstance(val, str):
                vs = val[:65] + ("…" if len(val) > 65 else "")
            else:
                vs = str(val)
            color = green if p.confidence >= 0.85 else yellow if p.confidence >= 0.65 else red
            tags = (f"  {red('BLOCKED')}" if p.blocked_by_guardrail else "") + \
                   (f"  {yellow('HITL')}" if p.requires_human_review else "")
            print(f"  {bold(p.field_name):<35} {vs}  {color(f'{p.confidence:.2f}')}{tags}")

    def _print_dashboard(self, customer_ids):
        print(f"\n{cyan('═' * 68)}")
        print(f"  {bold('FINAL CRM DASHBOARD — TechServ Solutions')}")
        print(cyan('═' * 68))
        print(f"  {'Customer':<28} {'Stage':<14} {'Revenue':<13} {'Likelihood':<11} Next Best Action")
        print(f"  {'─'*28} {'─'*14} {'─'*13} {'─'*11} {'─'*25}")
        for cid in customer_ids:
            crm = self.crm.get_crm(cid) or {}
            company = crm.get("company_name", cid)[:26]
            stage = crm.get("stage", "?")[:12]
            rev = crm.get("actual_revenue") or crm.get("expected_revenue", 0)
            rev_s = f"${rev:,}" if isinstance(rev, (int,float)) and rev > 0 else "—"
            lik = crm.get("conversion_likelihood", 0.5)
            lik_s = f"{lik:.0%}"
            nba = (crm.get("next_best_action") or "—")[:30]
            if stage == "Closed Won":
                print(f"  {green(company):<37} {green(stage):<23} {green(rev_s):<13} {green(lik_s):<11} {nba}")
            elif lik < 0.3:
                print(f"  {yellow(company):<37} {yellow(stage):<23} {rev_s:<13} {yellow(lik_s):<11} {nba}")
            else:
                print(f"  {company:<28} {stage:<14} {rev_s:<13} {lik_s:<11} {nba}")
        print(cyan('═' * 68))


# Keep old name as alias for backward compatibility
Simulation = CRMSimulation