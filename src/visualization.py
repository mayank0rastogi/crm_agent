"""
Customer Journey Graph Visualizer.

Builds a NetworkX graph showing the full journey of each customer:
  - Central customer node (colored by conversion outcome)
  - Call transcript nodes linked in sequence
  - CRM stage evolution per call
  - Wiki entries added after each call

Saves PNG per customer + one combined view.
Falls back to ASCII if networkx/matplotlib not installed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STAGE_COLORS = {
    "Qualification": "#AED6F1",
    "Discovery":     "#A9DFBF",
    "Demo":          "#F9E79F",
    "Proposal":      "#F0B27A",
    "Negotiation":   "#E59866",
    "Closed Won":    "#2ECC71",
    "Closed Lost":   "#E74C3C",
    "Nurture":       "#D7BDE2",
}

INTENT_COLORS = {
    "HIGH":   "#2ecc71",
    "MEDIUM": "#f39c12",
    "LOW":    "#e74c3c",
    "LOST":   "#7f8c8d",
    "CLOSED": "#3498db",
    "UNKNOWN": "#bdc3c7",
}


class CustomerJourneyGraph:
    """Builds and renders the journey graph for one customer."""

    def __init__(self, customer_id: str, company_name: str):
        self.customer_id = customer_id
        self.company_name = company_name
        self._calls: List[Dict] = []
        if HAS_NX:
            self.G = nx.DiGraph()
            self.G.add_node(customer_id, label=company_name, node_type="customer", color="#2C3E50")

    def add_call(
        self,
        call_number: int,
        call_id: str,
        stage_before: str,
        stage_after: str,
        wiki_entries_added: List[str],
        crm_fields_updated: List[str],
        conversion_likelihood: float,
        next_best_action: Optional[str] = None,
        meeting_date: str = "",
        meeting_type: str = "",
        key_facts: Optional[dict] = None,
    ) -> None:
        self._calls.append({
            "call_number": call_number,
            "call_id": call_id,
            "meeting_date": meeting_date,
            "meeting_type": meeting_type,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "wiki_entries": wiki_entries_added,
            "crm_fields": crm_fields_updated,
            "conversion_likelihood": conversion_likelihood,
            "next_best_action": next_best_action,
            "key_facts": key_facts or {},
        })
        if not HAS_NX:
            return

        call_node = f"call_{call_number}"
        stage_node = f"stage_{call_number}"

        self.G.add_node(call_node, label=f"Call {call_number}", node_type="call", color="#5DADE2")
        self.G.add_node(stage_node, label=stage_after, node_type="stage",
                        color=STAGE_COLORS.get(stage_after, "#BDC3C7"))
        self.G.add_edge(self.customer_id, call_node, label=f"#{call_number}")
        self.G.add_edge(call_node, stage_node, label="→")

        for i, entry in enumerate(wiki_entries_added):
            wnode = f"wiki_{call_number}_{i}"
            self.G.add_node(wnode, label=entry[:25], node_type="wiki", color="#E8DAEF")
            self.G.add_edge(call_node, wnode, label="wiki+")

        if call_number > 1:
            self.G.add_edge(f"call_{call_number - 1}", call_node, label="next")

    def save_png(self, output_path: str) -> bool:
        if not HAS_NX or not HAS_MPL or not self._calls:
            return False
        try:
            plt.figure(figsize=(14, 8))
            plt.title(f"Journey: {self.company_name}", fontsize=13, fontweight="bold")
            pos = nx.spring_layout(self.G, k=2.8, seed=42)
            node_colors = [self.G.nodes[n].get("color", "#BDC3C7") for n in self.G.nodes]
            labels = {n: self.G.nodes[n].get("label", n) for n in self.G.nodes}
            nx.draw_networkx(self.G, pos, node_color=node_colors, labels=labels,
                             font_size=7, node_size=500, arrows=True,
                             edge_color="#7F8C8D", connectionstyle="arc3,rad=0.1")
            legend = [
                mpatches.Patch(color="#2C3E50", label="Customer"),
                mpatches.Patch(color="#5DADE2", label="Call"),
                mpatches.Patch(color="#2ECC71", label="Closed Won"),
                mpatches.Patch(color="#D7BDE2", label="Nurture"),
                mpatches.Patch(color="#E8DAEF", label="Wiki Entry"),
            ]
            plt.legend(handles=legend, loc="upper left", fontsize=8)
            plt.axis("off")
            plt.tight_layout()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
            return True
        except Exception as e:
            print(f"  [Graph] PNG save failed: {e}")
            return False

    def print_ascii(self) -> None:
        """Print a narrative deal-story journey for this customer."""
        W = 68
        final_call = self._calls[-1] if self._calls else {}
        final_stage = final_call.get("stage_after", "")
        final_lik   = final_call.get("conversion_likelihood", 0)
        facts_last  = final_call.get("key_facts", {})
        contract_signed = facts_last.get("contract_signed", False)

        if final_stage == "Closed Won" or contract_signed or final_lik >= 1.0:
            outcome_tag = "★  DEAL CLOSED — WON"
        elif final_stage == "Closed Lost" or final_lik <= 0.05:
            outcome_tag = "✗  DEAL CLOSED — LOST"
        elif final_lik < 0.30:
            outcome_tag = "○  NURTURE  (low intent — re-engage in 90 days)"
        else:
            outcome_tag = "→  DEAL IN PROGRESS"

        print(f"\n  {'═' * W}")
        print(f"  DEAL JOURNEY  ─  {self.company_name}  ({self.customer_id})")
        print(f"  {outcome_tag}")
        print(f"  {'═' * W}")

        STAGE_ICONS = {
            "Qualification": "🔍", "Discovery": "📋", "Demo": "🖥",
            "Proposal": "📄", "Negotiation": "🤝",
            "Closed Won": "✅", "Closed Lost": "❌",
        }
        CALL_TYPE_LABELS = {
            "discovery_call":   "Discovery Call",
            "follow_up_call":   "Follow-up Call",
            "contract_signing": "Contract Signing",
            "evaluation_call":  "Evaluation Call",
            "loss_review":      "Loss Review Call",
            "":                 "Sales Call",
        }

        for i, call in enumerate(self._calls):
            is_last = (i == len(self._calls) - 1)
            branch  = "  └──" if is_last else "  ├──"
            cont    = "       " if is_last else "  │    "
            n_      = call["call_number"]
            date    = call.get("meeting_date", "")[:10]
            mtype   = CALL_TYPE_LABELS.get(call.get("meeting_type", ""), "Sales Call")
            sb, sa  = call["stage_before"], call["stage_after"]
            lik     = call["conversion_likelihood"]
            nba     = call.get("next_best_action") or "—"
            facts   = call.get("key_facts", {})
            s_icon  = STAGE_ICONS.get(sa, "•")

            stage_str = f"{sb} → {sa}" if sb != sa else f"{sa} (no change)"
            lik_str   = f"{lik:.0%}"

            print(f"  │")
            print(f"{branch} CALL {n_}  │  {date}  │  {mtype}")
            print(f"{cont}  {s_icon} Stage      :  {stage_str}   │  Likelihood: {lik_str}")

            # Key facts
            fact_parts = []
            if facts.get("users"):     fact_parts.append(f"{facts['users']} users")
            if facts.get("revenue"):   fact_parts.append(f"${facts['revenue']:,}/yr")
            if facts.get("duration"):  fact_parts.append(f"{facts['duration']}-year term")
            if facts.get("renewal"):   fact_parts.append(f"renews {facts['renewal']}")
            if facts.get("intent"):    fact_parts.append(f"intent={facts['intent']}")
            if facts.get("sentiment"): fact_parts.append(f"sentiment={facts['sentiment']}")
            if fact_parts:
                print(f"{cont}  📊 Key facts :  {', '.join(fact_parts)}")

            # Business events
            EVENT_MAP = {
                "stage":                                    "stage advanced",
                "expected_revenue":                         "revenue confirmed",
                "actual_revenue":                           "deal revenue locked",
                "close_date":                               "close date set",
                "deal_duration_years":                      "deal term confirmed",
                "renewal_date":                             "renewal date set",
                "num_users":                                "seat count captured",
                "custom_fields.security_questionnaire_sent":"security cleared ✓",
                "custom_fields.security_review_required":   "security review flagged",
                "custom_fields.proposal_sent":              "proposal sent ✓",
                "custom_fields.contract_signed":            "contract signed ✓",
                "conversion_likelihood":                    None,  # skip — shown in stage line
                "follow_up_tasks":                          None,
                "notes_history":                            None,
                "next_best_action":                         None,
                "follow_up_recommended":                    None,
                "follow_up_rationale":                      None,
            }
            events = [EVENT_MAP[f] for f in call.get("crm_fields", [])
                      if EVENT_MAP.get(f)]
            if events:
                print(f"{cont}  🔔 Events    :  {', '.join(events)}")

            # NBA
            nba_short = nba.split(":")[0].strip() if ":" in nba else nba[:55]
            print(f"{cont}  ➤ Next step  :  {nba_short}")

            # Wiki growth summary
            wcount = len(call.get("wiki_entries", []))
            if wcount:
                print(f"{cont}  📚 Wiki+     :  {wcount} facts added to customer knowledge base")

        # Outcome summary
        print(f"\n  {'═' * W}")
        if contract_signed or final_stage == "Closed Won":
            rev = facts_last.get("revenue")
            dur = facts_last.get("duration")
            rev_str = f"${rev:,}/yr" if rev else ""
            dur_str = f"  │  {dur}-year contract" if dur else ""
            ren_str = f"  │  renews {facts_last.get('renewal','')}" if facts_last.get("renewal") else ""
            print(f"  RESULT:  Deal closed  {rev_str}{dur_str}{ren_str}  │  100% conversion")
        elif final_stage == "Closed Lost" or final_lik <= 0.05:
            print(f"  RESULT:  Not converted  │  Likelihood at close: {final_lik:.0%}")
        elif final_lik < 0.30:
            print(f"  RESULT:  Nurture  │  Re-engage in 90 days  │  Likelihood: {final_lik:.0%}")
        else:
            print(f"  RESULT:  In progress  │  Likelihood: {final_lik:.0%}")
        print(f"  {'═' * W}\n")


def save_combined_graph(journeys: List["CustomerJourneyGraph"], output_path: str) -> bool:
    """Save a side-by-side view of all customer journeys."""
    if not HAS_NX or not HAS_MPL:
        return False
    n = len(journeys)
    if n == 0:
        return False
    try:
        fig, axes = plt.subplots(1, n, figsize=(12 * n, 9))
        if n == 1:
            axes = [axes]
        fig.suptitle("TechServ Solutions — All Customer Journeys", fontsize=14, fontweight="bold")
        for ax, journey in zip(axes, journeys):
            if not HAS_NX or not journey._calls:
                ax.set_visible(False)
                continue
            pos = nx.spring_layout(journey.G, k=2.5, seed=42)
            nc = [journey.G.nodes[nd].get("color", "#BDC3C7") for nd in journey.G.nodes]
            labs = {nd: journey.G.nodes[nd].get("label", nd) for nd in journey.G.nodes}
            nx.draw_networkx(journey.G, pos, ax=ax, node_color=nc, labels=labs,
                             font_size=6, node_size=400, arrows=True, edge_color="#7F8C8D")
            ax.set_title(journey.company_name, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"  [Graph] Combined graph failed: {e}")
        return False


def _stage_to_intent(stage: str) -> str:
    return {
        "Closed Won": "CLOSED", "Closed Lost": "LOST",
        "Negotiation": "HIGH", "Proposal": "HIGH", "Discovery": "HIGH",
        "Demo": "MEDIUM", "Qualification": "MEDIUM",
    }.get(stage, "UNKNOWN")


def ascii_graph(crm_data: Dict, wiki_changelog: List[Dict]) -> str:
    """Fallback ASCII representation when graph libs unavailable."""
    lines = ["  [TechServ Solutions CRM — ASCII Graph Mode]",
             "  Install networkx + matplotlib for visual output", ""]
    for cid, rec in crm_data.items():
        stage = rec.get("stage", "?")
        company = rec.get("company_name", cid)
        lines.append(f"  ◉ {company}  [{stage}]")
        for call in rec.get("calls", []):
            lines.append(f"    └── Call {call.get('call_number', '?')}")
    return "\n".join(lines)

