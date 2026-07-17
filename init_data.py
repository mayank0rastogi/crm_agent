"""
init_data.py — resets ALL data files to a clean starting state.
Run this before every demo:  python init_data.py
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
CUST = os.path.join(BASE, "data", "customers")


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  reset  {os.path.relpath(path, BASE)}")


# ── 1. Reset per-customer CRM records and wikis ──────────────────────────────
CUST_INITIAL = {
    "cust_001": {
        "crm": {
            "customer_id": "cust_001",
            "company_name": "XYZ Aviation Group",   # ← old name; agent detects rename in Call 1
            "stage": "Qualification",
            "expected_revenue": 0,
            "actual_revenue": None,
            "num_users": None,
            "deal_duration_years": None,
            "close_date": None,
            "renewal_date": None,
            "conversion_likelihood": 0.5,
            "loss_reason": None,
            "next_best_action": None,
            "follow_up_recommended": None,
            "follow_up_rationale": None,
            "follow_up_tasks": [],
            "notes_history": [],
            "owner_ae": "Alice Morgan",
            "custom_fields": {
                "security_questionnaire_sent": False,
                "security_review_required": False,
                "proposal_sent": False,
                "data_residency_requirement": None,
                "contract_signed": False
            },
            "created_date": "2025-01-10T00:00:00Z",
            "last_modified": "2025-01-10T00:00:00Z"
        },
        "wiki": {
            "version": "1.0",
            "customer_id": "cust_001",
            "description": "Customer wiki for XYZ Air Services (prev. XYZ Aviation Group) — starts empty, grows after each call.",
            "last_updated": "2025-01-10T00:00:00Z",
            "entries": []
        }
    },
    "cust_002": {
        "crm": {
            "customer_id": "cust_002",
            "company_name": "Pinnacle Logistics",
            "stage": "Qualification",
            "expected_revenue": 0,
            "actual_revenue": None,
            "num_users": None,
            "deal_duration_years": None,
            "close_date": None,
            "renewal_date": None,
            "conversion_likelihood": 0.5,
            "loss_reason": None,
            "next_best_action": None,
            "follow_up_recommended": None,
            "follow_up_rationale": None,
            "follow_up_tasks": [],
            "notes_history": [],
            "owner_ae": "Ben Carter",
            "custom_fields": {
                "security_questionnaire_sent": False,
                "security_review_required": False,
                "proposal_sent": False,
                "data_residency_requirement": None,
                "contract_signed": False
            },
            "created_date": "2025-01-15T00:00:00Z",
            "last_modified": "2025-01-15T00:00:00Z"
        },
        "wiki": {
            "version": "1.0",
            "customer_id": "cust_002",
            "description": "Customer wiki for Pinnacle Logistics — starts empty, grows after each call.",
            "last_updated": "2025-01-15T00:00:00Z",
            "entries": []
        }
    }
}

for cid, files in CUST_INITIAL.items():
    write(os.path.join(CUST, cid, "crm_opportunity.json"), files["crm"])
    write(os.path.join(CUST, cid, "wiki.json"),            files["wiki"])
    write(os.path.join(CUST, cid, "wiki_changelog.json"),  [])

# ── 2. Reset company wiki changelog (keep entries — they're static rules) ────
write(os.path.join(BASE, "data", "company", "company_wiki_changelog.json"), [])

# ── 3. Reset audit log ────────────────────────────────────────────────────────
write(os.path.join(BASE, "data", "crm", "audit_log.json"), [])

# ── 4. Reset entity graph to clean base state ─────────────────────────────────
write(os.path.join(BASE, "data", "wiki", "entity_graph.json"), {
    "version": "1.0",
    "description": "Knowledge graph — base customer data. Grows as agent discovers aliases.",
    "nodes": [
        {
            "canonical_id": "entity_001",
            "canonical_name": "XYZ Air Services",
            "aliases": ["xyz air services", "xyz air", "xyz aviation group", "xyz aviation", "xyz"],
            "domain": "xyzair.com",
            "deprecated_domains": ["xyzaviation.com"],
            "industry": "Aviation",
            "relationship_notes": "XYZ Aviation Group rebranded to XYZ Air Services in early 2025."
        },
        {
            "canonical_id": "entity_002",
            "canonical_name": "Pinnacle Logistics",
            "aliases": ["pinnacle", "pinnacle logistics"],
            "domain": "pinnaclelogistics.co.uk",
            "deprecated_domains": [],
            "industry": "Logistics",
            "relationship_notes": None
        }
    ],
    "relationships": [
        {
            "from": "XYZ Aviation Group",
            "to": "XYZ Air Services",
            "type": "renamed_to",
            "date": "2025-01-01",
            "confidence": 1.0
        }
    ]
})

# ── 5. Create output dir for graphs ──────────────────────────────────────────
os.makedirs(os.path.join(BASE, "output", "graphs"), exist_ok=True)
print(f"  ready  output/graphs/")

print("\n✓ All data reset to clean initial state.")
print("  Run 'python main.py' to start the simulation.")

