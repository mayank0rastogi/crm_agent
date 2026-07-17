"""
CRM Store — customer-centric JSON store.

EAS sales team CRM: each record is a customer with an embedded opportunity.
In production this maps to Salesforce/HubSpot API calls.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CRMStore:
    """
    Customer-centric CRM store.

    Single source of truth: data/crm/customers_crm.json
    Keys are customer_ids; each record contains the opportunity embedded.
    """

    def __init__(self,
                 crm_path: str = None,
                 audit_log_path: str = None,
                 # legacy compat params (ignored)
                 opportunities_path: str = None,
                 accounts_path: str = None):

        base = os.path.dirname(crm_path or audit_log_path or
                               os.path.join("data", "crm", "customers_crm.json"))
        if crm_path:
            self.crm_path = crm_path
        else:
            self.crm_path = os.path.join(base, "customers_crm.json")

        self.audit_path = audit_log_path or os.path.join(base, "audit_log.json")
        self._db: Dict[str, Dict] = {}
        self._audit: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.crm_path):
            with open(self.crm_path) as f:
                data = json.load(f)
            self._db = data.get("customers", {})
        if os.path.exists(self.audit_path):
            with open(self.audit_path) as f:
                self._audit = json.load(f)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.crm_path), exist_ok=True)
        with open(self.crm_path, "w") as f:
            json.dump({"customers": self._db}, f, indent=2)

    def _save_audit(self) -> None:
        with open(self.audit_path, "w") as f:
            json.dump(self._audit, f, indent=2)

    # ------------------------------------------------------------------
    # Seeding from customer profile files
    # ------------------------------------------------------------------

    def seed_customer(self, customer_data: Dict) -> None:
        """Upsert a customer record from a customer profile JSON."""
        cid = customer_data["customer_id"]
        if cid not in self._db:
            self._db[cid] = {
                "customer_id": cid,
                "company_name": customer_data.get("company_name", ""),
                "industry": customer_data.get("industry", ""),
                "employee_count": customer_data.get("employee_count"),
                "primary_contact": customer_data.get("primary_contact", {}),
                "opportunity": copy.deepcopy(customer_data.get("crm_opportunity", {})),
                "call_history": [],
                "last_modified": _now(),
            }
            self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_customer(self, customer_id: str) -> Optional[Dict]:
        rec = self._db.get(customer_id)
        return copy.deepcopy(rec) if rec else None

    def get_opportunity(self, customer_id: str) -> Optional[Dict]:
        rec = self.get_customer(customer_id)
        return rec.get("opportunity") if rec else None

    def list_customers(self) -> List[Dict]:
        return [copy.deepcopy(v) for v in self._db.values()]

    # Legacy compat — pipeline uses opportunity_id = "opp_customer_001"
    def get_opportunity_by_opp_id(self, opp_id: str) -> Optional[Dict]:
        for cid, rec in self._db.items():
            if rec.get("opportunity", {}).get("opportunity_id") == opp_id:
                return copy.deepcopy(rec)
        return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def apply_updates(self, customer_id: str, updates: List[Dict],
                      run_id: str, applied_by: str = "crm_agent") -> Dict:
        """
        Apply approved updates to a customer's opportunity.

        Each update: {"field": ..., "new_value": ..., "confidence": ..., "status": ...}
        NEEDS_HUMAN_REVIEW updates are deferred (not applied).
        """
        rec = self._db.get(customer_id)
        if not rec:
            return {"error": f"Customer '{customer_id}' not found."}

        applied, deferred = [], []
        opp = rec.setdefault("opportunity", {})
        cf  = opp.setdefault("custom_fields", {})

        for u in updates:
            field  = u["field"]
            val    = u["new_value"]
            status = u.get("status", "APPROVED")
            conf   = u.get("confidence", 1.0)

            if status == "NEEDS_HUMAN_REVIEW":
                deferred.append(u)
                continue
            if status in ("APPROVED", "REVISED"):
                old = self._get_nested(rec, field)
                self._set_nested(rec, field, val)
                applied.append({"field": field, "old_value": old,
                                 "new_value": val, "confidence": conf})

        rec["last_modified"] = _now()
        self._save()

        entry = {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "timestamp": _now(),
            "customer_id": customer_id,
            "applied_by": applied_by,
            "applied_updates": applied,
            "deferred_updates": deferred,
            "total_applied": len(applied),
            "total_deferred": len(deferred),
        }
        self._audit.append(entry)
        self._save_audit()
        return entry

    def record_call(self, customer_id: str, call_id: str, summary: str) -> None:
        """Append a call record to the customer's history."""
        rec = self._db.get(customer_id)
        if rec:
            rec.setdefault("call_history", []).append({
                "call_id": call_id, "summary": summary[:200], "recorded_at": _now()
            })
            self._save()

    def get_audit_history(self, customer_id: str) -> List[Dict]:
        return [e for e in self._audit if e.get("customer_id") == customer_id]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_nested(self, obj: Dict, path: str) -> Any:
        parts = path.split(".")
        val = obj
        for p in parts:
            val = val.get(p) if isinstance(val, dict) else None
        return val

    def _set_nested(self, obj: Dict, path: str, value: Any) -> None:
        parts = path.split(".")
        d = obj
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value


# CRM field schema (kept for guardrail compatibility)
CRM_SCHEMA: Dict[str, Dict] = {
    "stage":            {"type": str},
    "expected_revenue": {"type": float, "min": 0},
    "close_date":       {"type": str},
    "custom_fields":    {"type": dict},
    "follow_up_tasks":  {"type": list},
}


# (Old CRMStore class removed — new implementation above)

