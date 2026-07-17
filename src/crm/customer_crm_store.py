"""
Customer-centric CRM Store.

One JSON file per customer under data/customers/{id}/crm_opportunity.json.
The selling company (TechServ Solutions) owns this CRM.
Each record tracks the full lifecycle of a deal with a single customer.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUSTOMERS_DIR = os.path.join(BASE_DIR, "data", "customers")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "data", "crm", "audit_log.json")


class CustomerCRMStore:
    """
    Manages CRM opportunity records — one per customer.

    Storage layout:
      data/customers/{customer_id}/crm_opportunity.json
      data/customers/{customer_id}/profile.json
      data/crm/audit_log.json
    """

    def __init__(self, customers_dir: str = CUSTOMERS_DIR, audit_log_path: str = AUDIT_LOG_PATH):
        self.customers_dir = customers_dir
        self.audit_path = audit_log_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_crm(self, customer_id: str) -> Optional[Dict]:
        """Return the CRM opportunity record for a customer."""
        path = os.path.join(self.customers_dir, customer_id, "crm_opportunity.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def get_profile(self, customer_id: str) -> Optional[Dict]:
        """Return the customer profile (company info, contacts)."""
        path = os.path.join(self.customers_dir, customer_id, "profile.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list_customers(self) -> List[str]:
        """Return all customer IDs."""
        if not os.path.exists(self.customers_dir):
            return []
        return [
            d for d in os.listdir(self.customers_dir)
            if os.path.isdir(os.path.join(self.customers_dir, d))
        ]

    def get_transcripts(self, customer_id: str) -> List[Dict]:
        """Return all transcripts for a customer, sorted by call_number."""
        transcript_dir = os.path.join(self.customers_dir, customer_id, "transcripts")
        if not os.path.exists(transcript_dir):
            return []
        transcripts = []
        for fname in sorted(os.listdir(transcript_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(transcript_dir, fname)) as f:
                    transcripts.append(json.load(f))
        return sorted(transcripts, key=lambda t: t.get("call_number", 0))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def apply_updates(
        self,
        customer_id: str,
        updates: List[Dict],
        run_id: str,
        applied_by: str = "crm_agent",
    ) -> Dict:
        """
        Apply approved field updates to a customer's CRM record.
        Deferred (NEEDS_HUMAN_REVIEW) updates are logged but NOT applied.
        """
        crm = self.get_crm(customer_id)
        if not crm:
            return {"error": f"No CRM record for customer '{customer_id}'."}

        applied = []
        deferred = []
        now = datetime.now(timezone.utc).isoformat()

        for update in updates:
            field = update["field"]
            new_val = update["new_value"]
            status = update.get("status", "APPROVED")

            if status == "NEEDS_HUMAN_REVIEW":
                deferred.append(update)
                continue

            if status in ("APPROVED", "REVISED"):
                old_val = self._get_nested(crm, field)
                # Special case: notes_history is append-only
                if field == "notes_history":
                    existing = old_val if isinstance(old_val, list) else []
                    self._set_nested(crm, field, existing + [new_val])
                else:
                    self._set_nested(crm, field, new_val)
                applied.append({
                    "field": field,
                    "old_value": old_val,
                    "new_value": new_val,
                    "confidence": update.get("confidence", 1.0),
                })

        crm["last_modified"] = now

        # Persist
        path = os.path.join(self.customers_dir, customer_id, "crm_opportunity.json")
        with open(path, "w") as f:
            json.dump(crm, f, indent=2)

        # Audit log
        entry = {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "timestamp": now,
            "customer_id": customer_id,
            "applied_by": applied_by,
            "applied_updates": applied,
            "deferred_updates": deferred,
            "total_applied": len(applied),
            "total_deferred": len(deferred),
        }
        self._append_audit(entry)
        return entry

    def get_audit_history(self, customer_id: str) -> List[Dict]:
        """Return all audit entries for a customer."""
        log = self._load_audit()
        return [e for e in log if e.get("customer_id") == customer_id]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_nested(self, obj: Dict, path: str) -> Any:
        parts = path.split(".")
        val = obj
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return None
        return val

    def _set_nested(self, obj: Dict, path: str, value: Any) -> None:
        parts = path.split(".")
        d = obj
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    def _load_audit(self) -> List[Dict]:
        if not os.path.exists(self.audit_path):
            return []
        with open(self.audit_path) as f:
            return json.load(f)

    def _append_audit(self, entry: Dict) -> None:
        log = self._load_audit()
        log.append(entry)
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        with open(self.audit_path, "w") as f:
            json.dump(log, f, indent=2)
