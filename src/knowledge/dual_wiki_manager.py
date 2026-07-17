"""
Dual Wiki Manager.

Manages two wiki layers:
  1. Company wiki  — TechServ Solutions' internal rules, pricing, processes
                     (data/company/company_wiki.json)
  2. Customer wiki — what we know about each specific customer
                     (data/customers/{id}/wiki.json)

Both are append-only. Updates are logged to their respective changelogs.
Both wikis are queried together when building agent context.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DualWikiManager:
    """
    Manages company-level and customer-level knowledge bases.

    company_wiki  — universal rules, pricing, policies (shared across all customers)
    customer_wiki — customer-specific facts (contacts, preferences, history)
    """

    def __init__(self, customers_dir: Optional[str] = None, company_dir: Optional[str] = None):
        self.customers_dir = customers_dir or os.path.join(BASE_DIR, "data", "customers")
        self.company_dir = company_dir or os.path.join(BASE_DIR, "data", "company")
        self._company_wiki_path = os.path.join(self.company_dir, "company_wiki.json")
        self._company_changelog_path = os.path.join(self.company_dir, "company_wiki_changelog.json")

    # ------------------------------------------------------------------
    # Company wiki
    # ------------------------------------------------------------------

    def get_company_wiki(self) -> List[Dict]:
        """Return all company wiki entries."""
        return self._load_entries(self._company_wiki_path)

    def add_company_wiki_entry(
        self,
        category: str,
        subject: str,
        content: str,
        source: str = "agent",
        confidence: float = 0.9,
    ) -> Dict:
        """Add a new entry to the company wiki (append-only)."""
        return self._add_entry(
            self._company_wiki_path,
            self._company_changelog_path,
            category=category,
            subject=subject,
            content=content,
            source=source,
            confidence=confidence,
            extra={"scope": "company"},
        )

    # ------------------------------------------------------------------
    # Customer wiki
    # ------------------------------------------------------------------

    def get_customer_wiki(self, customer_id: str) -> List[Dict]:
        """Return all wiki entries for a specific customer."""
        path = self._customer_wiki_path(customer_id)
        return self._load_entries(path)

    def add_customer_wiki_entry(
        self,
        customer_id: str,
        category: str,
        subject: str,
        content: str,
        source: str = "agent",
        confidence: float = 0.9,
    ) -> Dict:
        """Add a new entry to a customer's wiki (append-only)."""
        wiki_path = self._customer_wiki_path(customer_id)
        changelog_path = os.path.join(self.customers_dir, customer_id, "wiki_changelog.json")
        return self._add_entry(
            wiki_path,
            changelog_path,
            category=category,
            subject=subject,
            content=content,
            source=source,
            confidence=confidence,
            extra={"customer_id": customer_id},
        )

    # ------------------------------------------------------------------
    # Combined context (for agent use)
    # ------------------------------------------------------------------

    def get_context_for_call(self, customer_id: str) -> Dict[str, List[Dict]]:
        """
        Return both company and customer wiki entries for a call.
        Used by agents to build their full context window.
        """
        return {
            "company_wiki": self.get_company_wiki(),
            "customer_wiki": self.get_customer_wiki(customer_id),
        }

    def get_context_for(self, company_name: str, stage: Optional[str] = None) -> List[Dict]:
        """
        Legacy-compatible method. Returns all company wiki entries plus
        any customer wiki entries that mention the company name.
        Used by ExtractionAgent when customer_id is not in context.
        """
        results = self.get_company_wiki()
        # Try to find a customer whose company matches
        for cid in os.listdir(self.customers_dir):
            wiki = self.get_customer_wiki(cid)
            for entry in wiki:
                if company_name.lower() in (
                    entry.get("subject", "") + entry.get("content", "")
                ).lower():
                    results.append(entry)
                    break
        return results

    def reload(self) -> None:
        """No-op — DualWikiManager reads from disk on every call (no in-memory cache)."""
        pass

    def search(self, query: str) -> List[Dict]:
        """Search company wiki only (for backward compatibility)."""
        return self.search_all("", query)
        """Full-text search across both wikis for a customer."""
        all_entries = self.get_company_wiki() + self.get_customer_wiki(customer_id)
        q = query.lower()
        return [
            e for e in all_entries
            if q in (e.get("content", "") + e.get("subject", "") + e.get("category", "")).lower()
        ]

    def get_customer_wiki_entry_count(self, customer_id: str) -> int:
        return len(self.get_customer_wiki(customer_id))

    def get_company_wiki_entry_count(self) -> int:
        return len(self.get_company_wiki())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _customer_wiki_path(self, customer_id: str) -> str:
        return os.path.join(self.customers_dir, customer_id, "wiki.json")

    def _load_entries(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("entries", [])

    def _save_entries(self, path: str, entries: List[Dict], extra_meta: Dict = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
        }
        if extra_meta:
            payload.update(extra_meta)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _add_entry(
        self,
        wiki_path: str,
        changelog_path: str,
        category: str,
        subject: str,
        content: str,
        source: str,
        confidence: float,
        extra: Dict = None,
    ) -> Dict:
        entries = self._load_entries(wiki_path)
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": f"wiki_{uuid.uuid4().hex[:8]}",
            "category": category,
            "subject": subject,
            "content": content,
            "effective_date": now[:10],
            "added_date": now,
            "confidence": confidence,
            "source": source,
            **(extra or {}),
        }
        entries.append(entry)
        self._save_entries(wiki_path, entries, extra)
        self._append_changelog(changelog_path, "ADD", entry)
        return entry

    def _append_changelog(self, path: str, action: str, entry: Dict) -> None:
        changelog: List[Dict] = []
        if os.path.exists(path):
            with open(path) as f:
                changelog = json.load(f)
        changelog.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "entry_id": entry.get("id"),
            "subject": entry.get("subject"),
            "category": entry.get("category"),
            "summary": entry.get("content", "")[:100],
        })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(changelog, f, indent=2)
