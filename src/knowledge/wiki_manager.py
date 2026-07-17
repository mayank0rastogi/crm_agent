"""
WikiManager — manages two wiki layers:

  1. company_knowledge.json   — EAS internal rules, policies, pricing (shared across all customers)
  2. customer_knowledge.json  — per-customer facts discovered from transcripts
  3. entity_graph.json        — alias/rebrand knowledge graph (grows from discoveries)

Both layers start EMPTY and grow as the agent processes transcripts.
All updates are append-only with a changelog for auditability.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WikiManager:
    def __init__(self, wiki_dir: str = None,
                 wiki_path: str = None, changelog_path: str = None):
        # Support both new (wiki_dir) and old (wiki_path) call signatures
        if wiki_dir:
            self.wiki_dir       = wiki_dir
        else:
            self.wiki_dir       = os.path.dirname(wiki_path) if wiki_path else "data/wiki"

        self.company_path   = os.path.join(self.wiki_dir, "company_knowledge.json")
        self.customer_path  = os.path.join(self.wiki_dir, "customer_knowledge.json")
        self.graph_path     = os.path.join(self.wiki_dir, "entity_graph.json")
        self.changelog_path = os.path.join(self.wiki_dir, "wiki_changelog.json")
        self._load()

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._company:   Dict = self._read(self.company_path,  {"version": "1.0", "entries": []})
        self._customer:  Dict = self._read(self.customer_path, {"version": "1.0", "customers": {}})
        self._graph:     Dict = self._read(self.graph_path,    {"nodes": [], "relationships": []})
        self._changelog: List = self._read(self.changelog_path, [])

    def reload(self) -> None:
        self._load()

    def _read(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    def _write(self, path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_company(self) -> None:
        self._company["last_updated"] = _now()
        self._write(self.company_path, self._company)

    def _save_customer(self) -> None:
        self._customer["last_updated"] = _now()
        self._write(self.customer_path, self._customer)

    def _save_graph(self) -> None:
        self._write(self.graph_path, self._graph)

    def _log(self, action: str, layer: str, subject: str, summary: str, entry_id: str) -> None:
        self._changelog.append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": _now(),
            "action": action,
            "layer": layer,
            "subject": subject,
            "entry_id": entry_id,
            "summary": summary[:120],
        })
        self._write(self.changelog_path, self._changelog)

    # ------------------------------------------------------------------
    # Company wiki (EAS internal rules / policies)
    # ------------------------------------------------------------------

    def add_company_entry(self, category: str, subject: str, content: str,
                          source: str = "agent", confidence: float = 0.9) -> Dict:
        entry = {
            "id": f"cw_{uuid.uuid4().hex[:6]}",
            "category": category,
            "subject": subject,
            "content": content,
            "added_date": _now(),
            "confidence": confidence,
            "source": source,
        }
        self._company["entries"].append(entry)
        self._save_company()
        self._log("ADD", "company", subject, content[:80], entry["id"])
        return entry

    # Legacy compat alias
    def add_entry(self, category: str, subject: str, content: str,
                  source: str = "agent", confidence: float = 0.9,
                  effective_date: str = None) -> Dict:
        return self.add_company_entry(category, subject, content, source, confidence)

    def get_company_context(self, keywords: List[str] = None) -> List[Dict]:
        if not keywords:
            return list(self._company.get("entries", []))
        kws = [k.lower() for k in keywords]
        return [
            e for e in self._company.get("entries", [])
            if any(kw in (e.get("content","") + e.get("subject","")).lower() for kw in kws)
        ]

    # Legacy compat
    def get_all(self) -> List[Dict]: return self.get_company_context()
    def get_by_category(self, c: str) -> List[Dict]:
        return [e for e in self._company.get("entries",[]) if e.get("category") == c]
    def get_context_for(self, company_name: str = "", stage: str = None,
                        customer_id: str = "") -> List[Dict]:
        return self.build_context(customer_id, company_name).get("company_rules", [])

    # ------------------------------------------------------------------
    # Customer wiki (per-customer facts)
    # ------------------------------------------------------------------

    def get_customer_knowledge(self, customer_id: str) -> Dict:
        return self._customer["customers"].get(customer_id, {})

    def update_customer_knowledge(self, customer_id: str, key: str, value: Any,
                                  source: str = "agent", call_id: str = "") -> None:
        if customer_id not in self._customer["customers"]:
            self._customer["customers"][customer_id] = {}
        old = self._customer["customers"][customer_id].get(key, {}).get("value")
        self._customer["customers"][customer_id][key] = {
            "value": value,
            "updated_at": _now(),
            "source": source,
            "call_id": call_id,
        }
        self._save_customer()
        self._log("UPDATE", "customer", customer_id, f"{key}: {old!r} → {value!r}", f"{customer_id}.{key}")

    def get_all_customer_knowledge(self) -> Dict:
        return self._customer.get("customers", {})

    # ------------------------------------------------------------------
    # Entity graph (alias / rebrand)
    # ------------------------------------------------------------------

    def add_entity_alias(self, canonical_name: str, alias: str,
                         relationship_type: str = "alias", confidence: float = 1.0,
                         source_call_id: str = "") -> None:
        node = self._find_entity_node(canonical_name)
        if node is None:
            node = {"canonical_id": f"ent_{uuid.uuid4().hex[:6]}",
                    "canonical_name": canonical_name,
                    "aliases": [], "added_date": _now()}
            self._graph["nodes"].append(node)
        if alias.lower() not in [a.lower() for a in node["aliases"]]:
            node["aliases"].append(alias)
        self._graph["relationships"].append({
            "from": alias, "to": canonical_name,
            "type": relationship_type, "confidence": confidence,
            "discovered_at": _now(), "source_call_id": source_call_id,
        })
        self._save_graph()
        self._log("ADD", "entity_graph", canonical_name,
                  f"alias '{alias}' → '{canonical_name}'", f"rel_{alias}")

    def resolve_entity(self, name: str) -> Optional[str]:
        name_lower = name.strip().lower()
        for node in self._graph.get("nodes", []):
            if node["canonical_name"].lower() == name_lower:
                return node["canonical_name"]
            if any(a.lower() == name_lower for a in node.get("aliases", [])):
                return node["canonical_name"]
        return None

    def _find_entity_node(self, canonical_name: str) -> Optional[Dict]:
        for node in self._graph.get("nodes", []):
            if node["canonical_name"].lower() == canonical_name.lower():
                return node
        return None

    def get_graph_snapshot(self) -> Dict:
        return {
            "nodes": len(self._graph.get("nodes", [])),
            "relationships": len(self._graph.get("relationships", [])),
            "entities": [
                {"canonical": n["canonical_name"], "aliases": n.get("aliases", [])}
                for n in self._graph.get("nodes", [])
            ],
        }

    # ------------------------------------------------------------------
    # Unified context builder
    # ------------------------------------------------------------------

    def build_context(self, customer_id: str, company_name: str = "") -> Dict:
        customer_facts = self.get_customer_knowledge(customer_id)
        company_rules  = self.get_company_context()
        resolved_name  = self.resolve_entity(company_name) if company_name else None
        return {
            "customer_id": customer_id,
            "customer_facts": customer_facts,
            "company_rules": company_rules,
            "resolved_company_name": resolved_name,
            "entity_graph_summary": self.get_graph_snapshot(),
            "total_wiki_entries": len(company_rules) + len(customer_facts),
        }

    def get_changelog(self, limit: int = 20) -> List[Dict]:
        return self._changelog[-limit:]

    def is_empty(self) -> bool:
        return (
            len(self._company.get("entries", [])) == 0 and
            len(self._customer.get("customers", {})) == 0 and
            len(self._graph.get("nodes", [])) == 0
        )



