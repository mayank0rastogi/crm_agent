"""
Entity Resolver — Knowledge Graph for company alias resolution.

Resolves aliases, old names, and abbreviations to canonical company names.
Data is stored in entity_graph.json and loaded at runtime.

Bonus feature: Implements the knowledge graph described in the assignment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EntityNode:
    canonical_id: str
    canonical_name: str
    aliases: List[str]
    domain: str
    deprecated_domains: List[str]
    industry: str
    relationship_notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "EntityNode":
        return cls(
            canonical_id=d["canonical_id"],
            canonical_name=d["canonical_name"],
            aliases=[a.lower() for a in d.get("aliases", [])],
            domain=d.get("domain", ""),
            deprecated_domains=d.get("deprecated_domains", []),
            industry=d.get("industry", ""),
            relationship_notes=d.get("relationship_notes"),
        )


@dataclass
class ResolutionResult:
    input_name: str
    canonical_name: str
    was_resolved: bool
    confidence: float
    reason: str
    node: Optional[EntityNode] = None


class EntityResolver:
    """
    Resolves company names (aliases, old names, typos) to canonical names.

    Uses an exact-match lookup against the entity graph.
    In a production system, this could be extended with fuzzy matching
    (e.g., Levenshtein distance) or embedding-based similarity.
    """

    def __init__(self, entity_graph_path: str):
        self._nodes: List[EntityNode] = []
        self._relationships: List[Dict] = []
        self._load(entity_graph_path)

        # Build reverse lookup: alias → node
        self._alias_index: Dict[str, EntityNode] = {}
        for node in self._nodes:
            self._alias_index[node.canonical_name.lower()] = node
            for alias in node.aliases:
                self._alias_index[alias.lower()] = node

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)
        self._nodes = [EntityNode.from_dict(n) for n in data.get("nodes", [])]
        self._relationships = data.get("relationships", [])

    def resolve(self, name: str) -> ResolutionResult:
        """
        Resolve a company name to its canonical form.

        Returns a ResolutionResult with the canonical name, confidence,
        and explanation.
        """
        if not name:
            return ResolutionResult(
                input_name=name,
                canonical_name=name,
                was_resolved=False,
                confidence=0.0,
                reason="Empty name provided.",
            )

        name_lower = name.strip().lower()

        # Exact match (canonical or alias)
        if name_lower in self._alias_index:
            node = self._alias_index[name_lower]
            was_resolved = node.canonical_name.lower() != name_lower

            return ResolutionResult(
                input_name=name,
                canonical_name=node.canonical_name,
                was_resolved=was_resolved,
                confidence=1.0,
                reason=(
                    f"Exact match in entity graph: '{name}' is a known alias of "
                    f"'{node.canonical_name}'."
                    if was_resolved
                    else f"'{name}' is already the canonical name."
                ),
                node=node,
            )

        # Partial match (name is a substring of a canonical name or alias)
        for alias, node in self._alias_index.items():
            if name_lower in alias or alias in name_lower:
                return ResolutionResult(
                    input_name=name,
                    canonical_name=node.canonical_name,
                    was_resolved=True,
                    confidence=0.75,
                    reason=(
                        f"Partial match: '{name}' partially matches alias '{alias}' "
                        f"of '{node.canonical_name}'."
                    ),
                    node=node,
                )

        # No match
        return ResolutionResult(
            input_name=name,
            canonical_name=name,
            was_resolved=False,
            confidence=0.5,
            reason=f"'{name}' not found in entity graph. Using as-is.",
        )

    def get_domain_for(self, canonical_name: str) -> Optional[str]:
        """Get the official domain for a canonical company name."""
        name_lower = canonical_name.lower()
        node = self._alias_index.get(name_lower)
        return node.domain if node else None

    def is_deprecated_domain(self, domain: str, canonical_name: str) -> bool:
        """Check if a domain is deprecated for a given canonical company."""
        node = self._alias_index.get(canonical_name.lower())
        if not node:
            return False
        return domain.lower() in [d.lower() for d in node.deprecated_domains]

    def get_all_aliases(self, canonical_name: str) -> List[str]:
        """Return all known aliases for a canonical company."""
        node = self._alias_index.get(canonical_name.lower())
        return node.aliases if node else []
