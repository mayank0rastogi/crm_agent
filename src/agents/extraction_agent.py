"""
Extraction Agent — Part 1 of the pipeline.

Responsibilities:
  1. Receive the meeting transcript + context (participants, opportunity metadata)
  2. Run entity extraction via the mock LLM
  3. Apply entity resolution (alias → canonical company name)
  4. Return a structured ExtractionResult with confidence scores
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from src.agents.base_agent import AgentResult, BaseAgent
from src.knowledge.entity_resolver import EntityResolver
from src.knowledge.wiki_manager import WikiManager
try:
    from src.knowledge.dual_wiki_manager import DualWikiManager as _DualWiki
except ImportError:
    _DualWiki = None
from src.llm_mock.mock_llm import ExtractionResult, MockLLM


class ExtractionAgent(BaseAgent):
    """
    Extracts structured information from a meeting transcript.

    Uses MockLLM for NLP-style extraction, then enriches results
    with entity resolution from the knowledge graph.
    """

    def __init__(
        self,
        llm: MockLLM,
        entity_resolver: EntityResolver,
        wiki_manager,  # WikiManager or DualWikiManager
    ):
        super().__init__("ExtractionAgent")
        self.llm = llm
        self.entity_resolver = entity_resolver
        self.wiki = wiki_manager

    def run(
        self,
        transcript: str,
        context: Dict,  # {"participants": [...], "meeting_date": ..., "opportunity_id": ...}
        run_id: Optional[str] = None,
    ) -> AgentResult:
        result = self._new_result(run_id)
        start = time.time()

        try:
            # Step 1: LLM extraction
            result.add_step("llm_extraction", "Extracting entities from transcript using MockLLM", "running")
            extraction: ExtractionResult = self.llm.extract_entities(transcript, context)
            result.add_step("llm_extraction", "Entity extraction complete", extraction.to_dict())

            # Step 2: Entity resolution — resolve company name aliases
            if extraction.company_name and extraction.company_name.value:
                raw_name = extraction.company_name.value
                resolution = self.entity_resolver.resolve(raw_name)

                if resolution.was_resolved:
                    # Update company name to canonical
                    extraction.company_name.value = resolution.canonical_name
                    extraction.company_name.reasoning += (
                        f" Entity resolved: '{raw_name}' → '{resolution.canonical_name}' "
                        f"(confidence: {resolution.confidence:.2f}). {resolution.reason}"
                    )
                    # Boost confidence if we have entity graph backing
                    extraction.company_name.confidence = min(
                        0.99,
                        extraction.company_name.confidence + 0.05
                    )
                    result.add_step(
                        "entity_resolution",
                        f"Resolved '{raw_name}' → '{resolution.canonical_name}'",
                        {
                            "input": raw_name,
                            "canonical": resolution.canonical_name,
                            "confidence": resolution.confidence,
                            "reason": resolution.reason,
                        },
                        confidence=resolution.confidence,
                    )
                else:
                    result.add_step(
                        "entity_resolution",
                        f"No alias found for '{raw_name}' — using as-is",
                        {"input": raw_name, "reason": resolution.reason},
                    )

            # Step 3: Enrich with wiki context
            # Supports both WikiManager (legacy) and DualWikiManager (new)
            company = (
                extraction.company_name.value
                if extraction.company_name and extraction.company_name.value
                else ""
            )
            customer_id = context.get("customer_id", "")
            if hasattr(self.wiki, "get_context_for_call") and customer_id:
                wiki_ctx = self.wiki.get_context_for_call(customer_id)
                wiki_entries = wiki_ctx.get("company_wiki", []) + wiki_ctx.get("customer_wiki", [])
            elif hasattr(self.wiki, "get_context_for"):
                wiki_entries = self.wiki.get_context_for(
                    company,
                    stage=extraction.opportunity_stage.value if extraction.opportunity_stage else None,
                )
            else:
                wiki_entries = []
            result.add_step(
                "wiki_enrichment",
                f"Loaded {len(wiki_entries)} relevant wiki entries",
                [e["id"] for e in wiki_entries],
            )

            result.output = {
                "extraction": extraction,
                "wiki_context": wiki_entries,
                "resolved_company": company,
            }
            result.status = "success"

        except Exception as exc:
            result.status = "error"
            result.error = str(exc)

        result.duration_ms = (time.time() - start) * 1000
        return result
