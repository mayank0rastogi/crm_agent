# Autonomous CRM Agent — TechServ Solutions

An autonomous AI pipeline that processes customer meeting transcripts and recommends validated CRM updates — built in **pure Python (zero external dependencies)**, using a Mock LLM, deterministic guardrails, a critique/review agent, and an interactive HITL simulation.

---

## The Scenario

**TechServ Solutions** is a SaaS company. Their sales reps manage enterprise deals. After every customer meeting the platform:

1. Reads the transcript, current CRM record, company wiki, and customer knowledge
2. Extracts structured information (company name, revenue, intent, actions, concerns)
3. Validates and proposes CRM updates with confidence scores
4. Routes low-confidence updates to a human reviewer
5. Applies approved updates and grows the knowledge base

### Two customers demonstrated

| Customer | Starting state | Journey | Outcome |
|---|---|---|---|
| `cust_001` XYZ Air Services *(prev. XYZ Aviation Group)* | Qualification, $0 revenue | 3 calls | **Closed Won — $375k/yr, 3-year** |
| `cust_002` Pinnacle Logistics | Qualification, $0 revenue | 1 call | **Nurture — low intent, re-engage Q3** |

Call 1 for cust_001 demonstrates the **entity rename detection** from the assignment:
- CRM has: `company_name: XYZ Aviation Group`
- Customer says in call: *"you may still have us listed as XYZ Aviation Group — we rebranded to XYZ Air Services"*
- Agent detects via `EntityResolver` → validates against `entity_graph.json` → updates CRM with `confidence: 0.99`

---

## Architecture

```
TechServ Solutions CRM  (one selling company, multiple customers)
│
│  Company Wiki  — pricing rules, process policies, compliance (shared, never reset)
│  Entity Graph  — alias/rebrand knowledge (XYZ Aviation Group → XYZ Air Services)
│
├── cust_001: XYZ Air Services
│     Customer Wiki   — grows after each call (empty at start)
│     Call 1 ──► Pipeline ──► Rebrand detected, Discovery stage, security flagged
│     Call 2 ──► Pipeline ──► Security cleared, Proposal stage
│     Call 3 ──► Pipeline ──► Contract signed, Closed Won, $375k locked
│
└── cust_002: Pinnacle Logistics
      Customer Wiki   — grows after each call
      Call 1 ──► Pipeline ──► LOW intent, budget mismatch → NURTURE
```

### Per-call pipeline

```
[Transcript + CRM + Company Wiki + Customer Wiki]
           │
           ▼
[1] ExtractionAgent          — Mock LLM extracts company, users, revenue, intent,
     │  + EntityResolver          stage, timeline, actions, concerns, sentiment.
     │                           Rebrand aliases resolved to canonical name.
     ▼
[2] CRMUpdateAgent           — Proposes field updates with confidence + evidence.
     │  + GuardrailsEngine        Stage transition validator, revenue range check,
     │  + RuleEngine              required-docs check, account name validation.
     │                           Declarative rules (JSON-configurable) block violations.
     ▼
[3] ReviewAgent              — Second-pass critique:
     │                           Hallucination detection, missing evidence, blocked
     │                           proposals → APPROVED / REVISED / REJECTED / NEEDS_HUMAN_REVIEW
     ▼
[4] HITL Gate                — Low-confidence updates shown interactively.
     │                           User: [A]pprove / [R]eject / [E]dit
     │                           Approved facts optionally saved to customer wiki.
     ▼
[5] CustomerCRMStore         — Approved updates written to JSON CRM record.
     │                           Full audit trail maintained.
     ▼
[6] Wiki Growth              — High-confidence facts auto-added to customer wiki
                               (contacts, intent, sentiment, NBA, deal outcome).
```

---

## Project Structure

```
crm-agent/
├── main.py                          # Interactive call-by-call demo runner
├── run_evaluation.py                # Evaluation framework (Part 4)
├── init_data.py                     # Reset all data to clean initial state
├── requirements.txt
│
├── data/
│   ├── company/
│   │   ├── company_info.json        # TechServ Solutions metadata
│   │   ├── company_wiki.json        # Rules, pricing, policies (never reset)
│   │   └── company_wiki_changelog.json
│   │
│   ├── customers/
│   │   ├── cust_001/                # XYZ Air Services (prev. XYZ Aviation Group)
│   │   │   ├── profile.json         # Company info, contacts, tier
│   │   │   ├── crm_opportunity.json # Live CRM record — updated per call
│   │   │   ├── wiki.json            # Customer wiki — EMPTY at start, grows per call
│   │   │   ├── wiki_changelog.json
│   │   │   └── transcripts/
│   │   │       ├── call_001.json    # Discovery + rebrand detection
│   │   │       ├── call_002.json    # Follow-up: security cleared, ready for proposal
│   │   │       └── call_003.json    # Contract signing → Closed Won
│   │   │
│   │   └── cust_002/                # Pinnacle Logistics
│   │       ├── profile.json
│   │       ├── crm_opportunity.json
│   │       ├── wiki.json
│   │       ├── wiki_changelog.json
│   │       └── transcripts/
│   │           └── call_001.json    # Low-intent discovery → Nurture
│   │
│   ├── crm/
│   │   └── audit_log.json           # Full audit trail of all CRM changes
│   │
│   └── wiki/
│       └── entity_graph.json        # Knowledge graph: alias → canonical name
│
└── src/
    ├── simulation.py                # Orchestrator (interactive, call-by-call)
    ├── display.py                   # Terminal pretty-printer
    │
    ├── llm_mock/
    │   └── mock_llm.py              # Simulated LLM — no external API
    │
    ├── agents/
    │   ├── base_agent.py            # Reasoning trace + timing base class
    │   ├── extraction_agent.py      # Part 1 — entity extraction + entity resolution
    │   ├── crm_update_agent.py      # Part 2 — proposals + guardrail checks
    │   └── review_agent.py          # Part 3 — critique, hallucination detection
    │
    ├── guardrails/
    │   ├── rule_engine.py           # Declarative rule evaluator (JSON-configurable)
    │   ├── stage_transitions.py     # Valid CRM stage transition map + entry requirements
    │   └── validators.py            # Revenue range, name change, email domain, docs
    │
    ├── knowledge/
    │   ├── dual_wiki_manager.py     # Two-layer wiki: company + per-customer
    │   ├── wiki_manager.py          # Legacy single-layer wiki (backward compat)
    │   └── entity_resolver.py       # Alias → canonical name via entity graph
    │
    ├── crm/
    │   └── customer_crm_store.py    # Customer-centric CRM read/write + audit
    │
    └── evaluation/
        ├── evaluator.py             # Evaluation metrics + scoring
        └── test_cases.py            # 5 manually crafted test cases
```

---

## Knowledge Layer Explained

### `dual_wiki_manager.py` — Two-layer wiki manager

Manages **two separate knowledge bases** that serve different purposes:

**Layer 1 — Company wiki** (`data/company/company_wiki.json`)
- TechServ Solutions' internal institutional knowledge: pricing, compliance rules, process policies
- Pre-loaded with 7 entries (pricing tiers, security requirements, follow-up policy, rebrand history)
- **Never reset** — this is the persistent company memory that survives all demo runs
- New entries can be appended (e.g. a new pricing rule) and persist forever
- Queried by all agents to validate decisions (e.g. "security questionnaire before proposal" → `cwiki_003`)

**Layer 2 — Customer wiki** (`data/customers/{id}/wiki.json`)
- What TechServ knows about **each specific customer** — discovered from transcripts
- **Starts completely empty** at `init_data.py` reset
- Grows automatically after every processed call:
  - Contacts seen in the meeting
  - Buying intent level per call
  - Sentiment trend
  - Next best action recommendation
  - Deal outcome (loss reason, contract status, etc.)
- Also grows from HITL approvals: user can save confirmed facts (revenue, deadline) directly to wiki
- Every addition is append-only and logged to `wiki_changelog.json` with timestamp

The `DualWikiManager` is the single interface agents use to query both layers at once via `get_context_for_call(customer_id)`.

---

### `wiki_manager.py` — Legacy wiki manager (kept for backward compatibility)

An earlier single-layer implementation used by `src/pipeline.py` (the old non-interactive pipeline used by the evaluation framework's `run_inline`). It reads from `data/wiki/` instead of the new `data/company/` + `data/customers/` structure. **Agents in the live simulation use `DualWikiManager` instead.**

---

### `entity_resolver.py` — Knowledge graph alias resolver

Implements the **Bonus: Knowledge Graph** feature from the assignment.

Resolves any company name variant — old names, abbreviations, misspellings — to its **canonical legal name** using the entity graph stored in `data/wiki/entity_graph.json`.

**How it works:**
1. `entity_graph.json` stores nodes (canonical names + all known aliases) and relationships (rename events with dates and confidence)
2. When `ExtractionAgent` extracts a company name from a transcript, it calls `EntityResolver.resolve(name)`
3. The resolver does an exact match lookup against all aliases across all nodes
4. If found: returns the canonical name, the confidence, and the reason
5. If not exact: tries partial/substring match at lower confidence

**The assignment example reproduced exactly:**
```
entity_graph.json:
  node: canonical_name="XYZ Air Services"
        aliases=["XYZ Aviation Group", "xyz aviation", ...]
        relationships: XYZ Aviation Group → XYZ Air Services (renamed_to, 2025-01-01)

Transcript: "you may still have us listed as XYZ Aviation Group"

EntityResolver.resolve("XYZ Aviation Group"):
  → canonical_name = "XYZ Air Services"
  → confidence     = 1.0
  → reason         = "Exact match in entity graph: 'XYZ Aviation Group' is a known alias"

CRM update: account_name  XYZ Aviation Group → XYZ Air Services  (conf: 0.99)
            reasoning: "Confirmed by entity graph (wiki ref: cwiki_000): renamed in 2025"
```

---

## CRM Fields Tracked Per Customer

| Field | Description |
|---|---|
| `stage` | Qualification → Discovery → Proposal → Negotiation → Closed Won/Lost |
| `expected_revenue` | Estimated deal value (updated per call) |
| `actual_revenue` | Confirmed on contract signing |
| `num_users` | Seat count |
| `deal_duration_years` | Contract length (e.g. 3-year term) |
| `close_date` | Hard deadline extracted from transcript |
| `renewal_date` | Populated when contract is signed |
| `conversion_likelihood` | 0–100%, driven by stage + intent + milestones hit |
| `next_best_action` | SEND_SECURITY_QUESTIONNAIRE / SEND_FORMAL_PROPOSAL / INITIATE_ONBOARDING / NURTURE |
| `loss_reason` | Why deal stalled or was lost |
| `follow_up_recommended` | Boolean + rationale |
| `notes_history` | Append-only meeting summaries per call |
| `follow_up_tasks` | Action items per call |
| `custom_fields.*` | security_questionnaire_sent, proposal_sent, contract_signed, etc. |

---

## Mock LLM Design

`src/llm_mock/mock_llm.py` — zero external API calls. Simulates LLM behavior using:

- **Regex + keyword pattern matching** for entity extraction (company names, users, revenue, dates)
- **Priority signal detection** for stage classification (e.g. "board signed off" → Closed Won regardless of score)
- **Weighted keyword scoring** for intent (HIGH/MEDIUM/LOW/LOST) and sentiment
- **Confidence scoring** per field (0.0–1.0) based on evidence strength (explicit vs inferred)
- **Merged custom-fields context** — NBA and likelihood use the proposed state of this call, not the stale CRM
- **Hallucination injection** (configurable rate) to stress-test the Review Agent

To swap in a real LLM: subclass `BaseLLM` and implement `extract_entities`, `propose_crm_updates`, `review_proposals`.

---

## Guardrails (Deterministic — Not LLM-Based)

Rules are pure Python + declarative JSON — they cannot be "reasoned around" by an LLM:

| Rule | Trigger | Severity |
|---|---|---|
| Security questionnaire before proposal | Stage → Proposal/Negotiation | ERROR |
| Enterprise deals >$100k need security review | Revenue > $100,000 | ERROR |
| Proposal must precede Negotiation | Stage → Negotiation | ERROR |
| EU data residency must be flagged | `data_residency` concern detected | ERROR |
| Revenue within tier bounds | Any revenue update | WARNING |
| Account name validated by entity graph | Account name change | WARNING |
| Email domain matches known canonical domain | Account name change with deprecated domain | NOTE |
| Stage regression warning | Moving backwards in pipeline | WARNING |

---

## Conversion Likelihood — How It's Calculated

Likelihood answers: *"how likely is this deal to close, given what just happened in this call?"*

| Situation | Likelihood |
|---|---|
| Contract signed / "board signed off" | **100%** |
| Deal lost (LOST intent) | **0%** |
| LOW intent, no decision maker, no budget | **15%** |
| Currently at Qualification stage + HIGH intent | **47%** (45% base + modifiers) |
| Currently at Discovery + stage advanced this call | **75%** (65% + 8% advance + 2% intent) |
| Currently at Proposal + security cleared + proposal sent | **97%** |

---

## Setup & Run

```bash
# No pip install needed for core — pure Python 3.8+
python init_data.py    # Reset all CRM + customer wikis to initial state (run before every demo)
python main.py         # Interactive call-by-call demo

# Optional: graph visualization
pip install networkx matplotlib
```

### Workflow in main.py

```
  1  XYZ Air Services — Call 1  (Discovery + rebrand: XYZ Aviation Group → XYZ Air Services)
  2  XYZ Air Services — Call 2  (Follow-up: security cleared, ready for proposal)
  3  XYZ Air Services — Call 3  (Contract signing → Closed Won)
  4  Pinnacle Logistics — Call 1  (Discovery / Low intent → Nurture)
  c  Show current CRM state (all customers)
  r  Reset all data
```

After each call: full CRM diff printed — `yellow` = changed field, `green` = new field.

---

## Assumptions

1. **Mock LLM** — all NLP done via pattern matching; swap `MockLLM` for any real LLM without changing agents
2. **CRM storage** — JSON files; maps 1:1 to Salesforce/HubSpot API calls in production
3. **Company wiki** is append-only and never reset — it's the permanent institutional memory
4. **Customer wiki** starts empty and grows per call — demonstrating wiki evolution from zero
5. **Revenue confidence** — values inferred (not explicitly stated) stay below 0.65 → routed to HITL
6. **Stage moves forward only** — regression requires explicit override
7. **Entity graph** starts pre-loaded with known aliases; grows as agent discovers new ones from transcripts

---

## Extensibility

| Task | How |
|---|---|
| Add a guardrail rule | Add entry to `BUILTIN_RULES` in `src/guardrails/rule_engine.py` |
| Add a new customer | Create `data/customers/cust_XXX/` with profile + crm + transcripts |
| Swap to a real LLM | Subclass `BaseLLM` in `src/llm_mock/mock_llm.py` |
| Add a CRM field | Add to `crm_opportunity.json` + `propose_crm_updates()` in `mock_llm.py` |
| Add an entity alias | Add to `entity_graph.json` → resolver picks it up at runtime |
| Add a test case | Add to `src/evaluation/test_cases.py` |

---

## Bonus Features Implemented

- **Knowledge graph** entity alias resolution — `XYZ Aviation Group` → `XYZ Air Services`
- **Confidence scoring** per extracted field (0.0–1.0) with evidence + reasoning
- **Human-in-the-loop** workflow — [A]pprove / [R]eject / [E]dit for low-confidence updates
- **Rule engine** combining declarative rules + LLM reasoning (neither can bypass the other)
- **Explainability** — every CRM update shows `reasoning` and `evidence` fields
- **Per-customer wiki** growing linearly from empty state — full audit trail
- **Conversion likelihood** driven by stage progress, milestones, and intent signals
- **Next best action** — SEND_SECURITY_QUESTIONNAIRE → SEND_FORMAL_PROPOSAL → INITIATE_ONBOARDING → NURTURE
- **Deal journey narrative** — printed after all calls: stage progression, events, outcome


An autonomous AI pipeline that processes customer meeting transcripts and recommends validated CRM updates. Built with **zero required external dependencies** (pure Python 3.8+), using a Mock LLM, deterministic guardrails, a critique/review agent, and an interactive human-in-the-loop simulation.

---

## The Scenario

**TechServ Solutions** is a SaaS company whose sales reps manage deals with enterprise customers. After every customer meeting, the AI pipeline:

1. Reads the transcript
2. Extracts structured information (company, revenue, intent, actions)
3. Proposes CRM field updates with confidence scores
4. Runs deterministic guardrail validation
5. Routes low-confidence updates to a human reviewer
6. Applies approved updates and grows the knowledge wiki

### Two customers demonstrated:

| Customer | Journey | Outcome |
|---|---|---|
| `cust_001` XYZ Air Services | 3 calls: discovery → follow-up → contract signing | **Closed Won — $375,000** |
| `cust_002` Pinnacle Logistics | 1 call: exploratory, no decision maker, budget mismatch | **Nurture — low intent** |

---

## Architecture

```
TechServ Solutions (ONE CRM, ONE selling company)
│
│  Company Wiki (rules, pricing, processes)
│  ─ grows as new knowledge is confirmed
│
├── cust_001: XYZ Air Services
│     ├── Customer Wiki (grows per call)
│     ├── Call 1 ──► ExtractionAgent ──► GuardrailsEngine ──► CRMUpdateAgent ──► ReviewAgent ──► HITL ──► CRM ✓
│     ├── Call 2 ──► (same pipeline) ──► CRM updated again ✓
│     └── Call 3 ──► Contract signed ──► Closed Won ✓
│
└── cust_002: Pinnacle Logistics
      ├── Customer Wiki (grows per call)
      └── Call 1 ──► LOW INTENT detected ──► NURTURE + follow-up recommended
```

### Pipeline stages per call

```
[Transcript]
     │
     ▼
[1] ExtractionAgent      — Mock LLM extracts: company, users, revenue, intent,
                           timeline, actions, concerns, sentiment
                           + EntityResolver maps aliases → canonical names
     │
     ▼
[2] CRMUpdateAgent       — Proposes field updates (stage, revenue, contacts,
     │                     next_best_action, conversion_likelihood, loss_reason…)
     │
     ├─► GuardrailsEngine  — Deterministic validation:
     │                       • Stage transition rules
     │                       • Revenue range checks
     │                       • Required docs before stage advance
     │                       • Account name / entity graph validation
     │
     └─► RuleEngine        — Declarative JSON-configurable business rules
                             (e.g. "security questionnaire before proposal")
     │
     ▼
[3] ReviewAgent          — Second-pass critique:
                           • Hallucination detection
                           • Missing evidence flags
                           • Rule violation re-check
                           → APPROVED / REVISED / REJECTED / NEEDS_HUMAN_REVIEW
     │
     ▼
[4] Human-in-the-Loop    — Interactive terminal prompts for low-confidence updates
                           User: [A]pprove / [R]eject / [E]dit
                           Approved facts optionally saved to customer wiki
     │
     ▼
[5] CRM Store            — Updates written to customer JSON record
                           Full audit log maintained

[6] Wiki Growth          — Auto-adds confirmed facts to customer wiki after each call
                           (contacts, intent, sentiment, loss reason, NBA…)
```

---

## Project Structure

```
crm-agent/
├── main.py                          # Interactive simulation runner
├── run_evaluation.py                # Evaluation suite
├── requirements.txt
│
├── data/
│   ├── company/
│   │   ├── company_info.json        # TechServ Solutions metadata
│   │   ├── company_wiki.json        # Company rules, pricing, policies
│   │   └── company_wiki_changelog.json
│   │
│   └── customers/
│       ├── cust_001/                # XYZ Air Services
│       │   ├── profile.json         # Company info, contacts, tier
│       │   ├── crm_opportunity.json # Live CRM record (updated per call)
│       │   ├── wiki.json            # Customer wiki (EMPTY at start, grows)
│       │   ├── wiki_changelog.json  # Audit trail of wiki additions
│       │   └── transcripts/
│       │       ├── call_001.json    # Discovery call
│       │       ├── call_002.json    # Follow-up (security cleared)
│       │       └── call_003.json    # Contract signing
│       │
│       └── cust_002/                # Pinnacle Logistics
│           ├── profile.json
│           ├── crm_opportunity.json
│           ├── wiki.json            # EMPTY at start, grows
│           ├── wiki_changelog.json
│           └── transcripts/
│               └── call_001.json   # Low-intent discovery
│
└── src/
    ├── simulation.py                # Interactive orchestrator (main entry)
    ├── visualization.py             # NetworkX journey graph (optional)
    ├── display.py                   # Terminal pretty-printer
    │
    ├── llm_mock/
    │   └── mock_llm.py              # Simulated LLM — pattern matching + confidence
    │
    ├── agents/
    │   ├── base_agent.py            # Reasoning trace + timing base class
    │   ├── extraction_agent.py      # Part 1 — entity extraction
    │   ├── crm_update_agent.py      # Part 2 — proposals + guardrails
    │   └── review_agent.py          # Part 3 — critique + approval
    │
    ├── guardrails/
    │   ├── rule_engine.py           # Declarative rule evaluator
    │   ├── stage_transitions.py     # Valid CRM stage transition map
    │   └── validators.py            # Revenue, name, domain, docs validators
    │
    ├── knowledge/
    │   ├── dual_wiki_manager.py     # Company wiki + per-customer wiki
    │   └── entity_resolver.py       # Alias → canonical name (knowledge graph)
    │
    ├── crm/
    │   └── customer_crm_store.py   # Customer-centric CRM read/write
    │
    └── evaluation/
        ├── evaluator.py             # Metrics: extraction, stage, guardrail, hallucination
        └── test_cases.py            # 5 manually crafted test cases
```

---

## Setup & Run

```bash
# Python 3.8+ — no install needed for core features
python main.py

# With graph visualization (optional)
pip install networkx matplotlib
python main.py
# → Saves PNG graphs to output/graphs/

# Run evaluation suite
python run_evaluation.py
```

---

## CRM Fields Tracked Per Customer

| Field | Description |
|---|---|
| `stage` | Qualification → Discovery → Proposal → Negotiation → Closed Won/Lost |
| `expected_revenue` | Estimated deal value (updated per call) |
| `actual_revenue` | Confirmed on signing |
| `num_users` | Seat count (extracted from transcript) |
| `deal_duration_years` | Contract length |
| `close_date` | Hard deadline extracted from transcript |
| `renewal_date` | Populated when contract is signed |
| `conversion_likelihood` | 0–100% based on buying intent signal |
| `next_best_action` | AI-recommended next step (SEND_PROPOSAL, NURTURE, etc.) |
| `loss_reason` | Why deal stalled or was lost |
| `follow_up_recommended` | Boolean + rationale |
| `follow_up_rationale` | Explanation of follow-up recommendation |
| `notes_history` | Append-only meeting summaries |
| `follow_up_tasks` | Action items extracted from each call |
| `custom_fields` | security_questionnaire_sent, proposal_sent, data_residency, etc. |

---

## Mock LLM Design

The `MockLLM` class simulates LLM behaviour without any API calls:

- **Entity extraction** — regex + keyword pattern matching on transcript text
- **Intent classification** — weighted keyword scoring (HIGH / MEDIUM / LOW / LOST)
- **Stage prediction** — decision-tree logic from transcript signals
- **Confidence scoring** — per-field 0.0–1.0 scores based on evidence strength
- **Hallucination injection** — configurable rate to stress-test the Review Agent
- **Intelligence fields** — derives `next_best_action`, `conversion_likelihood`, `loss_reason`

To swap in a real LLM: subclass `BaseLLM` in `src/llm_mock/mock_llm.py` and implement the same three methods.

---

## Wiki Architecture

Two wiki layers per run:

**Company wiki** (`data/company/company_wiki.json`)
- TechServ's internal rules, pricing, process policies
- Pre-loaded with 6 entries; append-only

**Customer wiki** (`data/customers/{id}/wiki.json`)
- Starts **empty** for each customer
- Auto-populated after every processed call (contacts, intent, sentiment, NBA)
- Human-approved facts added interactively via HITL
- Every addition logged to `wiki_changelog.json` with timestamp

---

## Guardrails (Deterministic — Not LLM-Based)

Rules are pure Python + declarative JSON — they cannot be "reasoned around" by an LLM:

| Rule | Trigger | Severity |
|---|---|---|
| Security questionnaire before proposal | Stage → Proposal/Negotiation | ERROR |
| Enterprise deals >$100k need security review | Revenue > $100,000 | ERROR |
| Proposal must precede Negotiation | Stage → Negotiation | ERROR |
| EU data residency must be flagged | `data_residency` concern detected | ERROR |
| Revenue within tier bounds | Any revenue update | WARNING |
| Account name validated by entity graph | Account name change | WARNING |
| Stage regression warning | Moving backwards in pipeline | WARNING |

---

## Human-in-the-Loop

Updates below **0.65 confidence** are flagged `NEEDS_HUMAN_REVIEW` and shown interactively:

```
  ⚠  REVIEW NEEDED
  Field: expected_revenue  │  Value: $375,000  │  Confidence: 0.62
  Reason: Revenue inferred from user count × pricing — not stated explicitly

  [A]pprove / [R]eject / [E]dit: _
```

On approval, the user is also offered to save the confirmed fact to the **customer wiki** for future calls.

---

## Evaluation Framework

`run_evaluation.py` tests 5 hand-crafted scenarios:

| Metric | How measured |
|---|---|
| Extraction accuracy | Field-level exact/fuzzy match vs. ground truth |
| Stage prediction accuracy | Correct stage classification rate |
| Action recall / precision | Jaccard match on required follow-up actions |
| Guardrail effectiveness | % of expected violations correctly caught |
| Hallucination rate | Fields flagged with no transcript evidence |
| Reviewer effectiveness | % of injected issues caught by Review Agent |

---

## Assumptions

1. **Mock LLM** — all NLP done via pattern matching; swap `MockLLM` for any real LLM client without changing agents
2. **CRM storage** — JSON files; maps 1:1 to Salesforce/HubSpot API calls in production
3. **Wiki is append-only** — entries never deleted; preserves full audit history
4. **Revenue confidence** — values inferred (not explicitly stated) are below auto-apply threshold and routed to HITL
5. **Stage progression** — only forward movement by default; regression requires explicit override
6. **One selling company** — TechServ Solutions owns the CRM; customers are tracked as `cust_001`, `cust_002`, etc.

---

## Extensibility

| Task | How |
|---|---|
| Add a guardrail rule | Add entry to `BUILTIN_RULES` in `src/guardrails/rule_engine.py` |
| Add a new customer | Create `data/customers/cust_XXX/` folder with profile + CRM + transcript files |
| Swap to a real LLM | Subclass `BaseLLM` in `src/llm_mock/mock_llm.py` |
| Add a new CRM field | Add to `crm_opportunity.json` schema and `propose_crm_updates()` in MockLLM |
| Add a test case | Add to `src/evaluation/test_cases.py` |

---

## Bonus Features Implemented

- **Knowledge graph** entity alias resolution (`data/wiki/entity_graph.json`)
- **Confidence scoring** per extracted field (0.0–1.0)
- **Human-in-the-loop** workflow with [A]pprove / [R]eject / [E]dit
- **Rule engine** combining declarative rules + LLM reasoning
- **Explainability** — every CRM update includes `reasoning` and `evidence` fields
- **Per-customer wiki** that grows linearly from empty state
- **Audit trail** — every change logged with timestamp, run ID, and confidence
- **NetworkX visualization** — customer journey graph (PNG + ASCII fallback)


An autonomous AI workflow that processes customer meeting transcripts and recommends validated CRM updates — built with **zero external dependencies**, using a mock LLM, deterministic guardrails, and a critique/review agent.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRM AGENT PIPELINE                           │
│                                                                     │
│  Meeting Transcript                                                 │
│  + CRM Opportunity   ──►  [1] Extraction Agent                     │
│  + Company Wiki              │  (Mock LLM + Regex)                 │
│  + Historical Context        │  Structured entities, intent,        │
│                              │  sentiment, buying signals           │
│                              ▼                                      │
│                      [2] Guardrails Engine (Deterministic)          │
│                              │  • Stage transition rules            │
│                              │  • Revenue range validation          │
│                              │  • Entity resolution (alias graph)   │
│                              │  • Domain mismatch detection         │
│                              │  • Required docs checklist           │
│                              │  • Business rule violations          │
│                              ▼                                      │
│                      [3] CRM Update Agent                           │
│                              │  (Mock LLM + Rule Engine)            │
│                              │  Proposes field updates with         │
│                              │  confidence scores + explainability  │
│                              ▼                                      │
│                      [4] Review Agent (Critique)                    │
│                              │  Second-pass validation              │
│                              │  Checks hallucinations, missing      │
│                              │  evidence, violated rules            │
│                              │  → APPROVE / REVISE / REJECT         │
│                              ▼                                      │
│                      [5] Human-in-the-Loop Gate                     │
│                              │  Low-confidence updates flagged      │
│                              │  for human review                    │
│                              ▼                                      │
│                      [6] CRM Store (JSON)                           │
│                              Audit trail, versioning, changelog     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
crm-agent/
├── main.py                      # Interactive demo runner
├── run_evaluation.py            # Evaluation framework runner
├── requirements.txt             # Zero external deps (stdlib only)
├── README.md
│
├── data/
│   ├── transcripts/             # Sample meeting transcripts (JSON)
│   ├── crm/                     # Mock CRM store (opportunities, accounts)
│   └── wiki/
│       ├── company_knowledge.json   # Company wiki — grows linearly
│       ├── entity_graph.json        # Knowledge graph: alias resolution
│       └── wiki_changelog.json      # Real-time wiki update log
│
└── src/
    ├── llm_mock/
    │   └── mock_llm.py          # Simulated LLM with confidence scoring
    ├── agents/
    │   ├── base_agent.py        # Agent base class with reasoning trace
    │   ├── extraction_agent.py  # Extracts structured info from transcript
    │   ├── crm_update_agent.py  # Proposes CRM field updates
    │   └── review_agent.py      # Critique + approval/rejection
    ├── guardrails/
    │   ├── rule_engine.py       # Declarative rule engine
    │   ├── validators.py        # Deterministic validation functions
    │   └── stage_transitions.py # Valid CRM stage transition map
    ├── knowledge/
    │   ├── wiki_manager.py      # Wiki CRUD with real-time updates
    │   └── entity_resolver.py   # Entity alias resolution (knowledge graph)
    ├── crm/
    │   └── crm_store.py         # CRM read/write with versioning
    └── evaluation/
        ├── evaluator.py         # Evaluation metrics + scoring
        └── test_cases.py        # Manually crafted test suite
```

---

## Key Design Decisions

### 1. Mock LLM (`src/llm_mock/mock_llm.py`)
Rather than calling an external API, the mock LLM simulates realistic LLM behavior using:
- **Regex + keyword pattern matching** for entity extraction
- **Weighted decision trees** for stage/intent classification
- **Confidence scoring** (0.0–1.0) per extracted field
- **Hallucination injection** (configurable rate) to test the reviewer
- **Template-based reasoning** for explainability strings

This lets the system run fully offline while testing real agentic behavior.

### 2. Deterministic Guardrails (not LLM-based)
Business rules are implemented as pure Python functions — not prompts. This is intentional:
- LLMs can "reason around" rules if they're in prompts
- Deterministic code guarantees rule enforcement
- Rules are declarative (JSON-configurable) via the rule engine

### 3. Entity Resolution (Bonus)
A lightweight knowledge graph (`data/wiki/entity_graph.json`) maps company aliases to their canonical legal name:
```json
{
  "canonical_name": "XYZ Air Services",
  "aliases": ["XYZ Aviation Group", "xyz aviation"],
  "relationships": [{"from": "XYZ Aviation Group", "to": "XYZ Air Services", "type": "renamed_to"}]
}
```
In Call 1, the CRM has `company_name: XYZ Aviation Group`. The customer says *"you may still have us listed as XYZ Aviation Group — we rebranded to XYZ Air Services"*. The `EntityResolver` resolves this alias to the canonical name and the guardrail validates it against the entity graph before updating the CRM.

### 4. Wiki as Living Knowledge
`company_knowledge.json` is append-only with versioned entries. Every update writes a `wiki_changelog.json` entry with timestamp. The wiki is queried by the extraction and update agents at runtime.

### 5. Human-in-the-Loop
Updates with confidence < 0.6 are flagged as `NEEDS_HUMAN_REVIEW` rather than auto-applied. The demo simulates a human approval prompt.

### 6. Audit Trail
Every pipeline run writes a full trace: input → extraction → guardrail results → proposed updates → review verdict → final applied changes. This is stored in `data/crm/audit_log.json`.

---

## Setup & Run

```bash
# No pip install needed — pure Python 3.8+
python init_data.py    # Reset all data to clean initial state (run before every demo)
python main.py         # Interactive call-by-call demo
```

### Python version
Python 3.8+ (no external packages required)

---

---

## Assumptions

1. LLM behavior is simulated deterministically — in production, swap `MockLLM` for any real LLM client
2. CRM is JSON files — in production, this maps to Salesforce/HubSpot API calls
3. Wiki updates are append-only (no deletions) to preserve audit history
4. Revenue inferred from user count stays below the auto-apply confidence threshold → routed to HITL
5. Stage can only advance forward (no regression) unless explicitly overridden
6. Entity graph starts pre-loaded with known aliases; grows as agent discovers new renames from transcripts

---

## Extensibility

- **Add a guardrail rule**: Add an entry to `BUILTIN_RULES` in `src/guardrails/rule_engine.py`
- **Add a new customer**: Create `data/customers/cust_XXX/` with profile + CRM + transcripts
- **Swap to a real LLM**: Subclass `BaseLLM` in `src/llm_mock/mock_llm.py`
- **Add a new CRM field**: Add to `crm_opportunity.json` and `propose_crm_updates()` in `mock_llm.py`
- **Add an entity alias**: Add to `data/wiki/entity_graph.json` — resolver picks it up at runtime