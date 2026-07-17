"""
Pretty-print pipeline results to the terminal.

No external dependencies — pure Python string formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ANSI color codes (fallback gracefully on non-ANSI terminals)
try:
    import sys
    COLORS = sys.stdout.isatty()
except Exception:
    COLORS = False

C = {
    "reset":  "\033[0m"  if COLORS else "",
    "bold":   "\033[1m"  if COLORS else "",
    "green":  "\033[32m" if COLORS else "",
    "yellow": "\033[33m" if COLORS else "",
    "red":    "\033[31m" if COLORS else "",
    "cyan":   "\033[36m" if COLORS else "",
    "blue":   "\033[34m" if COLORS else "",
    "dim":    "\033[2m"  if COLORS else "",
}


def _c(text: str, color: str) -> str:
    return f"{C[color]}{text}{C['reset']}"


def _header(title: str, width: int = 68) -> str:
    line = "═" * width
    return f"\n{_c(line, 'cyan')}\n  {_c(title, 'bold')}\n{_c(line, 'cyan')}"


def _section(title: str) -> str:
    return f"\n{_c('▶ ' + title, 'bold')}"


def _conf_badge(conf: float) -> str:
    if conf >= 0.85:
        return _c(f"({conf:.2f})", "green")
    elif conf >= 0.65:
        return _c(f"({conf:.2f})", "yellow")
    else:
        return _c(f"({conf:.2f})", "red")


def print_pipeline_result(result: Dict, show_trace: bool = False) -> None:
    """Print a full pipeline result in a human-readable format."""

    print(_header(f"CRM AGENT PIPELINE  ─  Run {result.get('run_id', '')}"))
    print(f"  Transcript : {result.get('transcript_id', '—')}")
    print(f"  Opportunity: {result.get('opportunity_id', '—')}")
    print(f"  Timestamp  : {result.get('timestamp', '—')}")
    timing = result.get("timing", {})
    total_ms = sum(timing.values())
    print(f"  Duration   : {total_ms:.1f} ms total")

    # ----------------------------------------------------------------
    # Extraction
    # ----------------------------------------------------------------
    print(_section("EXTRACTION AGENT"))
    extraction = result.get("extraction", {})
    if extraction:
        field_order = [
            "company_name", "contact_names", "num_users", "revenue_estimate",
            "timeline", "close_deadline", "buying_intent", "opportunity_stage",
            "requested_actions", "customer_concerns", "sentiment",
        ]
        max_key = max((len(k) for k in field_order), default=20)
        for key in field_order:
            if key in extraction:
                f = extraction[key]
                val = f.get("value")
                conf = f.get("confidence", 0)
                hallucinated = f.get("hallucinated", False)
                tag = _c(" [HALLUCINATED]", "red") if hallucinated else ""
                val_str = str(val) if not isinstance(val, list) else ", ".join(str(v) for v in val)
                print(f"  {key:<{max_key}}  {_c(val_str, 'bold')}  {_conf_badge(conf)}{tag}")
                if show_trace:
                    evidence = f.get("evidence", "")
                    if evidence:
                        print(f"  {' ' * max_key}  {_c('↳ ' + evidence[:80], 'dim')}")
    else:
        print("  (no extraction data)")

    # ----------------------------------------------------------------
    # Wiki context
    # ----------------------------------------------------------------
    wiki_ids = result.get("wiki_context_ids", [])
    if wiki_ids:
        print(_section(f"WIKI CONTEXT  ({len(wiki_ids)} entries loaded)"))
        print(f"  {_c(', '.join(wiki_ids), 'dim')}")

    # ----------------------------------------------------------------
    # Guardrails / Rule Engine
    # ----------------------------------------------------------------
    rule_summary = result.get("rule_summary", {})
    if rule_summary:
        total = rule_summary.get("total_rules_evaluated", 0)
        passed = rule_summary.get("passed", 0)
        errors = rule_summary.get("errors", 0)
        warnings = rule_summary.get("warnings", 0)
        blocking = rule_summary.get("blocking", False)

        status_str = _c("BLOCKING ERRORS", "red") if blocking else _c("OK", "green")
        print(_section(f"GUARDRAILS  [{status_str}]  {passed}/{total} rules passed"))

        for rule in rule_summary.get("details", []):
            icon = _c("✓", "green") if rule["passed"] else (
                _c("✗", "red") if rule["severity"] == "ERROR" else _c("⚠", "yellow")
            )
            wiki_ref = f"  [{_c(rule['wiki_ref'], 'dim')}]" if rule.get("wiki_ref") else ""
            print(f"  {icon}  {rule['message']}{wiki_ref}")

    # ----------------------------------------------------------------
    # Proposals
    # ----------------------------------------------------------------
    final_proposals = result.get("final_proposals", [])
    if final_proposals:
        print(_section(f"CRM UPDATE PROPOSALS  ({len(final_proposals)} total)"))

        verdict_colors = {
            "APPROVED": "green",
            "REVISED": "yellow",
            "REJECTED": "red",
            "NEEDS_HUMAN_REVIEW": "yellow",
        }

        for p in final_proposals:
            verdict = p.get("verdict", "?")
            color = verdict_colors.get(verdict, "reset")
            verdict_str = _c(f"[{verdict}]", color)
            conf = p.get("confidence", 0)

            old_val = p.get("old_value")
            new_val = p.get("new_value")

            # Format values
            if p["field"] == "expected_revenue":
                old_str = f"${old_val:,}" if isinstance(old_val, (int, float)) else str(old_val)
                new_str = f"${new_val:,}" if isinstance(new_val, (int, float)) else str(new_val)
            elif isinstance(new_val, list):
                new_str = "\n" + "\n".join(f"      • {v}" for v in new_val)
                old_str = str(old_val)
            else:
                old_str, new_str = str(old_val), str(new_val)

            print(f"\n  {verdict_str} {_c(p['field'], 'bold')}  {_conf_badge(conf)}")
            if old_val is not None and old_val != new_val:
                print(f"    {_c('Before:', 'dim')} {old_str}")
            print(f"    {_c('After: ', 'dim')} {_c(new_str, 'bold')}")
            print(f"    {_c('Reason:', 'dim')} {p.get('reasoning', '')[:160]}")

            if p.get("issues"):
                for issue in p["issues"]:
                    print(f"    {_c('⚠', 'yellow')} {issue}")

            if p.get("guardrail_blocked"):
                print(f"    {_c('🚫 BLOCKED:', 'red')} {p.get('guardrail_message', '')}")

    # ----------------------------------------------------------------
    # Review Agent
    # ----------------------------------------------------------------
    review = result.get("review_output", {})
    if review:
        overall = review.get("overall_verdict", "?")
        verdict_colors_r = {
            "APPROVED": "green",
            "APPROVED_WITH_WARNINGS": "yellow",
            "APPROVED_WITH_REJECTIONS": "yellow",
            "REVISED": "yellow",
            "REJECTED": "red",
            "NEEDS_HUMAN_REVIEW": "yellow",
        }
        ov_color = verdict_colors_r.get(overall, "reset")
        print(_section(f"REVIEW AGENT  →  {_c(overall, ov_color)}"))
        for note in review.get("review_notes", []):
            print(f"  {_c('⚠', 'yellow')} {note}")
        if not review.get("review_notes"):
            print(f"  {_c('No additional issues found.', 'dim')}")

    # ----------------------------------------------------------------
    # Human-in-the-Loop
    # ----------------------------------------------------------------
    hitl = result.get("hitl_decisions", {})
    if hitl:
        print(_section("HUMAN-IN-THE-LOOP"))
        for field, decision in hitl.items():
            action_color = "green" if decision["action"] == "approved" else "red"
            print(
                f"  {_c(field, 'bold')}: {_c(decision['action'].upper(), action_color)}  "
                f"{_c('(' + decision['reason'] + ')', 'dim')}"
            )

    # ----------------------------------------------------------------
    # Applied Updates
    # ----------------------------------------------------------------
    applied = result.get("applied_result", {})
    if applied and not applied.get("error"):
        n_applied = applied.get("total_applied", 0)
        n_deferred = applied.get("total_deferred", 0)
        print(_section(f"FINAL CRM UPDATE  —  {n_applied} applied, {n_deferred} deferred"))
        for u in applied.get("applied_updates", []):
            old = u["old_value"]
            new = u["new_value"]
            if u["field"] == "expected_revenue":
                old = f"${old:,}" if isinstance(old, (int, float)) else old
                new = f"${new:,}" if isinstance(new, (int, float)) else new
            elif isinstance(new, list):
                new = "[" + ", ".join(str(v) for v in new) + "]"
            print(f"  {_c('✓', 'green')} {_c(u['field'], 'bold')}  {old} → {_c(str(new), 'green')}")
        for u in applied.get("deferred_updates", []):
            print(f"  {_c('⏳', 'yellow')} {_c(u['field'], 'bold')}  (pending human review)")

    # ----------------------------------------------------------------
    # Reasoning traces (optional verbose mode)
    # ----------------------------------------------------------------
    if show_trace:
        traces = result.get("reasoning_traces", {})
        for agent_name, steps in traces.items():
            print(_section(f"REASONING TRACE — {agent_name}"))
            for step in steps:
                print(f"  {_c(step['step'], 'cyan')} — {step['description']}")

    print(f"\n{_c('═' * 68, 'cyan')}\n")


def print_evaluation_report(report: Dict) -> None:
    """Print the evaluation framework results."""
    print(_header("EVALUATION REPORT"))

    print(f"\n  {_c('Test cases:', 'bold')}        {report.get('total_test_cases')}")
    print(f"  {_c('Overall score:', 'bold')}     {_c(str(report.get('avg_overall_score', 0)), 'green')}")
    print(f"  {_c('Stage accuracy:', 'bold')}    {report.get('stage_accuracy', 0):.1%}")
    print(f"  {_c('Extraction acc:', 'bold')}    {report.get('avg_extraction_accuracy', 0):.1%}")
    print(f"  {_c('Action recall:', 'bold')}     {report.get('avg_action_recall', 0):.1%}")
    print(f"  {_c('Action precision:', 'bold')}  {report.get('avg_action_precision', 0):.1%}")
    print(f"  {_c('Guardrail eff.:', 'bold')}    {report.get('guardrail_effectiveness', 0):.1%}")
    print(f"  {_c('Hallucinations:', 'bold')}    {report.get('total_hallucinations', 0)}")
    print(f"  {_c('Reviewer eff.:', 'bold')}     {report.get('reviewer_effectiveness', 0):.1%}")

    print(_section("Per-Case Results"))
    for case in report.get("per_case", []):
        score = case["overall_score"]
        color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        print(
            f"\n  {_c(case['id'], 'bold')}  score={_c(str(score), color)}"
            f"  stage={'✓' if case['stage_correct'] else '✗'}"
            f"  recall={case['action_recall']:.0%}"
            f"  guardrails={case['guardrail_violations_caught']}"
        )
        print(f"  {_c(case['description'], 'dim')}")
        for note in case.get("notes", []):
            print(f"    {_c('→', 'yellow')} {note}")
        if case.get("hallucinations"):
            print(f"    {_c('HALLUCINATION:', 'red')} {case['hallucinations']}")

    print(f"\n{_c('═' * 68, 'cyan')}\n")
