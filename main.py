"""
TechServ Solutions — CRM Agent  |  Interactive Call-by-Call Demo

Run:  python init_data.py   (reset first)
      python main.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation import CRMSimulation

# ── ANSI colours ──────────────────────────────────────────────────────────────
try:
    _tty = sys.stdout.isatty()
except Exception:
    _tty = False

def _c(t, code): return f"\033[{code}m{t}\033[0m" if _tty else t
def bold(t):   return _c(t, "1")
def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def dim(t):    return _c(t, "2")
def magenta(t):return _c(t, "35")

# ── CRM state printer ─────────────────────────────────────────────────────────

def print_crm_state(crm: dict, previous: dict = None, title: str = "CRM STATE") -> None:
    """Print the full CRM record, highlighting fields that changed since 'previous'."""
    if not crm:
        print("  (no CRM data)")
        return

    DISPLAY_FIELDS = [
        ("stage",                 "Stage"),
        ("expected_revenue",      "Expected Revenue"),
        ("actual_revenue",        "Actual Revenue"),
        ("num_users",             "Users"),
        ("deal_duration_years",   "Deal Duration"),
        ("close_date",            "Close Date"),
        ("renewal_date",          "Renewal Date"),
        ("conversion_likelihood", "Likelihood"),
        ("next_best_action",      "Next Best Action"),
        ("loss_reason",           "Loss Reason"),
        ("follow_up_recommended", "Follow-up?"),
        ("follow_up_rationale",   "Follow-up Note"),
        ("follow_up_tasks",       "Follow-up Tasks"),
    ]
    CUSTOM_FIELDS = [
        ("security_review_required",   "Security Review Required"),
        ("security_questionnaire_sent","Security Questionnaire Sent"),
        ("proposal_sent",              "Proposal Sent"),
        ("contract_signed",            "Contract Signed"),
    ]

    print(f"\n  {cyan('─' * 64)}")
    print(f"  {bold(title)}  —  {bold(crm.get('company_name', '?'))}  ({crm.get('customer_id','')})")
    print(f"  {cyan('─' * 64)}")

    prev = previous or {}

    def _val_str(v):
        if v is None:                     return dim("—")
        if isinstance(v, bool):           return green("Yes") if v else dim("No")
        if isinstance(v, float):
            if 0 < v <= 1:                return f"{v:.0%}"
            return f"{v:,.0f}"
        if isinstance(v, (int,)):         return f"${v:,}" if "revenue" in "" else str(v)
        if isinstance(v, list):
            if not v:                     return dim("[]")
            return "\n" + "\n".join(f"      • {i}" for i in v)
        s = str(v)
        return s[:80] + ("…" if len(s) > 80 else "")

    def _row(label, new_val, old_val, is_currency=False):
        nstr = f"${new_val:,}" if is_currency and isinstance(new_val,(int,float)) and new_val else _val_str(new_val)
        changed = previous is not None and new_val != old_val and old_val is not None
        changed_new = previous is not None and new_val is not None and old_val is None
        if changed:
            ostr = f"${old_val:,}" if is_currency and isinstance(old_val,(int,float)) else _val_str(old_val)
            print(f"  {label:<30} {yellow(nstr)}  {dim(f'(was: {str(ostr)[:30]})')}")
        elif changed_new:
            print(f"  {label:<30} {green(nstr)}  {dim('← new')}")
        elif new_val is None or new_val == [] or new_val == False:
            print(f"  {label:<30} {dim(_val_str(new_val))}")
        else:
            print(f"  {label:<30} {nstr}")

    # Core fields
    for field, label in DISPLAY_FIELDS:
        val = crm.get(field)
        old = prev.get(field)
        is_rev = "revenue" in field
        if field == "conversion_likelihood":
            nstr = f"{val:.0%}" if isinstance(val, float) else dim("—")
            changed = previous is not None and val != old
            marker = yellow(" ← updated") if changed else ""
            print(f"  {label:<30} {nstr}{marker}")
        elif field == "follow_up_tasks":
            tasks = val or []
            label_str = f"  {label:<30}"
            if tasks:
                first = True
                for t in tasks:
                    if first:
                        print(f"{label_str} • {t}")
                        first = False
                    else:
                        print(f"  {' ' * 30} • {t}")
            else:
                print(f"{label_str} {dim('none')}")
        else:
            _row(label, val, old, is_currency=is_rev)

    # Custom fields
    print(f"  {dim('─' * 40)}")
    cf = crm.get("custom_fields", {})
    pcf = prev.get("custom_fields", {}) if previous else {}
    for field, label in CUSTOM_FIELDS:
        val = cf.get(field)
        old = pcf.get(field)
        changed = previous is not None and val != old
        vstr = green("✓ Yes") if val else dim("No")
        marker = yellow("  ← updated") if changed else ""
        print(f"  {label:<30} {vstr}{marker}")

    # Notes history count
    notes = crm.get("notes_history", [])
    print(f"  {'Notes history':<30} {len(notes)} entr{'y' if len(notes)==1 else 'ies'}")
    print(f"  {cyan('─' * 64)}\n")


def print_journey_summary(sim: CRMSimulation) -> None:
    """Print a compact multi-customer CRM dashboard."""
    states = sim.get_all_crm_states()
    print(f"\n  {cyan('═' * 68)}")
    print(f"  {bold('DEAL PIPELINE DASHBOARD  —  TechServ Solutions')}")
    print(f"  {cyan('═' * 68)}")
    print(f"  {'Customer':<28} {'Stage':<14} {'Revenue':<13} {'Likelihood':<11} Status")
    print(f"  {'─'*28} {'─'*14} {'─'*13} {'─'*11} {'─'*20}")
    for cid, crm in sorted(states.items()):
        company = crm.get("company_name", cid)[:26]
        stage   = crm.get("stage", "?")[:12]
        rev     = crm.get("actual_revenue") or crm.get("expected_revenue", 0)
        rev_s   = f"${rev:,}" if isinstance(rev,(int,float)) and rev>0 else "—"
        lik     = crm.get("conversion_likelihood", 0.5)
        lik_s   = f"{lik:.0%}"
        nba     = (crm.get("next_best_action") or "—")[:28]
        if stage == "Closed Won":
            print(f"  {green(company):<37} {green(stage):<23} {green(rev_s):<13} {green(lik_s):<11} {nba}")
        elif lik < 0.3:
            print(f"  {yellow(company):<37} {yellow(stage):<23} {rev_s:<13} {yellow(lik_s):<11} {nba}")
        else:
            print(f"  {company:<28} {stage:<14} {rev_s:<13} {lik_s:<11} {nba}")
    print(f"  {cyan('═' * 68)}\n")


# ── Call state tracker ────────────────────────────────────────────────────────

CALLS = [
    ("cust_001", 1, "XYZ Air Svcs — Call 1  (Discovery + rebrand: XYZ Aviation Group → XYZ Air Services)"),
    ("cust_001", 2, "XYZ Air Svcs — Call 2  (Follow-up: security cleared, ready for proposal)"),
    ("cust_001", 3, "XYZ Air Svcs — Call 3  (Contract signing → Closed Won)"),
    ("cust_002", 1, "Pinnacle Logistics — Call 1  (Discovery / Low intent → Nurture)"),
]


def build_menu(processed: set) -> str:
    lines = [
        f"\n  {cyan('═' * 68)}",
        f"  {bold('TechServ Solutions  —  CRM Agent  |  Call-by-Call Demo')}",
        f"  {cyan('─' * 68)}",
        f"  {dim('Process each call in order. CRM state is shown after each call.')}",
        f"  {cyan('─' * 68)}",
    ]
    for idx, (cid, call_num, desc) in enumerate(CALLS, 1):
        status = green("✓ DONE") if (cid, call_num) in processed else dim("pending")
        lines.append(f"  {bold(str(idx))}  {desc:<52}  [{status}]")
    lines += [
        f"  {cyan('─' * 68)}",
        f"  {bold('c')}  Show current CRM state (all customers)",
        f"  {bold('r')}  Reset all data  (re-run init_data.py)",
        f"  {bold('q')}  Quit",
        f"  {cyan('═' * 68)}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    processed: set = set()           # tracks which (cid, call_num) are done
    prev_crm: dict  = {}             # stores CRM state before each call for diff display

    sim = CRMSimulation(hallucination_rate=0.0)

    print(build_menu(processed))

    while True:
        try:
            choice = input(f"\n  {bold('Select call to process')} (1-{len(CALLS)}, c, r, q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break

        # ── Quit ──
        if choice in ("q", "quit", "exit"):
            print("  Goodbye.")
            break

        # ── Show CRM state ──
        if choice == "c":
            states = sim.get_all_crm_states()
            for cid, crm in sorted(states.items()):
                print_crm_state(crm, title=f"CURRENT CRM STATE  —  Call log: {len(crm.get('notes_history',[]))} entries")
            print_journey_summary(sim)
            print(build_menu(processed))
            continue

        # ── Reset ──
        if choice == "r":
            confirm = input("  Reset all data? This wipes all CRM updates and wikis. [y/N]: ").strip().lower()
            if confirm == "y":
                import subprocess
                subprocess.run([sys.executable, "init_data.py"], check=False)
                sim = CRMSimulation(hallucination_rate=0.0)
                processed.clear()
                prev_crm.clear()
                print(f"  {green('✓')} Reset complete.")
            print(build_menu(processed))
            continue

        # ── Process a call ──
        if not choice.isdigit() or not (1 <= int(choice) <= len(CALLS)):
            print(f"  Invalid option. Enter 1-{len(CALLS)}, c, r, or q.")
            continue

        idx = int(choice) - 1
        cid, call_num, desc = CALLS[idx]

        if (cid, call_num) in processed:
            print(f"  {yellow('⚠')} Call already processed. Choose another or reset.")
            continue

        # Check prerequisite: calls must be processed in order per customer
        if call_num > 1:
            prev_call = (cid, call_num - 1)
            if prev_call not in processed:
                print(f"  {yellow('⚠')} Process Call {call_num - 1} for this customer first.")
                continue

        # Snapshot CRM before this call for diff
        prev_crm[cid] = sim.get_crm_state(cid) or {}

        print(f"\n  {cyan('═' * 68)}")
        print(f"  {bold(f'PROCESSING:  {desc}')}")
        print(f"  {cyan('═' * 68)}")

        updated_crm = sim.run_single_call(cid, call_num, interactive=True)
        processed.add((cid, call_num))

        # Show CRM state after this call with diff
        if updated_crm:
            print_crm_state(
                updated_crm,
                previous=prev_crm.get(cid),
                title=f"CRM STATE AFTER CALL {call_num}",
            )

        # If all calls for this customer are done, print journey narrative
        total_calls = sum(1 for c, _, _ in CALLS if c == cid)
        calls_done  = sum(1 for (c, n) in processed if c == cid)
        if calls_done == total_calls:
            sim.finalize_journey(cid)

        print_journey_summary(sim)
        print(build_menu(processed))


if __name__ == "__main__":
    main()


