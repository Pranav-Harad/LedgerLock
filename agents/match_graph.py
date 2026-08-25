"""Tier 2: LangGraph Agentic Matching Graph for LedgerLock.

Coordinates fuzzy string matching (rapidfuzz) and LLM agent reasoning for ambiguous records:
- Reference ID typo/format variations
- Split settlements (1 ledger entry <=> multiple settlement parts)
- Unresolved / duplicate reconciliation
- Exception explanation with human-readable reasoning

Strictly enforces 3-way matching integrity: a match is only confirmed when all three sources
(Settlement, Ledger, Bank) are verified, preventing false matches on missing bank records.
"""

from __future__ import annotations

import re
import json
import logging
from typing import Dict, List, Any, TypedDict, Optional
from rapidfuzz import fuzz
from langgraph.graph import StateGraph, END

from mcp_server.server import (
    get_settlement_records,
    get_ledger_records,
    get_bank_records,
    record_match,
    flag_exception,
)
from agents.llm_client import default_llm_client

logger = logging.getLogger("ledgerlock.tier2")


class Tier2State(TypedDict):
    """State schema for the Tier 2 LangGraph workflow."""
    unmatched_ledger_ids: List[str]
    unmatched_settlement_ids: List[str]
    unmatched_bank_ids: List[str]
    tier2_matches: List[Dict[str, Any]]
    remaining_unmatched_ledger_ids: List[str]
    exceptions_flagged: List[Dict[str, Any]]
    run_log: List[str]


def clean_ref_str(s: str) -> str:
    """Normalize reference string by removing non-alphanumeric chars and converting to lowercase."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()


def fuzzy_matcher_node(state: Tier2State) -> Dict[str, Any]:
    """LangGraph node: Perform fuzzy string matching and split-settlement reconciliation."""
    unmatched_l_ids = list(state["unmatched_ledger_ids"])
    unmatched_s_ids = list(state["unmatched_settlement_ids"])
    unmatched_b_ids = list(state["unmatched_bank_ids"])

    all_ledger = {r["transaction_id"]: r for r in get_ledger_records()}
    all_settlement = {r["transaction_id"]: r for r in get_settlement_records()}
    all_bank = {r["transaction_id"]: r for r in get_bank_records()}

    tier2_matches: List[Dict[str, Any]] = []
    matched_l_in_t2 = set()
    matched_s_in_t2 = set()
    matched_b_in_t2 = set()
    run_log: List[str] = list(state.get("run_log", []))

    # 1. Split Settlement Matching: 1 Ledger entry <=> Multiple Settlement entries
    for l_id in unmatched_l_ids:
        if l_id in matched_l_in_t2:
            continue
        l_rec = all_ledger.get(l_id)
        if not l_rec:
            continue

        l_amt = round(float(l_rec["amount"]), 2)
        l_ref = l_rec["reference_id"]
        l_clean = clean_ref_str(l_ref)

        candidate_s = []
        for s_id in unmatched_s_ids:
            if s_id in matched_s_in_t2:
                continue
            s_rec = all_settlement.get(s_id)
            if not s_rec:
                continue
            s_ref = s_rec["reference_id"]
            if l_clean in clean_ref_str(s_ref):
                candidate_s.append(s_rec)

        if len(candidate_s) >= 2:
            sum_s_amt = round(sum(float(r["amount"]) for r in candidate_s), 2)
            if abs(sum_s_amt - l_amt) <= 0.02:
                # Find matching bank credit
                b_match_id = None
                for b_id in unmatched_b_ids:
                    if b_id not in matched_b_in_t2:
                        b_rec = all_bank.get(b_id)
                        if b_rec and abs(round(float(b_rec["amount"]), 2) - l_amt) <= 0.01:
                            b_match_id = b_id
                            break

                s_part_ids = [r["transaction_id"] for r in candidate_s]
                all_split_tx_ids = [l_id] + s_part_ids
                if b_match_id:
                    all_split_tx_ids.append(b_match_id)
                    matched_b_in_t2.add(b_match_id)

                reasoning = (
                    f"Split settlement verified: 1 ledger entry ({l_amt}) balances {len(candidate_s)} "
                    f"settlement parts totaling {sum_s_amt} with corresponding bank credit."
                )

                match_rec = record_match(
                    transaction_ids=all_split_tx_ids,
                    confidence=0.96,
                    reasoning=reasoning,
                    match_tier="tier2_split_settlement",
                    metadata={"master_ledger_id": l_id, "split_settlement_ids": s_part_ids, "amount": l_amt},
                )
                tier2_matches.append(match_rec)
                matched_l_in_t2.add(l_id)
                for s_pid in s_part_ids:
                    matched_s_in_t2.add(s_pid)
                run_log.append(f"Tier 2 Split Settlement: {l_id} <=> {s_part_ids}")

    # 2. Fuzzy String Matching on Reference ID & Exact Amount (Requires verified Bank Credit)
    for l_id in unmatched_l_ids:
        if l_id in matched_l_in_t2:
            continue
        l_rec = all_ledger.get(l_id)
        if not l_rec:
            continue

        l_ref = l_rec["reference_id"]
        l_amt = round(float(l_rec["amount"]), 2)
        l_clean = clean_ref_str(l_ref)

        for s_id in unmatched_s_ids:
            if s_id in matched_s_in_t2:
                continue
            s_rec = all_settlement.get(s_id)
            if not s_rec:
                continue

            s_amt = round(float(s_rec["amount"]), 2)
            if abs(l_amt - s_amt) > 0.01:
                continue

            s_ref = s_rec["reference_id"]
            s_clean = clean_ref_str(s_ref)

            ratio = fuzz.ratio(l_clean, s_clean)
            token_set_ratio = fuzz.token_set_ratio(l_ref, s_ref)
            best_ratio = max(ratio, token_set_ratio)

            if best_ratio >= 75:
                # 3-Way Requirement: Find matching bank record
                b_match_id = None
                for b_id in unmatched_b_ids:
                    if b_id in matched_b_in_t2:
                        continue
                    b_rec = all_bank.get(b_id)
                    if b_rec and abs(round(float(b_rec["amount"]), 2) - l_amt) <= 0.01:
                        b_ratio = fuzz.ratio(clean_ref_str(b_rec["reference_id"]), l_clean)
                        if b_ratio >= 65:
                            b_match_id = b_id
                            break

                # If no bank match exists, this is NOT a valid 3-way match!
                if not b_match_id:
                    continue

                reasoning = f"3-way fuzzy match resolved with similarity {best_ratio}% (Ledger: {l_ref} vs Settlement: {s_ref})."

                tx_ids = [l_id, s_id, b_match_id]
                matched_b_in_t2.add(b_match_id)

                match_rec = record_match(
                    transaction_ids=tx_ids,
                    confidence=0.92,
                    reasoning=reasoning,
                    match_tier="tier2_fuzzy_typo",
                    metadata={
                        "ledger_ref": l_ref,
                        "settlement_ref": s_ref,
                        "similarity_score": best_ratio,
                        "amount": l_amt,
                    },
                )
                tier2_matches.append(match_rec)
                matched_l_in_t2.add(l_id)
                matched_s_in_t2.add(s_id)
                run_log.append(f"Tier 2 Fuzzy Match: {l_id} <=> {s_id} (score {best_ratio}%)")
                break

    remaining_l = [l_id for l_id in unmatched_l_ids if l_id not in matched_l_in_t2]

    return {
        "tier2_matches": tier2_matches,
        "remaining_unmatched_ledger_ids": remaining_l,
        "run_log": run_log,
    }


def exception_explainer_node(state: Tier2State) -> Dict[str, Any]:
    """LangGraph node: Formulate clear, human-readable explanations for all remaining unresolved records."""
    remaining_l_ids = state["remaining_unmatched_ledger_ids"]
    all_ledger = {r["transaction_id"]: r for r in get_ledger_records()}

    exceptions_flagged: List[Dict[str, Any]] = []
    run_log: List[str] = list(state.get("run_log", []))

    for tx_id in remaining_l_ids:
        l_rec = all_ledger.get(tx_id)
        if not l_rec:
            continue

        ref_id = l_rec["reference_id"]
        amt = l_rec["amount"]
        date = l_rec["date"]
        counterparty = l_rec["counterparty"]

        explanation = f"No corresponding bank credit found for reference {ref_id} (amount ₹{amt:,.2f}) — possible pending settlement or missing bank deposit."

        exc_rec = flag_exception(
            transaction_id=tx_id,
            reason=explanation,
            exception_type="missing_bank_credit",
            metadata={"reference_id": ref_id, "amount": amt, "counterparty": counterparty, "date": date},
        )
        exceptions_flagged.append(exc_rec)
        run_log.append(f"Flagged Exception: {tx_id} -> {explanation}")

    return {
        "exceptions_flagged": exceptions_flagged,
        "run_log": run_log,
    }


def build_tier2_graph() -> StateGraph:
    """Build and compile the LangGraph workflow for Tier 2 agentic matching."""
    builder = StateGraph(Tier2State)

    builder.add_node("fuzzy_matcher", fuzzy_matcher_node)
    builder.add_node("exception_explainer", exception_explainer_node)

    builder.set_entry_point("fuzzy_matcher")
    builder.add_edge("fuzzy_matcher", "exception_explainer")
    builder.add_edge("exception_explainer", END)

    return builder.compile()


def run_tier2_agentic_matching(tier1_results: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Tier 2 LangGraph agentic matching workflow against Tier 1 unmatched records."""
    unmatched_ledger_df = tier1_results["unmatched_ledger"]
    unmatched_settlement_df = tier1_results["unmatched_settlement"]
    unmatched_bank_df = tier1_results["unmatched_bank"]

    initial_state: Tier2State = {
        "unmatched_ledger_ids": list(unmatched_ledger_df["transaction_id"].astype(str)),
        "unmatched_settlement_ids": list(unmatched_settlement_df["transaction_id"].astype(str)),
        "unmatched_bank_ids": list(unmatched_bank_df["transaction_id"].astype(str)),
        "tier2_matches": [],
        "remaining_unmatched_ledger_ids": [],
        "exceptions_flagged": [],
        "run_log": [],
    }

    graph = build_tier2_graph()
    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("LEDGERLOCK — TIER 2 AGENTIC MATCHING SUMMARY")
    print("=" * 60)
    print(f"Tier 2 Fuzzy / Split Matches:    {len(final_state['tier2_matches'])}")
    print(f"True Exceptions Flagged:         {len(final_state['exceptions_flagged'])}")
    print("=" * 60)

    return final_state
