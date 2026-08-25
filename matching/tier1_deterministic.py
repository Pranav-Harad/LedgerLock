"""Tier 1: Deterministic matching engine for LedgerLock.

Pure pandas-based 3-way matching across:
- Settlement Report
- Internal Ledger
- Bank Statement

Performs exact reference matching and date-tolerance window matching.
Strictly zero LLM calls.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


class Tier1DeterministicMatcher:
    """Zero-LLM deterministic 3-way matcher."""

    def __init__(self, date_tolerance_days: int = 3):
        self.date_tolerance_days = date_tolerance_days

    def match(
        self,
        settlement_df: pd.DataFrame,
        ledger_df: pd.DataFrame,
        bank_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Perform 3-way deterministic reconciliation.

        Returns a dictionary with:
            - 'matched_clean': DataFrame of exact matches
            - 'matched_with_date_variance': DataFrame of date-variance matches
            - 'all_matched': Combined DataFrame of matched records
            - 'unmatched_settlement': Unmatched settlement records
            - 'unmatched_ledger': Unmatched ledger records
            - 'unmatched_bank': Unmatched bank records
            - 'stats': Metrics dictionary
        """
        # Ensure working copies
        s_df = settlement_df.copy()
        l_df = ledger_df.copy()
        b_df = bank_df.copy()

        # Normalize dates
        s_df["dt"] = pd.to_datetime(s_df["date"])
        l_df["dt"] = pd.to_datetime(l_df["date"])
        b_df["dt"] = pd.to_datetime(b_df["date"])

        # Normalize reference_id strings
        s_df["ref_clean"] = s_df["reference_id"].astype(str).str.strip()
        l_df["ref_clean"] = l_df["reference_id"].astype(str).str.strip()
        b_df["ref_clean"] = b_df["reference_id"].astype(str).str.strip()

        # Step 1: Match on exact reference_id and exact amount (rounded to 2 decimal places)
        s_df["amt_round"] = s_df["amount"].round(2)
        l_df["amt_round"] = l_df["amount"].round(2)
        b_df["amt_round"] = b_df["amount"].round(2)

        # Merge Ledger and Settlement on exact (ref_clean, amt_round)
        ls_merged = pd.merge(
            l_df,
            s_df,
            on=["ref_clean", "amt_round"],
            suffixes=("_ledger", "_settlement"),
            how="inner",
        )

        # Merge with Bank on exact (ref_clean, amt_round)
        lsb_merged = pd.merge(
            ls_merged,
            b_df.add_suffix("_bank").rename(
                columns={
                    "ref_clean_bank": "ref_clean",
                    "amt_round_bank": "amt_round",
                }
            ),
            on=["ref_clean", "amt_round"],
            how="inner",
        )

        # Calculate date variances
        lsb_merged["date_diff_sl"] = (
            lsb_merged["dt_settlement"] - lsb_merged["dt_ledger"]
        ).dt.days.abs()
        lsb_merged["date_diff_bl"] = (
            lsb_merged["dt_bank"] - lsb_merged["dt_ledger"]
        ).dt.days.abs()
        lsb_merged["max_date_variance"] = lsb_merged[
            ["date_diff_sl", "date_diff_bl"]
        ].max(axis=1)

        # Filter within date tolerance
        valid_matches = lsb_merged[
            lsb_merged["max_date_variance"] <= self.date_tolerance_days
        ].copy()

        # Split into Clean vs Date Variance
        clean_mask = valid_matches["max_date_variance"] == 0
        matched_clean = valid_matches[clean_mask].copy()
        matched_date_variance = valid_matches[~clean_mask].copy()

        # Assign tier metadata
        matched_clean["match_tier"] = "tier1_clean"
        matched_clean["confidence"] = 1.0
        matched_clean["reasoning"] = (
            "Exact 3-way match on reference ID, amount, and transaction date."
        )

        matched_date_variance["match_tier"] = "tier1_date_variance"
        matched_date_variance["confidence"] = 0.98
        matched_date_variance["reasoning"] = matched_date_variance[
            "max_date_variance"
        ].apply(
            lambda v: f"3-way match on reference ID and amount with date variance of {v} day(s) (within tolerance)."
        )

        all_matched = pd.concat(
            [matched_clean, matched_date_variance], ignore_index=True
        )

        # Identify matched transaction_ids in each source to extract unmatched sets
        matched_ledger_ids = set(all_matched["transaction_id_ledger"])
        matched_settlement_ids = set(all_matched["transaction_id_settlement"])
        matched_bank_ids = set(all_matched["transaction_id_bank"])

        unmatched_ledger = (
            l_df[~l_df["transaction_id"].isin(matched_ledger_ids)]
            .drop(columns=["dt", "ref_clean", "amt_round"])
            .copy()
        )
        unmatched_settlement = (
            s_df[~s_df["transaction_id"].isin(matched_settlement_ids)]
            .drop(columns=["dt", "ref_clean", "amt_round"])
            .copy()
        )
        unmatched_bank = (
            b_df[~b_df["transaction_id"].isin(matched_bank_ids)]
            .drop(columns=["dt", "ref_clean", "amt_round"])
            .copy()
        )

        total_ledger_records = len(ledger_df)
        clean_count = len(matched_clean)
        variance_count = len(matched_date_variance)
        total_matched = len(all_matched)
        unmatched_count = len(unmatched_ledger)

        match_rate = (total_matched / total_ledger_records) * 100 if total_ledger_records > 0 else 0.0

        stats = {
            "total_ledger_records": total_ledger_records,
            "matched_clean_count": clean_count,
            "matched_date_variance_count": variance_count,
            "total_tier1_matched": total_matched,
            "unmatched_count": unmatched_count,
            "tier1_match_rate_pct": round(match_rate, 2),
        }

        return {
            "matched_clean": matched_clean,
            "matched_with_date_variance": matched_date_variance,
            "all_matched": all_matched,
            "unmatched_settlement": unmatched_settlement,
            "unmatched_ledger": unmatched_ledger,
            "unmatched_bank": unmatched_bank,
            "stats": stats,
        }


def run_tier1_matching(
    data_dir: Optional[str] = None,
    tolerance_days: int = 3,
) -> Dict[str, Any]:
    """Load files from data_dir and execute Tier 1 deterministic reconciliation."""
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )

    settlement_df = pd.read_csv(os.path.join(data_dir, "settlement_report.csv"))
    ledger_df = pd.read_csv(os.path.join(data_dir, "internal_ledger.csv"))
    bank_df = pd.read_csv(os.path.join(data_dir, "bank_statement.csv"))

    matcher = Tier1DeterministicMatcher(date_tolerance_days=tolerance_days)
    results = matcher.match(settlement_df, ledger_df, bank_df)

    stats = results["stats"]
    print("=" * 60)
    print("LEDGERLOCK — TIER 1 DETERMINISTIC RECONCILIATION SUMMARY")
    print("=" * 60)
    print(f"Total Base Records (Ledger):     {stats['total_ledger_records']}")
    print(f"Tier 1 Clean Exact Matches:      {stats['matched_clean_count']} ({stats['matched_clean_count']/stats['total_ledger_records']*100:.1f}%)")
    print(f"Tier 1 Date Variance Matches:    {stats['matched_date_variance_count']} ({stats['matched_date_variance_count']/stats['total_ledger_records']*100:.1f}%)")
    print(f"Total Tier 1 Matched:            {stats['total_tier1_matched']} ({stats['tier1_match_rate_pct']}%)")
    print(f"Remaining Unmatched for Tier 2:  {stats['unmatched_count']} ({stats['unmatched_count']/stats['total_ledger_records']*100:.1f}%)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_tier1_matching()
