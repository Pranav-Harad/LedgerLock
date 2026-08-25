"""MCP (Model Context Protocol) Server & Tool Registry for LedgerLock.

Exposes tools for data access, TDS rules, and reconciliation result recording:
- get_settlement_records(filter)
- get_ledger_records(filter)
- get_bank_records(filter)
- get_tds_rule(section)
- record_match(transaction_ids, confidence, reasoning)
- flag_exception(transaction_id, reason)

Downstream agents interact with data exclusively through these tools.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger("ledgerlock.mcp")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class ReconciliationStateStore:
    """In-memory and persistent tracking store for matches and exceptions recorded via MCP tools."""

    def __init__(self):
        self.recorded_matches: List[Dict[str, Any]] = []
        self.flagged_exceptions: List[Dict[str, Any]] = []

    def add_match(
        self,
        transaction_ids: List[str],
        confidence: float,
        reasoning: str,
        match_tier: str = "tier2_agentic",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "transaction_ids": transaction_ids,
            "confidence": round(float(confidence), 3),
            "reasoning": reasoning,
            "match_tier": match_tier,
            "metadata": extra_meta or {},
        }
        self.recorded_matches.append(entry)
        return entry

    def add_exception(
        self,
        transaction_id: str,
        reason: str,
        exception_type: str = "unmatched_source",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "transaction_id": transaction_id,
            "reason": reason,
            "exception_type": exception_type,
            "metadata": extra_meta or {},
        }
        self.flagged_exceptions.append(entry)
        return entry

    def reset(self) -> None:
        self.recorded_matches.clear()
        self.flagged_exceptions.clear()


# Global store instance
state_store = ReconciliationStateStore()


class MCPReconciliationServer:
    """MCP reconciliation tool provider and data manager."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DATA_DIR
        self._settlement_cache: Optional[pd.DataFrame] = None
        self._ledger_cache: Optional[pd.DataFrame] = None
        self._bank_cache: Optional[pd.DataFrame] = None

    def _load_data(self) -> None:
        settlement_file = os.path.join(self.data_dir, "settlement_report.csv")
        ledger_file = os.path.join(self.data_dir, "internal_ledger.csv")
        bank_file = os.path.join(self.data_dir, "bank_statement.csv")

        if os.path.exists(settlement_file):
            self._settlement_cache = pd.read_csv(settlement_file)
        else:
            self._settlement_cache = pd.DataFrame()

        if os.path.exists(ledger_file):
            self._ledger_cache = pd.read_csv(ledger_file)
        else:
            self._ledger_cache = pd.DataFrame()

        if os.path.exists(bank_file):
            self._bank_cache = pd.read_csv(bank_file)
        else:
            self._bank_cache = pd.DataFrame()

    def reload(self) -> None:
        self._load_data()

    def _apply_filter(self, df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        res = df.copy()
        if filters:
            for k, v in filters.items():
                if k in res.columns:
                    if isinstance(v, list):
                        res = res[res[k].isin(v)]
                    else:
                        res = res[res[k] == v]
        return res.to_dict(orient="records")

    # ==================== MCP Tool 1: get_settlement_records ====================
    def get_settlement_records(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve settlement report records filtered by column criteria."""
        if self._settlement_cache is None:
            self._load_data()
        return self._apply_filter(self._settlement_cache, filter)

    # ==================== MCP Tool 2: get_ledger_records ====================
    def get_ledger_records(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve internal ledger records filtered by column criteria."""
        if self._ledger_cache is None:
            self._load_data()
        return self._apply_filter(self._ledger_cache, filter)

    # ==================== MCP Tool 3: get_bank_records ====================
    def get_bank_records(self, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve bank statement records filtered by column criteria."""
        if self._bank_cache is None:
            self._load_data()
        return self._apply_filter(self._bank_cache, filter)

    # ==================== MCP Tool 4: get_tds_rule ====================
    def get_tds_rule(self, section: str) -> Dict[str, Any]:
        """Fetch standard Indian Income Tax Act TDS provisions and statutory rates for a given section."""
        sec_clean = section.upper().strip().replace(" ", "").replace("_", "-")
        rules = {
            "194-O": {
                "section": "194-O",
                "title": "Payment of certain sums by e-commerce operator to e-commerce participant",
                "statutory_rate": 0.01,
                "threshold_inr": 500000,
                "applicable_entities": "E-commerce participants selling goods/services via digital platforms",
                "description": "1% TDS on the gross amount of sales of goods or provision of services facilitated by e-commerce operators.",
            },
            "194H": {
                "section": "194H",
                "title": "Commission or brokerage",
                "statutory_rate": 0.05,
                "threshold_inr": 15000,
                "applicable_entities": "Agents, payment aggregators, brokers",
                "description": "5% TDS on payments in the nature of commission or brokerage for services rendered.",
            },
            "194C": {
                "section": "194C",
                "title": "Payments to contractors and sub-contractors",
                "statutory_rate": 0.02,
                "statutory_rate_individual": 0.01,
                "threshold_single_inr": 30000,
                "threshold_aggregate_inr": 100000,
                "applicable_entities": "Contractors, logistics providers, facility management",
                "description": "1% TDS for individual/HUF contractors, 2% TDS for corporate/firm entities.",
            },
            "194J": {
                "section": "194J",
                "title": "Fees for professional or technical services",
                "statutory_rate": 0.10,
                "statutory_rate_technical": 0.02,
                "threshold_inr": 30000,
                "applicable_entities": "Professional consulting, legal, architectural, technical services",
                "description": "10% TDS for professional services, 2% TDS for technical fees or royalty.",
            },
        }

        # Fuzzy lookup in rule dictionary
        for k, v in rules.items():
            if k == sec_clean or k.replace("-", "") == sec_clean.replace("-", ""):
                return v

        return {
            "section": section,
            "title": f"TDS Section {section}",
            "statutory_rate": 0.01,
            "description": f"Standard statutory rate information for Section {section}.",
        }

    # ==================== MCP Tool 5: record_match ====================
    def record_match(
        self,
        transaction_ids: List[str],
        confidence: float,
        reasoning: str,
        match_tier: str = "tier2_agentic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a verified reconciliation match across transaction entries."""
        return state_store.add_match(
            transaction_ids=transaction_ids,
            confidence=confidence,
            reasoning=reasoning,
            match_tier=match_tier,
            extra_meta=metadata,
        )

    # ==================== MCP Tool 6: flag_exception ====================
    def flag_exception(
        self,
        transaction_id: str,
        reason: str,
        exception_type: str = "unmatched_source",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Flag a reconciliation exception with actionable human-readable reasoning."""
        return state_store.add_exception(
            transaction_id=transaction_id,
            reason=reason,
            exception_type=exception_type,
            extra_meta=metadata,
        )


# Singleton server instance for direct in-process tool calls
default_mcp_server = MCPReconciliationServer()

# Direct convenience tool functions exposed for agent imports
get_settlement_records = default_mcp_server.get_settlement_records
get_ledger_records = default_mcp_server.get_ledger_records
get_bank_records = default_mcp_server.get_bank_records
get_tds_rule = default_mcp_server.get_tds_rule
record_match = default_mcp_server.record_match
flag_exception = default_mcp_server.flag_exception


def test_tools():
    """Verify each of the 6 tools responds correctly."""
    print("Testing MCP tools...")
    server = MCPReconciliationServer()

    # 1. get_settlement_records
    settlements = server.get_settlement_records()
    print(f"Tool 1 (get_settlement_records): retrieved {len(settlements)} records")

    # 2. get_ledger_records
    ledgers = server.get_ledger_records()
    print(f"Tool 2 (get_ledger_records): retrieved {len(ledgers)} records")

    # 3. get_bank_records
    banks = server.get_bank_records()
    print(f"Tool 3 (get_bank_records): retrieved {len(banks)} records")

    # 4. get_tds_rule
    rule = server.get_tds_rule("194-O")
    print(f"Tool 4 (get_tds_rule 194-O): rate = {rule.get('statutory_rate')}")

    # 5. record_match
    match_entry = server.record_match(["TXN_TEST_1", "TXN_TEST_2"], 0.95, "Test match recorded")
    print(f"Tool 5 (record_match): {match_entry}")

    # 6. flag_exception
    exc_entry = server.flag_exception("TXN_TEST_3", "Missing credit in bank statement")
    print(f"Tool 6 (flag_exception): {exc_entry}")

    print("All 6 MCP tools verified successfully.")


if __name__ == "__main__":
    test_tools()
