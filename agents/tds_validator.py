"""Tier 3: RAG-Grounded TDS Compliance Validator for LedgerLock.

Validates TDS deductions across all matched transactions against statutory Indian tax rules
retrieved from ChromaDB knowledge base using semantic search and structured metadata filtering.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any, Optional
import chromadb
from agents.llm_client import default_llm_client
from mcp_server.server import get_settlement_records

logger = logging.getLogger("ledgerlock.tds")

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "chroma_db")

# Fallback statutory reference
FALLBACK_RATES = {
    "194-O": {"rate": 0.01, "desc": "1% on gross sales for e-commerce participants"},
    "194H": {"rate": 0.05, "desc": "5% on commission or brokerage fees"},
    "194C": {"rate": 0.02, "desc": "2% for corporate contractor payments"},
    "194J": {"rate": 0.10, "desc": "10% for professional services"},
}


class TDSValidator:
    """RAG-grounded TDS validator querying local ChromaDB index."""

    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            client = chromadb.PersistentClient(path=self.chroma_dir)
            try:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                self.collection = client.get_collection(name="tds_rules", embedding_function=ef)
            except Exception:
                self.collection = client.get_collection(name="tds_rules")
        except Exception as e:
            logger.warning("ChromaDB initialization fallback: %s. Building index...", e)
            from rag.build_index import build_tds_index
            self.collection = build_tds_index(self.chroma_dir)

    def query_rule(self, section: str) -> Dict[str, Any]:
        """Query ChromaDB for the statutory rule and rate of a TDS section using metadata filters."""
        sec_clean = str(section).strip().upper()
        if not self.collection:
            self._init_chroma()

        try:
            # 1. Try structured metadata query
            results = self.collection.get(
                where={"section": sec_clean},
                limit=1,
            )
            if results and results["documents"] and len(results["documents"]) > 0:
                doc = results["documents"][0]
                meta = results["metadatas"][0] if results.get("metadatas") else {}
                return {
                    "section": meta.get("section", sec_clean),
                    "statutory_rate": float(meta.get("statutory_rate", 0.01)),
                    "doc_text": doc,
                }

            # 2. Fallback to vector query with section text
            vec_res = self.collection.query(
                query_texts=[f"Section {sec_clean} statutory rate"],
                n_results=1,
            )
            if vec_res and vec_res["documents"] and len(vec_res["documents"][0]) > 0:
                doc = vec_res["documents"][0][0]
                meta = vec_res["metadatas"][0][0] if vec_res.get("metadatas") else {}
                if meta.get("section") == sec_clean:
                    return {
                        "section": meta.get("section", sec_clean),
                        "statutory_rate": float(meta.get("statutory_rate", 0.01)),
                        "doc_text": doc,
                    }
        except Exception as e:
            logger.error("Error querying ChromaDB for section %s: %s", section, e)

        # Fallback statutory rates
        fb = FALLBACK_RATES.get(sec_clean, {"rate": 0.01, "desc": "Standard 1% rate"})
        return {"section": sec_clean, "statutory_rate": fb["rate"], "doc_text": fb["desc"]}

    def validate_transaction(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single transaction record's TDS compliance."""
        tx_id = record.get("transaction_id", "UNKNOWN")
        sec = record.get("tds_section", "194-O")
        applied_rate = round(float(record.get("tds_rate", 0.0)), 4)
        amount = round(float(record.get("amount", 0.0)), 2)
        tds_amount = round(float(record.get("tds_amount", 0.0)), 2)

        rule_data = self.query_rule(sec)
        expected_rate = round(float(rule_data["statutory_rate"]), 4)

        is_compliant = abs(applied_rate - expected_rate) < 0.0001

        if is_compliant:
            return {
                "transaction_id": tx_id,
                "is_compliant": True,
                "section": sec,
                "applied_rate": applied_rate,
                "expected_rate": expected_rate,
                "status": "COMPLIANT",
                "explanation": f"TDS deduction rate ({applied_rate * 100:.1f}%) complies with Section {sec} statutory rate.",
            }

        # Formulate audit explanation
        explanation = (
            f"TDS rate misapplied: {applied_rate * 100:.1f}% deducted under Section {sec}, "
            f"whereas statutory rate is {expected_rate * 100:.1f}%."
        )

        return {
            "transaction_id": tx_id,
            "is_compliant": False,
            "section": sec,
            "applied_rate": applied_rate,
            "expected_rate": expected_rate,
            "status": "NON_COMPLIANT",
            "expected_tds_amount": round(amount * expected_rate, 2),
            "variance_amount": round(tds_amount - (amount * expected_rate), 2),
            "explanation": explanation,
        }

    def validate_batch(self, settlement_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Validate a batch of settlement records."""
        records = settlement_records or get_settlement_records()
        compliant_list: List[Dict[str, Any]] = []
        violations_list: List[Dict[str, Any]] = []

        for rec in records:
            res = self.validate_transaction(rec)
            if res["is_compliant"]:
                compliant_list.append(res)
            else:
                violations_list.append(res)

        total_checked = len(records)
        violation_count = len(violations_list)
        compliance_rate = (len(compliant_list) / total_checked * 100) if total_checked > 0 else 100.0

        return {
            "total_checked": total_checked,
            "compliant_count": len(compliant_list),
            "violation_count": violation_count,
            "compliance_rate_pct": round(compliance_rate, 2),
            "compliant": compliant_list,
            "violations": violations_list,
        }


def run_tds_validation() -> Dict[str, Any]:
    """CLI / Execution wrapper for TDS validation."""
    validator = TDSValidator()
    results = validator.validate_batch()

    print("=" * 60)
    print("LEDGERLOCK — TIER 3 RAG TDS COMPLIANCE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Transactions Audited:      {results['total_checked']}")
    print(f"TDS Compliant Transactions:     {results['compliant_count']}")
    print(f"TDS Non-Compliant / Violations: {results['violation_count']}")
    print(f"TDS Compliance Rate:            {results['compliance_rate_pct']}%")
    print("-" * 60)
    print("Flagged TDS Violations:")
    for v in results["violations"]:
        print(f"  * {v['transaction_id']} [{v['section']}]: Applied {v['applied_rate']*100:.1f}% vs Expected {v['expected_rate']*100:.1f}% -> {v['explanation']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_tds_validation()
