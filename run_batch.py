"""CLI Entrypoint for LedgerLock Full Reconciliation Pipeline.

Executes end-to-end:
1. Tier 1: Deterministic Matching (pandas)
2. Tier 2: Agentic Fuzzy / Split Matching (LangGraph + RapidFuzz + LLM)
3. Tier 3: RAG TDS Compliance Validation (ChromaDB + SentenceTransformers)
4. Multi-Agent Audit & Report Compilation (CrewAI / Jinja2 HTML)
5. Quantitative Evaluation Harness (Ground Truth validation)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any

from mcp_server.server import state_store
from matching.tier1_deterministic import run_tier1_matching
from agents.match_graph import run_tier2_agentic_matching
from agents.tds_validator import run_tds_validation
from agents.audit_crew import run_audit_and_reporting
from eval.eval_harness import ReconciliationEvalHarness, print_eval_report
from observability.tracing import trace_span, langfuse_observer


def run_full_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """Execute the complete LedgerLock 3-way reconciliation pipeline."""
    start_time = time.time()

    # Reset in-memory state store for clean batch execution
    state_store.reset()

    with trace_span("ledgerlock.batch_run") as root_span:
        if verbose:
            print("\n" + "=" * 80)
            print("         LEDGERLOCK — FULL 3-WAY RECONCILIATION BATCH EXECUTION")
            print("=" * 80)

        # 1. Tier 1: Deterministic Engine
        with trace_span("pipeline.tier1_deterministic"):
            t1_results = run_tier1_matching()

        # 2. Tier 2: LangGraph Agentic Matching
        with trace_span("pipeline.tier2_agentic_langgraph"):
            t2_results = run_tier2_agentic_matching(t1_results)

        # 3. Tier 3: RAG TDS Compliance Validator
        with trace_span("pipeline.tier3_rag_tds_validation"):
            tds_results = run_tds_validation()

        # 4. Evaluation Harness against Ground Truth
        with trace_span("pipeline.eval_harness"):
            evaluator = ReconciliationEvalHarness()
            eval_results = evaluator.evaluate(
                tier1_results=t1_results,
                tier2_results=t2_results,
                tds_results=tds_results,
            )

        # 5. Multi-Agent Audit Crew & HTML Report Generation
        with trace_span("pipeline.audit_and_reporting"):
            report = run_audit_and_reporting(
                tier1_results=t1_results,
                tier2_results=t2_results,
                tds_results=tds_results,
                false_match_count=eval_results["false_match_count"],
            )

        duration = round(time.time() - start_time, 2)
        root_span.set_attribute("pipeline.duration_seconds", duration)
        root_span.set_attribute("pipeline.overall_match_rate", report["overall_match_rate"])
        root_span.set_attribute("pipeline.false_matches", eval_results["false_match_count"])

        if verbose:
            print_eval_report(eval_results)
            print(f"Batch run completed in {duration}s. Report saved to: {report['html_path']}\n")

        return {
            "tier1": t1_results["stats"],
            "tier2": {
                "matches_count": len(t2_results["tier2_matches"]),
                "exceptions_count": len(t2_results["exceptions_flagged"]),
            },
            "tds": {
                "compliance_rate_pct": tds_results["compliance_rate_pct"],
                "violations_count": len(tds_results["violations"]),
            },
            "eval": eval_results,
            "report": report,
            "duration_seconds": duration,
        }


if __name__ == "__main__":
    run_full_pipeline(verbose=True)
