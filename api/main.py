"""FastAPI REST API Service for LedgerLock Reconciliation Engine.

Endpoints:
- POST /reconcile/run: Executes batch reconciliation and returns JSON results
- GET /reconcile/report/html: Serves rendered HTML executive report
- GET /health: Health status check
- GET /telemetry/spans: Inspects collected OpenTelemetry trace spans
"""

from __future__ import annotations

import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from run_batch import run_full_pipeline
from observability.tracing import trace_span, get_collected_spans

app = FastAPI(
    title="LedgerLock Reconciliation API",
    description="Finance-ops 3-way reconciliation and TDS compliance agent for payment aggregators.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORT_HTML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "reconciliation_report.html",
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ledgerlock-api", "version": "0.1.0"}


@app.post("/reconcile/run")
def trigger_reconciliation() -> JSONResponse:
    """Execute full 3-way reconciliation batch run across all tiers."""
    try:
        with trace_span("api.trigger_reconciliation"):
            results = run_full_pipeline(verbose=False)
            return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation run failed: {str(e)}")


@app.get("/reconcile/report/html", response_class=HTMLResponse)
def get_html_report() -> HTMLResponse:
    """Retrieve and serve the rendered executive HTML report."""
    if not os.path.exists(REPORT_HTML_PATH):
        # Run pipeline if report does not exist yet
        run_full_pipeline(verbose=False)

    if not os.path.exists(REPORT_HTML_PATH):
        raise HTTPException(status_code=404, detail="HTML report not found.")

    with open(REPORT_HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/telemetry/spans")
def get_trace_spans() -> Dict[str, Any]:
    """Inspect in-memory OpenTelemetry trace spans."""
    spans = get_collected_spans()
    span_data = []
    for s in spans:
        span_data.append({
            "name": s.name,
            "trace_id": format(s.context.trace_id, "032x"),
            "span_id": format(s.context.span_id, "016x"),
            "status": str(s.status.status_code),
            "attributes": dict(s.attributes),
        })
    return {"total_spans": len(span_data), "spans": span_data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
