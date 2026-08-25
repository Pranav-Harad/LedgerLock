# LedgerLock — Finance-Ops Reconciliation Agent

LedgerLock is an automated 3-way financial reconciliation engine tailored for Indian payment aggregators. It reconciles settlement reports, internal ledgers, and bank statements, validates TDS compliance with Indian tax law using ChromaDB RAG, performs multi-agent auditing with CrewAI, and provides quantitative evaluation against ground-truth datasets.

## Architecture Overview

1. **Tier 1: Deterministic Engine (pandas)**: Resolves ~55-65% of records (clean matches & date-variance tolerance) in milliseconds with zero LLM overhead.
2. **Tier 2: Agentic Fuzzy Matcher (LangGraph + RapidFuzz + LLM)**: Resolves typos, split settlements, and duplicate entries, while flagging genuine missing records with human-readable explanations.
3. **Tier 3: RAG TDS Compliance Validator (ChromaDB + SentenceTransformers)**: Validates tax deductions against Sections 194-O, 194H, 194C, and 194J.
4. **Multi-Agent Audit (CrewAI)**: Independent auditor agent verifies 3-way amount balance integrity; report writer agent compiles an executive HTML dashboard.
5. **Tooling (MCP)**: Exposes modular data-access and match-recording tools via Model Context Protocol.
6. **Evaluation & Honesty**: Quantitative verification calculating match rates, exception precision/recall, and guaranteeing 0 false matches.

## Quickstart

### 1. Setup Environment
```bash
# Using uv (Python 3.11-3.12 recommended)
uv venv --python 3.12 .venv
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your preferred provider:
```bash
cp .env.example .env
```
- For **Gemini API**: Set `LLM_PROVIDER=gemini` and provide `GEMINI_API_KEY`.
- For **Ollama**: Set `LLM_PROVIDER=ollama` and ensure Ollama is running.
- For **Offline / Mock Mode**: Set `LLM_PROVIDER=mock`.

### 3. Generate Synthetic Dataset
```bash
python data/generate.py
```

### 4. Run Full Reconciliation Batch
```bash
python run_batch.py
```

### 5. Launch FastAPI Service
```bash
uvicorn api.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Trigger Reconciliation: `POST http://localhost:8000/reconcile/run`
- View Rendered HTML Report: `GET http://localhost:8000/reconcile/report/html`
