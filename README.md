# Attorney.AI 🏛️

> **Citation-first U.S. Legal Research RAG Assistant**  
> Every answer grounded in authoritative sources · Powered by CourtListener, GovInfo & eCFR · Not legal advice.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev)

---

## What Is Attorney.AI?

Attorney.AI is an open-source **RAG (Retrieval-Augmented Generation)** legal research assistant that:

- 🔍 **Retrieves** from real U.S. legal data sources — case opinions (CourtListener), statutes (GovInfo/U.S. Code), and federal regulations (eCFR)
- 📌 **Grounds every answer in citations** — no unsupported claims pass the verifier
- 📄 **Reviews contracts** — extracts 25 CUAD clause types with risk levels
- 🤖 **Uses HuggingFace Transformers** — Legal-BERT, DeBERTa-NLI, Legal-LED, BERT-NER
- ⚖️ **Never pretends to be a lawyer** — built-in disclaimers and uncertainty markers

---

## Architecture

```
User Question
     │
     ▼
Jurisdiction + Task Classifier  (GPT-4o-mini + zero-shot)
     │
     ▼
Legal NER                        (BERT-NER + regex: case names, citations, statutes)
     │
     ▼
Query Rewriter                   (legal issue, jurisdiction, date range, doc type)
     │
     ▼
Hybrid Retriever
  ├─ BM25 keyword search         (rank-bm25)
  ├─ Vector similarity           (Qdrant + OpenAI / BGE-large / Legal-BERT)
  └─ Metadata filters            (jurisdiction, court, source_type, date)
     │
     ▼
Cross-Encoder Reranker           (ms-marco-MiniLM-L-12-v2)
     │
     ▼
Citation-Grounded Generator      (GPT-4o-mini, strict citation-only prompt)
     │
     ▼
Citation Verifier                (DeBERTa NLI — PASS / FLAG / REJECT)
     │
     ▼
Response: answer + citations + verdict + disclaimer
```

---

## HuggingFace Transformer Models Used

| Module | Model | Task |
|---|---|---|
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Passage reranking |
| Legal NER | `dslim/bert-base-NER` | Entity extraction |
| ContractNLI | `cross-encoder/nli-deberta-v3-base` | Clause entailment |
| Summarizer | `nsi319/legal-led-base-16384` | Case brief generation |
| Local Embedder | `BAAI/bge-large-en-v1.5` | Offline embeddings |
| Legal Embedder | `nlpaueb/legal-bert-base-uncased` | Domain embeddings |

---

## Data Sources

| Source | Content | License |
|---|---|---|
| [CourtListener](https://www.courtlistener.com/api/) | U.S. case opinions | CC0 |
| [GovInfo](https://api.govinfo.gov) | U.S. Code, bills | Public Domain |
| [eCFR](https://www.ecfr.gov/api/) | Federal regulations | Public Domain |
| [Federal Register](https://www.federalregister.gov/api/v1/) | Proposed/final rules | Public Domain |
| SEC EDGAR | Contracts, filings | Public Domain |
| CUAD | Contract clauses (training) | CC BY 4.0 |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node 20+
- [Qdrant](https://qdrant.tech/) (via Docker)
- OpenAI API key (or use local BGE embeddings)

### 1. Clone & configure

```bash
git clone https://github.com/katkurigopi05/Attorney.AI.git
cd Attorney.AI
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Install & run backend

```bash
cd backend
pip install -e ".[dev]"
uvicorn main:app --reload --port 8000
```

### 4. Install & run frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Ingest legal data

```bash
cd backend
python scripts/ingest_courtlistener.py --courts scotus ca9 --pages 5
python scripts/ingest_ecfr.py --titles 1 2 3
```

Open **http://localhost:5173** — start researching!

---

## Project Structure

```
Attorney.AI/
├── backend/
│   ├── ingestion/          # Data fetchers: CourtListener, GovInfo, eCFR
│   │   ├── chunker.py      # Legal-aware section/clause chunker
│   │   └── metadata_schema.py  # Canonical chunk metadata
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # BM25 + vector RRF fusion
│   │   ├── local_embedder.py    # BGE / Legal-BERT offline embeddings
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── rag/
│   │   ├── pipeline.py          # Full 6-step RAG orchestrator
│   │   ├── legal_ner.py         # BERT NER + legal regex
│   │   ├── verifier.py          # Citation hallucination guard
│   │   └── summarizer.py        # Legal-LED case summarizer
│   ├── contract/
│   │   ├── contract_pipeline.py # CUAD clause extraction
│   │   └── nli_checker.py       # DeBERTa NLI entailment
│   └── api/routes/         # FastAPI endpoints
└── frontend/src/
    ├── pages/              # Home, Research, ContractReview
    └── components/         # SearchBar, AnswerBlock, CitationPanel
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health + index stats |
| `/api/research` | POST | Full RAG legal research query |
| `/api/search` | POST | Direct document search (no generation) |
| `/api/contract/analyze` | POST | Contract clause analysis (PDF/DOCX/TXT) |

Interactive docs: **http://localhost:8000/docs**

---

## Evaluation

Attorney.AI targets LegalBench-RAG metrics:

| Metric | Target |
|---|---|
| Precision@5 | ≥ 0.75 |
| Recall@10 | ≥ 0.80 |
| MRR | ≥ 0.70 |
| Citation Faithfulness | ≥ 0.90 |

Run evaluation:
```bash
python scripts/evaluate_rag.py --dataset legalbench_rag
```

---

## ⚠️ Disclaimer

**Attorney.AI is not a licensed legal service and does not provide legal advice.**  
All outputs are for research and informational purposes only.  
Always consult a licensed attorney for advice on your specific legal situation.

---

## License

MIT © 2024 — See [LICENSE](LICENSE)
