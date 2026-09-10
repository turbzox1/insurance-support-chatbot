# Insurance Support Chatbot

An internship project implementing a document-grounded insurance assistant with LangGraph,
Gemini, hybrid retrieval and an optional evaluation/observability layer.

## Run

Python 3.11 or 3.12 is recommended. The package supports Python 3.11–3.14.
From this repository's root, using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,observability]"
Copy-Item .env.example .env
# Edit .env and set GOOGLE_API_KEY. Set TAVILY_API_KEY for web search.
python -m src.retrieval.ingest
streamlit run src/app.py
```

The editable install makes imports work without editing sys.path. If activation is restricted,
use `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe` directly.
On Windows, if `python` opens the Store, use your installed Python executable or `py -3.12`.
The first ingestion downloads a Hugging Face embedding model; the first RAG request downloads
the cross-encoder. These require internet and disk space, but do not use Gemini quota.

CLI: `python -m src.chatbot.langgraph_chatbot`.
Older commands `python ingest.py`, `python chatbot.py` and `streamlit run app.py` remain supported.

## What runs

1. Streamlit passes the question and its own session history to the graph.
2. History is normalized and bounded to five verified turns. Independent calls have no history.
3. Clear domain, ambiguity and live-information cases use deterministic rules.
   Uncertain queries and dependent follow-ups use one combined Gemini analysis/rewrite call.
4. The router chooses RAG, web search, clarification or an insurance-only scope response.
5. RAG combines BM25 keyword matches and Chroma vector results with reciprocal rank fusion
   (`1 / (60 + rank)`). Chunk IDs preserve distinct source/page identities.
6. A `cross-encoder/ms-marco-MiniLM-L-6-v2` model reranks candidates. Semantic duplicate
   removal and a character budget compress the context using the shared embedding model.
7. A raw reranker-score threshold filters weak evidence. This is a heuristic, **not**
   calibrated confidence. If no evidence passes, configured web search is tried; otherwise
   the assistant abstains.
8. Gemini generates an answer plus evidence IDs. Unknown IDs and model-supplied citation
   markers/URLs are rejected. Source names, pages and URLs come only from retrieved metadata.
9. A separate Gemini verification checks every claim against the selected evidence.
   Failed verification removes the answer and its citations in favor of an explicit fallback.
10. Only verified answers enter conversation memory. Operational metrics are written locally.

The generation and verification steps reduce unsupported claims; they do not prove factual
correctness or eliminate hallucinations. Web snippets can also be incomplete or incorrect.
Rules are intentionally small and explainable; nuanced queries can be misclassified.
The system does not determine an individual's coverage without sufficient policy evidence.

### API

```python
from src.chatbot.langgraph_chatbot import app

first = app.invoke({"question": "What is Zero Co-pay?"})
second = app.invoke({
    "question": "Can I buy it separately?",
    "history": first["history"],
})
print(second["answer"], second["sources"], second["route"])
```

`session_id` is optional trace metadata; it does not fetch history from a shared store.
Callers must carry the returned `history`. Browser refresh/new sessions reset memory.
The result includes the actual workflow, errors, node timings, end-to-end latency,
attempted LLM calls, provider-reported tokens, retrieved documents and supplied evidence.

### LLM call budget

| Route | Normal agent calls |
| --- | --- |
| Clear clarification / non-insurance | 0 |
| Uncertain classification ending in clarification / non-insurance | 1 |
| RAG with usable evidence | 2: generation + verification |
| Live web answer with usable evidence | 2: generation + verification; 1 Tavily request |
| History-dependent / uncertain RAG or web | 3: combined analysis + generation + verification |
| Empty evidence / missing web configuration | 0 generation/verification calls |
| Evaluation with `--judge` | 1 additional judge call per case |

An empty generation or invalid citation skips verification. Provider failures stop the route;
Gemini automatic retries are disabled. Embedding, BM25, reranking and compression run locally.
`llm_calls` records actual attempted model invocations; providers may report tokens only
for successful requests. The judge is never called in normal chat.

## Documents and ingestion

The five supplied PDFs cover Zero Co-pay, BAGIC Family Health Care, an ICICI claim process,
Insurance Ombudsman Rules amended through 18 May 2021, and a 2019–2020 consumer affairs booklet.
They are historical/reference material, not a representation of every insurer's current terms.

Place PDF, UTF-8 TXT or DOCX files under `data/`. The loader preserves relative filenames and
zero-based PDF pages; displayed source pages are one-based. DOCX paragraphs and tables are read.
Scanned PDFs require external OCR: OCR is not implemented.

Ingestion extracts text, chunks with overlap, embeds with `BAAI/bge-small-en-v1.5`, and upserts
stable chunk IDs into the `insurance` Chroma collection. Repeated ingestion does not duplicate
chunks. After successful upserts, stale chunks are removed from that collection; unrelated
collections and the database directory are retained. Empty input leaves the index untouched.
Restart Streamlit after ingestion so its cached BM25 corpus is refreshed.
There is no document-upload UI.

## Tests and evaluation

```powershell
pytest -q
ruff check .
python -m src.evaluation.evaluation --validate
# Real runs consume quota. Begin with a small limit:
python -m src.evaluation.evaluation --run --limit 3
python -m src.evaluation.evaluation --run --judge --limit 3
# Full benchmark, with optional semantic judging:
python -m src.evaluation.evaluation --run --judge
```

The benchmark contains 45 cases across policy features, Zero Co-pay, claims, Ombudsman,
live information, ambiguity and unrelated requests. Static answer references were checked
against the supplied PDFs; source/page provenance appears in each case's notes. Live questions
deliberately have no frozen factual answer. This is a development benchmark, not an independently
annotated test set or evidence of achieved performance. Multi-turn behavior is tested separately.

Local reports are timestamped JSON under `evaluation/results/` with per-case answers,
evidence, errors and metric denominators:

- Routing accuracy: exact expected-vs-actual route.
- Source hit: expected source filename among retrieved chunks; **not** passage recall or precision.
- Correctness: optional semantic judge agreement with the reference.
- Groundedness / hallucination: optional cited-evidence support score and its complement.
- Retrieval relevance: optional judge assessment of retrieved text for RAG queries.
- End-to-end and node latency, error rate, model calls and provider token usage.

Skipped/unavailable scores are `null`, never substituted with success. Judge failures are
reported separately. One judge call scores all semantic dimensions; deterministic metrics
never consume judge calls. Judge scores are estimates and require human review.
By default the runner only validates the dataset. Tests block network sockets and use
mock models/web responses plus a real temporary Chroma index with deterministic test embeddings.
They do not establish live Gemini quality or real embedding retrieval accuracy.

## Optional observability

All credentials come from `.env` or environment variables. Never commit `.env`.

**LangSmith:** set `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` and `LANGSMITH_TRACING=true`
to trace graph nodes, model calls, errors, latency and tokens when reported by the provider.
To create a new dataset and experiment:

```powershell
python -m src.evaluation.evaluation --run --judge --langsmith --limit 3
```

Each execution uses a uniquely named dataset/experiment and runs examples sequentially.
The [LangSmith experiment API](https://docs.langchain.com/langsmith/evaluate-llm-application)
is used directly; the runner does not invent experiment links or scores.

**Langfuse:** install the `observability` extra and set `LANGFUSE_ENABLED=true`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`.
The [LangChain callback integration](https://langfuse.com/docs/integrations/langchain/tracing)
records graph/model activity and available token metadata. The evaluator attaches numeric
scores to the resulting trace ID. Initialization and score-upload failures are non-fatal;
missing keys disable it. Hosted traces are opt-in and include questions, answers and evidence.

Without either service, `logs/requests.jsonl` records only operational metadata, without
questions, answers or source contents. View it using:

```powershell
python -m src.evaluation.analytics_report
```

## Code map

| Location | Responsibility |
| --- | --- |
| `src/app.py`, `src/ui/app.py` | Streamlit entry point and session-owned UI |
| `src/chatbot/` | Query analysis, routing, state and LangGraph workflow |
| `src/retrieval/` | Loaders, ingestion, Chroma, BM25/RRF, reranking, compression |
| `src/services/` | Lazy model/web clients, memory, operational metrics, tracing |
| `src/config/` | Project-relative paths and environment settings |
| `src/evaluation/`, `evaluation/test_dataset.csv` | Benchmark, judge, experiments and reports |
| `tests/`, `.github/workflows/ci.yml` | Quota-free checks and GitHub CI |

## Honest interview / resume claims

Supported: implemented LangGraph routing, document ingestion, BM25 + dense RRF retrieval,
cross-encoder reranking, semantic deduplication, session-isolated conversational history,
source-ID validation, model-based verification, optional web search and evaluation integrations.

Do not claim measured accuracy, latency improvements, reduced hallucination percentages,
production reliability, regulatory compliance, calibrated confidence, fine-tuning, OCR,
persistent multi-user memory, or successful hosted experiments without actual evidence.
Langfuse/LangSmith integration code and mocked tests alone do not establish successful
delivery to a hosted account. No quality scores are published by this repository.
