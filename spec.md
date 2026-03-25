# Implementation Spec: Industrial Equipment Knowledge Base

## Goal

Build an end-to-end **hybrid RAG** (Retrieval-Augmented Generation) pipeline that
ingests PDF manuals for the **Carrier 30XA Air-Cooled Liquid Chiller**, stores
extracted knowledge in a vector + keyword index, and answers natural-language
questions with mandatory inline citations.

---

## Source Documents

| File | Pages | Focus |
|------|-------|-------|
| `carrier-30xa-iom.pdf` | 56 | Installation, operation & maintenance |
| `carrier-30xa-controls.pdf` | 206 | Controls, configuration & diagnostics |
| `carrier-30xa-installation.pdf` | 52 | Dimensions, clearances, setup |

**Scope:** All three PDFs are ingested. The pipeline generalises across documents.

---

## Architecture

```
PDFs
 │
 ├─[Text extractor]   pypdf.PdfReader  →  plain text per page
 └─[Table extractor]  pdfplumber       →  pipe-delimited table blocks per page
         │
         ▼
 [Chunker]  CharacterTextSplitter (sliding window: size=600, overlap=200)
         │
         ├─────────────────────────────────────┐
         ▼                                     ▼
 [Dense index]                         [Sparse index]
 OpenAI text-embedding-3-small         BM25 (rank_bm25)
 → ChromaDB (cosine, persistent)       → in-memory BM25Retriever
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
              [Hybrid Retriever]
              EnsembleRetriever (RRF fusion, weights 0.5/0.5)
              → top-K=10 candidates → deduplicated top-5
                        │
 User question + chat history
         │              │
         ▼              ▼
  [Question rewriter]   (CONDENSE_PROMPT + gpt-4o-mini)
  rewrites follow-up questions into standalone queries
         │
         ▼
  [LCEL Answer Chain]  wrapped in RunnableWithMessageHistory
  PromptTemplate + ChatOpenAI(gpt-4o-mini, temp=0)
  + StrOutputParser
  (mandatory: every statement cites [source: <file>, p.<N>])
  InMemoryChatMessageHistory  keyed by session_id
         │
         ▼
      Answer  →  appended to session history
```

**Why hybrid?**
Dense embeddings excel at semantic similarity; BM25 excels at exact keyword and
numeric matches (alarm codes, model numbers, operating limits). Fusing both via
Reciprocal Rank Fusion (RRF) reliably outperforms either alone on technical text.

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| PDF text | `pypdf` | Fast, no external deps |
| PDF tables | `pdfplumber` | Best OSS table extraction for PDF |
| Chunking | `langchain_text_splitters.CharacterTextSplitter` | Sliding window — uniform coverage, no boundary bias |
| Dense embeddings | `text-embedding-3-small` | Best cost/quality for technical text |
| Dense store | `chromadb` (local, persistent) | Zero infra, cosine search |
| Sparse index | `rank_bm25` via `BM25Retriever` | TF-IDF keyword matching |
| Fusion | `EnsembleRetriever` (RRF) | Standard hybrid fusion in LangChain |
| LLM | `gpt-4o-mini` (temp=0) | Deterministic, cheap, 128k context |
| Orchestration | LangChain LCEL | Composable, no deprecated chains |
| Session memory | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` | Zero-config, multi-session, LangChain 1.x standard |
| Secrets | `python-dotenv` + `.env` | No shell exports required |

---

## File Structure

```
interview-blokh-galina/
├── data/                         # PDFs (not committed)
├── chroma_db/                    # Persistent dense index (not committed)
├── src/
│   ├── __init__.py
│   ├── ingest.py                 # Extract → chunk → embed → persist
│   ├── query.py                  # Hybrid retrieve → generate → cite (single-turn)
│   ├── chat.py                   # Multi-turn chat with session memory
│   └── evaluate.py               # Run eval set, compute metrics, write CSV
├── results/                      # Eval CSV outputs (not committed)
├── download_data.py
├── pyproject.toml
├── .python-version               # 3.12
├── .env                          # OPENAI_API_KEY (gitignored)
├── spec.md
└── README.md
```

---

## Implementation Steps

### Step 1 — Ingest (`src/ingest.py`)

#### 1.1 Extraction

For each page of each PDF:

- **Text**: `pypdf.PdfReader` → `page.extract_text()`, strip whitespace
- **Tables**: `pdfplumber` → `page.extract_tables()` → serialise each table:

```
[TABLE]
col1 | col2 | col3
val1 | val2 | val3
[/TABLE]
```

Tag every extracted item with metadata: `source` (filename), `page` (1-indexed),
`content_type` (`"text"` or `"table"`).

#### 1.2 Chunking — sliding window

Use `CharacterTextSplitter(chunk_size=600, chunk_overlap=200, separator="\n")`.

- **Size 600**: fits ~150 tokens — keeps chunks focused, avoids diluting embeddings
- **Overlap 200**: one-third overlap so no boundary loses its context
- **No sentence splitting**: tables have no sentence structure; fixed window is safer

Each chunk inherits the page metadata of its source item.

#### 1.3 Indexing

- **Dense**: `OpenAIEmbeddings(model="text-embedding-3-small")` →
  `Chroma.from_documents(persist_directory="chroma_db/", collection_name="carrier_30xa")`
- **Sparse**: all chunk texts stored in memory as a `BM25Retriever`
  (serialised to `chroma_db/bm25_corpus.pkl` for reuse without re-parsing)

Run once; subsequent calls load from disk.

#### 1.4 Async ingestion (`--async` flag)

For large corpora, embedding is the bottleneck. The `--async` flag switches to
parallel batch embedding via `asyncio.gather`:

```python
# EMBED_BATCH_SIZE = 100  (chunks per request)
batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
results = await asyncio.gather(
    *[embeddings.aembed_documents(batch) for batch in batches]
)
```

Pre-computed embeddings are stored directly via the `chromadb.PersistentClient`
(bypassing LangChain's synchronous `Chroma.from_documents`). The collection is
created with `{"hnsw:space": "cosine"}` to match the default LangChain retriever
behaviour so `query.py` loads it transparently.

```bash
python -m src.ingest           # synchronous (default)
python -m src.ingest --async   # parallel batches
```

#### 1.5 Environment

```
OPENAI_API_KEY=sk-...   # in .env — loaded by python-dotenv at startup
```

---

### Step 2 — Query (`src/query.py`)

#### 2.1 Hybrid Retrieval

```python
dense_retriever  = chroma.as_retriever(search_kwargs={"k": 10})
sparse_retriever = BM25Retriever.from_documents(docs, k=10)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5],           # equal weight; tune if needed
)
```

RRF formula: `score(d) = Σ 1 / (rank_i(d) + 60)` over both ranked lists.
Returns deduplicated top-5 by fused score.

#### 2.2 LCEL Answer Chain

```python
chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | SYSTEM_PROMPT
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)
```

`format_docs` prepends each chunk with its source tag:
`[source: carrier-30xa-iom.pdf, p.12]\n<chunk text>` so the LLM has
explicit provenance for every passage it reads.

#### 2.3 System Prompt

```
You are a technical assistant for the Carrier 30XA Air-Cooled Liquid Chiller.
Answer using ONLY the information in the context below.

Rules:
1. Every factual statement MUST end with an inline citation [source: <file>, p.<N>].
2. Cite all pages if a fact appears across multiple pages.
3. If the answer is not in the context, say: "I could not find that information
   in the provided manuals." — do not guess or infer.
4. Do not add information not explicitly stated in the context.
```

#### 2.4 Source Attribution

After chain invocation, `ask()` independently queries the retriever and attaches
deduplicated source references (file + page) to the returned dict alongside the
LLM answer.

---

### Step 4 — Multi-Turn Chat (`src/chat.py`)

#### 4.1 Problem: follow-up questions break retrieval

A standalone retriever cannot resolve references in follow-up questions:

```
User: What refrigerant does the 30XA use?
User: What is its operating pressure?   ← "its" is ambiguous to the retriever
```

Without history, the second question retrieves irrelevant chunks.

#### 4.2 Solution: contextual question rewriting

Before retrieval, a cheap LLM call rewrites the user's message into a
self-contained question using the chat history:

```python
CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the conversation below and a follow-up question, "
        "rewrite the follow-up as a standalone question that can be "
        "understood without the conversation context.\n"
        "If the question is already standalone, return it unchanged.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Follow-up: {question}\n"
        "Standalone question:"
    ),
)
```

The rewritten question is passed to the hybrid retriever. The original question
(not the rewritten one) is shown to the user.

#### 4.3 Session summarisation

Long conversations cause the condense prompt to exceed the token budget and
degrade question rewriting quality. When the serialised history exceeds
`MAX_HISTORY_CHARS` (2000 chars ≈ 500 tokens):

1. Split messages: **old turns** (all except the last `MAX_RECENT_TURNS=2`) and
   **recent turns** (last 2 turns, kept verbatim)
2. Summarise old turns with a cheap LLM call:
   `SUMMARISE_PROMPT | gpt-4o-mini` → 2–3 sentence summary
3. Build condensed history: `[Earlier summary: ...]\n\n<recent turns>`

The condensed history is passed to `CONDENSE_PROMPT` instead of the full
history, keeping the token cost bounded regardless of session length.

#### 4.4 Session memory with `RunnableWithMessageHistory`

```python
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    conversational_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

- Each `session_id` gets an isolated conversation history
- History grows automatically each turn — no manual append needed
- The full pipeline per turn:
  1. Load history for `session_id`
  2. Summarise history if over `MAX_HISTORY_CHARS` (`_maybe_summarise_history`)
  3. Rewrite question using condensed history + `CONDENSE_PROMPT`
  4. Retrieve with hybrid retriever on the rewritten question
  5. Generate answer with mandatory citations
  6. Save `(user_msg, ai_msg)` to history

#### 4.5 Usage

```bash
python -m src.chat                    # interactive REPL (default session)
python -m src.chat --session alice    # named session
```

---

### Step 3 — Evaluate (`src/evaluate.py`)

Runs the full set of evaluation questions through the pipeline and computes both
automated and printable manual-review metrics.

#### 3.1 Automated Metrics (computed per run)

| Metric | How computed |
|--------|-------------|
| **Citation presence rate** | % of answers containing `[source:` pattern |
| **Not-found rate** | % of answers containing the fallback phrase |
| **Source diversity** | avg. unique source files per answer (should be > 1 for cross-doc questions) |
| **Retrieval spread** | avg. unique pages retrieved per question |
| **Precision@K** | keyword-based proxy — see §3.4 below |

These are fully automated — no reference answers needed.

#### 3.4 Retrieval Precision@K

Since no ground-truth relevance labels exist, precision@K is computed as a
**keyword-based proxy**:

1. Extract content keywords from the question (tokenise, lowercase, remove
   stopwords and tokens ≤ 2 chars)
2. For each of the top-K retrieved chunks, check if at least one keyword appears
   in `doc.page_content`
3. `Precision@K = hits / K`

```python
keywords = {t for t in question.lower().split() if t not in STOPWORDS and len(t) > 2}
hits = sum(1 for doc in docs[:K] if any(kw in doc.page_content.lower() for kw in keywords))
precision_at_k = hits / K
```

This is a conservative lower bound — a chunk may be highly relevant without
containing the exact keyword — but it is fully automated and flags obvious
retrieval failures (score ≈ 0). Reported in the summary printout and in
`results/eval.csv`.

#### 3.2 Manual Review Rubric (per question)

| Criterion | Pass condition |
|-----------|---------------|
| Cited | Every sentence has `[source: ..., p.N]` |
| Grounded | No fact in the answer is absent from the retrieved chunks |
| Correct | Answer matches the actual content of the cited page |
| Complete | All parts of the question are addressed |

Score each question 0–4 (one point per criterion). Target ≥ 3/4 on all questions.

#### 3.3 Evaluation Questions

Ten questions spanning all three documents:

1. What is the operating temperature range of the evaporator?
2. What is the minimum ambient temperature for operation?
3. Which sensors are connected to the main controller?
4. What are the steps to start the chiller for the first time?
5. What fault code is triggered by low oil pressure?
6. What refrigerant type is used in the Carrier 30XA?
7. What are the recommended clearances for installation?
8. How is the capacity control algorithm described?
9. What is the maximum operating pressure of the refrigerant circuit?
10. What type of compressor does the 30XA use?

Output: `results/eval.csv` with columns `question`, `answer`, `sources`,
`citation_present`, `not_found`.

---

## Key Design Decisions

### Hybrid retrieval (dense + BM25)

Dense embeddings capture semantic similarity but can miss exact numeric values
(e.g. "30 psi", "HFO-1234ze") that BM25 finds trivially. Hybrid retrieval via RRF
is the de-facto best practice for RAG over technical documents and adds no
inference cost — only a fast in-memory sort.

### Sliding window chunking

Every token appears in ≥1 chunk regardless of where page breaks or paragraph
boundaries fall. Critical for tables, which have no natural sentence boundaries.
Overlap of 200 chars (≈one table row) prevents a value from being split across
chunks with no context.

### Context tagging before LLM

Each chunk passed to the LLM is prefixed with its source tag:
`[source: file.pdf, p.N]`. This means citations in the answer are grounded in
what the model literally reads — not inferred from metadata appended afterward.

### Mandatory inline citations

Enforced via the prompt, not post-processing. Simpler, more robust. The LLM sees
the source tag on every passage, so citing it is the path of least resistance.

### Temperature = 0

Technical Q&A requires determinism. Hallucination risk is lowest at temp=0 when
the context is clear.

### LCEL over legacy chains

LangChain 1.x removed `RetrievalQA` and `langchain.chains`. LCEL (`|` pipe
composition) is the current standard — composable, typed, debuggable via
LangSmith if needed.

### Multi-turn memory: rewrite before retrieval

The most common failure mode in conversational RAG is retrieval degradation on
follow-up questions. Contextual question rewriting fixes this with a single
additional LLM call (~100 tokens) before every retrieval. This is cheaper and
more reliable than stuffing the full conversation history into the retriever query.

`RunnableWithMessageHistory` was introduced in LangChain 1.x as the replacement
for `ConversationChain` and `ConversationalRetrievalChain`. It wraps any LCEL
chain and handles history injection transparently, keyed by `session_id`.
`InMemoryChatMessageHistory` stores turns in RAM — sufficient for an interactive
session; swap for `FileChatMessageHistory` to persist across restarts.

### Local ChromaDB + pickle for BM25

Zero infra. `chroma_db/` holds the dense index; `bm25_corpus.pkl` holds the
serialised document list for BM25 reconstruction. Both are reproduced by running
`python -m src.ingest`.

### `.env` for secrets

`OPENAI_API_KEY` in `.env` (gitignored). `python-dotenv` loads it at module
import — no shell configuration required.

---

## Future Work / Improvements

### Retrieval quality

- **Cross-encoder re-ranking** — after hybrid retrieval, apply a `sentence-transformers`
  cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the
  top-10 candidates and keep the top-5. Adds ~50 ms per query; meaningfully
  improves precision on ambiguous questions.
- **Parent-document retrieval** — chunk with small windows for embedding but
  return the full parent page to the LLM for richer context. Reduces the risk of
  answers spanning a chunk boundary.
- **Metadata filtering** — allow filtering by source document at query time
  (e.g. `?source=carrier-30xa-controls.pdf`) to narrow retrieval scope.
- **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer
  first, embed it, and use that embedding for retrieval. Helps on vague questions
  where the user does not know the right keywords.
- **Upgrade to `text-embedding-3-large`** — if retrieval quality is poor on
  technical numeric values, the larger model provides better representation.

### Parsing & extraction

- **Diagram / image extraction** — use `pdfplumber` visual rendering + a
  vision model (GPT-4o) to caption diagrams and store captions as chunks.
- **Structured section detection** — detect numbered sections and warning blocks
  as distinct chunk types; store `section` metadata to enable section-level filtering.
- **OCR fallback** — for scanned pages with no selectable text, fall back to
  `pytesseract` or the OpenAI vision API.

### Evaluation

- **Reference answer set** — create 10–20 ground-truth `(question, expected_answer)`
  pairs by manually checking the PDFs; compute exact-match and ROUGE-L scores.
- **LLM-as-judge faithfulness** — use a second `gpt-4o-mini` call to score
  whether every claim in the answer is supported by the retrieved context.
  Produces a per-answer faithfulness score (0.0–1.0).

### Memory & sessions

- **Cross-session persistent memory** — swap `InMemoryChatMessageHistory` for
  `FileChatMessageHistory` (JSON) or `SQLChatMessageHistory` (SQLite) to
  preserve conversation history across restarts.

### Infrastructure

- **FastAPI / REST endpoint** — expose `query` and `chat` as HTTP endpoints so
  the pipeline can be called from external tools or a front-end.
- **Streaming responses** — use `chain.astream()` + server-sent events to stream
  tokens to the client rather than waiting for the full answer.
- **Docker / container packaging** — containerise the service so it runs
  identically across environments without manual venv setup.
