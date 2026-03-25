# Hybrid RAG Pipeline for Industrial Equipment Knowledge Base

A production-ready **Retrieval-Augmented Generation (RAG)** system that ingests PDF manuals for the Carrier 30XA Air-Cooled Liquid Chiller and answers technical questions with verifiable inline citations.

## 🎯 What It Does

- **Ingests** PDF manuals (text + tables) using sliding window chunking
- **Indexes** content with both dense embeddings (semantic) and sparse BM25 (keyword) 
- **Answers** natural language questions using hybrid retrieval with RRF fusion
- **Cites** every factual statement with `[source: <file>, p.<N>]` references
- **Supports** multi-turn conversations with contextual question rewriting
- **Evaluates** performance with automated metrics (citation rate, precision@K, etc.)

## 🏗️ Architecture

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
              Manual RRF fusion (weights 0.5/0.5)
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

**Why hybrid?** Dense embeddings excel at semantic similarity; BM25 excels at exact keyword and numeric matches (alarm codes, model numbers, operating limits). Fusing both via Reciprocal Rank Fusion (RRF) reliably outperforms either alone on technical text.

## 🛠️ Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| PDF text | `pypdf` | Fast, no external deps |
| PDF tables | `pdfplumber` | Best OSS table extraction for PDF |
| Chunking | `langchain_text_splitters.CharacterTextSplitter` | Sliding window — uniform coverage, no boundary bias |
| Dense embeddings | `text-embedding-3-small` | Best cost/quality for technical text |
| Dense store | `chromadb` (local, persistent) | Zero infra, cosine search |
| Sparse index | `rank_bm25` via `BM25Retriever` | TF-IDF keyword matching |
| Fusion | Manual RRF implementation | EnsembleRetriever not available in LangChain 1.2.13 |
| LLM | `gpt-4o-mini` (temp=0) | Deterministic, cheap, 128k context |
| Orchestration | LangChain LCEL | Composable, no deprecated chains |
| Session memory | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` | Zero-config, multi-session, LangChain 1.x standard |
| Secrets | `python-dotenv` + `.env` | No shell exports required |

## 📁 File Structure

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
├── pyproject.toml
├── .python-version               # 3.12
├── .env                          # OPENAI_API_KEY (gitignored)
├── start.md                      # This file
└── spec.md                       # Detailed technical specification
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
git clone <repo-url>
cd interview-blokh-galina

# Install dependencies (Python 3.12 required)
uv sync --all-groups

# Set up OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 2. Download Data

```bash
# Download the Carrier 30XA manuals
python download_data.py
```

This downloads three PDF manuals to the `data/` directory:
- `carrier-30xa-iom.pdf` (56 pages) - Installation, operation & maintenance
- `carrier-30xa-controls.pdf` (206 pages) - Controls, configuration & diagnostics  
- `carrier-30xa-installation.pdf` (52 pages) - Dimensions, clearances, setup

### 3. Option A: Fresh Ingestion (First-time setup)

```bash
# Download the Carrier 30XA manuals
python download_data.py

# Synchronous ingestion (default)
python -m src.ingest

# Or parallel batch ingestion for large corpora
python -m src.ingest --async
```

This:
- Downloads three PDF manuals to the `data/` directory
- Extracts text and tables from PDFs
- Chunks with sliding window (600 chars, 200 overlap)
- Creates dense ChromaDB embeddings and sparse BM25 index
- Persists to `chroma_db/` for reuse

### 3. Option B: Use Existing Database (Clone & Chat)

If you have a repository with the `chroma_db/` folder already included, you can skip ingestion entirely:

```bash
# Clone repo with existing ChromaDB
git clone <repo-url>
cd interview-blokh-galina

# Install dependencies
uv sync --all-groups

# Set up OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Start chatting immediately!
python -m src.chat
```

**What's already available in `chroma_db/`:**
- Dense embeddings: All text chunks embedded with OpenAI's text-embedding-3-small
- BM25 corpus: `chroma_db/bm25_corpus.pkl` for sparse keyword retrieval
- Metadata: Source file names and page numbers for citations

**When to re-ingest:** Only run `python -m src.ingest` if you:
- Add new PDF files to `data/`
- Update existing PDFs
- Want to change chunking parameters
- Need to use a different embedding model

### 4. Query the System

#### Single-Turn Questions
```bash
# One-off query
python -m src.query "What is the operating temperature range?"

# Interactive REPL
python -m src.query
```

#### Multi-Turn Chat
```bash
# Start chat session
python -m src.chat

# Sample conversation:
> What refrigerant does the 30XA use?
The 30XA uses refrigerant R-134a [source: carrier-30xa-iom.pdf, p.43].

> What is its operating pressure?
The maximum operating pressure of the refrigerant circuit must not exceed the specified maximum operating pressures, which can be verified by checking the instructions in the manual and the pressures given on the unit name plate [source: carrier-30xa-iom.pdf, p.7].

> What fault code indicates low oil pressure?
The fault code triggered by low oil pressure is Circuit B Low Oil Pressure and Circuit C Low Oil Pressure [source: carrier-30xa-controls.pdf, p.99].
```

### 5. Evaluate Performance

```bash
# Run evaluation with CSV output
python -m src.evaluate --csv

# View results
cat results/eval.csv
```

## 📊 Evaluation Metrics

The system tracks:

- **Citation presence rate**: % of answers with proper inline citations (target: 100%)
- **Not-found rate**: % of questions that couldn't be answered (lower is better)
- **Source diversity**: Average unique source files per answer (>1 = cross-document)
- **Page spread**: Average unique pages per answer
- **Precision@K**: Keyword-based proxy for retrieval precision

## 🔧 Key Features

### Hybrid Retrieval with RRF Fusion
- Dense retrieval captures semantic meaning
- Sparse retrieval finds exact matches (codes, numbers)
- Reciprocal Rank Fusion combines both optimally

### Contextual Question Rewriting
- Follow-up questions are rewritten as standalone queries
- Uses chat history to resolve references ("its", "that", etc.)
- Maintains retrieval quality across conversations

### Session Summarisation
- Long conversations are automatically summarized
- Keeps token usage bounded regardless of session length
- Preserves recent turns verbatim for context

### Mandatory Citations
- Every factual statement must cite source and page
- System prompt enforces citation discipline
- Post-processing validates citation presence

### Async Ingestion
- Parallel batch embedding for large document sets
- Bypasses LangChain's synchronous Chroma operations
- Maintains compatibility with synchronous query pipeline

## 🎯 Use Cases

- **Technical Support**: Answer customer questions about equipment operation
- **Field Service**: Provide technicians with instant access to manual information
- **Training**: Help new engineers learn equipment specifications and procedures
- **Compliance**: Ensure answers are sourced from official documentation
- **Knowledge Management**: Centralize dispersed technical documentation

## 🔍 Implementation Details

### Chunking Strategy
- **Size**: 600 characters (~150 tokens) keeps chunks focused
- **Overlap**: 200 characters prevents context loss at boundaries
- **Sliding window**: Uniform coverage without sentence boundary bias

### Retrieval Parameters
- **Top-K**: 10 candidates from each retriever
- **Final-K**: 5 documents after RRF fusion
- **RRF K**: 60 (standard constant for rank fusion)

### LLM Configuration
- **Model**: gpt-4o-mini (cost-effective, high quality)
- **Temperature**: 0 (deterministic answers)
- **Context window**: 128k tokens (ample for retrieved docs + prompt)

## 🚧 Future Work

- **Web Interface**: Streamlit/Gradio UI for non-technical users
- **Document Updates**: Incremental ingestion pipeline
- **Alternative Embeddings**: Local models for privacy
- **Evaluation Suite**: Human-in-the-loop rubric scoring
- **Performance Optimization**: Vector quantization, caching
- **Multi-Modal**: Image/diagram extraction and indexing
- **Cross-Document Reasoning**: Synthesis across multiple manuals

## 📝 License

This project is for demonstration and educational purposes. Carrier documentation is copyrighted by Carrier Corporation.

---

**Built with**: Python 3.12, LangChain 1.2.13, OpenAI APIs, ChromaDB, BM25
