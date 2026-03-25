"""
ingest.py — Parse PDFs, chunk, embed, and store in ChromaDB.

Usage:
    python -m src.ingest                     # ingest all PDFs in data/
    python -m src.ingest data/carrier-30xa-iom.pdf   # single file
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
from pathlib import Path

import chromadb
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100


def _clean_text(text: str) -> str:
    """Strip null bytes and non-printable control characters that break JSON serialisation."""
    return "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127)
    )


def _serialise_table(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-delimited text."""
    rows = []
    for row in table:
        cells = [str(cell or "").strip() for cell in row]
        rows.append(" | ".join(cells))
    return "[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Extract text and tables from every page, return list of content dicts."""
    reader = PdfReader(str(pdf_path))
    items: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for i, (pdf_page, plumber_page) in enumerate(
            zip(reader.pages, plumber_pdf.pages)
        ):
            page_num = i + 1

            text = _clean_text((pdf_page.extract_text() or "").strip())
            if text:
                items.append(
                    {
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_num,
                        "content_type": "text",
                    }
                )

            tables = plumber_page.extract_tables() or []
            for table in tables:
                if table:
                    serialised = _clean_text(_serialise_table(table))
                    items.append(
                        {
                            "text": serialised,
                            "source": pdf_path.name,
                            "page": page_num,
                            "content_type": "table",
                        }
                    )

    text_count = sum(1 for it in items if it["content_type"] == "text")
    table_count = sum(1 for it in items if it["content_type"] == "table")
    print(f"  Parsed {text_count} text pages, {table_count} tables from {pdf_path.name}")
    return items


def items_to_documents(items: list[dict]) -> list:
    """Sliding-window chunk each item, preserving source metadata."""
    from langchain_core.documents import Document

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n",
    )
    docs = []
    for item in items:
        chunks = splitter.split_text(item["text"])
        for j, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": item["source"],
                        "page": item["page"],
                        "content_type": item["content_type"],
                        "chunk": j,
                    },
                )
            )
    return docs


def ingest_pdfs(pdf_paths: list[Path]) -> Chroma:
    """Parse, chunk, embed, and persist all PDFs to ChromaDB + BM25 corpus."""
    bm25_path = CHROMA_DIR / "bm25_corpus.pkl"
    if CHROMA_DIR.exists() and bm25_path.exists():
        print(f"Index already exists at {CHROMA_DIR}. Delete it to re-ingest.")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="carrier_30xa",
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    all_docs = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        items = parse_pdf(pdf_path)
        docs = items_to_documents(items)
        print(f"  → {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"\nTotal chunks to embed: {len(all_docs)}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Persisting dense index to: {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="carrier_30xa",
    )

    print(f"Persisting BM25 corpus to: {bm25_path}")
    with open(bm25_path, "wb") as f:
        pickle.dump(all_docs, f)

    print("Done.")
    return vectorstore


async def _embed_all_async(texts: list[str]) -> list[list[float]]:
    """Embed all texts in parallel batches via asyncio.gather."""
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    print(f"  Embedding {len(batches)} batches in parallel...")
    results = await asyncio.gather(
        *[embeddings_model.aembed_documents(batch) for batch in batches]
    )
    return [emb for batch in results for emb in batch]


async def ingest_pdfs_async(pdf_paths: list[Path]) -> None:
    """Async variant: parallel embedding batches, raw chromadb client for storage."""
    bm25_path = CHROMA_DIR / "bm25_corpus.pkl"
    if CHROMA_DIR.exists() and bm25_path.exists():
        print(f"Index already exists at {CHROMA_DIR}. Delete it to re-ingest.")
        return

    all_docs = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        items = parse_pdf(pdf_path)
        docs = items_to_documents(items)
        print(f"  → {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"\nTotal chunks to embed: {len(all_docs)}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [doc.page_content for doc in all_docs]
    all_embeddings = await _embed_all_async(texts)

    print(f"Persisting dense index to: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="carrier_30xa",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[str(i) for i in range(len(all_docs))],
        embeddings=all_embeddings,
        documents=texts,
        metadatas=[doc.metadata for doc in all_docs],
    )

    print(f"Persisting BM25 corpus to: {bm25_path}")
    with open(bm25_path, "wb") as f:
        pickle.dump(all_docs, f)

    print("Done (async ingestion).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("paths", nargs="*", help="PDF paths (default: all in data/)")
    parser.add_argument(
        "--async", dest="use_async", action="store_true",
        help="Use async parallel embedding (faster for large corpora)",
    )
    args = parser.parse_args()

    if args.paths:
        pdf_paths = [Path(p) for p in args.paths]
    else:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}. Run python download_data.py first.")
        sys.exit(1)

    missing = [p for p in pdf_paths if not p.exists()]
    if missing:
        print(f"Files not found: {missing}")
        sys.exit(1)

    if args.use_async:
        asyncio.run(ingest_pdfs_async(pdf_paths))
    else:
        ingest_pdfs(pdf_paths)


if __name__ == "__main__":
    main()
