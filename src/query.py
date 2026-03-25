"""
query.py — Hybrid RAG: dense (ChromaDB) + sparse (BM25) retrieval via LCEL.

Usage:
    python -m src.query "What is the operating temperature range?"
    python -m src.query   # interactive REPL
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# EnsembleRetriever not available in LangChain 1.2.13; implementing RRF fusion manually
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
BM25_PATH = CHROMA_DIR / "bm25_corpus.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 10
FINAL_K = 5

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a technical assistant for the Carrier 30XA Air-Cooled Liquid Chiller.\n"
        "Answer using ONLY the information in the context below.\n"
        "Rules:\n"
        "1. Every factual statement MUST end with an inline citation in the format "
        "[source: <filename>, p.<page>].\n"
        "2. Cite all pages if a fact appears across multiple pages.\n"
        "3. If the answer is not in the context, respond with exactly: "
        "'I could not find that information in the provided manuals.'\n"
        "4. Do NOT guess or infer facts not explicitly stated in the context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (with inline citations):"
    ),
)


def _format_docs(docs: list[Document]) -> str:
    """Prefix every chunk with its source tag so the LLM can cite it directly."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[source: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _rrf_fusion(dense_docs: list[Document], sparse_docs: list[Document], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: combine two ranked lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = f"{doc.metadata.get('source', '?')}_{doc.metadata.get('page', '?')}_{doc.page_content[:50]}"
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(sparse_docs):
        doc_id = f"{doc.metadata.get('source', '?')}_{doc.metadata.get('page', '?')}_{doc.page_content[:50]}"
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
        doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:FINAL_K]]


def build_chain():
    """Build hybrid retriever (dense + BM25) and LCEL answer chain."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Index not found at {CHROMA_DIR}. Run `python -m src.ingest` first."
        )
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {BM25_PATH}. Run `python -m src.ingest` first."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="carrier_30xa",
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    with open(BM25_PATH, "rb") as f:
        corpus_docs: list[Document] = pickle.load(f)
    sparse_retriever = BM25Retriever.from_documents(corpus_docs, k=TOP_K)

    def _hybrid_retrieve(question: str) -> list[Document]:
        dense_docs = dense_retriever.invoke(question)
        sparse_docs = sparse_retriever.invoke(question)
        return _rrf_fusion(dense_docs, sparse_docs)

    hybrid_retriever = RunnableLambda(_hybrid_retrieve)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = (
        {"context": hybrid_retriever | _format_docs, "question": RunnablePassthrough()}
        | SYSTEM_PROMPT
        | llm
        | StrOutputParser()
    )
    return hybrid_retriever, chain


def ask(retriever_and_chain, question: str) -> dict:
    retriever, chain = retriever_and_chain
    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = [
        f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
        for doc in docs
    ]
    return {
        "question": question,
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "docs": docs,
    }


def print_result(result: dict) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {', '.join(result['sources']) or 'none'}")
    print("-" * 60)


def main() -> None:
    retriever_and_chain = build_chain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print_result(ask(retriever_and_chain, question))
        return

    print("Carrier 30XA Knowledge Base (hybrid RAG) — type 'quit' to exit\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        print_result(ask(retriever_and_chain, question))


if __name__ == "__main__":
    main()
