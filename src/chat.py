"""
chat.py — Multi-turn conversational RAG with session memory.

Uses contextual question rewriting to resolve follow-up references before
retrieval, and RunnableWithMessageHistory to manage per-session chat history.

Usage:
    python -m src.chat                    # interactive REPL, default session
    python -m src.chat --session alice    # named session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.query import build_chain, _format_docs

load_dotenv()

LLM_MODEL = "gpt-4o-mini"
MAX_HISTORY_CHARS = 2000
MAX_RECENT_TURNS = 2

SUMMARISE_PROMPT = PromptTemplate(
    input_variables=["history"],
    template=(
        "Summarise the following conversation in 2-3 sentences, preserving key "
        "technical facts and topics discussed:\n\n{history}\n\nSummary:"
    ),
)

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without the "
        "conversation context.\n"
        "If the question is already standalone, return it unchanged.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Follow-up: {question}\n"
        "Standalone question:"
    ),
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical assistant for the Carrier 30XA Air-Cooled Liquid Chiller.\n"
            "Answer using ONLY the information in the context below.\n"
            "Rules:\n"
            "1. Every factual statement MUST end with an inline citation "
            "[source: <filename>, p.<page>].\n"
            "2. Cite all pages if a fact appears across multiple pages.\n"
            "3. If the answer is not in the context, respond with exactly: "
            "'I could not find that information in the provided manuals.'\n"
            "4. Do NOT guess or infer facts not explicitly stated in the context.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def _format_history(history: BaseChatMessageHistory) -> str:
    """Serialise all chat history messages to plain text."""
    lines = []
    for msg in history.messages:
        role = "Human" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _maybe_summarise_history(
    history: BaseChatMessageHistory,
    llm: ChatOpenAI,
) -> str:
    """Return history text, summarising old turns when over the token budget."""
    full_text = _format_history(history)
    if len(full_text) <= MAX_HISTORY_CHARS:
        return full_text

    messages = history.messages
    keep_n = MAX_RECENT_TURNS * 2
    old_messages = messages[:-keep_n] if len(messages) > keep_n else []
    recent_messages = messages[-keep_n:] if len(messages) > keep_n else messages

    if not old_messages:
        return full_text

    old_text = "\n".join(
        f"{'Human' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in old_messages
    )
    summary = (SUMMARISE_PROMPT | llm | StrOutputParser()).invoke({"history": old_text})
    recent_text = "\n".join(
        f"{'Human' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in recent_messages
    )
    return f"[Earlier conversation summary: {summary}]\n\n{recent_text}"


def build_chat_chain():
    """Build the conversational chain: rewriter → hybrid retriever → answer."""
    hybrid_retriever, _ = build_chain()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    condense_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def rewrite_and_retrieve(inputs: dict) -> dict:
        question = inputs["question"]
        history = get_session_history(inputs.get("session_id", "default"))

        history_text = _maybe_summarise_history(history, condense_llm)
        if history_text.strip():
            standalone = (
                CONDENSE_PROMPT | condense_llm | StrOutputParser()
            ).invoke({"chat_history": history_text, "question": question})
        else:
            standalone = question

        docs = hybrid_retriever.invoke(standalone)
        return {
            "context": _format_docs(docs),
            "question": question,
            "chat_history": history.messages,
            "sources": list(dict.fromkeys(
                f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
                for doc in docs
            )),
        }

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        answer_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return rewrite_and_retrieve, chain_with_history


def chat(
    rewriter,
    chain_with_history,
    question: str,
    session_id: str = "default",
) -> dict:
    """Run one conversational turn. Returns answer + sources."""
    inputs = rewriter({"question": question, "session_id": session_id})
    answer = chain_with_history.invoke(
        {"context": inputs["context"], "question": inputs["question"]},
        config={"configurable": {"session_id": session_id}},
    )
    return {
        "question": question,
        "answer": answer,
        "sources": inputs["sources"],
    }


def print_turn(result: dict) -> None:
    print(f"\nA: {result['answer']}")
    print(f"   Sources: {', '.join(result['sources']) or 'none'}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn chat with the 30XA knowledge base")
    parser.add_argument("--session", default="default", help="Session ID (default: 'default')")
    args = parser.parse_args()

    print("Building hybrid retriever and chat chain...")
    rewriter, chain_with_history = build_chat_chain()

    print(
        f"\nCarrier 30XA — Conversational Mode  [session: {args.session}]\n"
        "Type 'quit' to exit, 'reset' to clear session history.\n"
    )

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "reset":
            _session_store.pop(args.session, None)
            print("Session history cleared.\n")
            continue

        result = chat(rewriter, chain_with_history, question, session_id=args.session)
        print_turn(result)


if __name__ == "__main__":
    main()
