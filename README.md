# Industrial Equipment Knowledge Base

## Background

Industrial facilities rely on technical manuals to operate and maintain their
equipment. These manuals contain critical information — operating limits,
component specifications, troubleshooting procedures, alarm codes — but the
knowledge is locked inside dense PDF documents. Operators need to quickly find
answers, understand relationships between components, and reason about system
behavior.

Your task is to build a system that turns technical equipment manuals into a
queryable knowledge base.

## Task

You are given three PDF manuals for the **Carrier 30XA Air-Cooled Liquid
Chiller** — the same piece of equipment documented from different angles:

| File | Pages | Content |
|------|-------|---------|
| `carrier-30xa-iom.pdf` | 56 | Installation, operation & maintenance |
| `carrier-30xa-controls.pdf` | 206 | Controls, configuration & diagnostics |
| `carrier-30xa-installation.pdf` | 52 | Installation — dimensions, clearances, setup |

**Build a pipeline that:**

1. **Parses** one or more of the PDFs and extracts structured information
2. **Builds a knowledge base** from the extracted content
3. **Answers questions** about the equipment using retrieval + LLM generation

You choose which manuals to work with and how deep to go. Working well with one
document is better than working poorly with all three. If your approach
generalizes across documents, that's a strong signal.

Your system should be able to handle questions like:
- *"What is the operating temperature range of the evaporator?"*
- *"Which sensors are connected to the main controller?"*

These are **examples only** — your system will be evaluated against a broader
set of questions covering specs, component relationships, operating procedures,
and diagnostics.

**Key challenges:**
- The PDFs contain tables, diagrams, and structured data — not just prose
- Some questions require connecting information across multiple sections
- Some questions require understanding relationships between components (e.g.,
  which sensors feed into which control loops)

## Data

Run the download script to fetch the manuals:

```bash
python download_data.py
```

This downloads the three PDFs into `data/`. All documents are publicly
available manuals for the Carrier 30XA chiller.

## What We're Looking For

1. **LLM pipeline design** — How you structure the end-to-end flow from PDF to
   answers. What models you choose, how you chain them, how you handle different
   content types.
2. **Problem decomposition** — How you break down the task, what you tackle
   first, how you make decisions independently.
3. **RAG & retrieval** — How you build the knowledge base and retrieve relevant
   context. Embedding strategy, chunking approach, how you construct context for
   the LLM.
4. **AI-assisted development** — How effectively you use AI tools to build the
   system. Do you understand the code you're producing?
5. **Evaluation** — How you measure whether your system actually works. What
   questions do you test with, how do you assess answer quality?

## Guidelines

- **Time:** ~90 minutes. Prioritize getting something working end-to-end over
  perfecting any single component.
- **AI tools:** Fully permitted.
- **Approach:** Entirely your choice — vector DB, graph DB, hybrid, or
  something else. Justify your decisions.
- **Code structure:** Organize however you see fit.
- **API keys:** Use your own API keys for any LLM services. If you don't have
  access to a specific model, use what you have and note what you'd prefer.
- **Output:** Your repo should include working code and a brief writeup (in
  README or a separate doc) explaining your approach and design decisions.

## Quick Start

```bash
pip install openai langchain pypdf chromadb

# Download the manuals
python download_data.py

# Verify you can load a PDF
python -c "
from pypdf import PdfReader

reader = PdfReader('data/carrier-30xa-iom.pdf')
print(f'Pages: {len(reader.pages)}')
print(f'First page preview:')
print(reader.pages[0].extract_text()[:500])
"
```
