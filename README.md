# PDF Knowledge Graph & RAG Chatbot

This repository implements two tasks:
1. Task 1: Build a knowledge graph from PDF documents via a GitHub Actions workflow.
2. Task 2: Integrate the generated knowledge graph into a Retrieval Augmented Generation (RAG) chatbot.

## Overview

Implemented components:
- `run_local_workflow.py`: Extracts text, metadata, entities, topics, builds a NetworkX graph, saves `knowledge_graph.json` (node-link format) and `knowledge_graph_visualization.png`.
- `.github/workflows/run_knowledge_graph.yml`: CI workflow that triggers on PDF or script changes, runs the builder, uploads artifacts, and optionally commits updates.
- `chatbot/knowledge_graph.py`: Loads the JSON graph (supports `links` format) and prepares mappings for integration.
- `chatbot/rag_chatbot.py`: RAG chatbot referencing vector store + (optional) knowledge graph lookups.

## Repository Structure

```
Engineer-Interview/
├── pdfs/                            # Provide your PDF files here
├── run_local_workflow.py            # Knowledge graph builder script
├── knowledge_graph.json             # Generated graph (after running builder)
├── knowledge_graph_visualization.png# Generated PNG visualization
├── chatbot/
│   ├── __init__.py
│   ├── rag_chatbot.py               # Chatbot implementation (RAG)
│   ├── knowledge_graph.py           # Knowledge graph loader
├── .github/
│   └── workflows/
│       └── run_knowledge_graph.yml  # GitHub Actions workflow (Task 1)
├── requirements.txt                 # Dependencies
├── solution.md                      # Filled solution documentation
├── SOLUTION_TEMPLATE.md             # Original template
├── .env.example
└── README.md
```

## Task 1: Knowledge Graph Workflow

### What It Does
- Scans `./pdfs` for `.pdf` files.
- Extracts text (pypdf), heuristics-based metadata (authors, institutions, year).
- Derives topics and entities via regex and frequency filtering.
- Builds a graph with node types: document, author, institution, topic, entity.
- Saves:
  - `knowledge_graph.json` (node-link, includes `document_data`)
  - `knowledge_graph_visualization.png` (spring layout, colored by type, black labels)
  - `knowledge_graph_degree_hist.png` 
  - `knowledge_graph_doc_topic_matrix.png` 
  - `knowledge_graph_enhanced.png` 

### Run Locally

```bash
python run_local_workflow.py
```

### GitHub Actions

Workflow: `.github/workflows/run_knowledge_graph.yml`
Triggers:
- Push/PR affecting `pdfs/**` or `run_local_workflow.py`
- Manual `workflow_dispatch`
Outputs: artifacts + optional commit of JSON/PNG.

## Task 2: RAG Chatbot

### Pipeline
- Load PDFs (LangChain loader produces one page per document chunk).
- Split text (character splitter with overlap).
- Generate embeddings (OpenAI).
- Store in Chroma vector store (`langchain-chroma`).
- Retrieve top-k chunks per query.
- Generate answer with ChatOpenAI (sources + structured formatting).
- Optional: Use knowledge graph mappings (`doc_topics`, `entity_docs`) to enrich responses.

### Run Chatbot

```bash
export OPENAI_API_KEY=sk-...
python -m chatbot.rag_chatbot
```

## Requirements

Install:

```bash
pip install -r requirements.txt
```

Ensure system Graphviz installed if using pygraphviz (optional visualization enhancements):
```bash
brew install graphviz   # macOS
```

## Files Generated

- `knowledge_graph.json`
- `knowledge_graph_visualization.png`
- `knowledge_graph_degree_hist.png` 
- `knowledge_graph_doc_topic_matrix.png` 
- `knowledge_graph_enhanced.png` 

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
# add OPENAI_API_KEY to .env
pip install -r requirements.txt
```

## Notes

- PDF page-level extraction allows granular chunk retrieval.
- Authors/institutions use regex heuristics (may miss edge cases).
- Graph JSON includes `links` (standard NetworkX node-link) + `document_data`.
- Labels rendered with black font for readability.

## Future Enhancements (Suggested)

- Add spaCy for NER to improve entity quality.
- Integrate KG signals into retrieval score re-ranking.
- Provide interactive graph exploration (e.g. pyvis).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Knowledge graph path warning | Ensure `knowledge_graph.json` exists after running builder |
| Missing PyPDF | `pip install pypdf` |
| Chroma telemetry logs | Set `ANONYMIZED_TELEMETRY=False` |
| OpenSSL warning (macOS) | Safe to ignore or suppress via `warnings.filterwarnings` |

## License

MIT License.

---
Good luck refining and extending the system.