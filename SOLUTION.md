# Solution Documentation

**Candidate Name**: Banula Perera  
**Date**: 21st Nov 2025 
**Time Spent**: 24 hrs

---

## Overview

This solution delivers:
- A local script to build a knowledge graph from PDFs in ./pdfs and generate a PNG visualization.
- A GitHub Actions workflow that runs the builder on pushes/PRs and uploads artifacts.
- A lightweight knowledge graph loader for the chatbot to consume knowledge_graph.json.

The pipeline extracts text and light-weight metadata (authors, institutions, year, keywords, abstract), derives entities and topics using regex + frequency heuristics, constructs a NetworkX graph, and saves it in node-link JSON (links) format alongside a visualization.

---

## Task 1: GitHub Actions Workflow

### Approach

I implemented a self-contained builder script (run_local_workflow.py) and a CI workflow (.github/workflows/run_knowledge_graph.yml).

- Built from scratch, aligned to the reference goals.
- Focused on deterministic, dependency-light heuristics (regex + frequency) to avoid heavy NLP models.
- Saved graph in node-link JSON (with links) so it’s easily ingested by NetworkX and downstream tools.

The workflow:
1. Checks out the repo, sets up Python 3.11.
2. Installs dependencies from requirements.txt (including graphviz for plotting).
3. Runs the builder to produce knowledge_graph.json and knowledge_graph_visualization.png.
4. Uploads artifacts and commits outputs back to the repo on push events.

### Implementation Details

#### Key Components

1. Document Processing:
   - Extracts text per page using pypdf (PdfReader). Pages with extractable text are concatenated.
   - Libraries: pypdf for robust PDF text extraction; NetworkX + matplotlib for graph and visualization.

2. Entity Extraction:
   - Regex patterns to capture simple domain-neutral tokens: chemicals/techniques (e.g., HPLC, GC-MS), organisms, and methods (study, analysis).
   - Section-like topics from header-looking lines; frequent-term topics from token frequency with stopword filtering.
   - Authors via capitalized-name regex (with hyphen/apostrophe/initial tolerance). Institutions via common institution keywords.

3. Knowledge Graph Construction:
   - Nodes:
     - document (label, path)
     - author, institution
     - topic
     - entity (category attribute: chemicals, organisms, methods)
   - Edges:
     - authored_by (document → author)
     - affiliated_with (document → institution)
     - discusses (document → topic)
     - mentions (document → entity)
   - Output: node-link JSON (nodes + links) plus an attached document_data section.

4. Visualization:
   - Spring layout with color by type and node size scaled by degree.
   - Labels on document and topic nodes; black label font for readability.
   - Saves as knowledge_graph_visualization.png.

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| PDF text quality varies (layout/encoding) | Fallback-safe extraction per page, ignore failures, proceed with partial text |
| Over/under extraction using regex | Use conservative patterns, dedupe while preserving order, constrain search windows for metadata |

### Testing

- [x] Tested locally with sample PDFs in ./pdfs
- [x] Verified knowledge_graph.json and PNG generated
- [x] Confirmed colors, labels, and sizes reflect node types/degree
- [x] Validated GitHub Actions workflow on push/PR and artifact upload

### Improvements Made

1. Deterministic dedupe (preserve original order) for authors and entities.
2. Node-link JSON format to maximize compatibility (links key).
3. Label coloring set to black to improve readability.

### Future Enhancements

1. Add spaCy/transformers for NER to improve entity precision.
2. Topic modeling (e.g., KeyBERT) for better topic extraction.
3. Add second visualization (doc–topic heatmap, degree histogram) and community detection for larger graphs.

---

## Task 2: RAG Chatbot System

### Approach

A minimal KnowledgeGraphIntegration class loads the knowledge graph JSON and exposes doc_topics and entity_docs mappings for retrieval-time enrichment. The chatbot can bias answers toward known document–topic relations and display related entities.

- Uses the builder’s node-link JSON output.
- Keeps integration modular so other retrievers/LLMs can be swapped without changing the KG loader.

### Implementation Details

#### Architecture

```
Documents → Loader → Splitter → Embeddings → Vector Store
                                                    ↓
User Query → Embeddings → Similarity Search → Context
                                                    ↓
                                          LLM with Context → Answer
```

#### Key Components

1. Document Loading & Processing:
   - Loader: (in LangChain) DirectoryLoader + PyPDFLoader (one Document per page).
   - Text splitter: character-based with overlap (e.g., chunk_size=500, overlap=100) to maintain context.
   - Rationale: page granularity supports precise citations; overlap preserves sentence continuity.

2. Embeddings:
   - Model: OpenAIEmbeddings via langchain-openai.
   - Rationale: strong performance and compatibility with vector stores.

3. Vector Store:
   - Type: Chroma (langchain-chroma).
   - Configuration: persisted directory, k=5 retrieval.
   - Rationale: lightweight local store; good dev ergonomics.

4. LLM Configuration:
   - Model: ChatOpenAI (e.g., gpt-3.5-turbo or compatible).
   - Temperature: 0.7 by default.
   - Rationale: balanced creativity vs. faithfulness to context.

5. Retrieval Strategy:
   - Search type: similarity.
   - k: 5 chunks by default.
   - Rationale: small k reduces token costs and noise.

6. Prompt Engineering:
   - Clear instructions to ground answers in retrieved context and to cite sources.
   - Structured output: Answer, Key Points, Sources, Related Topics.

### Knowledge Graph Integration

- Loads knowledge_graph.json (node-link) and builds:
  - doc_topics: document → [topics it discusses]
  - entity_docs: entity → [documents that mention it]
- Intended uses:
  - Re-rank retrieved chunks favoring documents aligned with user’s topic.
  - Include Related Topics and Entities from the KG in the answer footer.

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Path resolution of KG from chatbot/ subfolder | Resolve relative to script directory with fallback to project root |
| JSON schema differences (“links” vs “edges”) | Support node-link (“links”) first; fallback manual ingest if only “edges” exist |

### Testing

#### Test Questions & Results

| Question | Expected | Result | Quality (1-5) |
|----------|----------|--------|---------------|
| What topics do the PDFs discuss? | Topic list with doc references | Returns topics from KG and sample docs | 4 |
| Who authored document X? | Author list for X | Returns authors from metadata | 4 |
| What methods are mentioned? | Methods list grounded in text | Lists study/analysis/model, cites docs | 3 |

#### Edge Cases Tested

- [x] Questions about topics not in documents (responds with “not found in documents” guidance)
- [x] Follow-up questions (memory can be enabled or simplified)
- [x] Ambiguous questions (requests clarification)
- [ ] Very long questions
- [ ] Questions in different phrasings

### Response Quality Analysis

**Strengths**:
- Deterministic KG-backed related topics/entities.
- Simple pipeline with clear, debuggable steps.

**Weaknesses**:
- Regex/heuristics can miss nuanced entities.
- No advanced NER or coreference resolution yet.

### Improvements Made

1. Robust KG loader tolerating node-link or edges schemas.
2. Path resolution hardened to be independent of working directory.
3. Simple data structures (dicts) for quick integration with retriever.

### Future Enhancements

1. Incorporate KG signals into retrieval re-ranking.
2. Add NER (spaCy) and concept normalization (e.g., ScispaCy).
3. Expose an API endpoint for chatbot + KG results.

---

## Code Quality

### Best Practices Applied

- [x] Type hints in builder
- [x] Error handling around I/O and extraction
- [x] Meaningful names and modular functions
- [x] Minimal configuration via CLI flags
- [x] No hardcoded absolute paths

### Testing Strategy

- Run builder locally and in CI with sample PDFs.
- Validate JSON schema loads into NetworkX.
- Sanity-check visualization outputs and node counts.
- Integration test: chatbot loads KG and exposes doc_topics/entity_docs.

---

## Design Decisions & Trade-offs

### Decision 1: Heuristic extraction vs. heavy NLP

**Options Considered**:
1. spaCy/NER + model-based topic detection
2. Regex + frequency heuristics

**Choice**: Regex + frequency heuristics

**Rationale**: Faster, simpler, no heavy model dependencies; good for baseline.

**Trade-offs**: Lower precision/recall vs. full NER; limited domain generalization.

### Decision 2: Node-link JSON format

**Options Considered**:
1. Custom JSON (nodes + edges)
2. NetworkX node-link JSON (“links”)

**Choice**: Node-link

**Rationale**: First-class support in NetworkX; less glue code.

**Trade-offs**: Requires loader support when upstream exports only “edges”.

---

## Performance Considerations

### Bottlenecks Identified

1. Graph layout computation for large graphs: spring layout is O(n^2)ish; can slow with many nodes.
2. PDF extraction for large files: serial extraction can be slow.

### Optimization Opportunities

1. Use kamada_kawai or precomputed positions for repeated builds.
2. Parallelize PDF extraction (multiprocessing) for many files.
3. Cache parsed text and metadata per file (mtime-based).

---

## API Cost Management

- Estimated cost per query: low (depends on LLM model and chunk count).
- Strategies:
  - Keep k small (e.g., 5).
  - Persist vector store to avoid re-embedding.
  - Trim context to only relevant chunks.

---

## Learnings

### New Things Learned

1. Practical trade-offs of PDF text extraction libraries.
2. Node-link formats simplify KG interchange.
3. CI ergonomics for artifact publishing and updating.

### Challenges That Made Me Grow

1. Handling path resolution reliably across CLI, CI, and subpackages.
2. Designing heuristics that perform acceptably across varied PDFs.

### What I'd Do Differently Next Time

1. Start with modular NER hooks for easier upgrade to spaCy/transformers.
2. Add validation/metrics for extraction quality early.
3. Include an interactive KG explorer (pyvis) from the start.

---

## Additional Notes

- Labels in the PNG are set to black for readability.
- Graph JSON includes an additional document_data section for convenient metadata access.

---

## Questions for Discussion

1. Preferred NER stack (spaCy vs. transformers) for production?
2. Should we introduce OCR (e.g., Tesseract) for scanned PDFs?
3. How should KG signals influence retriever ranking for the chatbot?

---

## References

1. NetworkX node-link format: https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html
2. pypdf: https://pypdf.readthedocs.io/
3. Matplotlib: https://matplotlib.org/