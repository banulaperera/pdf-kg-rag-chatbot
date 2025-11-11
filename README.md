# Engineer Interview - PDF Knowledge Graph & RAG Chatbot

Welcome to the Standard Seed Corporation Engineer Interview! This repository contains a practical coding challenge to assess your skills in data processing, knowledge graph creation, and building RAG (Retrieval Augmented Generation) systems with LangChain.

## Overview

This interview consists of two main tasks:

1. **Task 1**: Create a GitHub Actions workflow to organize PDFs into a knowledge graph
2. **Task 2**: Build a chatbot system with RAG on the provided PDFs using LangChain

## Repository Structure

```
Engineer-Interview/
├── pdfs/                           # Directory containing sample PDF documents
│   ├── machine_learning_basics.txt
│   ├── natural_language_processing.txt
│   ├── deep_learning.txt
│   ├── data_science_fundamentals.txt
│   └── python_programming.txt
├── .github/
│   └── workflows/
│       └── organize_pdfs.yml       # TODO: Create this workflow (Task 1)
├── chatbot/                        # TODO: Create this directory (Task 2)
│   ├── __init__.py
│   ├── rag_chatbot.py             # Main chatbot implementation
│   ├── knowledge_graph.py          # Knowledge graph utilities
│   └── requirements.txt            # Python dependencies
├── requirements.txt                # Project-level dependencies
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## Task 1: GitHub Actions Workflow for PDF Organization

### Objective
Create a GitHub Actions workflow that processes the PDFs in the `pdfs/` directory and organizes them into a knowledge graph.

### Requirements

1. The workflow should trigger on:
   - Push to main branch (when PDFs are added/modified)
   - Manual workflow dispatch
   
2. The workflow should:
   - Read all PDF/text files from the `pdfs/` directory
   - Extract key information (topics, entities, relationships)
   - Build a knowledge graph structure
   - Save the knowledge graph as a JSON file in the repository
   - Create visualizations of the knowledge graph
   
3. Expected outputs:
   - `knowledge_graph.json` - Structured representation of the knowledge graph
   - `knowledge_graph_visualization.png` - Visual representation of the graph
   - GitHub Actions summary with statistics

### Hints
- Use Python with libraries like `PyPDF2`, `pdfplumber`, or `pypdf` for PDF processing
- Use `networkx` for building the knowledge graph
- Use `matplotlib` or `graphviz` for visualization
- Consider using NLP libraries like `spaCy` or `nltk` for entity extraction

## Task 2: RAG Chatbot System

### Objective
Build a chatbot system that uses Retrieval Augmented Generation (RAG) to answer questions based on the content of the PDFs.

### Requirements

1. **Document Processing**:
   - Load and parse all documents from the `pdfs/` directory
   - Split documents into manageable chunks
   - Create embeddings for the text chunks
   - Store embeddings in a vector database

2. **RAG Implementation**:
   - Use LangChain for the RAG pipeline
   - Implement semantic search to retrieve relevant chunks
   - Use an LLM to generate answers based on retrieved context
   - Include source citations in responses

3. **Chatbot Interface**:
   - Create a simple command-line interface or web interface
   - Allow users to ask questions about the documents
   - Display relevant answers with source references
   - Handle follow-up questions

4. **Code Quality**:
   - Well-structured, modular code
   - Proper error handling
   - Documentation and comments
   - Type hints where appropriate

### Hints
- Use LangChain's document loaders and text splitters
- Consider using FAISS or ChromaDB for vector storage
- Use OpenAI API, Anthropic, or local models (Ollama) for the LLM
- Implement proper error handling for API calls
- Consider adding conversation memory for follow-up questions

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- API key for your chosen LLM provider (OpenAI, Anthropic, etc.) or Ollama installed locally

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Standard-Seed-Corporation/Engineer-Interview.git
cd Engineer-Interview
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Chatbot

```bash
python -m chatbot.rag_chatbot
```

### Testing Your Setup

Before starting the interview, verify your environment is ready:

```bash
python test_setup.py
```

### Testing Locally

To test the knowledge graph workflow locally without pushing to GitHub:

```bash
python run_local_workflow.py
```

This will generate `knowledge_graph.json` and `knowledge_graph_visualization.png` locally.

## Evaluation Criteria

### Task 1 (GitHub Actions Workflow)
- ✅ Workflow correctly processes PDFs
- ✅ Knowledge graph is well-structured
- ✅ Visualization is clear and informative
- ✅ Workflow is efficient and follows best practices
- ✅ Code is clean and maintainable

### Task 2 (RAG Chatbot)
- ✅ Correctly implements RAG pattern
- ✅ Accurate retrieval of relevant information
- ✅ High-quality generated responses
- ✅ Good user experience
- ✅ Code quality and architecture
- ✅ Error handling and edge cases
- ✅ Documentation

## Submission Guidelines

1. Fork this repository
2. Implement both tasks
3. Test your implementation thoroughly
4. Create a pull request with your solution
5. Include a brief writeup explaining your approach and any design decisions

## Time Estimate

- Task 1: 2-3 hours
- Task 2: 3-4 hours
- Total: 5-7 hours

## Questions?

If you have any questions about the requirements, please create an issue in this repository.

## License

MIT License - See LICENSE file for details

---

Good luck! We're excited to see your solution! 🚀