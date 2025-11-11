# Project Overview: Engineer Interview Repository

## Summary

This repository is a technical interview challenge designed to assess candidates' skills in:
- GitHub Actions and CI/CD workflows
- Knowledge graph construction from unstructured data
- Retrieval Augmented Generation (RAG) systems
- LangChain framework usage
- Python programming and software engineering best practices

## Repository Structure

```
Engineer-Interview/
├── .github/
│   └── workflows/
│       └── organize_pdfs.yml      # GitHub Actions workflow for knowledge graph
├── pdfs/                           # Sample documents (5 technical articles)
│   ├── machine_learning_basics.txt
│   ├── natural_language_processing.txt
│   ├── deep_learning.txt
│   ├── data_science_fundamentals.txt
│   └── python_programming.txt
├── chatbot/                        # RAG chatbot implementation package
│   ├── __init__.py
│   ├── rag_chatbot.py             # Main RAG chatbot with LangChain
│   ├── knowledge_graph.py          # Knowledge graph utilities
│   └── requirements.txt            # Chatbot-specific dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # Main project documentation
├── INTERVIEW_GUIDE.md              # Detailed guide for candidates
├── SOLUTION_TEMPLATE.md            # Template for candidate solutions
├── requirements.txt                # Project-level dependencies
├── test_setup.py                   # Environment verification script
├── run_local_workflow.py           # Local workflow testing script
└── example_usage.py                # Example chatbot usage
```

## Components

### 1. Sample Documents (pdfs/)

Five technical documents covering:
- **Machine Learning Basics**: Introduction to ML concepts, types, and applications
- **Natural Language Processing**: NLP tasks, techniques, and challenges
- **Deep Learning**: Neural networks, architectures, and training
- **Data Science Fundamentals**: The data science process and best practices
- **Python Programming**: Python for data science with libraries and patterns

These are provided as text files (simulating PDFs) with structured content including headers, sections, and technical terminology for effective knowledge graph construction.

### 2. GitHub Actions Workflow (.github/workflows/organize_pdfs.yml)

**Purpose**: Automatically processes documents and builds a knowledge graph

**Features**:
- Triggers on push to main (when PDFs change) or manual dispatch
- Sets up Python 3.11 environment
- Installs dependencies (networkx, matplotlib, spaCy, etc.)
- Extracts text from documents
- Identifies entities and topics using NLP
- Constructs a NetworkX graph with:
  - Document nodes
  - Topic nodes
  - Entity nodes
  - Relationships between nodes
- Creates visualization (PNG)
- Exports JSON representation
- Uploads artifacts
- Commits results back to repository

**Output Files**:
- `knowledge_graph.json` - Structured graph data
- `knowledge_graph_visualization.png` - Visual representation
- GitHub Actions summary with statistics

### 3. RAG Chatbot System (chatbot/)

**Architecture**:
```
Documents → TextLoader → CharacterTextSplitter → Embeddings → ChromaDB
                                                                   ↓
User Query → Embeddings → Similarity Search → Retrieved Chunks → LLM → Answer
```

**Key Components**:

#### rag_chatbot.py
- **RAGChatbot class**: Main chatbot implementation
- **Document Loading**: Loads text files from pdfs/
- **Text Splitting**: Chunks documents with configurable overlap
- **Vector Store**: Uses ChromaDB with OpenAI embeddings
- **LLM Integration**: ChatOpenAI (gpt-3.5-turbo by default)
- **Conversational Memory**: Maintains chat history
- **Source Citations**: Returns source documents with answers
- **Interactive CLI**: Chat loop with commands:
  - `docs` - List available documents
  - `graph` - Show knowledge graph summary
  - `exit/quit` - Exit chatbot

#### knowledge_graph.py
- **KnowledgeGraphLoader class**: Utilities for working with the knowledge graph
- **Graph Loading**: Loads JSON graph into NetworkX
- **Document Queries**: Get documents, topics, relationships
- **Search**: Find documents by topic
- **Statistics**: Graph metrics and summaries

### 4. Helper Scripts

#### test_setup.py
Verifies the environment is correctly configured:
- ✅ Python version (3.8+)
- ✅ File structure
- ✅ Required dependencies
- ✅ Environment variables (.env file)
- ✅ Sample documents
- ✅ Module imports

#### run_local_workflow.py
Tests the knowledge graph workflow locally:
- Processes documents
- Builds knowledge graph
- Creates visualization
- Saves JSON output
- No need to push to GitHub

#### example_usage.py
Demonstrates programmatic chatbot usage:
- Initializes RAGChatbot
- Runs example queries
- Shows how to integrate into applications

### 5. Documentation

#### README.md
- Project overview
- Repository structure
- Task descriptions (Tasks 1 & 2)
- Setup instructions
- Running instructions
- Evaluation criteria
- Submission guidelines

#### INTERVIEW_GUIDE.md (9,716 characters)
Comprehensive guide including:
- Welcome and overview
- Time expectations (5-7 hours)
- Detailed requirements for each task
- What's already provided vs. what to build
- Evaluation criteria
- Tips for success
- Common pitfalls
- Resources and documentation links
- Support information

#### SOLUTION_TEMPLATE.md (5,954 characters)
Structured template for candidates to document:
- Approach and design decisions
- Implementation details
- Challenges and solutions
- Testing strategy
- Code quality practices
- Performance considerations
- Time breakdown
- Learnings

#### .env.example
Template for environment configuration:
- API keys (OpenAI, Anthropic)
- Model configuration
- Vector store settings
- Document processing parameters
- Knowledge graph output paths

## Technical Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **LangChain**: RAG implementation framework
- **OpenAI API**: Embeddings and LLM
- **ChromaDB**: Vector database
- **NetworkX**: Graph manipulation
- **Matplotlib**: Visualization

### Key Dependencies
```
langchain==0.1.0
langchain-community==0.0.10
langchain-openai==0.0.2
chromadb==0.4.22
openai==1.6.1
networkx==3.2.1
matplotlib==3.8.2
pypdf==3.17.4
python-dotenv==1.0.0
```

## Features

### Task 1: Knowledge Graph Workflow
✅ Automated processing via GitHub Actions  
✅ NLP-based entity and topic extraction  
✅ Graph structure with typed nodes (documents, topics, entities)  
✅ Relationship detection (discusses, mentions, related)  
✅ Visual and JSON output  
✅ GitHub Actions summary with statistics  
✅ Artifact upload  
✅ Auto-commit results  

### Task 2: RAG Chatbot
✅ Document loading and chunking  
✅ Semantic search with embeddings  
✅ Context-aware answer generation  
✅ Source citations  
✅ Conversational memory  
✅ Interactive CLI  
✅ Knowledge graph integration  
✅ Error handling  
✅ Configuration via environment variables  

## Interview Flow

1. **Candidate receives repository**
2. **Reads INTERVIEW_GUIDE.md** for detailed instructions
3. **Runs test_setup.py** to verify environment
4. **Completes Task 1**: GitHub Actions workflow
   - Option A: Improve reference implementation
   - Option B: Build from scratch
5. **Completes Task 2**: RAG chatbot
   - Option A: Improve reference implementation
   - Option B: Build from scratch
6. **Tests thoroughly**:
   - Run workflow (locally or via GitHub Actions)
   - Test chatbot with various questions
   - Verify edge cases
7. **Documents solution** using SOLUTION_TEMPLATE.md
8. **Submits via Pull Request**

## Evaluation Criteria

### Technical Skills (40%)
- Code quality and organization
- Understanding of RAG concepts
- LangChain proficiency
- GitHub Actions knowledge
- Python best practices

### Problem Solving (30%)
- Approach to challenges
- Edge case handling
- Debugging and testing
- Performance optimization

### Communication (20%)
- Code documentation
- Clear explanations
- Design decision rationale
- Professional presentation

### Creativity (10%)
- Innovative solutions
- Additional features
- Improvements over reference
- Unique approaches

## Design Decisions

### Why Text Files Instead of PDFs?
- Easier for testing and development
- Avoids PDF parsing complexity in the interview
- Focus on RAG and knowledge graph logic
- Candidates can still demonstrate PDF handling if desired

### Why Reference Implementation?
- Gives candidates multiple paths:
  1. Understand and improve (shows code reading skills)
  2. Build from scratch (shows design skills)
- Tests different competencies
- More realistic (often improving existing code)
- Reduces time pressure

### Why LangChain?
- Industry-standard RAG framework
- Good abstraction over LLM providers
- Extensive documentation
- Real-world relevant
- Tests framework learning ability

### Why GitHub Actions?
- Common in modern development
- Tests CI/CD understanding
- Practical automation skill
- Integration with repository

## Success Metrics

A successful submission will:
1. ✅ Complete both tasks
2. ✅ Have working code (passes basic tests)
3. ✅ Show understanding of RAG concepts
4. ✅ Demonstrate good code practices
5. ✅ Include documentation
6. ✅ Handle basic edge cases
7. ✅ Take reasonable time (5-7 hours)

## Future Enhancements

Potential additions for advanced candidates:
- Support for actual PDF files
- Web interface (Streamlit, Gradio)
- Multiple LLM provider support
- Advanced chunking strategies
- Hybrid search (keyword + semantic)
- Query routing
- Response streaming
- Conversation export
- User feedback loop
- Cost tracking
- Performance metrics
- Deployment configuration

## License

MIT License - See LICENSE file for details

## Support

Candidates can:
- Create issues for questions
- Reference documentation links in INTERVIEW_GUIDE.md
- Contact interviewers directly

---

**Last Updated**: 2025-11-11  
**Version**: 1.0.0  
**Maintainer**: Standard Seed Corporation
