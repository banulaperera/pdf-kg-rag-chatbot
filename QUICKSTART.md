# Quick Start Guide

Get started with the Engineer Interview Challenge in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key (or another LLM provider)

## Setup (5 minutes)

### 1. Clone and Navigate

```bash
git clone https://github.com/Standard-Seed-Corporation/Engineer-Interview.git
cd Engineer-Interview
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

### 5. Verify Setup

```bash
python test_setup.py
```

If all checks pass ✅, you're ready!

## Test the Reference Implementation (5 minutes)

### Test the Knowledge Graph Workflow Locally

```bash
python run_local_workflow.py
```

This will:
- Process the documents in `pdfs/`
- Build a knowledge graph
- Create `knowledge_graph.json`
- Create `knowledge_graph_visualization.png`

### Test the RAG Chatbot

```bash
python -m chatbot.rag_chatbot
```

Try asking:
- "What is machine learning?"
- "Explain neural networks"
- "What Python libraries are used for data science?"

Type `docs` to see available documents.  
Type `graph` to see knowledge graph info.  
Type `exit` to quit.

## Start the Interview

Now you're ready to begin! Read the full instructions:

```bash
cat INTERVIEW_GUIDE.md
```

Or jump straight to:

### Task 1: GitHub Actions Workflow
- Review `.github/workflows/organize_pdfs.yml`
- Decide to improve it or build from scratch
- Test your changes

### Task 2: RAG Chatbot
- Review `chatbot/rag_chatbot.py`
- Decide to improve it or build from scratch
- Test with various questions

## Common Commands

```bash
# Test your environment setup
python test_setup.py

# Build knowledge graph locally
python run_local_workflow.py

# Run the chatbot
python -m chatbot.rag_chatbot

# Run example queries programmatically
python example_usage.py

# Check code syntax
python -m py_compile chatbot/*.py

# Install additional dependencies
pip install <package-name>
```

## Quick Tips

💡 **Start Simple**: Get the basics working first  
💡 **Test Often**: Run tests after each change  
💡 **Read the Docs**: LangChain docs are excellent  
💡 **Ask Questions**: Create an issue if stuck  
💡 **Document**: Comment your code and explain decisions  

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### API Key Not Found
```bash
# Make sure .env exists and has your key
cat .env | grep OPENAI_API_KEY
```

### Permission Denied
```bash
chmod +x test_setup.py run_local_workflow.py
```

### Module Not Found (chatbot)
```bash
# Make sure you're in the repository root
pwd  # Should show .../Engineer-Interview
```

## Time Allocation

- ⏱️ **Setup**: 10 minutes
- ⏱️ **Understanding code**: 30-60 minutes  
- ⏱️ **Task 1**: 2-3 hours
- ⏱️ **Task 2**: 3-4 hours
- ⏱️ **Testing**: 30-60 minutes
- ⏱️ **Documentation**: 30-60 minutes

**Total**: 5-7 hours

## Need Help?

- 📖 Read [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) for detailed instructions
- 📊 Check [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture details
- 📝 Use [SOLUTION_TEMPLATE.md](SOLUTION_TEMPLATE.md) to document your solution
- ❓ Create an issue if you have questions

## Next Steps

1. ✅ Complete setup (you're here!)
2. 📖 Read INTERVIEW_GUIDE.md thoroughly
3. 🧪 Test reference implementations
4. 💻 Start coding!
5. 📝 Document your solution
6. 🚀 Submit your PR

Good luck! 🎉
