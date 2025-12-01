"""
RAG Chatbot Implementation using LangChain

This module implements a Retrieval Augmented Generation (RAG) chatbot that can
answer questions based on the PDF documents in the repository.

Features:
  - Improved prompts with context and instructions 
  - Source highlighting with document metadata
  - Knowledge graph integration for relationship awareness
  - Conversation memory and context management
  - Response confidence scoring
  - Query optimization and expansion
  - Caching and performance optimization
"""

import os
import sys
import re
import warnings
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import logging
from chatbot.knowledge_graph import KnowledgeGraphIntegration

# Suppress warnings
warnings.filterwarnings('ignore', message='.*urllib3 v2 only supports OpenSSL.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress ChromaDB telemetry logs
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )
    from langchain_community.document_loaders import (
        PyPDFLoader,
        DirectoryLoader,
        TextLoader
    )
    from langchain.schema import Document
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    import networkx as nx
    HAS_LANGCHAIN = True
except ImportError as e:
    logger.warning(f"LangChain dependencies not installed: {e}")
    logger.info("Install with: pip install langchain langchain-openai langchain-community chromadb python-dotenv")
    HAS_LANGCHAIN = False


@dataclass
class SourceInfo:
    """Information about a document source"""
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    relevance_score: float = 0.0
    text_snippet: str = ""


@dataclass
class ChatResponse:
    """Structured response from the chatbot"""
    answer: str
    sources: List[SourceInfo]
    confidence_score: float
    related_topics: List[str]
    follow_up_questions: List[str]
    metadata: Dict = None


class Chunker:
    """Text chunking strategies"""

    @staticmethod
    def sentence_aware_split(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.
        Ensures chunks don't break in the middle of sentences.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap
        if overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk
                    prev_chunk_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                    overlapped_chunks.append(prev_chunk_end + " " + chunk)
                else:
                    overlapped_chunks.append(chunk)
            chunks = overlapped_chunks

        return chunks

    @staticmethod
    def semantic_split(text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text based on semantic boundaries (paragraphs, sections).
        """
        # Split by major sections (headers)
        sections = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""

        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += "\n\n" + section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def hybrid_split(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Hybrid strategy combining semantic and sentence-aware splitting.
        """
        # First split by paragraphs
        sections = re.split(r'\n\n+', text)
        chunks = []

        for section in sections:
            # Then split by sentences within each section
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks

class RAGChatbot:
    """RAG Chatbot with advanced features"""

    def __init__(
        self,
        documents_dir: str = "../pdfs",
        vector_store_dir: str = ".chroma_db",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        chunking_strategy: str = "hybrid",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """Initialize the enhanced RAG chatbot"""
        load_dotenv()

        self.documents_dir = Path(documents_dir)
        self.vector_store_dir = vector_store_dir
        self.model_name = model_name
        self.temperature = temperature
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.documents = []
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.kg_integration = KnowledgeGraphIntegration()
        self.chunker = Chunker()

        logger.info(f"Initialized chatbot with {chunking_strategy} chunking strategy")

    def load_documents(self) -> List[Document]:
        """Load all documents from the documents directory"""
        logger.info(f"Loading documents from {self.documents_dir}")

        if not self.documents_dir.exists():
            logger.error(f"Documents directory not found: {self.documents_dir}")
            return []

        documents = []

        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )

        try:
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")

        # Load text files
        text_loader = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        try:
            text_docs = text_loader.load()
            documents.extend(text_docs)
            logger.info(f"Loaded {len(text_docs)} text documents")
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")

        self.documents = documents
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using the configured strategy
        """
        logger.info(f"Splitting documents using {self.chunking_strategy} strategy")

        split_docs = []

        for doc in documents:
            text = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', None)

            # Choose chunking strategy
            if self.chunking_strategy == "sentence_aware":
                chunks = self.chunker.sentence_aware_split(
                    text,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap
                )
            elif self.chunking_strategy == "semantic":
                chunks = self.chunker.semantic_split(text, chunk_size=self.chunk_size)
            else:  
                chunks = self.chunker.hybrid_split(
                    text,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap
                )

            # Create new documents from chunks
            for i, chunk in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        'source': source,
                        'page': page,
                        'chunk': i,
                        'total_chunks': len(chunks)
                    }
                ))

        logger.info(f"Created {len(split_docs)} document chunks")
        return split_docs

    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents"""
        logger.info("Creating vector store...")

        if not HAS_LANGCHAIN:
            logger.error("LangChain not installed")
            return

        embeddings = OpenAIEmbeddings()

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.vector_store_dir
        )

        logger.info("Vector store created successfully")

    def create_qa_chain(self):
        """Create the QA chain with improved prompts"""
        if not HAS_LANGCHAIN:
            logger.error("LangChain not installed")
            return

        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )

        # Improved system prompt
        system_prompt = """You are an expert assistant helping users understand technical documents.

IMPORTANT INSTRUCTIONS:
1. Always base your answers on the provided document excerpts
2. If information is not in the documents, clearly state that
3. Provide specific examples from the documents when possible
4. Maintain technical accuracy
5. Be concise but comprehensive

When providing an answer:
- Start with a direct answer to the question
- Support it with relevant excerpts from the documents
- Explain the significance and context
- Suggest related questions the user might find helpful

Format your response clearly with:
- **Answer**: Your main response
- **Key Points**: Bulleted key information
- **Sources**: Referenced document sections
- **Related Topics**: Other topics mentioned in the documents"""

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""System: {system_prompt}

Context from documents:
{context}

Question: {question}

Answer: """.replace("{system_prompt}", system_prompt)
        )

        # Create memory with output_key specified
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify which output to store in memory
        )
        self.memory = memory

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        logger.info("QA chain created successfully")

    def setup(self, force_rebuild: bool = False):
        """Set up the chatbot"""
        logger.info("Setting up chatbot...")

        # Check if vector store exists
        vector_store_exists = Path(self.vector_store_dir).exists()

        if force_rebuild or not vector_store_exists:
            logger.info("Building vector store from scratch...")
            documents = self.load_documents()
            if not documents:
                logger.error("No documents loaded")
                return

            split_docs = self.split_documents(documents)
            self.create_vector_store(split_docs)
        else:
            logger.info("Loading existing vector store...")
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=embeddings
            )

        self.create_qa_chain()
        logger.info("Chatbot setup complete!")

    def extract_sources(self, source_docs: List[Document]) -> List[SourceInfo]:
        """Extract and structure source information"""
        sources = []
        seen_sources = set()

        for i, doc in enumerate(source_docs):
            metadata = doc.metadata
            source_key = metadata.get('source', 'unknown')

            # Avoid duplicates
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            source_info = SourceInfo(
                document=Path(source_key).name,
                page=metadata.get('page'),
                section=metadata.get('section'),
                relevance_score=1.0 - (i * 0.15),  # Higher score for top results
                text_snippet=doc.page_content[:200] + "..."
            )
            sources.append(source_info)

        return sources

    def calculate_confidence(self, answer: str, sources: List[SourceInfo]) -> float:
        """Calculate confidence score for the answer"""
        
        # Base confidence on number of sources
        source_confidence = min(len(sources) / 3.0, 1.0)

        # Reduce confidence if answer contains uncertainty phrases
        uncertainty_phrases = [
            'not sure', 
            'unclear', 
            'unknown', 
            'uncertain', 
            'possibly', 
            'might be',
            'cannot find',
            'no information',
            'don\'t know',
            'not mentioned'
        ]
        
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
        
        # Calculate confidence (don't penalize too heavily)
        confidence = source_confidence * (1.0 - min(uncertainty_count * 0.15, 0.6))
        
        # Boost confidence if answer is detailed (longer answers with sources)
        if len(answer) > 200 and len(sources) > 0:
            confidence = min(confidence * 1.1, 1.0)
        
        return max(0.3, min(confidence, 1.0))  # Ensure minimum 30% confidence

    def generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []

        # Extract key terms from the original question
        keywords = question.lower().split()

        # Generate follow-up suggestions based on context
        follow_up_templates = [
            f"How does {keywords[0] if keywords else 'this'} relate to the documents?",
            f"Can you provide more details about {keywords[-1] if keywords else 'this'} topic?",
            "What are the practical applications of this concept?",
            "Are there any limitations or caveats mentioned in the documents?",
            "What similar topics are discussed in the documents?"
        ]

        follow_ups = follow_up_templates[:3]  # Return top 3
        return follow_ups

    def query(self, question: str) -> ChatResponse:
        """
        Query the chatbot and return a structured response
        """
        logger.info(f"Processing query: {question}")

        if not self.qa_chain or not self.vector_store:
            logger.error("Chatbot not properly initialized")
            return ChatResponse(
                answer="Error: Chatbot not initialized",
                sources=[],
                confidence_score=0.0,
                related_topics=[],
                follow_up_questions=[]
            )

        try:
            # Expand query using knowledge graph
            expanded_terms = self.kg_integration.expand_query(question)
            logger.info(f"Query expansion: {expanded_terms}")

            # Get answer from QA chain
            result = self.qa_chain({"question": question})

            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])

            # Extract structured source information
            sources = self.extract_sources(source_docs)

            # Calculate confidence
            confidence = self.calculate_confidence(answer, sources)

            # Generate related topics from knowledge graph
            related_topics = expanded_terms[1:4] if len(expanded_terms) > 1 else []

            # Generate follow-up questions
            follow_ups = self.generate_follow_up_questions(question, answer)

            response = ChatResponse(
                answer=answer,
                sources=sources,
                confidence_score=confidence,
                related_topics=related_topics,
                follow_up_questions=follow_ups,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model_name,
                    'chunking_strategy': self.chunking_strategy
                }
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return ChatResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                related_topics=[],
                follow_up_questions=[]
            )

    def format_response(self, response: ChatResponse) -> str:
        """Format the response for display"""
        output = []

        output.append("\n" + "="*70)
        output.append("ANSWER")
        output.append("="*70)
        output.append(response.answer)

        output.append("\n" + "-"*70)
        output.append(f"CONFIDENCE SCORE: {response.confidence_score:.1%}")
        output.append("-"*70)

        if response.sources:
            output.append("\n📚 SOURCES:")
            for i, source in enumerate(response.sources, 1):
                output.append(f"\n  [{i}] {source.document}")
                if source.page:
                    output.append(f"      Page: {source.page}")
                output.append(f"      Relevance: {source.relevance_score:.1%}")
                output.append(f"      Snippet: {source.text_snippet[:100]}...")

        if response.related_topics:
            output.append("\n🔗 RELATED TOPICS:")
            for topic in response.related_topics:
                output.append(f"  • {topic}")

        if response.follow_up_questions:
            output.append("\n❓ YOU MAY ALSO WANT TO KNOW:")
            for q in response.follow_up_questions:
                output.append(f"  • {q}")

        output.append("\n" + "="*70 + "\n")

        return "\n".join(output)

    def chat_loop(self):
        """Run interactive chat loop"""
        print("\n" + "="*70)
        print(" RAG CHATBOT - DOCUMENT KNOWLEDGE ASSISTANT")
        print("="*70)
        print("\nType your questions below. Press Ctrl+C or type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Thank you for using the chatbot.")
                    break

                # Process query
                response = self.query(user_input)

                # Format and display response
                formatted_response = self.format_response(response)
                print(formatted_response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Thank you for using the chatbot.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Error in chat loop: {e}")


def main():
    """Main entry point"""
    if not HAS_LANGCHAIN:
        print("Error: LangChain dependencies not installed")
        print("Install with: pip install langchain langchain-openai langchain-community chromadb python-dotenv")
        sys.exit(1)

    try:
        # Initialize chatbot
        chatbot = RAGChatbot(
            documents_dir="pdfs",
            chunking_strategy="hybrid",
            chunk_size=500,
            chunk_overlap=100
        )

        # Setup (will use cached vector store if available)
        chatbot.setup(force_rebuild=False)

        # Start interactive chat
        chatbot.chat_loop()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
