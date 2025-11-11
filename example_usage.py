"""
Example usage of the RAG Chatbot

This script demonstrates how to use the chatbot programmatically
without the interactive chat interface.
"""

import os
from chatbot.rag_chatbot import RAGChatbot


def main():
    """Example usage of the RAG chatbot"""
    
    # Initialize the chatbot
    print("Initializing RAG Chatbot...")
    chatbot = RAGChatbot(
        documents_dir="pdfs",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Setup the chatbot (load documents and create vector store)
    print("\nSetting up chatbot (this may take a moment)...")
    chatbot.setup()
    
    # Example questions
    example_questions = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the main components of a neural network?",
        "What is Natural Language Processing used for?",
        "What Python libraries are commonly used for data science?",
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    
    for question in example_questions:
        print(f"\n{'='*70}")
        result = chatbot.query(question)
        print(chatbot.format_response(result))
    
    print("\n" + "="*70)
    print("To start an interactive chat session, run:")
    print("  python -m chatbot.rag_chatbot")
    print("="*70)


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found!")
        print("Please copy .env.example to .env and add your OpenAI API key.")
        exit(1)
    
    main()
