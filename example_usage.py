"""
Example usage of the RAG Chatbot

This script demonstrates how to use the chatbot programmatically
with user input and proper exception handling.
"""

import os
import sys
from chatbot.rag_chatbot import RAGChatbot
from dotenv import load_dotenv


def get_user_question():
    """Get a single question from user input"""
    print("\n" + "="*70)
    print("ENTER YOUR QUESTION")
    print("="*70)
    print("Type your question below (or 'quit' to exit)")
    print("-"*70)
    
    try:
        question = input("\nQuestion: ").strip()
    
        if question.lower() == 'quit':
            print("Exiting...")
            sys.exit(0)
        
        if not question:
            print(" No question entered.")
            return None
        
        print(f" Question received")
        return question
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error getting input: {e}")
        return None


def main():
    """Example usage of the RAG chatbot with user input"""
    
    try:
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
        print(" Chatbot setup complete!")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Please ensure the 'pdfs' directory exists and contains PDF files.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n Error initializing chatbot: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)
    
    # Ask user if they want to use example questions or enter their own
    print("\n" + "="*70)
    print("Choose an option:")
    print("  1. Use example questions")
    print("  2. Enter your own questions")
    print("  3. Interactive chat mode")
    print("="*70)
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            # Example questions
            questions = [
                "What is machine learning?",
                "Explain the difference between supervised and unsupervised learning",
                "What are the main components of a neural network?",
                "What is Natural Language Processing used for?",
                "What Python libraries are commonly used for data science?",
            ]
            print(f"\n Using {len(questions)} example questions")
            
        elif choice == '2':
            # Get user questions
            question = get_user_question()
            
            if not question:
                print("\n No questions entered. Exiting...")
                sys.exit(0)

            questions = [question]
            print(f"\n Processing your question")
                
        elif choice == '3':
            # Interactive chat mode
            print("\nStarting interactive chat mode...")
            print("Type 'quit' or 'exit' to end the conversation\n")
            chatbot.chat_loop()
            return
            
        else:
            print(f"\n Invalid choice: '{choice}'")
            print("Please enter 1, 2, or 3")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

    except Exception as e:
        print(f"\n Error getting input: {e}")
        sys.exit(1)
    
    # Process questions
    print("\n" + "="*70)
    print("PROCESSING QUERIES")
    print("="*70)
    
    successful_queries = 0
    failed_queries = 0
    
    for i, question in enumerate(questions, 1):
        try:
            print(f"\n{'='*70}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*70}")
            
            result = chatbot.query(question)
            print(chatbot.format_response(result))
            successful_queries += 1
            
        except Exception as e:
            print(f"\n Error processing question: {question}")
            print(f"   Error details: {str(e)}")
            failed_queries += 1
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total questions: {len(questions)}")
    print(f"Successful: {successful_queries}")
    print(f"Failed: {failed_queries}")
    print("\n" + "="*70)
    print("To start an interactive chat session, run:")
    print("  python -m chatbot.rag_chatbot")
    print("Or choose option 3 when running this script")
    print("="*70)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print(" Error: OPENAI_API_KEY not found!")
        print("\nPlease follow these steps:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to the .env file")
        print("  3. Run this script again")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("Please check the logs and try again.")
        sys.exit(1)