#!/usr/bin/env python3
"""
Test Setup Script

This script helps candidates verify that their environment is set up correctly
before starting the interview challenges.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False


def check_file_structure():
    """Check if required files and directories exist"""
    print("\nChecking file structure...")
    
    required_items = {
        'directories': ['pdfs', 'chatbot', '.github/workflows'],
        'files': [
            'README.md',
            'requirements.txt',
            '.env.example',
            'INTERVIEW_GUIDE.md',
            'chatbot/__init__.py',
            'chatbot/rag_chatbot.py',
            'chatbot/knowledge_graph.py',
            '.github/workflows/organize_pdfs.yml'
        ]
    }
    
    all_good = True
    
    for directory in required_items['directories']:
        if Path(directory).exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (missing)")
            all_good = False
    
    for file in required_items['files']:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
            all_good = False
    
    return all_good


def check_env_file():
    """Check if .env file exists and has necessary keys"""
    print("\nChecking environment configuration...")
    
    if not Path('.env').exists():
        print("  ⚠️  .env file not found")
        print("     Run: cp .env.example .env")
        print("     Then edit .env and add your API keys")
        return False
    
    print("  ✅ .env file exists")
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if os.getenv('OPENAI_API_KEY'):
        # Don't print the actual key
        key = os.getenv('OPENAI_API_KEY')
        if key and key != 'your_openai_api_key_here' and len(key) > 20:
            print("  ✅ OPENAI_API_KEY is set")
            return True
        else:
            print("  ⚠️  OPENAI_API_KEY appears to be placeholder")
            return False
    else:
        print("  ⚠️  OPENAI_API_KEY not set in .env")
        return False


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        'langchain',
        'langchain_community',
        'langchain_openai',
        'chromadb',
        'openai',
        'networkx',
        'matplotlib',
        'python-dotenv',
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (not installed)")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    return True


def check_pdf_files():
    """Check if sample PDF/text files exist"""
    print("\nChecking sample documents...")
    
    pdf_dir = Path('pdfs')
    if not pdf_dir.exists():
        print("  ❌ pdfs/ directory not found")
        return False
    
    files = list(pdf_dir.glob('*.txt')) + list(pdf_dir.glob('*.pdf'))
    
    if len(files) == 0:
        print("  ❌ No documents found in pdfs/")
        return False
    
    print(f"  ✅ Found {len(files)} documents:")
    for f in files:
        print(f"     - {f.name}")
    
    return True


def test_imports():
    """Test if critical imports work"""
    print("\nTesting critical imports...")
    
    try:
        from chatbot import rag_chatbot
        print("  ✅ chatbot.rag_chatbot")
    except ImportError as e:
        print(f"  ❌ chatbot.rag_chatbot: {e}")
        return False
    
    try:
        from chatbot import knowledge_graph
        print("  ✅ chatbot.knowledge_graph")
    except ImportError as e:
        print(f"  ❌ chatbot.knowledge_graph: {e}")
        return False
    
    return True


def main():
    """Run all checks"""
    print("="*60)
    print("Engineer Interview - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("File Structure", check_file_structure),
        ("Sample Documents", check_pdf_files),
        ("Dependencies", check_dependencies),
        ("Module Imports", test_imports),
        ("Environment Config", check_env_file),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {check_name} check: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, passed_check in results.items():
        status = "✅" if passed_check else "❌"
        print(f"{status} {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to start the interview.")
        print("\nNext steps:")
        print("  1. Read INTERVIEW_GUIDE.md carefully")
        print("  2. Start with Task 1 (GitHub Actions workflow)")
        print("  3. Then complete Task 2 (RAG Chatbot)")
        print("  4. Document your solution in SOLUTION_TEMPLATE.md")
        print("\nGood luck! 🚀")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please resolve the issues above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - No .env file: cp .env.example .env (then edit)")
        print("  - Wrong Python version: Use Python 3.8+")
        return 1


if __name__ == "__main__":
    sys.exit(main())
