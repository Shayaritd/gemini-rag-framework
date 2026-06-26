"""
Enterprise RAG System with Gemini
Complete working script for the Enterprise-RAG-Framework
"""

import os
import sys
import time
from pathlib import Path

# ============================================================
# STEP 1: Set your Gemini API key
# ============================================================
# Option A: Set directly (uncomment and add your key)
# os.environ["GEMINI_API_KEY"] = "AIzaSyAH2K051HPQy8ZP6HjEgU7njqHBA_FNcKc"

# Option B: Load from .env file (recommended)
from dotenv import load_dotenv
load_dotenv()

# Check if API key is set
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found!")
    print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
    print("Or uncomment the direct key in the script.")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")

# ============================================================
# STEP 2: Add src to Python path
# ============================================================
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# STEP 3: Import the framework
# ============================================================
try:
    from src.enterprise_rag import RAGSystem
    print("✅ Imported RAGSystem successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ============================================================
# STEP 4: Create directories and sample document
# ============================================================
print("\n" + "=" * 60)
print("🚀 ENTERPRISE RAG SYSTEM WITH GEMINI")
print("=" * 60)

os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/index", exist_ok=True)

# Create sample document if it doesn't exist
sample_path = Path("data/documents/sample.txt")
if not sample_path.exists():
    print("\n📝 Creating sample document...")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("""
Enterprise RAG Framework - Production-Grade Document Q&A System

ABOUT:
Enterprise RAG is a production-grade system for building document question-answering applications using Retrieval Augmented Generation.

KEY FEATURES:
1. Hybrid Retrieval: Combines BM25 keyword search with dense vector embeddings
2. Multi-Format Support: PDF, DOCX, TXT, HTML, Markdown, CSV
3. Smart Chunking: Recursive, sliding window, and semantic chunking
4. Cross-Encoder Reranking: Improves retrieval quality
5. Metadata Filtering: Filter by category, author, date, etc.
6. Evaluation Suite: Precision, recall, faithfulness metrics
7. Production Ready: Monitoring, logging, and security features

SUPPORTED LLM PROVIDERS:
- Google Gemini (free tier - 250 requests/day)
- OpenAI (GPT-3.5, GPT-4)
- Local models via Ollama (Llama, Qwen, Mistral)
- Anthropic Claude

BEST PRACTICES:
- Use hybrid search for optimal results
- Implement metadata filtering for access control
- Monitor performance metrics regularly
- Use caching for frequently asked questions

USE CASES:
- Customer support document Q&A
- Internal knowledge base search
- Technical documentation assistant
- Legal document analysis
- Research paper summarization
""")
    print("✅ Sample document created!")

# ============================================================
# STEP 5: Initialize RAG System with Gemini
# ============================================================
print("\n🔧 Initializing RAG System with Gemini...")

try:
    rag = RAGSystem(
        vector_store_config={
            "type": "faiss",
            "index_path": "data/index",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        retrieval_config={
            "type": "hybrid",
            "top_k": 5,
            "use_reranker": False
        },
        generation_config={
            "model": "gemini-2.5-flash",
            "provider": "gemini",
            "config": {
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
    )
    print("✅ RAG System initialized successfully!")
    
except Exception as e:
    print(f"❌ Error initializing RAG: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# STEP 6: Show available methods
# ============================================================
print("\n📋 Available methods:")
methods = [m for m in dir(rag) if not m.startswith("_")]
for method in methods:
    print(f"   - {method}")

# ============================================================
# STEP 7: Index the document
# ============================================================
print("\n📄 Indexing document...")

try:
    # Read the sample document
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Try different methods to add documents
    if hasattr(rag, 'add_documents'):
        rag.add_documents([{
            'text': content,
            'metadata': {'source': 'sample.txt'}
        }])
        print("✅ Documents added via 'add_documents'")
    elif hasattr(rag, 'index_documents'):
        rag.index_documents([{
            'text': content,
            'metadata': {'source': 'sample.txt'}
        }])
        print("✅ Documents added via 'index_documents'")
    elif hasattr(rag, 'add'):
        rag.add(content)
        print("✅ Documents added via 'add'")
    else:
        print("⚠️ No document indexing method found")
    
    # Save index if available
    if hasattr(rag, 'save_index'):
        rag.save_index()
        print("✅ Index saved!")
    elif hasattr(rag, 'save'):
        rag.save()
        print("✅ Index saved!")
        
except Exception as e:
    print(f"⚠️ Note: {e}")

# ============================================================
# STEP 8: Interactive Q&A
# ============================================================
print("\n" + "=" * 60)
print("💬 INTERACTIVE Q&A (type 'quit' to exit)")
print("=" * 60 + "\n")

question_count = 0

while True:
    try:
        query = input("❓ Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(f"\n👋 Goodbye! Asked {question_count} questions.")
            break
        
        if not query:
            continue
        
        question_count += 1
        print("⏳ Generating answer with Gemini...")
        
        start_time = time.time()
        
        # Try different query methods
        if hasattr(rag, 'query'):
            response = rag.query(query)
            elapsed = time.time() - start_time
            
            # Handle different response formats
            if isinstance(response, dict):
                answer = response.get('answer', response.get('text', str(response)))
                sources = response.get('sources', [])
            else:
                answer = str(response)
                sources = []
            
            print(f"\n📝 Answer: {answer}")
            
            if sources:
                print(f"\n📚 Sources: {len(sources)}")
                for i, src in enumerate(sources[:3], 1):
                    if isinstance(src, dict):
                        title = src.get('title', src.get('filename', f'Document {i}'))
                        score = src.get('score', 0)
                        print(f"   {i}. {title} (confidence: {score:.2f})")
            
            print(f"\n⏱️ Time: {elapsed:.2f}s")
            print(f"📊 Questions used: {question_count}/250 (daily limit)")
            print("-" * 60 + "\n")
            
        else:
            print("⚠️ No 'query' method found!")
            break
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

print("\n✅ Session complete!")