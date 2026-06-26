"""
Enterprise RAG System with Gemini + ChromaDB
Works perfectly on Windows
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Set API key directly if .env not working
os.environ["GEMINI_API_KEY"] = "AIzaSyAH2K051HPQy8ZP6HjEgU7njqHBA_FNcKc"

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("🚀 ENTERPRISE RAG WITH GEMINI + CHROMADB")
print("=" * 60)

# Create sample document
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/index", exist_ok=True)

sample_path = Path("data/documents/sample.txt")
if not sample_path.exists():
    print("\n📝 Creating sample document...")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("""
Enterprise RAG Framework - Production-Grade Document Q&A System

KEY FEATURES:
1. Hybrid Retrieval: BM25 keyword search + vector embeddings
2. Multi-Format Support: PDF, DOCX, TXT, HTML, Markdown
3. Smart Chunking: Recursive, sliding window, semantic
4. Cross-Encoder Reranking: Improves quality
5. Evaluation Suite: Precision, recall, faithfulness
6. Production Ready: Monitoring, logging, security

SUPPORTED PROVIDERS:
- Google Gemini (free - 250 requests/day)
- OpenAI GPT-3.5, GPT-4
- Local models via Ollama

BEST PRACTICES:
- Use hybrid search for optimal results
- Implement metadata filtering
- Monitor performance metrics
- Cache frequent queries

USE CASES:
- Customer support Q&A
- Internal knowledge base
- Technical documentation
- Legal document analysis
""")
    print("✅ Sample created!")

# Import RAGSystem
try:
    from src.enterprise_rag import RAGSystem
    print("✅ Imported RAGSystem")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Initialize RAG System with ChromaDB
print("\n🔧 Initializing RAG System with Gemini + ChromaDB...")

try:
    rag = RAGSystem(
        vector_store_config={
            "type": "chroma",
            "index_path": "data/index",
            "collection_name": "enterprise_rag",
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
    print("✅ RAG System initialized!")
    
except Exception as e:
    print(f"❌ Error initializing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Show available methods
print("\n📋 Available methods:")
for method in dir(rag):
    if not method.startswith("_"):
        print(f"   - {method}")

# Index document
print("\n📄 Indexing document...")

try:
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Try different add methods
    if hasattr(rag, 'add_documents'):
        rag.add_documents([{
            'text': content,
            'metadata': {'source': 'sample.txt'}
        }])
        print("✅ Documents added via 'add_documents'")
    elif hasattr(rag, 'add'):
        rag.add(content)
        print("✅ Documents added via 'add'")
    elif hasattr(rag, 'index_documents'):
        rag.index_documents([{
            'text': content,
            'metadata': {'source': 'sample.txt'}
        }])
        print("✅ Documents added via 'index_documents'")
    else:
        print("⚠️ No document add method found")
    
    # Save index
    if hasattr(rag, 'save_index'):
        rag.save_index()
        print("✅ Index saved!")
    elif hasattr(rag, 'save'):
        rag.save()
        print("✅ Index saved!")
        
except Exception as e:
    print(f"⚠️ Note: {e}")

# Interactive Q&A
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
        
        if hasattr(rag, 'query'):
            response = rag.query(query)
            elapsed = time.time() - start_time
            
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
                        print(f"   {i}. {title} (relevance: {score:.2f})")
            
            print(f"\n⏱️ Time: {elapsed:.2f}s")
            print(f"📊 Questions used: {question_count}/250 (daily limit)")
            print("-" * 60 + "\n")
            
        else:
            print("❌ No query method found!")
            print("   Available methods:", [m for m in dir(rag) if not m.startswith("_")])
            break
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

print("\n✅ Session complete!")