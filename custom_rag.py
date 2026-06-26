"""
Custom RAG System with Gemini
Bypasses framework issues - works on Windows
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
os.environ["GEMINI_API_KEY"] = "AIzaSyAH2K051HPQy8ZP6HjEgU7njqHBA_FNcKc"

print("=" * 60)
print("🚀 CUSTOM RAG SYSTEM WITH GEMINI")
print("=" * 60)

try:
    import google.generativeai as genai
    print("✅ Gemini SDK loaded")
except ImportError:
    print("❌ Install: pip install google-generativeai")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers loaded")
except ImportError:
    print("❌ Install: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
    print("✅ FAISS loaded")
    faiss_available = True
except ImportError:
    print("⚠️ FAISS not available, using numpy fallback")
    faiss_available = False

try:
    from rank_bm25 import BM25Okapi
    print("✅ BM25 loaded")
except ImportError:
    print("❌ Install: pip install rank-bm25")
    sys.exit(1)

class SimpleRAG:
    def __init__(self, index_path="data/index", model_name="all-MiniLM-L6-v2"):
        print(f"\n🔧 Initializing SimpleRAG...")
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        print("   Loading embedding model...")
        self.embedder = SentenceTransformer(f"sentence-transformers/{model_name}")
        self.dim = self.embedder.get_sentence_embedding_dimension()
        print(f"   ✅ Embedding dimension: {self.dim}")
        
        self.documents = []
        self.metadata = []
        self.texts = []
        
        if faiss_available:
            self.index = faiss.IndexFlatL2(self.dim)
            print("   ✅ Using FAISS index")
        else:
            self.index = None
            print("   ⚠️ Using numpy fallback")
        
        print("   Configuring Gemini...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("   ✅ Gemini configured!")
        
        self.load_index()
        print("✅ SimpleRAG ready!\n")
    
    def add_document(self, text, metadata=None):
        self.documents.append(text)
        self.metadata.append(metadata or {"source": "unknown"})
        self.texts.append(text)
    
    def index_documents(self):
        if not self.documents:
            print("   ⚠️ No documents to index")
            return
        
        print(f"   Indexing {len(self.documents)} documents...")
        embeddings = self.embedder.encode(self.documents, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32')
        
        if faiss_available:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(embeddings_np)
        else:
            self.embeddings = embeddings_np
        
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        self.save_index()
        print(f"   ✅ Indexed {len(self.documents)} documents")
    
    def save_index(self):
        print("   💾 Saving index...")
        with open(self.index_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        if faiss_available and self.index is not None:
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        print("   ✅ Index saved!")
    
    def load_index(self):
        docs_file = self.index_path / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.metadata = data["metadata"]
                    self.texts = self.documents.copy()
                
                if faiss_available:
                    index_file = self.index_path / "faiss.index"
                    if index_file.exists():
                        self.index = faiss.read_index(str(index_file))
                
                tokenized = [doc.split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized)
                print(f"   ✅ Loaded {len(self.documents)} documents")
                return True
            except Exception as e:
                print(f"   ⚠️ Failed to load index: {e}")
                return False
        return False
    
    def search(self, query, top_k=5):
        if not self.documents:
            return []
        
        query_emb = self.embedder.encode([query])[0].astype('float32')
        query_emb = np.array([query_emb])
        
        vector_results = []
        if faiss_available and self.index is not None:
            distances, indices = self.index.search(query_emb, min(top_k * 2, len(self.documents)))
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    vector_results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(1 / (1 + distances[0][i]))
                    })
        else:
            if hasattr(self, 'embeddings'):
                similarities = np.dot(self.embeddings, query_emb.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]
                for idx in top_indices:
                    if idx < len(self.documents):
                        vector_results.append({
                            'text': self.documents[idx],
                            'metadata': self.metadata[idx],
                            'score': float(similarities[idx])
                        })
        
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        bm25_results = []
        for idx in bm25_indices:
            if idx < len(self.documents):
                bm25_results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(bm25_scores[idx] / 10)
                })
        
        all_results = {}
        for r in vector_results:
            all_results[r['text']] = r
        for r in bm25_results:
            if r['text'] in all_results:
                all_results[r['text']]['score'] = (all_results[r['text']]['score'] + r['score']) / 2
            else:
                all_results[r['text']] = r
        
        results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        return results
    
    def query(self, question):
        print(f"\n❓ Question: {question}")
        print("   Searching documents...")
        
        results = self.search(question, top_k=3)
        
        if not results:
            return {
                'answer': "No relevant information found.",
                'sources': []
            }
        
        context = "\n\n".join([f"Source {i+1}:\n{doc['text']}" for i, doc in enumerate(results)])
        print("   Generating answer with Gemini...")
        
        prompt = f"""Based on the following documents, answer the question accurately.
If the answer is not in the documents, say you don't know.

Documents:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'answer': response.text.strip(),
                'sources': results
            }
        except Exception as e:
            return {
                'answer': f"Error: {e}",
                'sources': results
            }

# Main execution
os.makedirs("data/documents", exist_ok=True)
sample_path = Path("data/documents/sample.txt")

if not sample_path.exists():
    print("\n📝 Creating sample document...")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("""
Enterprise RAG Framework - Document Q&A System

FEATURES:
- Hybrid Retrieval: BM25 + Vector Search
- Multi-Format Support: PDF, DOCX, TXT, HTML
- Gemini Integration (free tier - 250 requests/day)
- Smart Chunking
- Evaluation Suite
- Production Ready

BEST PRACTICES:
- Use hybrid search
- Implement caching
- Monitor performance
""")
    print("✅ Sample created!")

# Initialize RAG
rag = SimpleRAG(index_path="data/index")

# Load documents
print("\n📚 Loading documents...")
with open(sample_path, "r", encoding="utf-8") as f:
    content = f.read()
rag.add_document(content, {"source": "sample.txt"})
rag.index_documents()

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
        start = time.time()
        response = rag.query(query)
        elapsed = time.time() - start
        
        print(f"\n📝 Answer: {response['answer']}")
        
        if response.get('sources'):
            print(f"\n📚 Sources: {len(response['sources'])}")
            for i, src in enumerate(response['sources'][:2], 1):
                title = src.get('metadata', {}).get('source', 'Document')
                score = src.get('score', 0)
                print(f"   {i}. {title} (relevance: {score:.2f})")
        
        print(f"\n⏱️ Time: {elapsed:.2f}s")
        print(f"📊 Questions used: {question_count}/250 (daily limit)")
        print("-" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

print("\n✅ Session complete!")