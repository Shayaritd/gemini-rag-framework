# 🚀 Gemini RAG Framework

<div align="center">

<img src="https://img.shields.io/badge/AI-RAG%20Framework-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Powered%20By-Google%20Gemini-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge">

<h3>
*Enterprise-grade Document Q&A System powered by Google Gemini AI
</h3>

<p>
Build intelligent document question-answering systems using 
<strong>Google Gemini AI + Hybrid Search + Vector Retrieval</strong>
</p>

</div>

---

# 🌟 Overview

**Gemini RAG Framework** is a production-ready Retrieval-Augmented Generation system that allows users to ask questions from their private documents and receive accurate AI-generated answers with source references.

The framework combines:

* 🧠 Google Gemini Large Language Models
* 🔍 Hybrid Search (BM25 + Vector Search)
* 📚 Intelligent document processing
* 🎯 Context-aware retrieval
* 📊 Retrieval evaluation metrics

Unlike traditional chatbots that rely only on LLM knowledge, this system retrieves relevant information from your documents first and then generates grounded responses.

---

# 🎯 Problem Statement

Large Language Models are powerful but have limitations:

❌ Cannot access private company documents
❌ May hallucinate answers
❌ Cannot remember custom knowledge bases
❌ Lack source verification

This project solves these problems using **Retrieval-Augmented Generation (RAG).**

---

# ✨ Key Features

## 🔍 Advanced Hybrid Retrieval

Combines multiple search strategies:

### 1. Vector Search

Understands semantic meaning.

Example:

Query:

```
How can I reset my password?
```

Can find:

```
Steps to recover account credentials
```

even without exact keyword matching.

---

### 2. BM25 Keyword Search

Provides accurate keyword matching.

Useful for:

* Technical documents
* Code files
* Product manuals

---

### 3. Reciprocal Rank Fusion (RRF)

Combines:

```
Vector Search Results
          +
BM25 Results
          |
          ↓
   Final Ranked Context
```

---

# 🤖 Gemini AI Integration

Powered by:

## Google Gemini 2.5 Flash

Capabilities:

✅ Fast responses
✅ Large context window
✅ High-quality reasoning
✅ Cost-efficient inference
✅ Source-grounded generation

---

# 📄 Document Intelligence

Supports:

| Format   | Supported |
| -------- | --------- |
| PDF      | ✅         |
| DOCX     | ✅         |
| TXT      | ✅         |
| HTML     | ✅         |
| Markdown | ✅         |

Features:

* Automatic text extraction
* Metadata preservation
* Smart chunking
* Document indexing

---

# 🧩 System Architecture

```
                 USER QUESTION
                       |
                       ↓

              Query Processing

                       |
          -------------------------
          |                       |
          ↓                       ↓

   Vector Retriever        BM25 Retriever

          |                       |

          -----------+------------

                      ↓

              Hybrid Ranking
              (RRF Fusion)

                      ↓

              Relevant Documents

                      ↓

             Gemini 2.5 Flash

                      ↓

              Final Answer
          + Source References

```

---

# ⚙️ How It Works

## Step 1: Document Ingestion

```
Documents
   |
   ↓
Text Extraction
   |
   ↓
Chunk Creation
   |
   ↓
Embedding Generation
   |
   ↓
Vector Database
```

---

## Step 2: User Query

```
User Question

      ↓

Search Engine

      ↓

Relevant Context

      ↓

Gemini LLM

      ↓

Final Answer
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Shayaritd/gemini-rag-framework.git

cd gemini-rag-framework
```

---

## Create Virtual Environment

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Gemini API

Create:

```
.env
```

Add:

```
GEMINI_API_KEY=your_api_key_here
```

Get API key:

Google AI Studio:

https://ai.dev/

⚠️ Never commit API keys to GitHub.

---

# ▶️ Running The Application

Start RAG system:

```bash
python custom_rag.py
```

Example:

```
Question:
What are the main features of this framework?


Answer:

The framework provides:

1. Hybrid Search
2. Gemini AI integration
3. Multi-document support
4. Source attribution
5. Evaluation metrics


Sources:

document.pdf
page 5
```

---

# 📁 Project Structure

```
gemini-rag-framework/

│
├── data/
│   └── documents/
│
├── embeddings/
│
├── retrieval/
│   ├── vector_search.py
│   ├── bm25.py
│
├── generation/
│   └── gemini.py
│
├── evaluation/
│
├── custom_rag.py
│
├── requirements.txt
│
└── README.md

```

---

# 🛠️ Technology Stack

| Technology            | Purpose             |
| --------------------- | ------------------- |
| Python                | Backend Development |
| Google Gemini         | LLM Generation      |
| LangChain             | RAG Pipeline        |
| FAISS                 | Vector Search       |
| BM25                  | Keyword Retrieval   |
| Sentence Transformers | Embeddings          |
| dotenv                | Security Management |

---

# 📊 Performance

Example:

| Metric              | Result           |
| ------------------- | ---------------- |
| Retrieval Method    | Hybrid Search    |
| LLM                 | Gemini 2.5 Flash |
| Response Time       | ~2-3 seconds     |
| Document Types      | 5+               |
| Daily Free Requests | 250              |

---

# 🧪 Evaluation

The framework supports:

### Retrieval Evaluation

* Precision
* Recall
* Ranking quality

### Generation Evaluation

* Faithfulness
* Answer relevance
* Context accuracy

---

# 🌍 Real World Applications

## Enterprise Knowledge Assistant

Employees can query:

* Company policies
* Documentation
* Internal reports

## Education Assistant

Students can ask questions from:

* Notes
* Research papers
* Books

## Customer Support AI

Automates:

* FAQs
* Product support
* Troubleshooting

---

# 🗺️ Roadmap

## Completed ✅

* Gemini Integration
* Hybrid Search
* Document Processing
* Source Attribution

## Upcoming 🚀

* Web UI using React
* Multi-user authentication
* Cloud deployment
* Conversation memory
* Advanced analytics

---

# 🤝 Contributing

Contributions are welcome!

Steps:

```bash
git clone your-fork

git checkout -b feature-name

git commit -m "Added feature"

git push origin feature-name
```

Create a Pull Request.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you find this project useful:

⭐ Star the repository

🐛 Report issues

💡 Suggest improvements

<div align="center">

Built with ❤️ using <strong>Python + Gemini AI + RAG</strong>

</div>
