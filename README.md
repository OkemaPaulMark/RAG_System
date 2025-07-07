# 📚 Intelligent RAG System

A Streamlit-powered application that uses **Retrieval-Augmented Generation (RAG)** to let you **upload, index, and chat with your documents** (TXT, PDF, DOCX).  
Powered by **LangChain**, **ChromaDB**, and **Groq LLMs**, this app helps you get accurate, contextual answers based on the content of your documents.

---

## 🚀 Features

✅ Upload `.txt`, `.pdf`, or `.docx` files  
✅ Automatically chunk and embed your document content  
✅ Store and retrieve chunks using **ChromaDB** (Dockerized)  
✅ Ask questions and get contextual answers via **Groq’s LLaMA 3**  
✅ Chat memory with **ConversationBufferMemory**  
✅ Clean, interactive UI built in **Streamlit**


---

## 🛠️ Tech Stack

| Component        | Technology               |
|------------------|---------------------------|
| Frontend         | Streamlit                |
| Backend          | LangChain                |
| Embeddings       | Azure OpenAI Embeddings  |
| Vector DB        | ChromaDB (Docker)        |
| LLM              | Groq LLaMA 3 (8b, 8192)   |
| File Handling    | python-docx, PyMuPDF     |
| Memory           | ConversationBufferMemory |
| Containerization | Docker, Docker Compose   |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OkemaPaulMark/RAG_System.git
cd RAG_System
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your `.env` File

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-azure-endpoint.openai.azure.com/
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the App Using Docker Compose

```bash
docker-compose up --build
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📁 File Structure

```bash
.
├── main.py                  # Streamlit frontend
├── rag_pipeline.py          # RAG logic and functions
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env                     # API keys
└── doc/
    └── the_art_of_war.txt   # Default document
```

---

## Sample Use Cases

- Upload a research paper and ask specific questions about sections.
- Upload your class pdf and aslk it questions
- Feed it training manuals or technical docs for quick answers.

---

## Clean Shutdown

To stop and clean:

```bash
docker-compose down --remove-orphans
```

---

## Author

**Okema Paul Mark**  
    Email: paulokema342@gmail.com  


---
