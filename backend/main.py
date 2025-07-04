from rag_pipeline import load_static_document, store_chunks, answer_question, chunk_text
from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag_pipeline import (
    load_static_document,
    store_chunks,
    answer_question,
    CHROMA_DIR  # make sure this is imported or redefined here
)

app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1: Store chunks only if vector store not already created
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("📄 No existing vectorstore found. Chunking and storing now...")
    chunks = load_static_document()
    store_chunks(chunks)
else:
    print("✅ Vectorstore already exists. Skipping chunking and storing.")

#root method
@app.get("/")
def root():
    return {"message": "Connected successfully to backend"}


# Schema for incoming question
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(payload: Question):
    answer = answer_question(payload.query)
    return {"answer": answer}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(status_code=400, detail="Only plain text files are supported.")
    
    content = await file.read()
    text = content.decode("utf-8")

    # Chunk & store
    chunks = chunk_text(text)  # You can reuse your existing chunk_text function
    store_chunks(chunks)

    return {"status": "success", "message": f"Uploaded and indexed {file.filename}"}