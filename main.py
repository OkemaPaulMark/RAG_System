import streamlit as st
from rag_pipeline import load_static_document, store_chunks, answer_question, chunk_text, CHROMA_DIR
from docx import Document
import fitz  # PyMuPDF
import os

st.set_page_config(page_title="Intelligent RAG System")

# Initialize vector store once
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    st.info("📄 No existing vectorstore found. Chunking and storing now...")
    chunks = load_static_document()
    store_chunks(chunks)
    st.success("✅ Vectorstore initialized.")


# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for File Upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        file_text = ""

        try:
            # Handle .txt
            if uploaded_file.name.endswith(".txt"):
                file_text = uploaded_file.read().decode("utf-8")

            # Handle .pdf
            elif uploaded_file.name.endswith(".pdf"):
                pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                file_text = "\n".join([page.get_text() for page in pdf])

            # Handle .docx
            elif uploaded_file.name.endswith(".docx"):
                doc = Document(uploaded_file)
                file_text = "\n".join([para.text for para in doc.paragraphs])

            else:
                st.error("Unsupported file type.")
                st.stop()

            st.success(f"Uploaded: {uploaded_file.name}")

            with st.expander("📖 Preview"):
                st.text(file_text[:1000])

            if st.button("📤 Upload & Index"):
                with st.spinner("Processing..."):
                    chunks = chunk_text(file_text)
                    store_chunks(chunks)
                    st.success("✅ File indexed into vectorstore.")

        except Exception as e:
            st.error(f"Failed to process file: {e}")


# Main Title
st.title("📚 Ask My Document AI")
st.markdown("Upload and query your document. This app uses RAG with LangChain and Groq.")

# Chat Display
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# Chat Input
user_input = st.chat_input("Ask a question about your document...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = answer_question(user_input)
            except Exception as e:
                response = f"Error: {e}"
            st.markdown(response)
            st.session_state.chat_history.append(("Bot", response))

# Clear chat button
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
