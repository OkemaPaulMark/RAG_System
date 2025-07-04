import streamlit as st  # type: ignore
import requests

st.set_page_config(page_title="Intelligent RAG System")

BACKEND_URL = "http://localhost:8000/ask"  # Adjust this if deployed elsewhere

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        with st.expander("📖 Preview File"):
            content = uploaded_file.read().decode("utf-8")
            st.text(content[:1000])
            uploaded_file.seek(0)

        if st.button("📤 Upload & Index Document"):
            with st.spinner("Uploading and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "text/plain")}
                    response = requests.post("http://localhost:8000/upload", files=files)
                    if response.status_code == 200:
                        st.success("Document uploaded and indexed successfully!")
                        # Optionally clear chat or reset state here
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

# Main Title and Instructions
st.title("📚 Ask My Document AI")
st.markdown("Upload any large .txt document and start chatting with it. Ask anything. Get smart answers based on your document’s content.")

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# Chat input at the bottom
user_input = st.chat_input("Ask a question about the document...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(("You", user_input))

    # Call backend API and show response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = requests.post(BACKEND_URL, json={"query": user_input})
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer returned.")
                else:
                    answer = f"❌ Error: {response.status_code} - {response.text}"
            except Exception as e:
                answer = f"⚠️ Could not connect to backend: {e}"

        st.markdown(answer)
        st.session_state.chat_history.append(("Bot", answer))

# Clear chat history button
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
