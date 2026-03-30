import streamlit as st
from pdf_processor import load_and_split_pdf
from vector_store import create_vector_store
from qa_chain import build_qa_chain, get_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;'>AI Document Assistant</h1>
<p style='text-align: center; color: gray;'>
Ask questions from your PDFs instantly
</p>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📁 Your Documents")
    st.caption("Upload files to start chatting")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.qa_chain is None:
        with st.spinner("Reading and indexing PDFs..."):
            all_chunks = []

            for file in uploaded_files:
                chunks = load_and_split_pdf(file)
                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No readable text found in PDF(s).")
                st.info("Try a text-based (non-scanned) PDF.")
                st.stop()

            vs = create_vector_store(all_chunks)
            st.session_state.qa_chain = build_qa_chain(vs)
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.pdf_count = len(uploaded_files)

        st.success("Documents ready!")

    # Show stats
    if st.session_state.chunk_count:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.chunk_count)
        col2.metric("PDFs", st.session_state.pdf_count)

    st.divider()

    # Download chat history
    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Bot"
            chat_text += f"{role}:\n{msg['content']}\n\n"

        st.download_button(
            label="⬇️Download chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

    # Clear everything
    if st.button("🗑️ Clear chat + reload"):
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.chunk_count = 0
        st.session_state.pdf_count = 0
        st.rerun()

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if st.session_state.qa_chain:
    if question := st.chat_input("Ask a question about your PDF..."):

        # User message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_answer(
                    st.session_state.qa_chain, question
                )

            st.write(result["answer"])

            # ---------------- SOURCES ----------------
            if result.get("sources"):
                with st.expander("Sources used"):
                    for i, doc in enumerate(result["sources"], 1):
                        page = doc.metadata.get("page", 0) + 1
                        source = doc.metadata.get("source", "document")
                        st.markdown(f"**Source {i} — Page {page}**")
                        st.caption(doc.page_content[:200] + "...")
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })

else:
    st.markdown("""
    <div style='text-align:center; margin-top:50px; color:gray;'>
    📂 Upload a PDF in the sidebar to start chatting
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Built with using RAG · Gemini · FAISS
</p>
""", unsafe_allow_html=True)