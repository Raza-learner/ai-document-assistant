import streamlit as st
from pdf_processor import load_and_split_pdf
from vector_store import create_vector_store
from qa_chain import build_qa_chain, get_answer

st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 AI Document Assistant")
st.caption("Upload one or more PDFs and ask questions — powered by RAG + Gemini")

# --- Session state (ALWAYS before any usage) ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs (one or more)",
        type="pdf",
        accept_multiple_files=True
    )

    # Process PDFs when uploaded and not yet indexed
    if uploaded_files and st.session_state.qa_chain is None:
        with st.spinner("Reading and indexing PDFs..."):
            all_chunks = []
            for file in uploaded_files:
                chunks = load_and_split_pdf(file)
                all_chunks.extend(chunks)
            try:
                if not all_chunks:
                    st.error("❌ No readable text found in the uploaded PDF(s).")
                    st.info(
                    "👉 This usually happens if the PDF is scanned or image-based.\n\n"
                    "Try uploading a text-based PDF."
                    )
                    st.stop()

                vs = create_vector_store(all_chunks)
                st.session_state.qa_chain = build_qa_chain(vs)

            except Exception as e:
                st.error("⚠️ Something went wrong while processing the document.")
                st.caption(str(e))
                st.stop()
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.pdf_count = len(uploaded_files)
        st.success(
            f"Ready! {len(all_chunks)} chunks from "
            f"{len(uploaded_files)} PDF(s)"
        )

    # Metrics
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
            label="Download chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

    # Clear everything
    if st.button("Clear chat + reload"):
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.chunk_count = 0
        st.session_state.pdf_count = 0
        st.rerun()

# --- Chat interface ---

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept new question only if PDFs are indexed
if st.session_state.qa_chain:
    if question := st.chat_input("Ask a question about your PDF..."):

        # Show and store user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        # Get and show assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                result = get_answer(st.session_state.qa_chain, question)
            st.markdown(result["answer"])

            # Show page sources
            if result["sources"]:
                with st.expander("Sources used to answer this"):
                    for i, doc in enumerate(result["sources"], 1):
                        page = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Source {i} — Page {page}**")
                        st.caption(doc.page_content[:250] + "...")
                        st.divider()

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )

else:
    st.info("Upload a PDF in the sidebar to get started.")