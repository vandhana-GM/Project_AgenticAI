import os
import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import load_vector_store, search_query

st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG System")
st.write("Upload a PDF, build embeddings, and ask questions.")

# Create uploads folder
os.makedirs("uploads", exist_ok=True)

# Sidebar
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDF Uploaded Successfully")

    if st.sidebar.button("Build Vector Database"):

        with st.spinner("Processing PDF..."):

            docs = load_pdf(pdf_path)

            chunks = split_documents(docs)

            embedding_model = get_embedding_model()

            create_vector_store(
                chunks,
                embedding_model
            )

        st.success("Vector Database Created Successfully!")

# Question Section

st.header("Ask Questions")

query = st.text_input(
    "Enter your question"
)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a question")
    else:

        embedding_model = get_embedding_model()

        db = load_vector_store(
            embedding_model
        )

        results = search_query(
            query,
            db
        )

        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(results):

            with st.expander(
                f"Chunk {i+1}"
            ):
                st.write(doc.page_content)