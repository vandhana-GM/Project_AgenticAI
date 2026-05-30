from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import load_vector_store, search_query

PDF_PATH = "data/sample.pdf"


def build_pipeline():

    print("Loading PDF...")

    docs = load_pdf(PDF_PATH)

    print("Splitting Text...")

    chunks = split_documents(docs)

    print("Creating Embeddings...")

    embedding_model = get_embedding_model()

    print("Creating FAISS DB...")

    create_vector_store(
        chunks,
        embedding_model
    )

    print("Pipeline Completed!")


def ask_question():

    embedding_model = get_embedding_model()

    db = load_vector_store(
        embedding_model
    )

    while True:

        query = input("\nAsk Question (exit to stop): ")

        if query.lower() == "exit":
            break

        results = search_query(
            query,
            db
        )

        print("\nTop Relevant Chunks:\n")

        for idx, doc in enumerate(results, 1):

            print(f"\nChunk {idx}")
            print("-" * 50)
            print(doc.page_content)


if __name__ == "__main__":

    build_pipeline()

    ask_question()