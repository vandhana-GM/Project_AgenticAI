from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    db.save_local("vector_db")

    print("Vector Database Saved!")

    return db