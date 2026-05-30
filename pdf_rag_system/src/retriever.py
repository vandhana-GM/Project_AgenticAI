from langchain_community.vectorstores import FAISS


def load_vector_store(embeddings):

    db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def search_query(query, db):

    results = db.similarity_search(
        query,
        k=3
    )

    return results