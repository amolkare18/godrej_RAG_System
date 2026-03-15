# from Embeddings import get_vector_store
from Embeddings_2_test import get_vector_store

_vector_store = None


def _get_vector_store():
    """Load Pinecone vector store once"""
    global _vector_store

    if _vector_store is None:
        _vector_store = get_vector_store()

    return _vector_store


def retrieve_documents(query, k=3):
    """
    Retrieve documents and similarity scores
    """

    vector_store = _get_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    docs = []
    scores = []

    for doc, score in results:
        docs.append(doc)
        scores.append(score)

    return docs, scores


if __name__ == "__main__":

    query = "What problem does LSTM solve?"

    docs, scores = retrieve_documents(query)

    for doc, score in zip(docs, scores):
        print("\n---- Retrieved Chunk ----\n")
        print("Score:", score)
        print("Chunk ID:", doc.metadata.get("chunk_id"))
        print(doc.page_content[:500])