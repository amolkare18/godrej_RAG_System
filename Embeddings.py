from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_NAME = "rag-index"
_vector_store = None


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )


def get_vector_store():
    global _vector_store

    if _vector_store is None:
        embeddings = _get_embeddings()

        _vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace="papers"
        )

    return _vector_store


def build_index():
    from Chunking_2 import get_chunks_2

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()

    if stats["total_vector_count"] > 0:
        print("Index already contains embeddings.")
        return

    chunks = get_chunks_2()

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = "research-paper"

    embeddings = _get_embeddings()

    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME,
        namespace="papers"
    )

    print(f"Embeddings stored in Pinecone: {len(chunks)} chunks")


if __name__ == "__main__":
    build_index()