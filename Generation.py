import streamlit as st
import json
from Retrieval import retrieve_documents
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.title("RAG Question Answering System")

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

llm = load_llm()

K = 3

def precision_at_k(docs, keywords, k):
    relevant = 0
    for doc in docs[:k]:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in keywords):
            relevant += 1
    return relevant / k

def recall_at_k(docs, keywords):
    retrieved_relevant = 0
    for doc in docs:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in keywords):
            retrieved_relevant += 1
    total_relevant = len(keywords)
    return retrieved_relevant / total_relevant

query = st.text_input("Ask your question")

if query:
    docs, scores = retrieve_documents(query)

    if not docs:
        st.warning("No relevant documents retrieved.")
        st.stop()

    context = "\n\n".join([doc.page_content[:800] for doc in docs])

    prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

Rules:
1. Use only the context.
2. If the answer is not in the context say:
"The provided documents do not contain enough information."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else response

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Context")

    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        st.markdown(f"### Chunk {i}")
        st.write(f"Similarity Score: {score:.4f}")
        st.write(doc.page_content[:800])

st.divider()
st.header("Retrieval Evaluation")

if st.button("Run Evaluation"):

    with open("evaluation_queries.json") as f:
        queries = json.load(f)

    results = []
    precision_scores = []
    recall_scores = []

    for item in queries:

        q = item["query"]
        keywords = item["relevant_keywords"]

        docs, scores = retrieve_documents(q)

        p = precision_at_k(docs, keywords, K)
        r = recall_at_k(docs, keywords)

        precision_scores.append(p)
        recall_scores.append(r)

        results.append({
            "Query": q,
            "Precision@3": round(p,3),
            "Recall@3": round(r,3)
        })

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    st.table(results)

    st.write(f"Average Precision@3: {avg_precision:.3f}")
    st.write(f"Average Recall@3: {avg_recall:.3f}")