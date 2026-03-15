import json
import pandas as pd
import streamlit as st
from Retrieval import retrieve_documents

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


def run_evaluation():

    with open("evaluation_queries.json") as f:
        queries = json.load(f)

    results = []

    precision_scores = []
    recall_scores = []

    for item in queries:

        query = item["query"]
        keywords = item["relevant_keywords"]

        docs, scores = retrieve_documents(query)

        p = precision_at_k(docs, keywords, K)
        r = recall_at_k(docs, keywords)

        precision_scores.append(p)
        recall_scores.append(r)

        results.append({
            "Query": query,
            "Precision@3": round(p,3),
            "Recall@3": round(r,3)
        })

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    df = pd.DataFrame(results)

    return df, avg_precision, avg_recall


st.title("RAG Retrieval Evaluation")

if st.button("Run Evaluation"):

    table, avg_p, avg_r = run_evaluation()

    st.subheader("Evaluation Results")
    st.dataframe(table)

    st.subheader("Average Metrics")

    st.write(f"Average Precision@3: **{avg_p:.3f}**")
    st.write(f"Average Recall@3: **{avg_r:.3f}**")