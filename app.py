import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

sentiment_model = pipeline("sentiment-analysis")

intent_model = pipeline(
    "text-classification",
    model="models/intent_model",
    tokenizer="models/intent_model"
)

with open("rag/company_docs.txt") as f:
    docs = f.readlines()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

def retrieve_context(query):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), 1)
    return docs[I[0][0]]

st.title("ZENDS AI Customer Support Copilot")

msg = st.text_input("Enter Customer Message")

if st.button("Analyze"):
    intent = intent_model(msg)[0]["label"]
    sentiment = sentiment_model(msg)[0]["label"]
    context = retrieve_context(msg)

    st.subheader("Intent")
    st.write(intent)
    st.subheader("Sentiment")
    st.write(sentiment)
    st.subheader("Policy")
    st.write(context)
