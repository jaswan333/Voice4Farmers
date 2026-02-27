import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from app.config import HF_TOKEN

if HF_TOKEN:
    login(HF_TOKEN)

INDEX_PATH = "faiss_index.index"
DOCS_PATH = "documents.pkl"

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
documents = None


def load_vector_store():
    global index, documents
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run build_index.py first.")

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)


def search(query, k=3):

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])

    top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0

    return results, top_score
