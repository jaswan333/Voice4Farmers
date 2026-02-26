import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
INDEX_PATH = "faiss_index.index"
DOCS_PATH = "documents.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(DATA_FOLDER, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

print("Creating embeddings...")
embeddings = embed_model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print("FAISS index created successfully.")