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
            text = f.read()

            for i in range(0, len(text), 400):
                chunk = text[i:i+500]
                documents.append(chunk)

print("Creating embeddings...")
embeddings = embed_model.encode(
    documents,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print("FAISS index created successfully.")
print("Total chunks indexed:", len(documents))