import os
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
INDEX_PATH = "faiss_index.index"
DOCS_PATH = "documents.pkl"

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

print("Reading data files...")

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(DATA_FOLDER, filename)

        # Extract crop name from filename
        crop_name = filename.replace(".txt", "").strip().capitalize()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

            # Split based on pattern: "\nCropName - "
            pattern = rf"\n{crop_name} - "
            sections = re.split(pattern, text)

            for i, section in enumerate(sections):
                if i == 0:
                    continue  # Skip text before first problem

                clean_section = f"{crop_name} - " + section.strip()

                # Avoid very small chunks
                if len(clean_section) > 200:
                    documents.append(clean_section)

print("Total cleaned chunks:", len(documents))

if len(documents) == 0:
    raise ValueError("No valid documents found. Check your data format.")

print("Creating embeddings...")

embeddings = embed_model.encode(
    documents,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

# Using Inner Product since embeddings are normalized
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print("FAISS index created successfully.")
print("Total chunks indexed:", len(documents))