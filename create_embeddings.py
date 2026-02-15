import pickle
from documents import documents
from embeddings import get_embedding

doc_embeddings = []

print("Generating embeddings...")

for doc in documents:
    emb = get_embedding(doc["content"])
    doc_embeddings.append(emb)

with open("doc_embeddings.pkl", "wb") as f:
    pickle.dump(doc_embeddings, f)

print("Embeddings saved successfully!")
