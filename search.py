import pickle
from embeddings import get_embedding, vector_search, rerank
from documents import documents

# Load embeddings
with open("doc_embeddings.pkl", "rb") as f:
    doc_embeddings = pickle.load(f)

query = input("Enter your search query: ")

# Get query embedding
query_embedding = get_embedding(query)

# Vector search
top_docs = vector_search(query_embedding, doc_embeddings, documents)

# Re-rank
reranked = rerank(query, top_docs, documents)

print("\nTop Results:\n")

for idx, score in reranked:
    print(f"Score: {score:.2f}")
    print(documents[idx]["content"])
    print("-" * 50)
