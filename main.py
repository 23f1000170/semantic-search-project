from fastapi import FastAPI
import time
import pickle
from documents import documents
from embeddings import get_embedding, vector_search, rerank

app = FastAPI()

with open("doc_embeddings.pkl", "rb") as f:
    doc_embeddings = pickle.load(f)

@app.get("/")
def root():
    return {"message": "Semantic Search API is running"}

@app.post("/search")
def search(payload: dict):
    start = time.time()

    query = payload["query"]
    k = payload.get("k", 8)
    rerankK = payload.get("rerankK", 5)

    if not query:
        return {"results": [],
                "reranked": False,
                "metrics": {
                    "latency": 0,
                    "totalDocs": len(documents)}}
    
    query_embedding = get_embedding(query)

    initial_results = vector_search(
        query_embedding,
        doc_embeddings,
        documents,
        k
    )

    if payload.get("rerank", True):
        final_results = rerank(query, initial_results, documents)[:rerankK]
        reranked_flag = True
    else:
        final_results = initial_results[:rerankK]
        reranked_flag = False


    

    results = [
        {
            "id": documents[idx]["id"],
            "score": max(0.0, min(float(score), 1.0)),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        }
        for idx, score in final_results
    ]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
