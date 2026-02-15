import numpy as np
from openai import OpenAI
import pickle
import os

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://aipipe.org/openai/v1")


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def vector_search(query_embedding, doc_embeddings, documents, k=8):
    scores = []

    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def rerank(query, candidates, documents):
    reranked = []

    for idx, _ in candidates:
        doc = documents[idx]

        prompt = f"""
        Query: "{query}"
        Document: "{doc['content']}"

        Rate the relevance of this document to the query on a scale of 0-10.
        Respond with ONLY a number.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_score = response.choices[0].message.content.strip()

        try:
            score = float(raw_score) / 10
        except:
            print("Warning: Invalid LLM output ->", raw_score)
            score = 0

        reranked.append((idx, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked



#echo $env:OPENAI_API_KEY in powershell  ->It will print your saved key.
