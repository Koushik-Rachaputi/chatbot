import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_docs(query_embedding, doc_embeddings, documents, top_k=3):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [documents[i] for i in top_indices]
