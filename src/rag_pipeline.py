from .retrieval import retrieve_relevant_docs
from .generate import generate_text
from .embedding_model import EmbeddingModel
from .utils import load_documents
import numpy as np

def rag_pipeline(query, documents):
    # Step 1: Initialize the embedding model
    embedding_model = EmbeddingModel()
    
    # Step 2: Generate document embeddings
    doc_embeddings = embedding_model.get_doc_embeddings(documents)
    
    # Step 3: Generate query embedding
    query_embedding = embedding_model.get_query_embedding(query)

    # Step 4: Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(query_embedding, doc_embeddings, documents)

    # Step 5: Combine documents into a single prompt
    prompt = query + "\n\n" + "\n\n".join(relevant_docs)

    # Step 6: Generate response using the base model
    generated_response = generate_text(prompt)
    return generated_response
