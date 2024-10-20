import os
import subprocess

# Define the project structure directly
project_structure = {
    "data": {
        "documents": {
            "doc1.txt": "This is the content of document 1.",
            "doc2.txt": "This is the content of document 2."
        },
        "embeddings": {
            # Placeholder for embedding files
        }
    },
    "src": {
        "main.py": """from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_documents
from rag_pipeline import rag_pipeline
import os
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()

# Load documents once at startup
documents = load_documents(config['paths']['document_folder'])

class Query(BaseModel):
    query: str

@app.post("/generate/")
def generate_response(query: Query):
    response = rag_pipeline(query.query, documents)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
        "retrieval.py": """import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_docs(query_embedding, doc_embeddings, documents, top_k=3):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [documents[i] for i in top_indices]
""",
        "generate.py": """import requests
import os

def generate_text(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={{api_key}}'
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("contents", [])[0].get("parts", [])[0].get("text", "")
""",
        "rag_pipeline.py": """from retrieval import retrieve_relevant_docs
from generate import generate_text
from embedding_model import EmbeddingModel
from utils import load_documents
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
    prompt = query + "\\n\\n" + "\\n\\n".join(relevant_docs)

    # Step 6: Generate response using the base model
    generated_response = generate_text(prompt)
    return generated_response
""",
        "embedding_model.py": """from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_doc_embeddings(self, documents):
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings

    def get_query_embedding(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding.squeeze().numpy()
""",
        "utils.py": """import os

def load_documents(doc_folder):
    documents = []
    for filename in os.listdir(doc_folder):
        with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents
"""
    },
    "config": {
        "config.yaml": """paths:
  document_folder: 'data/documents/'
""",
    },
    ".env": """# Place any API keys here if required
GEMINI_API_KEY=your_api_key_here
""",
    ".gitignore": """.env
__pycache__/
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3
*.egg-info/
*.egg
dist/
build/
*.log
*.so
*.zip
*.tar.gz
*.tgz
*.h5
*.hdf5
data/embeddings/
""",
    "requirements.txt": """fastapi
uvicorn
requests
numpy
torch
transformers
python-dotenv
scikit-learn
""",
    "README.md": """# RAG Project

This project implements a Retrieval-Augmented Generation (RAG) model using Hugging Face transformers for embedding and text generation.

## Project Structure

- **data/**: Contains documents and embeddings.
- **src/**: Contains the source code for retrieval, generation, and pipeline logic.
- **config/**: Contains configuration files.
- **.env**: Store sensitive information like API keys (if needed).
- **requirements.txt**: Lists dependencies for the project.
"""
}

# Function to create directories and files
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)  # Recur for subdirectories
        else:
            with open(path, 'w') as file:
                file.write(content)
            print(f"Created file: {path}")

# Create the project structure in the current working directory
create_structure(os.getcwd(), project_structure)

# Create a virtual environment
venv_path = os.path.join(os.getcwd(), "venv")
subprocess.run(["python", "-m", "venv", venv_path])

print("Project structure created successfully in the current folder.")
print("Virtual environment created at:", venv_path)

