from fastapi import FastAPI
from pydantic import BaseModel
from .utils import load_documents
from .rag_pipeline import rag_pipeline
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
