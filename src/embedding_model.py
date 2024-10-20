from transformers import AutoTokenizer, AutoModel
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
