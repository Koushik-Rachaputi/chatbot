import os

def load_documents(doc_folder):
    documents = []
    for filename in os.listdir(doc_folder):
        with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents
