# context_engine/memory.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import threading

# Load sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dimension of embeddings
dim = model.get_sentence_embedding_dimension()

# Create FAISS index (for similarity search)
index = faiss.IndexFlatL2(dim)

# Simple in-memory storage
memory_texts = []
lock = threading.Lock()

def add_to_memory(item: dict):
    """Add a user or assistant message to memory."""
    text = item.get("text")
    emb = model.encode([text])
    with lock:
        index.add(np.array(emb, dtype='float32'))
        memory_texts.append(text)

def get_relevant_context(query: str, top_k=3) -> str:
    """Retrieve most relevant past messages."""
    if len(memory_texts) == 0:
        return ""
    q_emb = model.encode([query]).astype('float32')
    D, I = index.search(q_emb, k=min(top_k, len(memory_texts)))
    results = [memory_texts[i] for i in I[0] if i != -1]
    return " ".join(results)
