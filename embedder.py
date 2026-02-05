import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Load model globally (once)
# Automatically handles device selection (CUDA/CPU) if device is not specified, 
# but we will follow the explicit device handling requirement.
def get_device() -> torch.device:
    """
    Returns the appropriate device for torch (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize device and model
DEVICE = get_device()
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME, device=str(DEVICE))

def embed_text(text: str) -> np.ndarray:
    """
    Converts a single string into a normalized dense vector embedding.
    Returns: a float32 numpy array.
    """
    # encode returns a numpy array by default
    embedding = MODEL.encode(
        text, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    return embedding.astype(np.float32)

def embed_batch(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Converts a list of strings into a batch of normalized dense vector embeddings.
    Returns: a float32 numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = MODEL.encode(
        texts, 
        batch_size=batch_size, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)

if __name__ == "__main__":
    # CLI test block
    sample = "python backend developer with fastapi experience"
    vec = embed_text(sample)
    print("shape:", vec.shape)
    print("device:", get_device())
