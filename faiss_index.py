import os
import faiss
import numpy as np
from typing import Tuple, Optional

def build_index(vectors: np.ndarray) -> faiss.Index:
    """
    Creates a FAISS index using Inner Product (cosine similarity for normalized vectors).
    Assumes vectors are already normalized and of shape (N, 384).
    """
    dimension = vectors.shape[1]
    # IndexFlatIP is used for inner product search. 
    # Since inputs are normalized, IP is equivalent to Cosine Similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors.astype("float32"))
    return index

def save_index(index: faiss.Index, path: str) -> None:
    """
    Saves the FAISS index to the specified disk path.
    """
    faiss.write_index(index, path)

def load_index(path: str) -> faiss.Index:
    """
    Loads a FAISS index from the specified disk path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found at: {path}")
    return faiss.read_index(path)

def search(index: faiss.Index, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searches the index for the top k nearest neighbors.
    Query can be (384,) or (1, 384).
    Returns (scores, indices).
    """
    # Ensure query is 2D (1, dimension)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    scores, ids = index.search(query.astype("float32"), k)
    return scores, ids

if __name__ == "__main__":
    # CLI test block
    import numpy as np
    
    # Create 1000 fake normalized vectors
    d = 384
    nb = 1000
    vecs = np.random.rand(nb, d).astype("float32")
    # Normalize for cosine similarity
    faiss.normalize_L2(vecs)

    # Build index
    index = build_index(vecs)

    # Search using the first vector as query
    query = vecs[0]
    scores, ids = search(index, query, k=5)

    print("top ids:", ids)
    print("scores:", scores)
