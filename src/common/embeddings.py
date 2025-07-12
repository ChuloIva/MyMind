# Placeholder for text-embedding-3-small helper
import numpy as np

def embed_batch(nodes: list[str]) -> list[list[float]]:
    """
    Placeholder function to simulate embedding a batch of text nodes.
    In a real implementation, this would call an embedding model.
    """
    # Simulate embedding by returning random vectors
    return np.random.rand(len(nodes), 768).tolist()
