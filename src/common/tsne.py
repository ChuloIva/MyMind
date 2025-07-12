import umap
import numpy as np

def reduce_dimensions(vectors: list[list[float]], n_components: int = 2) -> np.ndarray:
    """
    Reduces the dimensionality of vectors using UMAP.
    """
    return umap.UMAP(n_components=n_components).fit_transform(np.array(vectors))
