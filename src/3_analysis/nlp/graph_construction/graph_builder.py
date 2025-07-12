from common.embeddings import embed_batch
import umap
import numpy as np

def build(nodes: list[str]):
    vecs = embed_batch(nodes)
    xy = umap.UMAP(n_components=2).fit_transform(np.array(vecs))
    return [{"id": n, "x": float(x), "y": float(y)} for n,(x,y) in zip(nodes, xy)]