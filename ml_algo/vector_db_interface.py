import pickle
from sklearn.neighbors import KDTree
import numpy as np

class VectorDatabase:
    # ANDREY TODO
    def __init__(self, path_to_index):
        self.full_index = None
        with open(path_to_index, "rb") as f:
            self.full_index = pickle.load(f)

        self.indexes = [i[0] for i in self.full_index]
        self.embs = [i[1].reshape(-1) for i in self.full_index]
        self.tree = KDTree(np.array(self.embs), metric="euclidean")

    def query(self, emb, k=5):
        emb = emb.reshape(-1)
        dist, inds = self.tree.query([emb], k=k)
        inds = list(inds[0])
        best_embs = [self.embs[i] for i in list(inds)]
        return list(zip(inds, best_embs))
