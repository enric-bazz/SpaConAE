"""k-nearest-neighbour spatial graph construction."""

import numpy as np
import pynndescent
import torch_geometric
from scipy import sparse as sp


def get_knn(location, n_neighbors):
    index = pynndescent.NNDescent(location, n_neighbors=n_neighbors)
    neighbor_idx, neighbor_dist = index.neighbor_graph
    return neighbor_idx, neighbor_dist


def knn_to_adj(knn_idx, knn_dist):
    n_samples, n_neighbors = knn_idx.shape
    row = np.repeat(np.arange(n_samples), n_neighbors)
    col = knn_idx.flatten()
    data = np.ones(n_samples * n_neighbors)
    return sp.csr_matrix((data, (row, col)), shape=(n_samples, n_samples))


def location_to_edge(location, n_neighbors):
    """``edge_index`` (shape ``[2, E]``) of the kNN graph of ``location``."""
    idx, dist = get_knn(location, n_neighbors)
    adj = knn_to_adj(idx, dist)
    return torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
