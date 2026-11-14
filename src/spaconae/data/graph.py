"""Spatial k-nearest-neighbour graph.

The graph is built once on the whole dataset and stored as an array of
neighbour indices. Mini-batches take their edges from it, so every phase and
the inference see the same neighbourhoods.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def knn_neighbors(location, k, sections=None):
    """Indices ``[n_spots, k]`` of the k nearest spots of each spot (itself excluded).

    With ``sections`` (one label per spot) neighbours are searched only among
    the spots of the same section.
    """
    location = np.asarray(location)
    if sections is None:
        groups = [np.arange(len(location))]
    else:
        sections = np.asarray(sections)
        groups = [np.flatnonzero(sections == s) for s in np.unique(sections)]

    neighbors = np.empty((len(location), k), dtype=np.int64)
    for members in groups:
        if len(members) <= k:
            raise ValueError(f"A section has {len(members)} spots, too few for {k} neighbours")
        # Without an argument, kneighbors() leaves each point out of its own neighbours
        local = NearestNeighbors(n_neighbors=k).fit(location[members]).kneighbors(return_distance=False)
        neighbors[members] = members[local]
    return neighbors


def full_edge_index(neighbors):
    """``edge_index`` (``[2, n * k]``) of the whole graph: neighbour -> spot."""
    n, k = neighbors.shape
    target = np.repeat(np.arange(n), k)
    return torch.from_numpy(np.stack([neighbors.ravel(), target]))


def seed_subgraph(neighbors, seeds):
    """Subgraph needed to encode the spots ``seeds`` with one graph layer.

    Returns ``(nodes, edge_index)``: ``nodes`` are global spot indices, the
    seeds first, then the neighbours that are not seeds; ``edge_index`` holds
    the neighbour -> seed edges in positions of ``nodes``. After one
    message-passing layer the first ``len(seeds)`` nodes have the same output
    as in the full graph; the other nodes lack their own neighbours and must
    be discarded.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    source = neighbors[seeds].ravel()
    nodes = np.concatenate([seeds, np.setdiff1d(source, seeds)])

    order = np.argsort(nodes)
    source_local = order[np.searchsorted(nodes[order], source)]
    target_local = np.repeat(np.arange(len(seeds)), neighbors.shape[1])
    return nodes, torch.from_numpy(np.stack([source_local, target_local]))
