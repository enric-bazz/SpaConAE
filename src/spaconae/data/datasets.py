"""Datasets and collate functions for the two modalities.

Every item carries its *global* index, which is what addresses the rows and
columns of the alignment matrix. Tensors are returned on CPU; Lightning moves
them to the training device.
"""

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import Dataset

from .graph import seed_subgraph


def _dense_float32(x):
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    return x.astype("float32")


class RNADataset(Dataset):
    """Returns ``(expression, index, label)``; label is -1 when unavailable."""

    def __init__(self, data_fit, labels=None):
        self.data_fit = _dense_float32(data_fit)
        if labels is not None:
            assert len(labels) == len(self.data_fit)
        self.labels = labels

    def __len__(self):
        return len(self.data_fit)

    def __getitem__(self, index):
        label = -1 if self.labels is None else self.labels[index]
        return self.data_fit[index], index, label


class SpatialDataset(Dataset):
    """Spots with their spatial graph.

    Items are spot indices. ``collate`` turns a batch of them (the seeds) into
    ``(x, edge_index, index, label)``, where ``x`` holds the seeds followed by
    their neighbours in the graph and ``edge_index`` the edges between them
    (see ``seed_subgraph``). ``index`` and ``label`` refer to the seeds only.
    """

    def __init__(self, data_fit, neighbors, labels=None):
        self.data_fit = _dense_float32(data_fit)
        assert len(self.data_fit) == len(neighbors)
        if labels is not None:
            assert len(labels) == len(self.data_fit)
        self.neighbors = neighbors
        self.labels = labels

    def __len__(self):
        return len(self.data_fit)

    def __getitem__(self, index):
        return index

    def collate(self, seeds):
        seeds = np.asarray(seeds, dtype=np.int64)
        nodes, edge_index = seed_subgraph(self.neighbors, seeds)
        x = torch.from_numpy(self.data_fit[nodes])
        idx = torch.from_numpy(seeds)
        if self.labels is None:
            y = torch.full((len(seeds),), -1, dtype=torch.long)
        else:
            y = torch.as_tensor(self.labels[seeds], dtype=torch.long)
        return x, edge_index, idx, y


class RNACollate:
    """Batches to ``(x, index, label)``."""

    def __call__(self, batch):
        batch_fit, batch_idx, batch_label = zip(*batch)
        x = torch.from_numpy(np.vstack(batch_fit))
        idx = torch.tensor(batch_idx, dtype=torch.long)
        y = torch.tensor(batch_label, dtype=torch.long)
        return x, idx, y
