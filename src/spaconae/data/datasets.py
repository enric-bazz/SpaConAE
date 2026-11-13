"""Datasets and collate functions for the two modalities.

Every item carries its *global* index, which is what addresses the rows and
columns of the alignment matrix. Tensors are returned on CPU; Lightning moves
them to the training device.
"""

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import Dataset

from .graph import location_to_edge


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
    """Returns ``(expression, location, index, label)``."""

    def __init__(self, data_fit, location, labels=None):
        self.data_fit = _dense_float32(data_fit)
        assert len(self.data_fit) == len(location)
        if labels is not None:
            assert len(labels) == len(self.data_fit)
        self.location = np.asarray(location)
        self.labels = labels

    def __len__(self):
        return len(self.data_fit)

    def __getitem__(self, index):
        label = -1 if self.labels is None else self.labels[index]
        return self.data_fit[index], self.location[index], index, label


class RNACollate:
    """Batches to ``(x, index, label)``."""

    def __call__(self, batch):
        batch_fit, batch_idx, batch_label = zip(*batch)
        x = torch.from_numpy(np.vstack(batch_fit))
        idx = torch.tensor(batch_idx, dtype=torch.long)
        y = torch.tensor(batch_label, dtype=torch.long)
        return x, idx, y


class SpatialCollate:
    """Batches to ``(x, edge_index, index, label)``.

    As in the notebooks, the kNN graph is built among the spots of the batch.
    """

    def __init__(self, knn=10):
        self.knn = knn

    def __call__(self, batch):
        batch_fit, batch_location, batch_idx, batch_label = zip(*batch)
        x = torch.from_numpy(np.vstack(batch_fit))
        edge_index = location_to_edge(np.vstack(batch_location), self.knn)
        idx = torch.tensor(batch_idx, dtype=torch.long)
        y = torch.tensor(batch_label, dtype=torch.long)
        return x, edge_index, idx, y
