"""Lightning data modules for the preprocessed scRNA-seq / spatial pair."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import CombinedLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

from .datasets import RNACollate, RNADataset, SpatialDataset, _dense_float32
from .graph import full_edge_index, knn_neighbors

@dataclass
class PairedData:
    """In-memory arrays of the two modalities on their shared genes."""

    X_sc: np.ndarray
    X_st: np.ndarray
    location: np.ndarray
    neighbors: np.ndarray  # [n_spots, k] spatial graph, see ``knn_neighbors``
    y_sc: Optional[np.ndarray] = None
    y_st: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None

    @property
    def n_cells(self):
        return self.X_sc.shape[0]

    @property
    def n_spots(self):
        return self.X_st.shape[0]

    @property
    def n_genes(self):
        return self.X_sc.shape[1]

    def full_batch(self):
        """Whole datasets as tensors plus the spatial graph of all spots."""
        return torch.from_numpy(self.X_sc), torch.from_numpy(self.X_st), full_edge_index(self.neighbors)


def load_pair(data_dir, st_file, sc_file, knn, label_key=None, section_key=None):
    """Read a preprocessed spatial / scRNA-seq pair.

    Both ``.h5ad`` files must hold the same genes in the same order, with
    ``X`` already normalised. The spatial file needs 2D coordinates in
    ``obsm["spatial"]``, from which the graph of the ``knn`` nearest spots is
    built. If the spots come from several tissue sections, ``section_key``
    names the ``obs`` column of the spatial file that tells them apart, and
    neighbours are searched within each section. If ``label_key`` is given, it must be a column of
    ``obs`` in both files with a shared set of categories (needed only for the
    label-transfer probe).
    """
    data_dir = Path(data_dir)
    adata_st = ad.read_h5ad(data_dir / st_file)
    adata_sc = ad.read_h5ad(data_dir / sc_file)

    if list(adata_st.var_names) != list(adata_sc.var_names):
        raise ValueError(f"{st_file} and {sc_file} must have the same var_names in the same order")
    if "spatial" not in adata_st.obsm:
        raise ValueError(f'{st_file} has no coordinates in obsm["spatial"]')

    sections = None
    if section_key is not None:
        if section_key not in adata_st.obs:
            raise ValueError(f"{st_file} has no obs column '{section_key}'")
        sections = adata_st.obs[section_key].to_numpy()
    location = np.asarray(adata_st.obsm["spatial"])

    y_sc = y_st = classes = None
    if label_key is not None:
        for name, adata in ((st_file, adata_st), (sc_file, adata_sc)):
            if label_key not in adata.obs:
                raise ValueError(f"{name} has no obs column '{label_key}'")
        unknown = set(adata_st.obs[label_key]) - set(adata_sc.obs[label_key])
        if unknown:
            raise ValueError(f"Labels of {st_file} missing from {sc_file}: {sorted(unknown)}")
        le = LabelEncoder()
        y_sc = le.fit_transform(adata_sc.obs[label_key])
        y_st = le.transform(adata_st.obs[label_key])
        classes = le.classes_

    return PairedData(
        X_sc=_dense_float32(adata_sc.X),
        X_st=_dense_float32(adata_st.X),
        location=location,
        neighbors=knn_neighbors(location, knn, sections),
        y_sc=y_sc,
        y_st=y_st,
        classes=classes,
    )


class PairedDataModule(L.LightningDataModule):
    """Independent shuffled mini-batches of the two modalities.

    Each training batch is ``{"st": (x, edge_index, idx, y), "sc": (x, idx, y)}``;
    the spatial part is described in ``SpatialDataset``.
    An epoch ends when the shorter loader is exhausted, like ``zip`` in the
    notebooks.
    """

    def __init__(self, data: PairedData, batch_size_st=64, batch_size_sc=64, num_workers=0):
        super().__init__()
        self.data = data
        self.batch_size_st = batch_size_st
        self.batch_size_sc = batch_size_sc
        self.num_workers = num_workers

    def train_dataloader(self):
        st_set = SpatialDataset(self.data.X_st, self.data.neighbors)
        st_loader = DataLoader(
            st_set, batch_size=self.batch_size_st, collate_fn=st_set.collate,
            drop_last=True, shuffle=True, num_workers=self.num_workers,
        )
        sc_loader = DataLoader(
            RNADataset(self.data.X_sc),
            batch_size=self.batch_size_sc, collate_fn=RNACollate(),
            drop_last=True, shuffle=True, num_workers=self.num_workers,
        )
        return CombinedLoader({"st": st_loader, "sc": sc_loader}, mode="min_size")


class LabelTransferDataModule(L.LightningDataModule):
    """scRNA-seq train/validation split and spatial test set for the probe."""

    def __init__(self, data: PairedData, batch_size_sc=64, batch_size_st=32, val_fraction=0.2, num_workers=0):
        super().__init__()
        if data.y_sc is None or data.y_st is None:
            raise ValueError("Label transfer needs labels in both modalities")
        self.data = data
        self.batch_size_sc = batch_size_sc
        self.batch_size_st = batch_size_st
        self.val_fraction = val_fraction
        self.num_workers = num_workers

    def setup(self, stage=None):
        full = RNADataset(self.data.X_sc, self.data.y_sc)
        n_val = int(round(self.val_fraction * len(full)))
        self.train_set, self.val_set = random_split(full, [len(full) - n_val, n_val])
        self.test_set = SpatialDataset(self.data.X_st, self.data.neighbors, self.data.y_st)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size_sc, collate_fn=RNACollate(),
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size_sc, collate_fn=RNACollate(),
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size_st, collate_fn=self.test_set.collate,
                          shuffle=False, num_workers=self.num_workers)
