"""Tests of the input-format checks."""

import anndata as ad
import numpy as np
import pytest
import torch

from spaconae import CrossModalAE
from spaconae.data import SpatialDataset, full_edge_index, knn_neighbors, load_pair, seed_subgraph


def test_load_pair_shapes_and_labels(data_dir):
    data = load_pair(data_dir, "st.h5ad", "sc.h5ad", knn=4, label_key="Cell_class")
    assert data.X_sc.dtype == "float32" and data.X_sc.shape[1] == data.X_st.shape[1]
    assert data.location.shape == (data.n_spots, 2)
    assert len(data.classes) == 3 and len(data.y_st) == data.n_spots

    assert data.neighbors.shape == (data.n_spots, 4)

    assert load_pair(data_dir, "st.h5ad", "sc.h5ad", knn=4).classes is None


def test_load_pair_rejects_mismatched_genes(data_dir):
    adata = ad.read_h5ad(data_dir / "sc.h5ad")
    adata[:, ::-1].copy().write(data_dir / "sc_reordered.h5ad")
    with pytest.raises(ValueError, match="same var_names"):
        load_pair(data_dir, "st.h5ad", "sc_reordered.h5ad", knn=4)


def test_load_pair_rejects_missing_coordinates_and_labels(data_dir):
    with pytest.raises(ValueError, match="spatial"):
        load_pair(data_dir, "sc.h5ad", "sc.h5ad", knn=4)
    with pytest.raises(ValueError, match="no obs column"):
        load_pair(data_dir, "st.h5ad", "sc.h5ad", knn=4, label_key="missing")
    with pytest.raises(ValueError, match="no obs column"):
        load_pair(data_dir, "st.h5ad", "sc.h5ad", knn=4, section_key="missing")


def test_knn_neighbors_are_exact_and_exclude_the_spot():
    rng = np.random.default_rng(0)
    location = rng.uniform(size=(50, 2))
    neighbors = knn_neighbors(location, 5)

    dist = np.linalg.norm(location[:, None] - location[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    assert np.array_equal(np.sort(neighbors, 1), np.sort(np.argsort(dist, 1)[:, :5], 1))


def test_knn_neighbors_stay_within_sections():
    rng = np.random.default_rng(0)
    location = rng.uniform(size=(40, 2))  # the sections overlap in 2D
    sections = np.repeat([0.1, 0.2], 20)
    neighbors = knn_neighbors(location, 3, sections)
    assert (sections[neighbors] == sections[:, None]).all()

    with pytest.raises(ValueError, match="too few"):
        knn_neighbors(location, 20, sections)


def test_seed_subgraph_lists_seeds_first():
    neighbors = knn_neighbors(np.random.default_rng(0).uniform(size=(30, 2)), 3)
    seeds = np.array([7, 2, 19])
    nodes, edge_index = seed_subgraph(neighbors, seeds)

    assert np.array_equal(nodes[:3], seeds) and len(set(nodes)) == len(nodes)
    source, target = edge_index.numpy()
    assert np.array_equal(nodes[source].reshape(3, 3), neighbors[seeds])
    assert np.array_equal(target, np.repeat(np.arange(3), 3))


def test_batched_spots_are_encoded_as_in_the_full_graph():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20)).astype("float32")
    neighbors = knn_neighbors(rng.uniform(size=(60, 2)), 4)
    encoder = CrossModalAE(d_input=20, d_hidden=16, d_latent=8).ST_encoder.eval()

    with torch.no_grad():
        z_full = encoder(torch.from_numpy(X), full_edge_index(neighbors))
        dataset = SpatialDataset(X, neighbors)
        for seeds in rng.permutation(60).reshape(4, 15):
            x, edge_index, idx, _ = dataset.collate(list(seeds))
            assert x.shape[0] > 15 and torch.equal(idx, torch.from_numpy(seeds))
            assert torch.allclose(encoder(x, edge_index, n_seeds=15), z_full[idx], atol=1e-6)
