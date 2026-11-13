"""Tests of the input-format checks."""

import anndata as ad
import pytest

from spaconae.data import load_pair


def test_load_pair_shapes_and_labels(data_dir):
    data = load_pair(data_dir, "st.h5ad", "sc.h5ad", label_key="Cell_class")
    assert data.X_sc.dtype == "float32" and data.X_sc.shape[1] == data.X_st.shape[1]
    assert data.location.shape == (data.n_spots, 2)
    assert len(data.classes) == 3 and len(data.y_st) == data.n_spots

    assert load_pair(data_dir, "st.h5ad", "sc.h5ad").classes is None


def test_load_pair_rejects_mismatched_genes(data_dir):
    adata = ad.read_h5ad(data_dir / "sc.h5ad")
    adata[:, ::-1].copy().write(data_dir / "sc_reordered.h5ad")
    with pytest.raises(ValueError, match="same var_names"):
        load_pair(data_dir, "st.h5ad", "sc_reordered.h5ad")


def test_load_pair_rejects_missing_coordinates_and_labels(data_dir):
    with pytest.raises(ValueError, match="spatial"):
        load_pair(data_dir, "sc.h5ad", "sc.h5ad")
    with pytest.raises(ValueError, match="no obs column"):
        load_pair(data_dir, "st.h5ad", "sc.h5ad", label_key="missing")
