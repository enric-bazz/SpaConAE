"""Tiny synthetic scRNA-seq / spatial pair and a matching config."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

N_CELLS, N_SPOTS, N_GENES, N_TYPES = 120, 80, 20, 3


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(N_GENES)]
    centers = rng.normal(size=(N_TYPES, N_GENES)) * 2

    def make(n):
        y = np.arange(n) % N_TYPES
        X = (centers[y] + rng.normal(size=(n, N_GENES))).astype("float32")
        obs = pd.DataFrame({"Cell_class": [f"type{t}" for t in y]}, index=[f"c{i}" for i in range(n)])
        return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    adata_sc = make(N_CELLS)
    adata_st = make(N_SPOTS)
    adata_st.obsm["spatial"] = rng.uniform(size=(N_SPOTS, 2))
    adata_st.obs["section"] = np.repeat(["a", "b"], N_SPOTS // 2)

    out = tmp_path / "data"
    out.mkdir()
    adata_sc.write(out / "sc.h5ad")
    adata_st.write(out / "st.h5ad")
    return out


@pytest.fixture
def cfg(tmp_path, data_dir):
    return {
        "seed": 0,
        "data": {"dir": str(data_dir), "st_file": "st.h5ad", "sc_file": "sc.h5ad", "label_key": "Cell_class",
                 "knn": 4, "section_key": "section"},
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "accelerator": "cpu",
        "progress_bar": False,
        "model": {"d_hidden": 16, "d_latent": 8, "dropout": 0.1},
        "pretrain": {
            "batch_size_st": 16, "batch_size_sc": 16,
            "lambda_recon": 0.1, "lambda_consistence": 1.0, "lr": 1e-3, "max_epochs": 2,
        },
        "align": {
            "encode_data": True,
            "lambda_g1": 1.0, "lambda_g2": 1.0, "lambda_r": 1e-6, "lambda_d": 2.0,
            "lr": 1e-2, "max_epochs": 5, "patience": 50, "min_delta": 0.05, "relative_delta": True,
            "output_file": "ewc_alignment.pt",
        },
        "finetune": {
            "alignment_file": "ewc_alignment.pt", "continual_alignment": True,
            "batch_size_st": 16, "batch_size_sc": 32,
            "lambda_recon": 1e-2, "lambda_contrast": 1.0, "lambda_ewc": 1.0, "lambda_consistence": 1.0,
            "sim_threshold": 0.7, "lr": 1e-3, "max_epochs": 2, "negatives_patience": 10,
            "output_suffix": "_finetuned", "save_alignment": True,
        },
        "probe": {
            "weights_suffix": "_finetuned", "batch_size_sc": 16, "batch_size_st": 16,
            "val_fraction": 0.2, "classifier_hidden": 16, "lr": 1e-2, "max_epochs": 3,
        },
    }
