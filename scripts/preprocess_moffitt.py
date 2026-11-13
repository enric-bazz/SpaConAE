"""Preprocessing of the Moffitt et al. (2018) hypothalamus datasets.

Port of ``notebooks/Data Pre-Processing.ipynb``. Dataset-specific, hence a
script and not part of the package: it produces the two ``.h5ad`` files in the
format ``spaconae`` expects (see "Input format" in the README).

Extra requirements: ``pip install scanpy squidpy pandas openpyxl``

Inputs
    * MERFISH data: downloaded by ``squidpy.datasets.merfish``.
    * scRNA-seq counts: GEO series GSE113576 (three files, ~250 MB).
    * scRNA-seq cell labels: Table S1 of the paper
      (``aau5324_moffitt_table-s1.xlsx``), to be provided by the user.

Example
    python scripts/preprocess_moffitt.py \
        --labels-xlsx data/raw/aau5324_moffitt_table-s1.xlsx \
        --raw-dir data/raw --out-dir data/processed
"""

import argparse
import re
import urllib.request
from pathlib import Path

import numpy as np

GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE113nnn/GSE113576/suppl/"
GEO_FILES = ("GSE113576_barcodes.tsv.gz", "GSE113576_genes.tsv.gz", "GSE113576_matrix.mtx.gz")

_LABEL_COLUMNS = {
    "Cell name": "Cell_ID",
    "Replicate number": "Replicate_number",
    "Cell class (determined from clustering of all cells)": "Cell_class",
    "Non-neuronal cluster (determined from clustering of all cells)": "Non-neuronal_cluster_ID",
    "Neuronal cluster (determined from clustering of inhibitory or excitatory neurons)": "Neuronal_cluster_ID",
}


def download_geo(raw_dir):
    """Download the GSE113576 files that are not already in ``raw_dir``."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in GEO_FILES:
        target = raw_dir / name
        if not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(GEO_URL + name, target)
    return raw_dir


def load_scrna(raw_dir, labels_xlsx):
    """scRNA-seq counts from GEO with the cell annotations of Table S1."""
    import anndata as ad
    import pandas as pd
    import scipy.io

    raw_dir = Path(raw_dir)
    matrix = scipy.io.mmread(raw_dir / GEO_FILES[2]).T.tocsc()
    genes = pd.read_csv(raw_dir / GEO_FILES[1], sep="\t", header=None)
    barcodes = pd.read_csv(raw_dir / GEO_FILES[0], sep="\t", header=None)

    adata_sc = ad.AnnData(X=matrix)
    adata_sc.var_names = genes[1].astype(str)
    adata_sc.obs_names = barcodes[0].astype(str)
    adata_sc = adata_sc[:, ~adata_sc.var_names.duplicated()].copy()

    df = pd.read_excel(labels_xlsx, header=1).rename(columns=_LABEL_COLUMNS)
    df["Cell_ID"] = df["Cell_ID"].astype(str)
    adata_sc.obs = adata_sc.obs.join(df.set_index("Cell_ID"), how="left")
    return adata_sc


def merge_subtypes(label):
    """Collapse the numbered MERFISH subtypes onto the scRNA-seq class names."""
    if re.match(r"Endothelial \d+", label):
        return "Endothelial"
    if re.match(r"OD Mature \d+", label):
        return "Mature oligodendrocyte"
    if re.match(r"OD Immature \d+", label):
        return "Immature oligodendrocyte"
    return label


def sample_spatial_patch(adata, sample_size=10000, seed=42):
    """The ``sample_size`` cells closest to a randomly chosen centre cell."""
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm["spatial"]
    np.random.seed(seed)
    center_coord = coords[np.random.choice(adata.n_obs)].reshape(1, -1)

    nbrs = NearestNeighbors(n_neighbors=int(sample_size), algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(center_coord)
    return adata[indices[0]].copy()


def preprocess_moffitt(
    labels_xlsx, raw_dir, out_dir, n_spots=10000, seed=42, download=True,
    st_file="moffit_spatial_train.h5ad", sc_file="moffit_scRNA_train.h5ad",
):
    """Write the two training ``.h5ad`` files to ``out_dir``.

    Steps: shared genes -> shared cell types (without 'Ambiguous') -> spatial
    patch of ``n_spots`` cells -> total-count normalisation (scRNA-seq only,
    MERFISH is already normalised), log1p and per-gene z-score.
    """
    import scanpy as sc
    import squidpy as sq

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if download:
        download_geo(raw_dir)

    adata_st = sq.datasets.merfish(path=raw_dir / "merfish.h5ad")
    adata_sc = load_scrna(raw_dir, labels_xlsx)

    # Shared genes
    common_genes = adata_st.var_names.intersection(adata_sc.var_names)
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()

    # Shared cell types
    adata_st.obs["Cell_class"] = adata_st.obs["Cell_class"].astype(str).apply(merge_subtypes)
    common_types = set(adata_sc.obs["Cell_class"].unique()) & set(adata_st.obs["Cell_class"].unique())
    common_types.discard("Ambiguous")
    adata_sc = adata_sc[adata_sc.obs["Cell_class"].isin(common_types)].copy()
    adata_st = adata_st[adata_st.obs["Cell_class"].isin(common_types)].copy()

    # Spatial patch
    adata_st = sample_spatial_patch(adata_st, sample_size=n_spots, seed=seed)

    # Normalisation
    sc.pp.log1p(adata_st)
    sc.pp.scale(adata_st)

    sc.pp.normalize_total(adata_sc)
    sc.pp.log1p(adata_sc)
    sc.pp.scale(adata_sc)

    adata_st.write(out_dir / st_file)
    adata_sc.write(out_dir / sc_file)
    print(f"scRNA-seq: {adata_sc.shape}, spatial: {adata_st.shape} -> {out_dir}")
    return adata_sc, adata_st


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Moffitt et al. scRNA-seq and MERFISH data")
    parser.add_argument("--labels-xlsx", required=True, help="Table S1 of the paper (.xlsx)")
    parser.add_argument("--raw-dir", default="data/raw", help="folder for the downloaded raw files")
    parser.add_argument("--out-dir", default="data/processed", help="folder for the two .h5ad files")
    parser.add_argument("--n-spots", type=int, default=10000, help="size of the spatial patch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", help="GEO files are already in --raw-dir")
    args = parser.parse_args()
    preprocess_moffitt(
        args.labels_xlsx, args.raw_dir, args.out_dir,
        n_spots=args.n_spots, seed=args.seed, download=not args.no_download,
    )
