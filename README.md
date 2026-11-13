# SpaConAE
Spatial Contrastive Autoencoder for joint scRNA-seq and spatial transcriptomics representation learning and cell-to-spot alignment

SpaConAE embeds scRNA-seq cells and spatial transcriptomics spots in a shared latent space. Two modality-specific autoencoders are aligned through a Tangram-style cell-to-spot matrix, which defines the positive pairs of a contrastive objective.

## Repository layout

```
src/spaconae/     Python package (PyTorch Lightning)
  data/           datasets, kNN spatial graph, data modules, preprocessing
  nn/             encoders, decoders, alignment matrix, classifier
  losses.py       consistency, alignment, contrastive and continual losses
  model.py        CrossModalAE: the two autoencoders
  stages/         one LightningModule per training phase + the evaluation probe
  callbacks.py    stopping criteria
  pipeline.py     runs the phases from a config
  cli.py          `spaconae` command
configs/          YAML configurations (values from the notebooks)
notebooks/        original Colab notebooks, kept as the reference implementation
tests/            unit tests and an end-to-end smoke test on synthetic data
```

## Method in brief

| Phase | Module | What is trained | Loss |
|---|---|---|---|
| 1. AE initialisation | `AEPretrainModule` | both autoencoders | MSE reconstruction + cross-decoder consistency |
| 2. Alignment | `AlignmentModule` | matrix `M` (cells x spots), full batch | Tangram objective on latents (or shared genes) |
| 3. Contrastive fine-tuning | `ContrastiveModule` | autoencoders, optionally `M` with an EWC penalty | reconstruction + consistency + filtered contrastive loss |
| Evaluation | `LabelTransferModule` | classifier on frozen scRNA-seq latents | cross-entropy; tested on spatial latents |

The scRNA-seq encoder is an MLP; the spatial encoder is an MLP followed by a graph attention layer on the k-nearest-neighbour graph of the spots. Both decoders are MLPs.

## Installation

```bash
pip install -e .                  # training and evaluation
pip install -e ".[preprocess]"    # adds scanpy/squidpy for the preprocessing step
pip install -e ".[dev]"           # adds pytest
```

Install the PyTorch build that matches your hardware first (see pytorch.org); the CPU-only build is much smaller than the default CUDA one.

## Usage

Each phase reads and writes the same files the notebooks exchanged through Google Drive, so phases can be run separately and checkpoints produced by the notebooks can be reused.

```bash
spaconae preprocess --config configs/moffitt.yaml   # writes data/processed/*.h5ad
spaconae pretrain   --config configs/moffitt.yaml   # -> checkpoints/*_weights.pth
spaconae align      --config configs/moffitt.yaml   # -> checkpoints/ewc_alignment.pt
spaconae finetune   --config configs/moffitt.yaml   # -> checkpoints/*_weights_finetuned.pth
spaconae probe      --config configs/moffitt.yaml   # label transfer scRNA-seq -> spatial
spaconae all        --config configs/moffitt.yaml   # pretrain, align, finetune, probe
```

`--data-dir`, `--checkpoint-dir`, `--log-dir` and `--accelerator` override the corresponding config entries. Metrics are written as CSV under `logs/<phase>/`.

`configs/moffitt.yaml` aligns in latent space and keeps refining `M` during fine-tuning; `configs/moffitt_gene_space.yaml` aligns on the shared genes and keeps `M` frozen.

### Data

Preprocessing needs Table S1 of Moffitt et al. (2018), `aau5324_moffitt_table-s1.xlsx`, in `data/raw/`. The MERFISH data (squidpy) and the scRNA-seq counts (GEO GSE113576, about 250 MB) are downloaded to `data/raw/`. Alternatively, copy the two `moffit_*_train.h5ad` files written by the preprocessing notebook to `data/processed/`.

Disk usage to keep in mind: the alignment matrix has `n_cells x n_spots` entries (about 1.1 GB for 28,390 x 10,000), so `ewc_alignment.pt` is about 2.2 GB and the fine-tuned matrix another 1.1 GB (`finetune.save_alignment: false` skips it).

## Tests

```bash
pytest
```

## Differences from the notebooks

The first version of the package reproduces the behaviour of the notebooks. Metrics are logged without the loss coefficients. Later changes:
