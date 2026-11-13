"""End-to-end smoke test of the four phases on synthetic data (CPU, seconds)."""

from pathlib import Path

import torch

from spaconae import CrossModalAE
from spaconae.data import load_pair
from spaconae.nn import LEGACY_KEY
from spaconae.pipeline import run_align, run_finetune, run_pretrain, run_probe

from conftest import N_CELLS, N_SPOTS


def test_full_pipeline(cfg):
    ckpt = Path(cfg["checkpoint_dir"])

    run_pretrain(cfg)
    for name in ("spatial_encoder", "scrna_encoder", "spatial_decoder", "scrna_decoder"):
        assert (ckpt / f"{name}_weights.pth").exists()

    run_align(cfg)
    ewc = torch.load(ckpt / "ewc_alignment.pt")
    assert ewc["optimal_params"][LEGACY_KEY].shape == (N_CELLS, N_SPOTS)
    assert ewc["importance"][LEGACY_KEY].shape == (N_CELLS, N_SPOTS)

    run_finetune(cfg)
    assert (ckpt / "scrna_encoder_weights_finetuned.pth").exists()
    assert (ckpt / "alignment_matrix_finetuned.pth").exists()

    results = run_probe(cfg)
    assert 0.0 <= results["test_acc"] <= 1.0


def test_gene_space_alignment_without_continual(cfg):
    cfg["align"].update(encode_data=False, lambda_r=0.0, output_file="ewc_noAE.pt")
    cfg["finetune"].update(alignment_file="ewc_noAE.pt", continual_alignment=False, save_alignment=False)

    run_pretrain(cfg)
    run_align(cfg)
    module = run_finetune(cfg)
    assert not module.AlignMat.M.requires_grad


def test_alignment_uses_pretrained_encoders(cfg):
    run_pretrain(cfg)
    module = run_align(cfg)

    data = load_pair(cfg["data_dir"])
    full_sc, full_st, edge_index = data.full_batch(knn=cfg["align"]["knn"])
    ae = CrossModalAE(d_input=data.n_genes, **cfg["model"]).load_weights(cfg["checkpoint_dir"]).eval()
    z_sc, z_st = ae.encode(full_sc, full_st, edge_index)

    assert torch.allclose(module.S.cpu(), z_sc, atol=1e-5)
    assert torch.allclose(module.G.cpu(), z_st, atol=1e-5)
