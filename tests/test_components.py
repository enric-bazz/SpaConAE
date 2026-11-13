"""Unit tests of the building blocks."""

import torch

from spaconae import CrossModalAE
from spaconae.data import location_to_edge
from spaconae.losses import AlignmentMatrixLoss, ContinualLoss, ContrastiveLoss
from spaconae.nn import AlignmentMatrix


def test_autoencoder_shapes_and_weight_roundtrip(tmp_path):
    torch.manual_seed(0)
    ae = CrossModalAE(d_input=20, d_hidden=16, d_latent=8)
    x_sc, x_st = torch.randn(10, 20), torch.randn(12, 20)
    edge_index = location_to_edge(torch.rand(12, 2).numpy(), 4)

    z_sc, z_st, sc_recon, st_recon = ae(x_sc, x_st, edge_index)
    assert z_sc.shape == (10, 8) and z_st.shape == (12, 8)
    assert sc_recon.shape == x_sc.shape and st_recon.shape == x_st.shape

    ae.save_weights(tmp_path, suffix="_x")
    other = CrossModalAE(d_input=20, d_hidden=16, d_latent=8).load_weights(tmp_path, suffix="_x")
    ae.eval(), other.eval()
    assert torch.allclose(ae.encode(x_sc, x_st, edge_index)[1], other.encode(x_sc, x_st, edge_index)[1])


def test_state_dict_keys_match_notebooks():
    ae = CrossModalAE(d_input=155)
    assert set(ae.SC_encoder.state_dict()) == {
        "model.0.weight", "model.0.bias", "model.3.weight", "model.3.bias", "model.5.weight", "model.5.bias",
    }
    assert set(ae.ST_encoder.state_dict()) == {
        "pre_fc.0.weight", "pre_fc.0.bias", "pre_fc.3.weight", "pre_fc.3.bias",
        "post_fc.weight", "post_fc.bias",
        "gat_layer.conv_layers.0.att_src", "gat_layer.conv_layers.0.att_dst",
        "gat_layer.conv_layers.0.bias", "gat_layer.conv_layers.0.lin.weight",
    }


def test_alignment_matrix_batch_keeps_global_normalisation():
    M = AlignmentMatrix(n_cells=6, n_spots=5)
    full = M.get_M_full()
    assert torch.allclose(full.sum(1), torch.ones(6))
    cells, spots = torch.tensor([4, 1]), torch.tensor([0, 3, 2])
    assert torch.equal(M.get_M_batch(cells, spots), full[cells][:, spots])


def test_alignment_loss_is_minimal_for_perfect_projection():
    G = torch.rand(5, 7) + 0.1
    M_probs = torch.full((6, 5), 1 / 5)
    loss_fn = AlignmentMatrixLoss(n_spots=5, lambda_r=0.0)
    total, _, gv, vg, density = loss_fn(M_probs, G, G)
    assert torch.isclose(gv, torch.tensor(1.0)) and torch.isclose(vg, torch.tensor(1.0))
    assert torch.isclose(density, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(total, torch.tensor(-2.0), atol=1e-5)


def test_contrastive_loss_outputs():
    torch.manual_seed(0)
    z = torch.randn(8, 4)
    loss, pos, neg, n_neg = ContrastiveLoss(sim_threshold=0.5)(z.clone(), z)
    assert loss.ndim == 0
    assert abs(pos - 1.0) < 1e-5  # identical pairs
    assert 0 <= n_neg <= 7


def test_contrastive_loss_averages_over_all_pairs():
    # Pair 0 is opposite, the others identical; threshold -1 selects no negatives
    z_st = torch.eye(4)
    z_proj = torch.eye(4)
    z_proj[0] = -z_proj[0]
    loss, pos, _, n_neg = ContrastiveLoss(sim_threshold=-1.0)(z_proj, z_st)
    assert n_neg == 0
    assert torch.isclose(loss, torch.tensor((1 - (-1)) ** 2 / 4))
    assert abs(pos - 0.5) < 1e-6


def test_continual_loss_is_zero_at_reference_and_weighted_by_importance():
    M = AlignmentMatrix(n_cells=3, n_spots=4)
    importance = torch.zeros(3, 4)
    importance[0, 0] = 2.0
    ewc = ContinualLoss(lambda_ewc=3.0, importance_matrix=importance, M_reference=M.M.detach())

    assert ewc(M.M).item() == 0.0
    with torch.no_grad():
        M.M += 1.0  # the reference must not move with the parameter
    assert torch.isclose(ewc(M.M), torch.tensor(3.0 / 2 * 2.0 * 1.0))
