"""Loss functions of the three training phases."""

import torch
import torch.nn.functional as F
from torch import nn


def consistence_loss(x_sc, x_st, z_sc, z_st, decoder_sc, decoder_st):
    """stAI cross-decoder consistency.

    Cosine distance between each input and its reconstruction through the
    decoder of the *other* modality.
    """
    loss_st = (1 - torch.sum(F.normalize(decoder_sc(z_st), p=2) * F.normalize(x_st, p=2), 1)).mean()
    loss_sc = (1 - torch.sum(F.normalize(decoder_st(z_sc), p=2) * F.normalize(x_sc, p=2), 1)).mean()
    return loss_st + loss_sc


class AlignmentMatrixLoss(nn.Module):
    """Unconstrained Tangram objective.

    Terms: gene-wise and spot-wise cosine similarity between projected and
    observed expression, entropy regulariser on the rows of M, and KL
    divergence between the spot density and a uniform prior.
    """

    def __init__(self, n_spots, lambda_g1=1.0, lambda_g2=1.0, lambda_r=1.0, lambda_d=1.0):
        super().__init__()
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d
        self.kl_loss = nn.KLDivLoss(reduction="sum", log_target=False)
        self.register_buffer("density_prior", torch.ones(n_spots) / n_spots)

    def forward(self, M_probs, G_pred, G):
        gv_term = self.lambda_g1 * F.cosine_similarity(G_pred, G, dim=0).mean()  # genes over voxels
        vg_term = self.lambda_g2 * F.cosine_similarity(G_pred, G, dim=1).mean()  # voxels over genes
        expression_term = gv_term + vg_term

        # sum(M log M) is the negative entropy; it enters the loss with a minus sign
        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        d_pred = torch.log(M_probs.sum(axis=0) / M_probs.shape[0])
        density_term = self.lambda_d * self.kl_loss(d_pred, self.density_prior)

        total_loss = -expression_term - regularizer_term + density_term
        return total_loss, regularizer_term, gv_term, vg_term, density_term


class ContrastiveLoss(nn.Module):
    """Filtered siamese loss between spots and projected scRNA pseudo-spots.

    Row ``i`` of both inputs is a positive pair. The other pseudo-spots of the
    batch are negatives, penalised only when their cosine similarity to the
    spot is below ``sim_threshold``.
    """

    def __init__(self, sim_threshold=0.6):
        super().__init__()
        self.sim_threshold = sim_threshold

    def forward(self, scrna_batch_projection, spatial_batch):
        batchsize_spatial = spatial_batch.size(0)

        spatial_norm = F.normalize(spatial_batch, dim=1)
        scrna_proj_norm = F.normalize(scrna_batch_projection, dim=1)

        sim_pos = torch.sum(spatial_norm * scrna_proj_norm, dim=1)  # [B]
        sim_neg = torch.matmul(spatial_norm, scrna_proj_norm.T)  # [B, B]
        # Exclude the positives from the negative pool
        sim_neg.fill_diagonal_(1e5)

        losses = []
        pos_sim_vec = []
        neg_sim_vec = []
        number_of_negatives = []

        for i in range(batchsize_spatial):
            negatives = sim_neg[i]
            positive = sim_pos[i]
            pos_sim_vec.append(positive.detach())

            loss_pos = (1 - positive) ** 2
            loss_neg = torch.clamp(self.sim_threshold - negatives, min=0) ** 2
            loss_neg_scalar = loss_neg.mean()

            losses.append(loss_pos + loss_neg_scalar)

            selected = negatives[self.sim_threshold > negatives]
            if selected.numel() > 0:
                mean_val = torch.mean(selected)
            else:
                mean_val = torch.tensor(0.0, device=negatives.device)
            neg_sim_vec.append(mean_val.detach())
            number_of_negatives.append(selected.numel() * 1.0)

        loss = torch.stack(losses).mean()

        positives_avg_sim = torch.stack(pos_sim_vec).mean()
        negatives_avg_sim = torch.stack(neg_sim_vec).mean()
        number_negatives_avg = torch.tensor(number_of_negatives).float().mean().round()

        return loss, positives_avg_sim.item(), negatives_avg_sim.item(), number_negatives_avg.item()


class ContinualLoss(nn.Module):
    """EWC-style penalty keeping M close to its Phase-2 solution."""

    def __init__(self, lambda_ewc, importance_matrix, M_reference):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.register_buffer("importance_matrix", importance_matrix, persistent=False)
        self.register_buffer("M_reference", M_reference, persistent=False)

    def forward(self, M_current):
        loss = torch.sum(self.importance_matrix * (M_current - self.M_reference).pow(2))
        return (self.lambda_ewc / 2.0) * loss
