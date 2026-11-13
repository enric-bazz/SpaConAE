"""Phase 3: contrastive fine-tuning of the autoencoders (and optionally of M)."""

import lightning as L
import torch
from torch import nn

from ..losses import ContinualLoss, ContrastiveLoss, consistence_loss
from ..model import CrossModalAE
from ..nn import AlignmentMatrix


class ContrastiveModule(L.LightningModule):
    """The ``scCLR`` model of the notebooks.

    scRNA-seq latents of the batch are projected onto the spots of the batch
    with the corresponding block of M; each pseudo-spot is the positive of its
    spot. With ``continual_alignment=True`` M keeps training under an EWC
    penalty, otherwise it is frozen.
    """

    def __init__(
        self,
        ae: CrossModalAE,
        initial_matrix,
        importance_matrix=None,
        continual_alignment=False,
        lambda_recon=1.0,
        lambda_contrast=1.0,
        lambda_ewc=1.0,
        lambda_consistence=1.0,
        sim_threshold=0.8,
        lr=1e-3,
    ):
        super().__init__()
        self.ae = ae
        self.lambda_recon = lambda_recon
        self.lambda_contrast = lambda_contrast
        self.lambda_ewc = lambda_ewc
        self.lambda_consistence = lambda_consistence
        self.continual_alignment = continual_alignment
        self.lr = lr

        self.AlignMat = AlignmentMatrix(initial_state=initial_matrix)
        if not continual_alignment:
            for param in self.AlignMat.parameters():
                param.requires_grad = False
        elif importance_matrix is None:
            raise ValueError("Importance matrix missing for continual learning")

        self.recon_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss(sim_threshold)
        if continual_alignment:
            self.continual_loss = ContinualLoss(lambda_ewc, importance_matrix, initial_matrix)
        else:
            self.continual_loss = None

        # number of selected negatives in the last batch, read by the stopping callback
        self.last_number_neg = None

    def training_step(self, batch, batch_idx):
        x_st, edge_index, spot_ids, _ = batch["st"]
        x_sc, cell_ids, _ = batch["sc"]

        z_sc, z_st, sc_recon, st_recon = self.ae(x_sc, x_st, edge_index)

        recon_sc = self.recon_loss(x_sc, sc_recon)
        recon_st = self.recon_loss(x_st, st_recon)
        consistence = consistence_loss(x_sc, x_st, z_sc, z_st, self.ae.SC_decoder, self.ae.ST_decoder)

        M_full = self.AlignMat.get_M_full()  # [n_cells, n_spots]

        if self.continual_loss is not None:
            # NOTE: as in the notebook, the penalty receives the softmaxed
            # matrix while the reference holds logits.
            continual = self.continual_loss(M_full)
        else:
            continual = torch.zeros((), device=x_st.device)

        # Pseudo-spots: batch cells projected onto batch spots
        M_batch = M_full[cell_ids][:, spot_ids]  # [B_sc, B_st]
        z_sc_proj = torch.matmul(M_batch.t(), z_sc)  # [B_st, d_latent]

        contrastive, pos_avg, neg_avg, number_neg = self.contrastive_loss(z_sc_proj, z_st)
        self.last_number_neg = number_neg

        loss = (
            self.lambda_recon * (recon_sc + recon_st)
            + self.lambda_contrast * contrastive
            + self.lambda_ewc * continual
            + self.lambda_consistence * consistence
        )

        self.log_dict(
            {
                "total": loss,
                "recon": recon_sc + recon_st,
                "consistence": consistence,
                "contrastive": contrastive,
                "continual": continual,
                "pos_sim": pos_avg,
                "neg_sim": neg_avg,
                "n_negatives": number_neg,
            },
            on_step=False, on_epoch=True, prog_bar=True, batch_size=x_st.size(0),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
