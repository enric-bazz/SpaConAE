"""Phase 1: autoencoder initialisation (reconstruction + consistency)."""

import lightning as L
import torch
from torch import nn

from ..losses import consistence_loss
from ..model import CrossModalAE


class AEPretrainModule(L.LightningModule):
    def __init__(self, ae: CrossModalAE, lambda_recon=0.1, lambda_consistence=1.0, lr=1e-3):
        super().__init__()
        self.ae = ae
        self.lambda_recon = lambda_recon
        self.lambda_consistence = lambda_consistence
        self.lr = lr
        self.recon_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x_st, edge_index, spot_ids, _ = batch["st"]
        x_sc, _, _ = batch["sc"]

        # x_st holds the batch spots followed by their graph neighbours
        n_seeds = spot_ids.numel()
        z_sc, z_st, sc_recon, st_recon = self.ae(x_sc, x_st, edge_index, n_seeds)
        x_st = x_st[:n_seeds]

        recon_sc = self.recon_loss(x_sc, sc_recon)
        recon_st = self.recon_loss(x_st, st_recon)
        consistence = consistence_loss(x_sc, x_st, z_sc, z_st, self.ae.SC_decoder, self.ae.ST_decoder)

        loss = self.lambda_recon * (recon_sc + recon_st) + self.lambda_consistence * consistence

        self.log_dict(
            {"total": loss, "recon_sc": recon_sc, "recon_st": recon_st, "consistence": consistence},
            on_step=False, on_epoch=True, prog_bar=True, batch_size=x_st.size(0),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
