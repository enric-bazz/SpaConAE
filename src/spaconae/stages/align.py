"""Phase 2: full-batch initialisation of the alignment matrix."""

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..losses import AlignmentMatrixLoss
from ..model import CrossModalAE
from ..nn import LEGACY_KEY, AlignmentMatrix


class AlignmentModule(L.LightningModule):
    """Learns M so that ``M^T S`` reproduces the spatial data ``G``.

    With ``encode_data=True`` S and G are the latent representations given by
    the encoders of ``ae``; otherwise they are the expression matrices on the
    shared genes (as in Tangram). One optimisation step uses the whole
    datasets, so an epoch is a single step.
    """

    def __init__(
        self,
        ae: CrossModalAE,
        full_scrna,
        full_spatial,
        full_edge_index,
        encode_data=True,
        lambda_g1=1.0,
        lambda_g2=1.0,
        lambda_r=1.0,
        lambda_d=1.0,
        lr=1e-3,
    ):
        super().__init__()
        self.encode_data = encode_data
        self.lr = lr

        if encode_data:
            # NOTE: as in the notebook, the datasets are encoded here, at
            # construction time, with the encoders in their current state and
            # in training mode.
            with torch.no_grad():
                S = ae.SC_encoder(full_scrna)
                G = ae.ST_encoder(full_spatial, full_edge_index)
        else:
            S, G = full_scrna, full_spatial

        # Static inputs; not stored in checkpoints
        self.register_buffer("S", S.to(torch.float32), persistent=False)
        self.register_buffer("G", G.to(torch.float32), persistent=False)

        n_cells, n_spots = full_scrna.shape[0], full_spatial.shape[0]
        self.alignment = AlignmentMatrix(n_cells=n_cells, n_spots=n_spots)
        self.alignment_loss = AlignmentMatrixLoss(n_spots, lambda_g1, lambda_g2, lambda_r, lambda_d)

    def compute_loss(self):
        M_probs, G_pred = self.alignment.project(self.S)
        return self.alignment_loss(M_probs, G_pred, self.G)

    def training_step(self, batch, batch_idx):
        loss, entropy_reg, gv, vg, density = self.compute_loss()
        lam = self.alignment_loss
        self.log_dict(
            {
                "total": loss,
                # reported without their coefficients
                "entropy_reg": entropy_reg / lam.lambda_r if lam.lambda_r else entropy_reg,
                "genes_voxels": gv / lam.lambda_g1 if lam.lambda_g1 else gv,
                "voxels_genes": vg / lam.lambda_g2 if lam.lambda_g2 else vg,
                "density": density / lam.lambda_d if lam.lambda_d else density,
            },
            on_step=False, on_epoch=True, prog_bar=True, batch_size=1,
        )
        return loss

    def train_dataloader(self):
        # Full-batch optimisation: a single dummy item per epoch
        return DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_importance(self):
        """EWC importance of M: squared gradient of the alignment loss."""
        self.eval()
        self.zero_grad()
        with torch.enable_grad():
            loss, *_ = self.compute_loss()
            loss.backward()
        importance = self.alignment.M.grad.detach().pow(2)
        self.zero_grad()
        return importance

    def ewc_state(self):
        """Importance and optimal logits, in the format of the notebooks."""
        return {
            "importance": {LEGACY_KEY: self.compute_importance().cpu()},
            "optimal_params": {LEGACY_KEY: self.alignment.M.detach().clone().cpu()},
        }
