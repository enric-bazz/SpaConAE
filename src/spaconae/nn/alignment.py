"""Tangram-style cell-to-spot alignment matrix."""

import torch
import torch.nn.functional as F
from torch import nn

# Key under which the notebooks stored the matrix in ``ewc_alignment*.pt``.
LEGACY_KEY = "AlignmentMatrix.M"


class AlignmentMatrix(nn.Module):
    """Learnable logits ``M`` of shape ``[n_cells, n_spots]``.

    A row-wise softmax turns each row into a distribution of one cell over all
    spots. Rows and columns are indexed by *global* cell and spot ids.
    """

    def __init__(self, n_cells=None, n_spots=None, initial_state=None):
        super().__init__()
        if initial_state is None:
            if n_cells is None or n_spots is None:
                raise ValueError("Provide either (n_cells, n_spots) or initial_state")
            initial_state = torch.randn(n_cells, n_spots, dtype=torch.float32)
        self.M = nn.Parameter(initial_state.clone().to(torch.float32))

    @property
    def n_cells(self):
        return self.M.shape[0]

    @property
    def n_spots(self):
        return self.M.shape[1]

    def get_M_full(self):
        return F.softmax(self.M, dim=1)

    def get_M_batch(self, cell_ids, spot_ids):
        # Softmax over ALL spots first, then slice: the batch block keeps the
        # global normalisation.
        return self.get_M_full()[cell_ids][:, spot_ids]

    def project(self, S):
        """Return ``(M_probs, M_probs^T @ S)`` for a full matrix ``S`` of cells."""
        M_probs = self.get_M_full()
        return M_probs, torch.matmul(M_probs.t(), S)
