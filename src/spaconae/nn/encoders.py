"""Modality-specific encoders.

Attribute names (``model``, ``pre_fc``, ``gat_layer``, ``post_fc``) match the
notebooks so that the original ``.pth`` state dicts load unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


def _init_linear_normal(module: nn.Module, std: float = 0.1) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            torch.nn.init.normal_(m.bias, mean=0, std=std)


class EncoderSC(nn.Module):
    """MLP encoder for scRNA-seq (one hidden layer more than stAI)."""

    def __init__(self, d_input, d_hidden, d_latent, dropout=0.0):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )
        _init_linear_normal(self)

    def forward(self, x):
        return self.model(x)


class GATLayer(nn.Module):
    def __init__(self, d_input, num_heads=4, n_layers=1, dropout=0.0, residual=True):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList(
            [GATConv(d_input, d_input, heads=num_heads, concat=False) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.residual:
            resid = x
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = self.dropout(F.leaky_relu(x))
        if self.residual:
            x += resid
        return x


class EncoderST(nn.Module):
    """MLP -> GAT (residual) -> linear encoder for spatial data.

    Unlike the other modules, its linear layers keep the PyTorch default
    initialisation, as in the notebooks.
    """

    def __init__(self, d_input, d_hidden, d_latent, num_heads=1, n_layers=1, dropout=0.0, residual=True):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(d_input, d_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent, bias=True),
        )
        self.post_fc = nn.Linear(d_latent, d_latent, bias=True)
        self.gat_layer = GATLayer(
            d_input=d_latent, num_heads=num_heads, n_layers=n_layers, dropout=dropout, residual=residual
        )

    def forward(self, x, edge_index):
        z = self.pre_fc(x)
        z = self.gat_layer(z, edge_index)
        z = self.post_fc(z)
        return z
