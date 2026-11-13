"""MLP decoders (identical architecture for both modalities)."""

from torch import nn

from .encoders import _init_linear_normal


class _MLPDecoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.model = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_input),
        )
        _init_linear_normal(self)

    def forward(self, z):
        return self.model(z)


class DecoderSC(_MLPDecoder):
    """Decoder for scRNA-seq."""


class DecoderST(_MLPDecoder):
    """Decoder for spatial transcriptomics."""
