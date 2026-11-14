"""Core network: two modality-specific autoencoders sharing a latent space."""

from pathlib import Path

import torch
from torch import nn

from .nn import DecoderSC, DecoderST, EncoderSC, EncoderST

# File names used by the notebooks; kept so existing checkpoints load as is.
_WEIGHT_FILES = {
    "ST_encoder": "spatial_encoder_weights{suffix}.pth",
    "SC_encoder": "scrna_encoder_weights{suffix}.pth",
    "ST_decoder": "spatial_decoder_weights{suffix}.pth",
    "SC_decoder": "scrna_decoder_weights{suffix}.pth",
}


class CrossModalAE(nn.Module):
    """scRNA-seq MLP autoencoder + spatial GAT autoencoder.

    Both encoders map the shared genes to a latent space of size ``d_latent``.
    Submodule names follow the notebooks (``ST_encoder``, ``SC_encoder``, ...).
    """

    def __init__(self, d_input, d_hidden=64, d_latent=32, dropout=0.1):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_latent = d_latent

        self.ST_encoder = EncoderST(
            d_input=d_input, d_hidden=d_hidden, d_latent=d_latent,
            num_heads=1, n_layers=1, dropout=dropout, residual=True,
        )
        self.ST_decoder = DecoderST(d_latent=d_latent, d_hidden=d_hidden, d_input=d_input)
        self.SC_encoder = EncoderSC(d_input=d_input, d_hidden=d_hidden, d_latent=d_latent, dropout=dropout)
        self.SC_decoder = DecoderSC(d_latent=d_latent, d_hidden=d_hidden, d_input=d_input)

    def forward(self, x_sc, x_st, edge_index_st, n_seeds=None):
        """Encode and reconstruct both modalities (spatial: see ``EncoderST.forward``)."""
        z_sc = self.SC_encoder(x_sc)
        z_st = self.ST_encoder(x_st, edge_index_st, n_seeds)
        return z_sc, z_st, self.SC_decoder(z_sc), self.ST_decoder(z_st)

    @torch.no_grad()
    def encode(self, x_sc, x_st, edge_index_st):
        """Latents of both modalities, without gradients."""
        return self.SC_encoder(x_sc), self.ST_encoder(x_st, edge_index_st)

    def save_weights(self, directory, suffix=""):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, pattern in _WEIGHT_FILES.items():
            torch.save(getattr(self, name).state_dict(), directory / pattern.format(suffix=suffix))

    def load_weights(self, directory, suffix="", decoders=True, map_location="cpu"):
        """Load encoder (and optionally decoder) weights saved by ``save_weights``."""
        directory = Path(directory)
        for name, pattern in _WEIGHT_FILES.items():
            if name.endswith("decoder") and not decoders:
                continue
            state = torch.load(directory / pattern.format(suffix=suffix), map_location=map_location)
            getattr(self, name).load_state_dict(state)
        return self
