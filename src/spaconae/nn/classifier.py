"""MLP classifier used for the downstream label-transfer probe."""

from torch import nn


class Classifier(nn.Module):
    def __init__(self, d_input, d_hidden, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        return self.model(x)
