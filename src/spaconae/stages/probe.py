"""Downstream evaluation: label transfer from scRNA-seq to spatial data."""

import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ..model import CrossModalAE
from ..nn import Classifier


class LabelTransferModule(L.LightningModule):
    """Classifier trained on frozen scRNA-seq latents, tested on spatial latents."""

    def __init__(self, ae: CrossModalAE, n_classes, classifier_hidden=64, lr=1e-3, class_names=None):
        super().__init__()
        self.SC_encoder = ae.SC_encoder
        self.ST_encoder = ae.ST_encoder
        for p in self.SC_encoder.parameters():
            p.requires_grad = False
        for p in self.ST_encoder.parameters():
            p.requires_grad = False

        self.classifier = Classifier(d_input=ae.d_latent, d_hidden=classifier_hidden, n_classes=n_classes)
        self.lr = lr
        self.n_classes = n_classes
        self.class_names = None if class_names is None else [str(c) for c in class_names]

        self._val_outputs = []
        self._test_outputs = []
        self.test_report = None

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        logits = self.classifier(self.SC_encoder(x))
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True,
                      prog_bar=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        logits = self.classifier(self.SC_encoder(x))
        self.log("val_loss", F.cross_entropy(logits, y), on_epoch=True, prog_bar=True, batch_size=len(y))
        self._val_outputs.append((logits.argmax(1).cpu(), y.cpu()))

    def on_validation_epoch_end(self):
        preds = torch.cat([p for p, _ in self._val_outputs])
        labels = torch.cat([y for _, y in self._val_outputs])
        self._val_outputs.clear()
        self.log("val_acc", accuracy_score(labels, preds), prog_bar=True)
        self.log("val_f1", f1_score(labels, preds, average="macro"))

    def test_step(self, batch, batch_idx):
        x, edge_index, _, y = batch
        logits = self.classifier(self.ST_encoder(x, edge_index, n_seeds=len(y)))
        self._test_outputs.append((logits.argmax(1).cpu(), y.cpu()))

    def on_test_epoch_end(self):
        preds = torch.cat([p for p, _ in self._test_outputs])
        labels = torch.cat([y for _, y in self._test_outputs])
        self._test_outputs.clear()
        self.log("test_acc", accuracy_score(labels, preds))
        self.log("test_f1", f1_score(labels, preds, average="macro"))
        self.test_report = classification_report(
            labels, preds, labels=list(range(self.n_classes)),
            target_names=self.class_names, zero_division=0,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
