"""Stopping criteria used by the training phases."""

import lightning as L


class LossPlateauStopping(L.Callback):
    """Stop when the monitored loss stops improving (alignment phase).

    An epoch counts as an improvement when the loss drops by more than
    ``min_delta`` (or ``best * min_delta`` with ``relative=True``).
    """

    def __init__(self, monitor="total", patience=5, min_delta=0.0, relative=False):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.relative = relative
        self._reset()

    def _reset(self):
        self.best_loss = float("inf")
        self.counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # NOTE: as in the notebook, where the stopper was instantiated inside
        # the training loop, its state is reset at every epoch.
        self._reset()

        val_loss = float(trainer.callback_metrics[self.monitor])
        delta = self.best_loss * self.min_delta if self.relative else self.min_delta

        if val_loss < self.best_loss - delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True


class NegativeCountStopping(L.Callback):
    """Stop the contrastive phase after ``patience`` epochs without negatives.

    Reads the number of selected negatives of the last batch of the epoch.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.last_number_neg == 0:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
        else:
            self.counter = 0
