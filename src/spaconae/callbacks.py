"""Stopping criteria used by the training phases."""

import lightning as L


class LossPlateauStopping(L.Callback):
    """Stop when the monitored loss stops improving (alignment phase).

    An epoch counts as an improvement when the loss drops below the best value
    by more than ``min_delta``, or by more than ``abs(best) * min_delta`` with
    ``relative=True``. Training stops after ``patience`` consecutive epochs
    without improvement.
    """

    def __init__(self, monitor="total", patience=5, min_delta=0.0, relative=False):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.relative = relative
        self.best_loss = None
        self.counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics[self.monitor])

        if self.best_loss is None:
            self.best_loss = val_loss
            return

        # abs(): the alignment loss is negative for most of the training
        delta = abs(self.best_loss) * self.min_delta if self.relative else self.min_delta

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
