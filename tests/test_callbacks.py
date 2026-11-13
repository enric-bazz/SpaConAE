"""Tests of the stopping criteria."""

from types import SimpleNamespace

from spaconae.callbacks import LossPlateauStopping, NegativeCountStopping


def _run(callback, losses):
    """Feed a sequence of epoch losses; return the epoch at which training stops."""
    trainer = SimpleNamespace(callback_metrics={}, should_stop=False)
    for epoch, loss in enumerate(losses, start=1):
        trainer.callback_metrics["total"] = loss
        callback.on_train_epoch_end(trainer, None)
        if trainer.should_stop:
            return epoch
    return None


def test_plateau_stops_after_patience_epochs_without_improvement():
    stopper = LossPlateauStopping(patience=3, min_delta=0.0)
    assert _run(stopper, [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]) == 5


def test_plateau_keeps_going_while_improving():
    stopper = LossPlateauStopping(patience=3, min_delta=0.01, relative=True)
    assert _run(stopper, [1.0 * 0.9**i for i in range(50)]) is None


def test_relative_delta_with_negative_loss():
    # improvements of 0.1% of |best| are below the 1% threshold
    stopper = LossPlateauStopping(patience=2, min_delta=0.01, relative=True)
    assert _run(stopper, [-1.0, -1.001, -1.002, -1.003]) == 3
    # improvements of 10% are above it
    stopper = LossPlateauStopping(patience=2, min_delta=0.01, relative=True)
    assert _run(stopper, [-1.0, -1.1, -1.21, -1.34]) is None


def test_negative_count_stopping():
    stopper = NegativeCountStopping(patience=2)
    trainer = SimpleNamespace(should_stop=False)
    for n_neg in (5, 0, 3, 0, 0):
        stopper.on_train_epoch_end(trainer, SimpleNamespace(last_number_neg=n_neg))
    assert trainer.should_stop
