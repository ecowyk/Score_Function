"""Validation-only plateau scheduling and early stopping.

Checkpoint ranking uses the exact loss minimum separately. This controller uses
a relative improvement threshold to avoid resetting patience for tiny changes.
"""

import math


class ValidationPlateau:
    def __init__(
        self, *, factor, lr_patience, stop_patience, relative_improvement, minimum_learning_rate
    ):
        if not 0 < factor < 1 or not 0 <= relative_improvement < 1:
            raise ValueError("Invalid plateau factor or improvement threshold")
        if not 1 <= lr_patience < stop_patience or minimum_learning_rate <= 0:
            raise ValueError("Invalid patience or minimum learning rate")
        self.factor = factor
        self.lr_patience = lr_patience
        self.stop_patience = stop_patience
        self.relative_improvement = relative_improvement
        self.minimum_learning_rate = minimum_learning_rate
        self.best = None
        self.bad_checks = 0
        self.lr_bad_checks = 0
        self.reductions = 0

    def observe(self, loss, learning_rate):
        if not math.isfinite(loss) or learning_rate <= 0:
            raise ValueError("Nonfinite loss or invalid learning rate")
        improved = self.best is None or loss < self.best * (1 - self.relative_improvement)
        if improved:
            self.best = loss
            self.bad_checks = self.lr_bad_checks = 0
        else:
            self.bad_checks += 1
            self.lr_bad_checks += 1
        stop = self.bad_checks >= self.stop_patience
        reduced = False
        if not stop and self.lr_bad_checks >= self.lr_patience:
            new_lr = max(self.minimum_learning_rate, learning_rate * self.factor)
            reduced = new_lr < learning_rate
            learning_rate = new_lr
            self.reductions += int(reduced)
            self.lr_bad_checks = 0
        return {
            "learning_rate": learning_rate,
            "stop": stop,
            "significant_improvement": improved,
            "bad_checks": self.bad_checks,
            "lr_reduced": reduced,
            "reductions": self.reductions,
        }
