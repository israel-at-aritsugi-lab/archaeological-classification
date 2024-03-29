"""
__author__: Abhishek Thakur
"""

import numpy as np
import torch

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        self.val_score = np.Inf if self.mode == "min" else -np.Inf

        if self.tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )

    def __call__(self, epoch_score, model, model_path):
        score = -1.0 * epoch_score if self.mode == "min" else np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            message = f"Early stopping counter: {self.counter}/{self.patience}"

            if self.tpu:
                xm.master_print(message)
            else:
                print(message)

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            message = f"Validation score improved: {self.val_score} --> {epoch_score}. Saving model!"

            if self.tpu:
                xm.master_print(message)
            else:
                print(message)

            if self.tpu:
                xm.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score
