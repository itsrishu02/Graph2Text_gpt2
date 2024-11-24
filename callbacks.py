import logging
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from utils import save_json
from pytorch_lightning.utilities import rank_zero_info

def count_trainable_parameters(model):
    """Counts the number of trainable parameters in the model."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)


class Seq2SeqLoggingCallback(pl.Callback):
    """Callback for logging and saving metrics during training and evaluation."""

    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True):
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics

        # Filter out unwanted keys
        new_metrics = {k: v for k, v in metrics.items() if all(exclude not in k for exclude in ["log", "progress_bar", "preds"])}

        # Log metrics
        trainer.logger.log_metrics(new_metrics)
        results_file = Path(pl_module.hparams.output_dir) / (f"{type_path}_results.txt" if type_path == "test" else f"{type_path}_results/{trainer.global_step:05d}.txt")
        generations_file = Path(pl_module.hparams.output_dir) / (f"{type_path}_generations.txt" if type_path == "test" else f"{type_path}_generations/{trainer.global_step:05d}.txt")

        results_file.parent.mkdir(exist_ok=True, parents=True)
        generations_file.parent.mkdir(exist_ok=True, parents=True)

        # Write metrics to file
        with open(results_file, "a+") as writer:
            for key, val in sorted(new_metrics.items()):
                try:
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    writer.write(f"{key}: {val:.6f}\n")
                except Exception:
                    pass

        if save_generations and "preds" in metrics:
            generations_file.write_text("\n".join(metrics["preds"]))

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        for key in sorted(metrics):
            if key not in ["log", "progress_bar", "preds"]:
                rank_zero_info(f"{key} = {metrics[key]}\n")


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Returns a ModelCheckpoint callback for saving the best model based on the specified metric."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(f"Supported metrics are rouge2, bleu, and loss. Got: {metric}")

    # Using `filepath` instead of `dirpath` for compatibility with older PyTorch Lightning versions
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
    )
    return checkpoint_callback



def get_early_stopping_callback(metric="bleu", patience=3):
    """Returns an EarlyStopping callback based on the specified metric and patience."""
    return EarlyStopping(
        monitor=f"val_{metric}",
        mode="max" if metric != "loss" else "min",
        patience=patience,
        verbose=True,
    )
