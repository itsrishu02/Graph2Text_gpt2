#!/usr/bin/env python

import argparse
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import tempfile
import json

from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import (
    Seq2SeqDataset,
    Seq2SeqDataCollator,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    save_json,
)
from utils_graph2text import convert_text, eval_meteor, eval_bleu, eval_chrf, eval_meteor_test_webnlg, eval_chrf_test_webnlg
from lightning_base import BaseTransformer, add_generic_args, generic_train

logger = logging.getLogger(__name__)

import re
def clean_predictions(predictions):
    """
    Cleans predictions by removing repetitive phrases, correcting formatting,
    and ensuring proper sentence structure.

    Args:
        predictions (list or str): Either a list of prediction strings or a single string.

    Returns:
        list or str: Cleaned predictions in the same format as the input.
    """
    def clean_single(prediction):
        if not prediction:
            return ""

        # Use a safe regex for repeated words or phrases
        prediction = re.sub(r'\b(\w+)( \1\b)+', r'\1', prediction)

        # Remove excessive spaces
        prediction = re.sub(r'\s+', ' ', prediction)

        # Remove stray punctuations
        prediction = re.sub(r'\s+([.,;!?])', r'\1', prediction)

        # Capitalize the first letter of the sentence if not already done
        prediction = prediction.strip()
        if prediction and not prediction[0].isupper():
            prediction = prediction[0].upper() + prediction[1:]

        return prediction

    # If input is a list, clean each string in the list
    if isinstance(predictions, list):
        return [clean_single(pred) for pred in predictions]

    # If input is a single string, clean it directly
    return clean_single(predictions)



class Graph2TextModule(BaseTransformer):
    mode = "graph2text"
    loss_names = ["loss"]
    metric_names = ["bleu", "meteor", "chrf"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, mode=self.mode, **kwargs)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.dataset_class = Seq2SeqDataset
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "additional_special_tokens": ["<H>", "<R>", "<T>", "<|startoftext|>", "<|endoftext|>"]}
        )
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.eval_beams = self.hparams.eval_beams or 1
        self.eval_max_length = self.hparams.eval_max_gen_length or 384

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

    def calc_generative_metrics(self, preds, target) -> dict:
        meteor_score = eval_meteor(preds, target)
        chrf_score = float(eval_chrf(preds, target))
        bleu_score = float(eval_bleu(preds, target))
        return {"bleu": bleu_score, "meteor": meteor_score, "chrf": chrf_score}

    def _generative_step(self, batch: dict) -> dict:
        """Generate outputs and calculate metrics."""
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.eval_max_length,
            num_beams=self.eval_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds = clean_predictions(preds)

        filtered_labels = torch.where(
            batch["labels"] != -100,
            batch["labels"],
            torch.tensor(self.tokenizer.pad_token_id).to(batch["labels"].device),
        )
        target = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        generative_metrics = self.calc_generative_metrics(preds, target)
        base_metrics.update(generative_metrics)
        return base_metrics


    def validation_epoch_end(self, outputs):
        """Log validation metrics."""
        preds = flatten_list([x["preds"] for x in outputs])
        targets = flatten_list([x["target"] for x in outputs])
        metrics = self.calc_generative_metrics(preds, targets)
        print(f"Validation Metrics: {metrics}")

        self.metrics = metrics

        # Convert metrics to torch.Tensors for compatibility with PyTorch Lightning
        bleu_tensor = torch.tensor(metrics["bleu"], dtype=torch.float32)
        meteor_tensor = torch.tensor(metrics["meteor"], dtype=torch.float32)
        chrf_tensor = torch.tensor(metrics["chrf"], dtype=torch.float32)

        if self.trainer.is_global_zero:
            # Use self.logger to log metrics
            if self.logger:
                self.logger.experiment.add_scalar("val_bleu", metrics["bleu"], self.current_epoch)
                self.logger.experiment.add_scalar("val_meteor", metrics["meteor"], self.current_epoch)
                self.logger.experiment.add_scalar("val_chrf", metrics["chrf"], self.current_epoch)

        # Update callback_metrics with tensors (optional, if required for compatibility)
        self.trainer.callback_metrics.update({
            "val_bleu": bleu_tensor,
            "val_meteor": meteor_tensor,
            "val_chrf": chrf_tensor,
        })

        # Return metrics as tensors
        return {
            "val_bleu": bleu_tensor,
            "val_meteor": meteor_tensor,
            "val_chrf": chrf_tensor,
        }


    def test_epoch_end(self, outputs):
        """Log test metrics."""
        preds = flatten_list([x["preds"] for x in outputs])
        targets = flatten_list([x["target"] for x in outputs])

        metrics = self.calc_generative_metrics(preds, targets)
        print(f"Test Metrics: {metrics}")

        metrics_save_path = Path(self.hparams.output_dir) / "test_metrics.json"
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Test metrics saved to {metrics_save_path}")

        predictions_save_path = Path(self.hparams.output_dir) / "test_predictions.json"
        with open(predictions_save_path, "w") as f:
            json.dump({"predictions": preds, "targets": targets}, f, indent=4)
        print(f"Predictions and targets saved to {predictions_save_path}")

        return {"metrics": metrics}

    def _step(self, batch: dict) -> tuple:
        """Calculate the loss for training or evaluation."""
        loss, _ = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return (loss,)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        dataset = Seq2SeqDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            type_path=type_path,
        )
        collator = Seq2SeqDataCollator(
            tokenizer=self.tokenizer,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test_seen", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        return parser



def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = Graph2TextModule(args)
    if args.do_train:
        generic_train(
            model,
            args,
            logging_callback=Seq2SeqLoggingCallback(),
            checkpoint_callback=get_checkpoint_callback(
                args.output_dir, model.default_val_metric, save_top_k=args.save_top_k
            ),
            early_stopping_callback=get_early_stopping_callback(
                model.default_val_metric, args.early_stopping_patience
            ),
        )

    if args.do_predict:
        trainer = pl.Trainer.from_argparse_args(args, max_epochs=2)
        if args.test_checkpoint_path:
            print(f"Loading checkpoint from: {args.test_checkpoint_path}")
            model = model.load_from_checkpoint(args.test_checkpoint_path)
        trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_generic_args(parser, os.getcwd())
    parser = Graph2TextModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
